# ============================================================
#  verifier.py  —  Hallucination detection via Groq LLM
#
#  Uses Groq's llama-3.3-70b — extremely fast (~500 tok/s)
#  Verifies each claim in the answer against retrieved chunks.
# ============================================================

import re
import json
from dataclasses import dataclass, field
from typing import List, Tuple

from groq import Groq

from .config import config
from .ingestor import Chunk


@dataclass
class ClaimVerification:
    claim:           str
    is_grounded:     bool
    confidence:      float
    supporting_chunk_id: str | None
    supporting_text: str | None
    reason:          str


@dataclass
class VerificationResult:
    answer:               str
    claims:               List[ClaimVerification]
    grounding_score:      float
    is_reliable:          bool
    ungrounded_claims:    List[str]
    should_refuse:        bool
    refusal_reason:       str
    retrieval_similarity: float
    warning:              str


class HallucinationVerifier:
    """
    Verifies answer claims against retrieved evidence using Groq LLM.
    Groq runs llama-3.3-70b at ~500 tokens/sec — verification is fast.
    """

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Get a free key at: https://console.groq.com\n"
                "Then: export GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=config.GROQ_API_KEY)

    # ── Public ───────────────────────────────────────────────

    def verify(
        self,
        answer:  str,
        chunks:  List[Tuple[Chunk, float]],
        query:   str,
        max_sim: float,
    ) -> VerificationResult:

        # Refusal check
        if max_sim < config.REFUSAL_SIMILARITY_THRESHOLD:
            return self._build_refusal(
                answer, [],
                reason=(
                    f"No relevant content found in the document "
                    f"(similarity={max_sim:.2f}, threshold={config.REFUSAL_SIMILARITY_THRESHOLD}). "
                    f"This question may be outside the scope of the uploaded document."
                ),
                max_sim=max_sim,
            )

        claims_text = self._split_into_claims(answer)
        if not claims_text:
            return self._build_refusal(answer, [], "Answer could not be parsed.", max_sim)

        context       = self._format_context(chunks)
        verifications = self._verify_claims_batch(claims_text, context, chunks)

        grounded_count  = sum(1 for v in verifications if v.is_grounded)
        total           = len(verifications)
        grounding_score = (grounded_count / total * 100) if total > 0 else 0.0
        is_reliable     = grounding_score >= (config.GROUNDING_THRESHOLD * 100)
        ungrounded      = [v.claim for v in verifications if not v.is_grounded]
        warning         = self._build_warning(grounding_score, ungrounded, is_reliable)

        return VerificationResult(
            answer               = answer,
            claims               = verifications,
            grounding_score      = round(grounding_score, 1),
            is_reliable          = is_reliable,
            ungrounded_claims    = ungrounded,
            should_refuse        = False,
            refusal_reason       = "",
            retrieval_similarity = max_sim,
            warning              = warning,
        )

    # ── Claim splitting ───────────────────────────────────────

    def _split_into_claims(self, text: str) -> List[str]:
        """Split answer into individual verifiable claims.

        Handles prose sentences, numbered lists (1. ...), and bullet points (- ...).
        """
        claims = []

        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip list markers: "1.", "2)", "-", "•", "*"
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line).strip()
            if not line:
                continue
            # Split remaining line into sentences
            for sent in re.split(r'(?<=[.!?])\s+', line):
                sent = sent.strip()
                if len(sent) >= 20:
                    claims.append(sent)

        # Fallback: treat whole text as one block if no claims extracted
        if not claims:
            for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
                sent = sent.strip()
                if len(sent) >= 20:
                    claims.append(sent)

        return claims[:12]

    # ── Batch verification ────────────────────────────────────

    def _verify_claims_batch(
        self, claims: List[str], context: str, chunks: List[Tuple[Chunk, float]]
    ) -> List[ClaimVerification]:
        results = []
        batch_size = config.VERIFICATION_BATCH_SIZE
        for i in range(0, len(claims), batch_size):
            batch = claims[i : i + batch_size]
            results.extend(self._call_groq_verifier(batch, context, chunks))
        return results

    def _call_groq_verifier(
        self,
        claims:  List[str],
        context: str,
        chunks:  List[Tuple[Chunk, float]],
    ) -> List[ClaimVerification]:

        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

        prompt = f"""You are a strict fact-checker. Verify whether each claim is directly supported by the context.

CONTEXT:
{context}

CLAIMS TO VERIFY:
{numbered}

Respond with ONLY a JSON array. Each element:
{{
  "claim_index": <int, 1-based>,
  "is_grounded": <bool>,
  "confidence": <float 0.0-1.0>,
  "supporting_text": <exact quote from context, or "">,
  "reason": <one sentence>
}}

Rules:
- is_grounded = true ONLY if context explicitly states or clearly implies the claim
- If unsure → false (prefer caution over hallucination)
- supporting_text must be a direct quote, not a paraphrase
- No extra text outside the JSON array"""

        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            data    = json.loads(raw)
            results = []

            for item in data:
                idx         = item.get("claim_index", 1) - 1
                if not (0 <= idx < len(claims)):
                    continue
                is_grounded = bool(item.get("is_grounded", False))
                sup_text    = item.get("supporting_text", "")
                sup_chunk   = self._find_supporting_chunk(sup_text, chunks) if (sup_text and is_grounded) else None

                results.append(ClaimVerification(
                    claim               = claims[idx],
                    is_grounded         = is_grounded,
                    confidence          = float(item.get("confidence", 0.5)),
                    supporting_chunk_id = sup_chunk,
                    supporting_text     = sup_text if is_grounded else None,
                    reason              = item.get("reason", ""),
                ))

            return results

        except Exception as e:
            return [
                ClaimVerification(
                    claim=c, is_grounded=False, confidence=0.0,
                    supporting_chunk_id=None, supporting_text=None,
                    reason=f"Verification error: {str(e)[:80]}"
                )
                for c in claims
            ]

    # ── Helpers ──────────────────────────────────────────────

    def _format_context(self, chunks: List[Tuple[Chunk, float]]) -> str:
        parts = []
        for i, (chunk, score) in enumerate(chunks):
            parts.append(
                f"[Passage {i+1} | {chunk.doc_name} p.{chunk.page_num+1} | "
                f"relevance={score:.2f}]\n{chunk.text}"
            )
        return "\n\n".join(parts)

    def _find_supporting_chunk(self, sup_text: str, chunks) -> str | None:
        for chunk, _ in chunks:
            if sup_text[:50].lower() in chunk.text.lower():
                return chunk.chunk_id
        return None

    def _build_warning(self, score: float, ungrounded: List[str], reliable: bool) -> str:
        if reliable:
            return ""
        n = len(ungrounded)
        return (
            f"⚠️ Low confidence answer (grounding score: {score:.0f}%). "
            f"{n} claim{'s' if n != 1 else ''} could not be verified. "
            f"Treat with caution."
        )

    def _build_refusal(self, answer, claims, reason, max_sim) -> VerificationResult:
        return VerificationResult(
            answer="", claims=[], grounding_score=0.0,
            is_reliable=False, ungrounded_claims=[],
            should_refuse=True, refusal_reason=reason,
            retrieval_similarity=max_sim, warning="",
        )
