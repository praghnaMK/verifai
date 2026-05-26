# ============================================================
#  main.py  —  VerifAI FastAPI REST API
#
#  Endpoints:
#  POST /v1/ingest        Upload and index a document
#  POST /v1/query         Ask a question with hallucination verification
#  GET  /v1/stats         Index stats
#  POST /v1/eval/run      Run benchmark evaluation
#  GET  /v1/health        Health check
# ============================================================

import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.core.rag_engine import VerifAIEngine
from backend.core.config import config

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(
    title="VerifAI",
    description="Self-verifying RAG engine with hallucination detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend at / ───────────────────────────────────────
_FRONTEND_HTML = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

@app.get("/", include_in_schema=False)
async def serve_ui():
    if not _FRONTEND_HTML.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(content=_FRONTEND_HTML.read_text(encoding="utf-8"))

# ── Singleton engine ──────────────────────────────────────────

engine = VerifAIEngine()

# Try loading existing index on startup
@app.on_event("startup")
async def startup():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    try:
        engine.load_index()
    except Exception:
        pass  # No index yet — created on first ingest

# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k:    Optional[int] = None

class ClaimResult(BaseModel):
    claim:           str
    is_grounded:     bool
    confidence:      float
    supporting_text: Optional[str]
    reason:          str

class QueryResponse(BaseModel):
    question:         str
    answer:           str
    grounding_score:  float
    is_reliable:      bool
    should_refuse:    bool
    refusal_reason:   str
    warning:          str
    claims:           List[ClaimResult]
    ungrounded_claims: List[str]
    retrieval_similarity: float
    latency_ms:       int
    sources:          List[dict]

class IngestResponse(BaseModel):
    doc_name:      str
    chunks:        int
    total_indexed: int
    status:        str

class StatsResponse(BaseModel):
    chunks_indexed: int
    docs_loaded:    List[str]
    is_ready:       bool

class EvalRequest(BaseModel):
    qa_pairs: List[dict]   # [{"question": str, "expected_answer": str}]

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "index_ready": engine.is_ready}


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF, TXT, or MD document and add it to the index.
    Supports multi-document ingestion — call multiple times.
    """
    allowed = {".pdf", ".txt", ".md"}
    suffix  = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}"
        )

    # Save to temp file
    tmp_path = Path(config.UPLOAD_DIR) / file.filename
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = engine.ingest(tmp_path)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Query the indexed documents.
    Returns answer + per-claim hallucination verification.
    """
    if not engine.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a document via POST /v1/ingest first."
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result       = engine.query(req.question, top_k=req.top_k)
    verification = result.verification

    # Format claims for response
    claims_out = [
        ClaimResult(
            claim           = c.claim,
            is_grounded     = c.is_grounded,
            confidence      = c.confidence,
            supporting_text = c.supporting_text,
            reason          = c.reason,
        )
        for c in verification.claims
    ]

    # Format source citations
    sources = [
        {
            "doc_name":  chunk.doc_name,
            "page":      chunk.page_num + 1,
            "score":     round(score, 3),
            "preview":   chunk.text[:120] + "...",
            "chunk_id":  chunk.chunk_id,
        }
        for chunk, score in result.retrieved_chunks[:5]
    ]

    return QueryResponse(
        question              = result.query,
        answer                = verification.answer if not verification.should_refuse else "",
        grounding_score       = verification.grounding_score,
        is_reliable           = verification.is_reliable,
        should_refuse         = verification.should_refuse,
        refusal_reason        = verification.refusal_reason,
        warning               = verification.warning,
        claims                = claims_out,
        ungrounded_claims     = verification.ungrounded_claims,
        retrieval_similarity  = round(verification.retrieval_similarity, 3),
        latency_ms            = result.latency_ms,
        sources               = sources,
    )


@app.get("/v1/stats", response_model=StatsResponse)
async def stats():
    """Return index statistics."""
    s = engine.stats
    return StatsResponse(
        chunks_indexed = s["chunks_indexed"],
        docs_loaded    = s["docs_loaded"],
        is_ready       = engine.is_ready,
    )


@app.post("/v1/eval/run")
async def run_eval(req: EvalRequest):
    """
    Run benchmark evaluation on a set of Q&A pairs.
    Returns precision, grounding rate, refusal rate, avg latency.
    """
    if not engine.is_ready:
        raise HTTPException(status_code=400, detail="No index loaded.")

    results         = []
    grounding_scores = []
    refusals        = 0
    latencies       = []

    for pair in req.qa_pairs:
        question = pair.get("question", "")
        if not question:
            continue

        result = engine.query(question)
        v      = result.verification

        grounding_scores.append(v.grounding_score)
        latencies.append(result.latency_ms)
        if v.should_refuse:
            refusals += 1

        results.append({
            "question":        question,
            "answer":          v.answer,
            "grounding_score": v.grounding_score,
            "is_reliable":     v.is_reliable,
            "should_refuse":   v.should_refuse,
            "latency_ms":      result.latency_ms,
            "ungrounded":      v.ungrounded_claims,
        })

    n = len(results)
    return {
        "total_questions":    n,
        "avg_grounding_score": round(sum(grounding_scores) / n, 1) if n else 0,
        "reliable_answers":   sum(1 for r in results if r["is_reliable"]),
        "refusal_count":      refusals,
        "avg_latency_ms":     round(sum(latencies) / n) if n else 0,
        "results":            results,
    }
