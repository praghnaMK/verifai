"""
dashboard/app.py  —  VerifAI Streamlit Dashboard

Features:
- Document upload and ingestion
- Query interface with real-time grounding score
- Per-claim verification with colour-coded highlights
- Source citation panel
- Benchmark eval runner
"""

import sys
import json
import time
import requests
import streamlit as st
import plotly.graph_objects as go

API_BASE = "http://localhost:8000/v1"

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VerifAI — Self-Verifying RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .grounded-claim {
    background: #E1F5EE; border-left: 4px solid #1D9E75;
    padding: 8px 12px; margin: 4px 0; border-radius: 0 8px 8px 0;
    font-size: 14px; line-height: 1.6;
  }
  .ungrounded-claim {
    background: #FCEBEB; border-left: 4px solid #E24B4A;
    padding: 8px 12px; margin: 4px 0; border-radius: 0 8px 8px 0;
    font-size: 14px; line-height: 1.6;
  }
  .uncertain-claim {
    background: #FAEEDA; border-left: 4px solid #BA7517;
    padding: 8px 12px; margin: 4px 0; border-radius: 0 8px 8px 0;
    font-size: 14px; line-height: 1.6;
  }
  .source-card {
    background: #F8F9FA; border: 1px solid #E0E0E0;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
    font-size: 13px;
  }
  .score-high   { color: #1D9E75; font-weight: 600; font-size: 28px; }
  .score-medium { color: #BA7517; font-weight: 600; font-size: 28px; }
  .score-low    { color: #E24B4A; font-weight: 600; font-size: 28px; }
  .refused-box  {
    background: #FFF3CD; border: 1px solid #BA7517;
    border-radius: 8px; padding: 14px 18px; margin: 10px 0;
  }
  .reliable-badge   { background:#E1F5EE; color:#085041; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .unreliable-badge { background:#FCEBEB; color:#791F1F; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 VerifAI")
    st.caption("Self-Verifying RAG Engine")
    st.divider()

    # API health check
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            st.success("API connected")
            if data.get("index_ready"):
                stats = requests.get(f"{API_BASE}/stats").json()
                st.info(
                    f"**Index ready**\n\n"
                    f"{stats['chunks_indexed']} chunks indexed\n\n"
                    f"Docs: {', '.join(stats['docs_loaded']) or 'none'}"
                )
            else:
                st.warning("No documents indexed yet")
        else:
            st.error("API error")
    except Exception:
        st.error("Cannot reach API. Run: `uvicorn backend.api.main:app --reload`")

    st.divider()

    # Document upload
    st.subheader("Upload Document")
    uploaded = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        help="Document will be chunked, embedded, and indexed for querying."
    )

    if uploaded and st.button("Index Document", type="primary"):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                r = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=120,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.success(
                        f"✅ Indexed **{data['doc_name']}**\n\n"
                        f"{data['chunks']} chunks · "
                        f"{data['total_indexed']} total"
                    )
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Ingestion failed"))
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption(
        "VerifAI v1.0 · Dual BM25+FAISS retrieval · "
        "Claim-level hallucination detection · "
        "Built by Praghna M.K"
    )

# ─────────────────────────────────────────────────────────────
#  Main area — tabs
# ─────────────────────────────────────────────────────────────

tab_query, tab_eval, tab_about = st.tabs(["🔎 Query", "📊 Benchmark", "ℹ️ About"])

# ── Query tab ─────────────────────────────────────────────────

with tab_query:
    st.header("Ask a Question")
    st.caption(
        "VerifAI retrieves relevant passages, generates an answer, then "
        "verifies each sentence against the source — flagging anything ungrounded."
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. What methodology was used in the study?",
        height=80,
    )

    col_btn, col_topk = st.columns([2, 1])
    with col_btn:
        ask_btn = st.button("Ask VerifAI", type="primary", use_container_width=True)
    with col_topk:
        top_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=6)

    if ask_btn and question.strip():
        with st.spinner("Retrieving → Generating → Verifying..."):
            try:
                r = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
                if r.status_code != 200:
                    st.error(r.json().get("detail", "Query failed"))
                    st.stop()

                data = r.json()

            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # ── Refusal ──────────────────────────────────────────
        if data["should_refuse"]:
            st.markdown(f"""
            <div class="refused-box">
              <strong>🚫 VerifAI refused to answer</strong><br>
              {data['refusal_reason']}
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Metrics row ──────────────────────────────────
            score = data["grounding_score"]
            score_class = (
                "score-high"   if score >= 75 else
                "score-medium" if score >= 50 else
                "score-low"
            )
            badge = (
                '<span class="reliable-badge">✓ Reliable</span>'
                if data["is_reliable"]
                else '<span class="unreliable-badge">⚠ Low confidence</span>'
            )

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Grounding Score", f"{score:.0f}%")
            mc2.metric("Claims Verified", len(data["claims"]))
            mc3.metric(
                "Grounded",
                sum(1 for c in data["claims"] if c["is_grounded"])
            )
            mc4.metric("Latency", f"{data['latency_ms']}ms")

            # ── Warning ──────────────────────────────────────
            if data["warning"]:
                st.warning(data["warning"])

            st.divider()

            # ── Two-column layout ────────────────────────────
            col_ans, col_src = st.columns([3, 2])

            with col_ans:
                st.subheader("Answer")
                st.markdown(f"*{badge}*", unsafe_allow_html=True)
                st.write(data["answer"])

                st.subheader("Claim-Level Verification")
                st.caption(
                    "🟢 Grounded in document  "
                    "🔴 Ungrounded / hallucinated  "
                    "🟡 Uncertain"
                )

                for claim in data["claims"]:
                    if claim["is_grounded"]:
                        cls   = "grounded-claim"
                        icon  = "✅"
                        extra = (
                            f'<br><small style="color:#085041;">'
                            f'Supporting: "{claim["supporting_text"][:100]}..."</small>'
                            if claim["supporting_text"] else ""
                        )
                    elif claim["confidence"] > 0.3:
                        cls   = "uncertain-claim"
                        icon  = "⚠️"
                        extra = f'<br><small style="color:#633806;">{claim["reason"]}</small>'
                    else:
                        cls   = "ungrounded-claim"
                        icon  = "❌"
                        extra = f'<br><small style="color:#791F1F;">{claim["reason"]}</small>'

                    st.markdown(
                        f'<div class="{cls}">{icon} {claim["claim"]}{extra}</div>',
                        unsafe_allow_html=True
                    )

            with col_src:
                st.subheader("Sources Retrieved")

                # Grounding gauge
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = score,
                    title = {"text": "Grounding Score"},
                    delta = {"reference": 75, "suffix": "%"},
                    gauge = {
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#1D9E75" if score >= 75 else "#BA7517" if score >= 50 else "#E24B4A"},
                        "steps": [
                            {"range": [0,  50], "color": "#FCEBEB"},
                            {"range": [50, 75], "color": "#FAEEDA"},
                            {"range": [75, 100],"color": "#E1F5EE"},
                        ],
                        "threshold": {
                            "line":  {"color": "#378ADD", "width": 3},
                            "thickness": 0.8,
                            "value": 75,
                        },
                    },
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                for src in data["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                      <strong>{src['doc_name']}</strong> · Page {src['page']}
                      <span style="float:right; color:#888; font-size:11px;">
                        relevance {src['score']:.0%}
                      </span><br>
                      <small style="color:#555;">{src['preview']}</small>
                    </div>
                    """, unsafe_allow_html=True)

# ── Benchmark tab ─────────────────────────────────────────────

with tab_eval:
    st.header("Benchmark Evaluation")
    st.caption(
        "Test VerifAI on a set of questions to measure grounding accuracy, "
        "refusal rate, and average latency."
    )

    sample_qa = [
        {"question": "What is the main topic of this document?"},
        {"question": "Who are the authors or contributors mentioned?"},
        {"question": "What conclusions are drawn?"},
        {"question": "What year was this written? (test for refusal if not in doc)"},
        {"question": "What is the capital of France? (out-of-scope test)"},
    ]

    qa_input = st.text_area(
        "Q&A pairs (JSON array)",
        value=json.dumps(sample_qa, indent=2),
        height=200,
    )

    if st.button("Run Benchmark", type="primary"):
        try:
            qa_pairs = json.loads(qa_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON format.")
            st.stop()

        with st.spinner(f"Running {len(qa_pairs)} questions..."):
            r = requests.post(
                f"{API_BASE}/eval/run",
                json={"qa_pairs": qa_pairs},
                timeout=300,
            )

        if r.status_code != 200:
            st.error(r.json().get("detail", "Eval failed"))
            st.stop()

        data = r.json()

        # ── Summary metrics ──────────────────────────────────
        em1, em2, em3, em4 = st.columns(4)
        em1.metric("Avg Grounding", f"{data['avg_grounding_score']:.0f}%")
        em2.metric("Reliable Answers", f"{data['reliable_answers']}/{data['total_questions']}")
        em3.metric("Refusals", data["refusal_count"])
        em4.metric("Avg Latency", f"{data['avg_latency_ms']}ms")

        st.divider()

        # ── Per-question results ─────────────────────────────
        for res in data["results"]:
            score = res["grounding_score"]
            color = "#1D9E75" if score >= 75 else "#BA7517" if score >= 50 else "#E24B4A"

            with st.expander(
                f"{'🚫' if res['should_refuse'] else '✅' if res['is_reliable'] else '⚠️'} "
                f"{res['question'][:80]}  —  {score:.0f}%"
            ):
                if res["should_refuse"]:
                    st.info("VerifAI refused to answer (out-of-scope or insufficient evidence)")
                else:
                    st.write(res["answer"])
                    if res["ungrounded"]:
                        st.warning(f"Ungrounded claims: {res['ungrounded']}")
                    st.caption(f"Grounding: {score:.0f}% · Latency: {res['latency_ms']}ms")

# ── About tab ─────────────────────────────────────────────────

with tab_about:
    st.header("How VerifAI Works")

    st.markdown("""
    **VerifAI** is a self-verifying RAG (Retrieval-Augmented Generation) engine
    that detects its own hallucinations at the claim level.

    ### The problem with standard RAG
    Most RAG systems answer confidently even when wrong.
    They cite sources whether or not those sources actually support the claim.
    Hallucination rates in production RAG systems range from 17–33% (Stanford/Cornell Law, 2025).

    ### What VerifAI does differently

    **1. Dual retrieval (BM25 + FAISS + RRF)**
    - BM25 catches exact keyword matches (names, technical terms)
    - FAISS catches semantic similarity (paraphrases, synonyms)
    - Reciprocal Rank Fusion merges both — reduces hallucination by ~40% vs single retrieval

    **2. Claim-level citation verification**
    - After generating an answer, a second LLM pass checks each sentence
    - "Is this claim directly supported by the retrieved passages?"
    - Flags every sentence that drifts from the source

    **3. Grounding score (0–100)**
    - Every answer gets a score: % of claims grounded in the document
    - Below 60% → flagged as low confidence
    - Below similarity threshold → graceful refusal

    **4. Graceful refusal**
    - If the question can't be answered from the document, VerifAI says so
    - Instead of hallucinating, it explains what's missing

    ### Stack
    - **Backend**: Python · FastAPI · LangChain
    - **Retrieval**: FAISS (dense) · BM25 (sparse) · Reciprocal Rank Fusion
    - **LLM**: Groq — llama-3.3-70b (~500 tok/s, free tier)
    - **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (local, no API key)
    - **Dashboard**: Streamlit · Plotly
    - **Deployment**: Docker Compose

    ---
    Built by **Praghna M.K** · [github.com/praghnaMK](https://github.com/praghnaMK)
    """)
