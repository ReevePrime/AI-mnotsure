"""
Streamlit UI — two-column layout.
Left:  WhatsApp-style chat with DeepEval quality badges on each response.
Right: observability panel — retrieved chunks ranked by similarity,
       per-chunk confidence bars, and a diagnostic label.
"""

import math
import asyncio
import streamlit as st
import anthropic
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from chatbot import ask

##########################################################################
#                             PAGE CONFIG
##########################################################################

st.set_page_config(
    page_title="Moorepay Knowledge Assistant",
    page_icon="💼",
    layout="wide",
)

##########################################################################
#                        DEEPEVAL JUDGE + METRICS
##########################################################################

class AnthropicJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._client = anthropic.Anthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        kwargs = {"system": "Respond only with valid JSON, no other text."} if schema else {}
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text

    async def a_generate(self, prompt: str, schema=None) -> str:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.model


@st.cache_resource
def _load_eval_stack():
    """Instantiated once per session and reused across all questions."""
    judge = AnthropicJudge("claude-haiku-4-5-20251001")
    return [
        ("Answer Relevancy",     AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)),
        ("Faithfulness",         FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)),
        ("Contextual Relevancy", ContextualRelevancyMetric(threshold=0.7, model=judge, include_reason=True)),
    ]


def evaluate_answer(question: str, answer: str, context_used: list[str]) -> dict:
    metrics = _load_eval_stack()
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context_used,
    )
    results = {}
    for name, metric in metrics:
        metric.measure(test_case)
        score = metric.score
        results[name] = {
            "score": score,
            "passed": score is not None and score >= metric.threshold,
            "reason": getattr(metric, "reason", None),
        }
    return results


##########################################################################
#                          RENDERING HELPERS
##########################################################################

def render_metric_badges(metrics: dict):
    """Compact three-card row shown beneath each assistant message."""
    cols = st.columns(3)
    for col, (name, data) in zip(cols, metrics.items()):
        score = data["score"]
        passed = data["passed"]
        score_str = f"{score:.2f}" if score is not None else "N/A"
        color = "#198754" if passed else "#dc3545"
        label = "PASS" if passed else "FAIL"
        short = name.replace("Contextual ", "Ctx. ")
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:5px 4px;border-radius:6px;"
                f"border:1px solid {color}33;background:{color}11'>"
                f"<div style='font-size:0.68em;color:#666;margin-bottom:2px'>{short}</div>"
                f"<div style='font-weight:700;font-size:0.82em;color:{color}'>{label}</div>"
                f"<div style='font-size:0.72em;color:#888'>{score_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def chunk_confidence(distance: float) -> float:
    return math.exp(-distance)


def render_chunk(rank: int, chunk: dict):
    conf = chunk_confidence(chunk["distance"])
    if conf >= 0.65:
        label, color = "Strong match", "#198754"
    elif conf >= 0.40:
        label, color = "Partial match", "#fd7e14"
    else:
        label, color = "Weak match", "#dc3545"

    with st.expander(
        f"#{rank}  Chunk {chunk['index']}  —  {conf:.0%}  ({label})",
        expanded=(rank == 1),
    ):
        st.progress(conf)
        st.markdown(
            f"<p style='margin:4px 0 8px;font-size:0.82em;"
            f"color:{color};font-weight:600'>{label}"
            f"&nbsp;·&nbsp;<span style='color:#888;font-weight:400'>"
            f"L2 distance: {chunk['distance']:.3f}</span></p>",
            unsafe_allow_html=True,
        )
        st.write(chunk["text"])


_PRIORITY = ["Contextual Relevancy", "Faithfulness", "Answer Relevancy"]

_DIAGNOSES = {
    "Contextual Relevancy": (
        "error",
        "**Root cause: Retrieval** — The retrieved chunks weren't relevant to the "
        "question. The answer may lack grounding in the source document.",
    ),
    "Faithfulness": (
        "error",
        "**Root cause: Generation** — The answer contains claims not supported by "
        "the retrieved context. Possible hallucination.",
    ),
    "Answer Relevancy": (
        "warning",
        "**Root cause: Relevance** — The answer doesn't fully address the question asked.",
    ),
}


def render_diagnostic(metrics: dict):
    failures = [k for k in _PRIORITY if k in metrics and not metrics[k]["passed"]]
    if not failures:
        st.success("**No issues detected** — all metrics passed.")
        return
    level, msg = _DIAGNOSES[failures[0]]
    getattr(st, level)(msg)
    if len(failures) > 1:
        st.caption(f"Also failing: {', '.join(failures[1:])}")


##########################################################################
#                           SESSION STATE INIT
##########################################################################

if "messages" not in st.session_state:
    # Each entry: {role, content, metrics?, retrieved?}
    st.session_state.messages = []
if "last_obs" not in st.session_state:
    # {result, metrics} for the observability panel; always the latest response
    st.session_state.last_obs = None

##########################################################################
#             PROCESS NEW INPUT (before rendering so it shows immediately)
##########################################################################

# st.chat_input at the top level → Streamlit pins it to the bottom of the page.
prompt = st.chat_input("Ask about Moorepay...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Retrieving and generating answer..."):
        result = ask(prompt)

    metrics = None
    if result["retrieved"]:
        with st.spinner("Evaluating answer quality (3 LLM-judge calls)..."):
            metrics = evaluate_answer(prompt, result["answer"], result["context_used"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "metrics": metrics,
        "retrieved": result["retrieved"],
    })
    st.session_state.last_obs = {"result": result, "metrics": metrics}

##########################################################################
#                               LAYOUT
##########################################################################

st.title("💼 Moorepay Knowledge Assistant")

col_chat, col_obs = st.columns([3, 2])

# ── LEFT: Chat ────────────────────────────────────────────────────────────
with col_chat:
    st.caption(
        "Answers are grounded in a curated source document — not the model's training data. "
        "Quality metrics are evaluated by an LLM judge after each response."
    )

    with st.container(height=560, border=False):
        if not st.session_state.messages:
            st.markdown(
                "<div style='color:#aaa;text-align:center;padding-top:80px'>"
                "Type a question below to get started.</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant" and msg.get("metrics"):
                        st.markdown(
                            "<div style='margin-top:10px'></div>",
                            unsafe_allow_html=True,
                        )
                        render_metric_badges(msg["metrics"])

# ── RIGHT: Observability ──────────────────────────────────────────────────
with col_obs:
    st.subheader("Observability")

    if st.session_state.last_obs is None:
        st.info("Ask a question to see retrieval details and diagnostics here.")
    else:
        obs = st.session_state.last_obs
        retrieved = obs["result"]["retrieved"]

        if not retrieved:
            st.warning(
                "No chunks were retrieved — the query may be out of scope "
                "or the similarity threshold filtered everything out."
            )
        else:
            st.markdown(
                f"**Retrieved chunks** &nbsp; "
                f"<span style='color:#888'>{len(retrieved)} / k=5 · ranked by similarity</span>",
                unsafe_allow_html=True,
            )
            for i, chunk in enumerate(retrieved):
                render_chunk(i + 1, chunk)

        if obs["metrics"]:
            st.divider()
            st.markdown("**Diagnostic**")
            render_diagnostic(obs["metrics"])
