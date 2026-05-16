"""
Streamlit UI
"""

import streamlit as st
from chatbot import ask

##########################################################################
#                             CONFIG
##########################################################################

st.set_page_config(
    page_title="Moorepay Knowledge Assistant",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Moorepay Knowledge Assistant")
st.caption(
    "Ask anything about Moorepay. Answers are grounded in a curated source "
    "document — not the model's training data."
)

##########################################################################
#                                 INPUT
##########################################################################

question = st.text_input(
    label="Your question",
    placeholder="e.g. When was Moorepay founded?",
)

run = st.button("Ask", type="primary")


if run and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = ask(question)

    # Two-column layout:
    # We want the answer on the left,
    # and the retrieved chunks on the right
    col_answer, col_context = st.columns([3, 2])

    with col_answer:
        st.subheader("Answer")
        st.write(result["answer"])

        # Confidence indicator
        st.subheader("Retrieval Confidence")
        confidence = result["confidence"]
        st.progress(confidence)

        # Colour-coded label
        if confidence >= 0.65:
            st.success(f"{confidence:.0%} — Strong match in source document")
        elif confidence >= 0.40:
            st.warning(
                f"{confidence:.0%} — Partial match — answer may be incomplete")
        else:
            st.error(
                f"{confidence:.0%} — Weak match — treat answer with caution")

        st.caption(
            "Confidence reflects how closely the retrieved passages matched "
            "your query. It does not guarantee the answer is correct."
        )

    with col_context:
        st.subheader(f"Retrieved Chunks ({len(result['retrieved'])} of k=3)")

        if not result["retrieved"]:
            st.info("No chunks met the similarity threshold for this query.")
        else:
            for i, r in enumerate(result["retrieved"]):
                with st.expander(
                    f"Chunk {r['index']} — distance: {r['distance']:.3f}",
                    # we expand the top result by default for visibility
                    expanded=(i == 0)
                ):
                    st.write(r["text"])

elif run and not question.strip():
    st.warning("Please enter a question first.")
