# src/chatbot.py

# The RAG pipeline (not the UI)
# Returns a structured result dict.

import math
import anthropic
from dotenv import load_dotenv
from retriever import load_index, load_chunks, retrieve

load_dotenv()
client = anthropic.Anthropic()

LLM_MODEL = "claude-haiku-4-5-20251001"

##########################################################################
#                          PROMPT TEMPLATE
##########################################################################

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about
Moorepay, a UK payroll and HR software company.

Answer using ONLY the context provided. Do not use outside knowledge.
If the answer is not present in the context, respond with:
"I don't have enough information in the provided context to answer that."

Be concise and factual."""


def build_prompt(question: str, chunks: list[str]) -> str:
    context_block = "\n---\n".join(chunks)
    return f"""CONTEXT:
---
{context_block}
---

QUESTION:
{question}"""


##########################################################################
#                        CONFIDENCE EVALUATION
##########################################################################

def mean_confidence(results: list[dict]) -> float:
    """
    Returns a float between 0.0 and 1.0.
      - Below 0.5: strong semantic match
      - 0.5 – 1.0: related but not tightly matched
      - Above 1.5: probably not very relevant
    """
    if not results:
        return 0.0
    scores = [math.exp(-r["distance"]) for r in results]
    return round(sum(scores) / len(scores), 3)


##########################################################################
#                          MAIN PIPELINE
##########################################################################

# We keep load_index() and load_chunks() outside the main function
# So they load once at runtime and app.py doesn't reload them on every question.
_index = load_index()
_chunks = load_chunks()


def ask(question: str) -> dict:
    """
    RAG pipeline. We return a dict with the answer and some metadata about the retrieval.
    The dict contains "retrieved" (the chunks returned by the retriever),
    "confidence" (a score from 0.0 to 1.0 based on the distances of the retrieved chunks),
    and "context_used" (the text of the retrieved chunks, for prompt building and debugging
    """

    # We retrieve the relevant chunks
    results = retrieve(question, _index, _chunks)

    # If nothing relevant was found, we STATE it clearly in the answer and skip the LLM step entirely to save tokens.
    # This is to prevent the LLM from trying to answer based on empty/irrelevant context (hallucination).
    if not results:
        return {
            "answer": "I don't have enough information in the provided context to answer that.",
            "retrieved": [],
            "confidence": 0.0,
            "context_used": []
        }

    # We build the prompt with retrieved chunks as context
    context_texts = [r["text"] for r in results]
    user_message = build_prompt(question, context_texts)

    # Then we call the LLM
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
        # WE KEEP IT TEMPERATURE AT 0 TO ENSURE NO RANDOMNESS
        # Also at 0 the LLM will always produce the same answer for the same input.
        temperature=0,
    )
    answer = response.content[0].text.strip()

    # We return the structured result
    return {
        "answer": answer,
        "retrieved": results,
        "confidence": mean_confidence(results),
        "context_used": context_texts
    }
