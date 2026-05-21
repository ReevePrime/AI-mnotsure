# 🤔 AI-mnotsure

> *Most chatbots answer with confidence. This one answers — then immediately questions every life choice that led to that answer.*

---

## What is this?

**AI-mnotsure** is a RAG (Retrieval-Augmented Generation) chatbot with a built-in eval suite that measures hallucination, faithfulness, and retrieval quality. Because healthy self-doubt is a feature, not a bug.

This project was built as a portfolio piece for a role at **Moorepay**, demonstrating production-ready AI engineering practices including RAG pipelines, evaluation frameworks, and deployment.

---

## Features

- 💬 **RAG Chatbot** — Retrieval-augmented responses grounded in real documents
- 🔍 **Built-in Eval Suite** — Measures hallucination, faithfulness, and retrieval quality
- 📊 **Metrics Dashboard** — Visualise model self-doubt in real time
- 🧪 **Honest by Design** — Uncertainty is surfaced, not hidden

---

## Live Demo

👉 **[ai-mnotsure.streamlit.app](https://ai-mnotsure.streamlit.app/)**

---

## Evaluation Notes

> ⚠️ **Important caveat on metrics**
>
> Due to cost limitations with using OpenAI for testing with [DeepEval](https://docs.confident-ai.com/), the evaluation metrics were generated using a **local LLM (Qwen2.5:14b)**. Results may be less accurate than a production setup using OpenAI or Anthropic models. This repo is primarily a portfolio project.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| RAG Framework | LangChain / LlamaIndex |
| Evaluation | DeepEval |
| Local LLM | Qwen2.5:14b (via Ollama) |

---

## About This Project

This project was built specifically as a portfolio submission for a position at **Moorepay**. It showcases:

- End-to-end RAG pipeline design
- Automated evaluation and hallucination detection
- Practical deployment via Streamlit Cloud
- Honest documentation of limitations
