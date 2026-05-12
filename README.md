# RAG Document Q&A System

> Ask questions about your own documents. Get answers with sources. No hallucination.

![CI](https://github.com/icy2701/RAG-DOCUMENT-QA/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/docker/image-size/aisi27/rag-document-qa/v1.0?logo=docker&label=docker)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-enabled-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/icy2701/RAG-DOCUMENT-QA)

---

Ask questions about your own documents and get answers with source attribution — built on LangChain, FAISS, and a local language model.

---

## What It Does

Upload `.txt` documents, ask questions in plain English, and get answers pulled directly from your files. Every response includes the source document it came from. New documents can be uploaded live without restarting the server.

---

## How It Works
question → embedding → FAISS retrieves top-k chunks → flan-t5 generates answer → response with sources

---

## Stack

LangChain · FAISS · sentence-transformers · flan-t5 · FastAPI · Docker · GitHub Actions

---

## Getting Started

```bash
git clone https://github.com/icy2701/RAG-DOCUMENT-QA.git
cd RAG-DOCUMENT-QA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/vectorstore.py   # builds the FAISS index from documents in data/
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API interactively.

---

## API

| Method | Endpoint | Description |
| GET | /health | Server status |
| POST | /ask | Ask a question, get an answer with sources |
| POST | /upload | Upload a .txt file and index it live |

**POST /ask**
```json
{ "question": "What is overfitting?", "top_k": 3 }
```
```json
{
  "answer": "Overfitting occurs when a model learns the training data too well...",
  "sources": [{ "source": "data/ml_basics.txt" }]
}
```

---

## Tests

```bash
pytest tests/test_api.py -v
```

---

## Docker

```bash
docker pull aisi27/rag-document-qa:v1.0
docker run -p 8000:8000 aisi27/rag-document-qa:v1.0
```

---
