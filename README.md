# RAG Document Q&A System

> Ask questions about your own documents. Get answers with sources. No hallucination.

![CI](https://github.com/icy2701/RAG-DOCUMENT-QA/actions/workflows/ci.yml/badge.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/aisi27/rag-document-qa?logo=docker)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-enabled-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/icy2701/RAG-DOCUMENT-QA)

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

---

## About

Standard language models answer from training memory. They cannot read your private files and they hallucinate — giving confident but wrong answers with no evidence.

This project solves that. It is a Retrieval-Augmented Generation (RAG) system that:
- Takes your own documents as input
- Searches them mathematically by meaning, not keywords
- Generates answers grounded in your actual content
- Returns the exact source file for every answer so you can verify it

Built from scratch as a learning project covering the full ML engineering stack — from data ingestion to a containerised, tested, CI/CD-backed REST API.

---

## Features

- Ask questions about your documents in plain English
- Answers sourced directly from your files — no hallucination
- Upload new documents live via API without restarting the server
- Source attribution on every answer — see exactly which file was used
- Interactive API docs auto-generated at /docs
- Fully containerised with Docker
- Automated tests with pytest
- CI pipeline runs on every push

---

## Architecture

User sends question via POST /ask
|
FastAPI server (src/main.py)
|
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
converts question to 384-number vector
|
FAISS Index
searches 24+ vectors for top 3 closest matches
returns most semantically relevant chunks
|
Prompt Template:
"Answer based on this context: {chunks}
Question: {question}"
|
google/flan-t5-base
reads prompt, generates answer from context only
|
{ "answer": "...", "sources": ["data/ml_basics.txt", ...] }

Live upload pipeline (POST /upload):
User uploads .txt file
|
Saved to data/
|
Chunked into 500-char pieces (50-char overlap)
|
Embedded into vectors
|
FAISS.merge_from() — merged into live index
No server restart needed
|
{ "chunks_added": N, "total_vectors": N }

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | Orchestration — document loading, chunking, vector store |
| FAISS | Vector database — stores and searches embeddings |
| sentence-transformers all-MiniLM-L6-v2 | Text to 384-number meaning vectors |
| google/flan-t5-base | Free local language model — no API key needed |
| FastAPI | REST API framework |
| Uvicorn | ASGI web server |
| Pydantic | Input validation and schema definition |
| pytest | Automated testing |
| Docker | Containerisation |
| GitHub Actions | CI/CD pipeline |

---

## Prerequisites

Make sure you have these installed before starting:

- Python 3.11 or higher — python.org/downloads
- Git — git-scm.com
- Docker Desktop (for Docker usage) — docker.com/products/docker-desktop

---

## Installation

```bash
git clone https://github.com/icy2701/RAG-DOCUMENT-QA.git
cd RAG-DOCUMENT-QA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/vectorstore.py
```

The last command builds the FAISS index from the sample documents. It downloads the embedding model (~80MB) on first run.

---

## Usage

Start the server:

```bash
uvicorn src.main:app --reload
```

The server loads models on startup (~30 seconds first time). When you see:

All models ready. Server accepting requests.
INFO: Uvicorn running on http://127.0.0.1:8000

Open http://127.0.0.1:8000/docs to use the interactive API.

---

## API Reference

### GET /health
Returns server status.

Response:
```json
{"status": "ok"}
```

---

### POST /ask
Ask a question about your documents.

Request body:
```json
{
  "question": "What is overfitting?",
  "top_k": 3
}
```

Response:
```json
{
  "answer": "Overfitting occurs when a model learns the training data too well, including its noise, and performs poorly on new unseen data.",
  "sources": [
    {"source": "data/ml_basics.txt"},
    {"source": "data/ml_basics.txt"},
    {"source": "data/ml_basics.txt"}
  ]
}
```

---

### POST /upload
Upload a new .txt document and index it live.

Request: multipart/form-data with a .txt file

Response:
```json
{
  "message": "Document 'neural_networks.txt' uploaded and indexed successfully.",
  "chunks_added": 8,
  "total_vectors": 32
}
```

---

## Running Tests

```bash
pytest tests/test_api.py -v
```

Expected output:
tests/test_api.py::test_health_returns_200 PASSED
tests/test_api.py::test_ask_returns_200 PASSED
tests/test_api.py::test_ask_response_has_answer_key PASSED
tests/test_api.py::test_ask_response_has_sources_key PASSED
tests/test_api.py::test_ask_rejects_missing_question PASSED
5 passed

---

## Docker

Pull and run from Docker Hub:

```bash
docker pull aisi27/rag-document-qa:v1.0
docker run -p 8000:8000 aisi27/rag-document-qa:v1.0
```

Or build locally:

```bash
docker build -t rag-document-qa .
docker run -p 8000:8000 rag-document-qa
```

---

## Project Structure
rag-document-qa/
├── src/
│   ├── ingest.py          # document loading and chunking
│   ├── vectorstore.py     # FAISS index builder
│   ├── rag_chain.py       # full RAG pipeline
│   └── main.py            # FastAPI server
├── data/
│   ├── python_programming.txt
│   ├── ml_basics.txt
│   └── rptu_specialisations.txt
├── tests/
│   └── test_api.py
├── faiss_index/           # generated — do not edit manually
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
└── README.md

