# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions using your own documents as the knowledge base.

## What is RAG?

Instead of relying on what an AI learned during training, RAG lets it search your documents first, then generate an answer based on what it finds. Think of it as an open-book exam — the AI looks up the answer rather than guessing from memory.

The pipeline works in three stages:
1. Index — split documents into chunks and convert each chunk into a vector (embedding) that captures its meaning
2. Retrieve — when a question comes in, find the chunks whose vectors are closest to the question's vector
3. Generate — feed those chunks to a language model so it answers using your actual content

## Project Structure

- src/ — application source code
- data/ — sample documents the system searches through
- tests/ — test files
- docs/ — documentation

## Sample Documents

- data/python_programming.txt — overview of Python as a language
- data/ml_basics.txt — core machine learning concepts and algorithms
- data/rptu_specialisations.txt — RPTU computer science specialisation tracks

## Tech Stack

- LangChain — orchestration framework
- FAISS — vector database for fast similarity search
- sentence-transformers (all-MiniLM-L6-v2) — text to vector conversion
- PyTorch + HuggingFace Transformers — neural network engine
- FastAPI + Uvicorn — HTTP API layer
- pytest — testing

## Setup

```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Test Embeddings

```bash
python3 test_embed.py
# Expected output: (384,)
```

