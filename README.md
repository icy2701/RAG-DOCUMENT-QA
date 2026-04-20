# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions using your own documents as the knowledge base — not from the AI's memory, from YOUR files.

## What is RAG?

Think of it like an open-book exam. A regular AI answers from memory (closed-book). RAG lets the AI open the book first:

1. Index — split documents into chunks and convert each chunk into a vector (embedding) that captures its meaning
2. Retrieve — when a question comes in, find the chunks whose meaning is closest to the question
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

- LangChain — orchestration framework that connects all components
- FAISS — vector database for fast similarity search
- sentence-transformers (all-MiniLM-L6-v2) — converts text into 384-number vectors
- PyTorch + HuggingFace Transformers — neural network engine
- FastAPI + Uvicorn — HTTP API layer
- pytest — testing

## Setup

git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Test Embeddings

python3 test_embed.py
Expected output: (384,)

## Progress

### Day 1 — Project Setup
- Created folder structure: src/, data/, tests/, docs/
- Set up virtual environment and installed all dependencies
- Created 3 sample documents covering Python, ML basics, and RPTU specialisations
- Verified embedding technology works — SentenceTransformer produces (384,) vectors
- Initialised Git repository and pushed to GitHub

### Day 2 — Document Loading and Chunking (src/ingest.py)
- Loads all .txt files from data/ using LangChain's DirectoryLoader
- Splits each document into chunks of 500 characters with 50 character overlap
- 3 documents split into 24 chunks, average size 345 characters
- Chunk overlap ensures sentences at boundaries are not lost between chunks
- Metadata (source filename) is preserved in every chunk for traceability

### Day 3 — FAISS Vector Store (src/vectorstore.py)
- Loads the all-MiniLM-L6-v2 embedding model from HuggingFace (~80MB, cached after first run)
- Converts all 24 chunks into 384-number vectors using HuggingFaceEmbeddings
- Stores all vectors in a FAISS index — a catalogue system for meaning-based search
- Saves the index to disk in faiss_index/ so it persists between runs
- Tested retrieval with "what is machine learning?" — all 3 results correctly returned from ml_basics.txt
- k=3 means return the 3 nearest vectors to the query vector
- FAISS always returns k results even for irrelevant queries — similarity thresholding to be added later

### Day 4 — flan-t5 LLM Integration (src/rag_chain.py)
- Loaded saved FAISS index from disk instead of rebuilding every run
- Integrated google/flan-t5-base as the language model — free, local, no API key needed
- Built the complete RAG pipeline in one ask() function:
  question → embed → FAISS retrieves top 3 chunks → format as context → prompt flan-t5 → answer
- Prompt template instructs flan-t5 to answer ONLY from provided context, not training memory
- Tested 5 questions — all retrieved correct sources (ML questions → ml_basics.txt, Python questions → python_programming.txt)
- Answers are short due to flan-t5 being a small model — improves with larger models later
- AutoTokenizer converts text to numbers, model.generate() thinks, tokenizer.decode() converts back to text
