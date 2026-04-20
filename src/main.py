import os
import sys
import shutil

# Add src/ to Python's path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager

# FastAPI is the web framework
from fastapi import FastAPI, UploadFile, File, HTTPException

# UploadFile — FastAPI's type for receiving uploaded files over HTTP
# File       — used as a default value to tell FastAPI this parameter
#              is a file upload, not a regular form field
# HTTPException — lets us return proper error responses with status codes
#                 for example 400 Bad Request if someone uploads a PDF

# BaseModel for defining the shape of JSON input
from pydantic import BaseModel

# LangChain components — same as before
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# flan-t5 components — same as before
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── GLOBAL VARIABLES ─────────────────────────────────────────────────────────

# These are None at start and get filled during the lifespan startup
# They stay in memory for the entire life of the server
# Every endpoint reads from these — no reloading per request
embeddings = None
vectorstore = None
tokenizer = None
model = None

# ── LIFESPAN ──────────────────────────────────────────────────────────────────

# Everything before yield runs on startup
# Everything after yield runs on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, vectorstore, tokenizer, model

    print("Server starting — loading models...")

    # Load embedding model — must match the one used to build the index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    # Build path to saved FAISS index
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

    # Load the saved FAISS index from disk
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"FAISS index loaded with {vectorstore.index.ntotal} vectors.")

    # Load flan-t5 tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print("flan-t5 loaded.")

    print("All models ready. Server accepting requests.")
    yield
    print("Server shutting down.")

# ── CREATE APP ────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)

# ── INPUT SCHEMA ──────────────────────────────────────────────────────────────

# Defines the required shape of a /ask request body
# question is required, top_k defaults to 3
class Question(BaseModel):
    question: str
    top_k: int = 3

# ── HEALTH ENDPOINT ───────────────────────────────────────────────────────────

# Simple liveness check — returns 200 if server is running
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── ASK ENDPOINT ──────────────────────────────────────────────────────────────

@app.post("/ask")
def ask(question_input: Question):

    # Convert question to vector, search FAISS for top_k closest chunks
    results = vectorstore.similarity_search(
        question_input.question,
        k=question_input.top_k
    )

    # Join retrieved chunks into one context block
    context = "\n\n".join([r.page_content for r in results])

    # Build the prompt — instruct flan-t5 to answer from context only
    prompt = f"""Answer the question in detail using the context below. Write at least 3 sentences.
Context: {context}
Question: {question_input.question}
Detailed Answer:"""

    # Tokenize the prompt into numbers the model understands
    # truncation=True handles flan-t5's 512 token input limit
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate the answer — model thinks here
    outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode the output numbers back to readable English
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract source filenames from chunk metadata
    sources = [r.metadata for r in results]

    return {
        "answer": answer,
        "sources": sources
    }

# ── UPLOAD ENDPOINT ───────────────────────────────────────────────────────────

# @app.post("/upload") creates a new endpoint at /upload
# UploadFile is FastAPI's type for receiving files over HTTP
# File(...) tells FastAPI this is a required file upload field
# The whole function is async because file reading is I/O bound —
# async lets the server handle other requests while waiting for the file
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # We use global because we need to modify the vectorstore variable
    # that was set during startup — without global we'd create a new
    # local variable and the global one would stay unchanged
    global vectorstore

    # ── VALIDATION ────────────────────────────────────────────────────────────

    # Check that the uploaded file is a .txt file
    # file.filename is the original name of the file the user uploaded
    # We only support .txt for now — PDFs and Word docs come later
    # HTTPException with status_code=400 means "Bad Request" —
    # the server understood the request but the input is invalid
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported. Please upload a plain text file."
        )

    # ── SAVE FILE TO DISK ─────────────────────────────────────────────────────

    # Build the path where we'll save the uploaded file
    # We save it into data/ so it's alongside our existing documents
    # os.path.basename() extracts just the filename from any path
    # preventing path traversal attacks like "../../etc/passwd"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(BASE_DIR, "data", os.path.basename(file.filename))

    # Read the file bytes from the upload and write them to disk
    # await file.read() reads the entire file content as bytes
    # "wb" means write binary — correct for any file type
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    print(f"File saved to {save_path}")

    # ── CHUNK THE NEW DOCUMENT ────────────────────────────────────────────────

    # Load the saved file using TextLoader
    # We load just this one file — not the whole data/ folder
    # This is much faster than reprocessing all documents
    loader = TextLoader(save_path, encoding="utf-8")
    new_docs = loader.load()

    # Split into chunks using same settings as our original pipeline
    # chunk_size=500 and chunk_overlap=50 must match what we used before
    # so new chunks are consistent with existing ones in the index
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    new_chunks = splitter.split_documents(new_docs)
    print(f"New document split into {len(new_chunks)} chunks.")

    # ── EMBED AND MERGE INTO FAISS ────────────────────────────────────────────

    # Build a brand new small FAISS index from just the new chunks
    # FAISS.from_documents() embeds all new chunks and creates a new index
    # We use the same embeddings model as the main index —
    # both indexes must use the same model to be in the same vector space
    new_store = FAISS.from_documents(new_chunks, embeddings)
    print(f"New FAISS store built with {new_store.index.ntotal} vectors.")

    # merge_from() adds all vectors from new_store into vectorstore
    # The existing vectors in vectorstore are completely untouched
    # The new vectors are appended to the end
    # This is like adding new pages to an existing filing cabinet
    # without disturbing any of the existing pages
    vectorstore.merge_from(new_store)
    print(f"Merged. Total vectors now: {vectorstore.index.ntotal}")

    # ── SAVE UPDATED INDEX TO DISK ────────────────────────────────────────────

    # Save the updated index back to disk
    # This ensures the new document survives a server restart
    # Without this, the merged vectors exist in memory only —
    # if the server restarts it would load the old index without the new doc
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
    vectorstore.save_local(FAISS_PATH)
    print("Updated FAISS index saved to disk.")

    # Return a success response with details about what was processed
    return {
        "message": f"Document '{file.filename}' uploaded and indexed successfully.",
        "chunks_added": len(new_chunks),
        "total_vectors": vectorstore.index.ntotal
    }