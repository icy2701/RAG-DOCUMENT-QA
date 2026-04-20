import os
import sys

# Add src/ to Python's path so it can find our other files
# os.path.abspath(__file__) → full path of this file
# os.path.dirname(...)      → goes up to src/ folder
# sys.path.append(...)      → adds src/ to Python's search list
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# asynccontextmanager lets us write the startup and shutdown logic
# of our server in one single function using yield
# Everything before yield = startup, everything after yield = shutdown
from contextlib import asynccontextmanager

# FastAPI is the framework that creates our web server
# It handles receiving requests and sending responses over HTTP
from fastapi import FastAPI

# BaseModel lets us define exactly what shape incoming data must be
# If someone sends wrong data, FastAPI automatically rejects it
# before it even reaches our code
from pydantic import BaseModel

# Same imports we used in rag_chain.py
# We need these to load our models and FAISS index when server starts
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── GLOBAL VARIABLES ─────────────────────────────────────────────────────────

# We declare these as None now and fill them during startup
# They live outside all functions so every endpoint can access them
# Think of these as empty shelves — the lifespan function puts models on them
# once at startup and they stay there for the entire life of the server
embeddings = None
vectorstore = None
tokenizer = None
model = None

# ── LIFESPAN FUNCTION ─────────────────────────────────────────────────────────

# @asynccontextmanager turns this function into a lifespan manager
# FastAPI calls this function when the server starts and stops
# Everything BEFORE yield runs on startup — we load all models here
# Everything AFTER yield runs on shutdown — we could clean up here
# yield is the dividing line between startup and shutdown logic
# This replaces the old @app.on_event("startup") which is now deprecated
# (deprecated means still works but will be removed in future versions)
@asynccontextmanager
async def lifespan(app: FastAPI):

    # global tells Python we want to modify the global variables above
    # without this keyword Python would create new local variables
    # inside this function and the globals would stay None forever
    global embeddings, vectorstore, tokenizer, model

    print("Server starting — loading models...")

    # Load the embedding model — must be the same one used to BUILD the index
    # all-MiniLM-L6-v2 produces 384-number vectors
    # If we used a different model here, queries would be in a different
    # vector space than our stored chunks — search results would be garbage
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    # Build the path to faiss_index/ regardless of where the script runs from
    # os.path.abspath(__file__)              → /Users/.../src/main.py
    # os.path.dirname(...)                   → /Users/.../src/
    # os.path.dirname(...dirname(...))       → /Users/.../rag-document-qa/
    # os.path.join(BASE_DIR, "faiss_index")  → /Users/.../rag-document-qa/faiss_index
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

    # Load the FAISS index we saved in vectorstore.py
    # Much faster than rebuilding from scratch every time the server starts
    # allow_dangerous_deserialization=True is required because FAISS uses
    # pickle format to save — LangChain forces you to confirm you trust the file
    # Since we created it ourselves, we trust it completely
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"FAISS index loaded with {vectorstore.index.ntotal} vectors.")

    # Load flan-t5 tokenizer — converts text to numbers the model understands
    # Already downloaded from Day 4 so loads from local cache instantly
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # Load flan-t5 model — the neural network that generates answers
    # ForSeq2SeqLM means sequence-to-sequence — text in, text out
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print("flan-t5 loaded.")

    print("All models ready. Server accepting requests.")

    # yield is the dividing line
    # The server is now running and handling requests
    # Everything above ran at startup, everything below runs at shutdown
    yield

    # Shutdown logic goes here if needed
    # For example: closing database connections, saving state, etc.
    # We don't need anything here for this project
    print("Server shutting down.")

# ── CREATE THE APP ────────────────────────────────────────────────────────────

# FastAPI() creates the entire web application
# title= is the name shown on the /docs page
# lifespan=lifespan tells FastAPI to use our lifespan function
# for startup and shutdown — this is how models get loaded
app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)

# ── DEFINE INPUT SCHEMA ───────────────────────────────────────────────────────

# This class defines exactly what a valid /ask request must contain
# question: str    → must be a text string, required, no default
# top_k: int = 3   → must be a number, optional, defaults to 3
# Pydantic validates this automatically — wrong types or missing required
# fields are rejected with a 422 error before our code even runs
class Question(BaseModel):
    question: str
    top_k: int = 3

# ── HEALTH ENDPOINT ───────────────────────────────────────────────────────────

# @app.get("/health") registers this function as the handler for GET /health
# GET means this endpoint can be accessed by typing the URL in a browser
# /health is the standard "is the server alive?" check in all real APIs
# Monitoring systems ping this every few seconds to confirm uptime
# If it returns anything other than 200, alerts fire
@app.get("/health")
def health_check():
    # We return a Python dictionary
    # FastAPI automatically converts it to JSON: {"status": "ok"}
    # HTTP 200 is returned automatically when a function completes without error
    return {"status": "ok"}

# ── ASK ENDPOINT ──────────────────────────────────────────────────────────────

# @app.post("/ask") registers this function as the handler for POST /ask
# POST is used when we're sending data TO the server (our question)
# GET is for receiving data FROM the server (like loading a webpage)
# FastAPI reads the incoming JSON, validates it against the Question schema,
# and passes it to this function already parsed as a Question object
@app.post("/ask")
def ask(question_input: Question):

    # RETRIEVE — find the most relevant chunks from FAISS
    # question_input.question → the actual question string
    # question_input.top_k    → how many results to return (default 3)
    # similarity_search converts the question to a vector internally
    # then finds the top_k stored vectors closest to it in meaning
    results = vectorstore.similarity_search(
        question_input.question,
        k=question_input.top_k
    )

    # FORMAT CONTEXT — join chunk texts into one block
    # r.page_content is the actual text of each retrieved chunk
    # "\n\n".join() puts double newlines between chunks as separators
    # This combined text is the "open book" we hand to flan-t5
    context = "\n\n".join([r.page_content for r in results])

    # BUILD PROMPT — the structured instruction we give flan-t5
    # "based on the context below" tells the model to use OUR documents
    # not its own training memory
    # "Write at least 3 sentences" encourages longer detailed answers
    # "Detailed Answer:" signals where the model should start generating
    prompt = f"""Answer the question in detail using the context below. Write at least 3 sentences.
Context: {context}
Question: {question_input.question}
Detailed Answer:"""

    # TOKENIZE — convert the prompt text into numbers the model understands
    # return_tensors="pt" → return PyTorch tensors (the format the model needs)
    # truncation=True     → if prompt exceeds max_length, cut it down gracefully
    # max_length=512      → flan-t5-base has a hard limit of 512 input tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # GENERATE — run flan-t5 on the tokenized prompt
    # **inputs unpacks the dictionary of tensors into the function
    # max_new_tokens=512 allows up to 512 tokens in the answer (~400 words)
    # The model generates tokens one at a time until it decides it's done
    # or hits the max_new_tokens limit
    outputs = model.generate(**inputs, max_new_tokens=512)

    # DECODE — convert the model's number output back to readable English
    # outputs[0] gets the first generated sequence
    # skip_special_tokens=True removes internal tokens like <pad> and </s>
    # that would appear as garbage text if left in
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # BUILD SOURCES — extract metadata from each retrieved chunk
    # r.metadata is the dictionary attached to each chunk
    # it contains {"source": "data/ml_basics.txt"} for each chunk
    # returning this lets the user verify WHERE the answer came from
    sources = [r.metadata for r in results]

    # RETURN RESPONSE — FastAPI converts this dictionary to JSON automatically
    # Final response looks like:
    # {
    #   "answer": "Overfitting occurs when a model learns...",
    #   "sources": [
    #     {"source": "data/ml_basics.txt"},
    #     {"source": "data/ml_basics.txt"},
    #     {"source": "data/ml_basics.txt"}
    #   ]
    # }
    return {
        "answer": answer,
        "sources": sources
    }