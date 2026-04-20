import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── CREATE THE APP 

# This single line creates the entire web application
app=FastAPI(title="RAG Document QA")

# ── DEFINE INPUT SCHEMA ───────────────────────────────────────────────────────

# This class defines exactly what a valid /ask request must contain
# question: str       → must be a text string, required
# top_k: int = 3      → must be a number, optional, defaults to 3 if not provided
# If someone sends a request missing the question field FastAPI
# automatically returns a clear error — our code never even runs

class Question(BaseModel):
    question:str
    top_k:int=3

# ── GLOBAL VARIABLES ─────────────────────────────────────────────────────────

# We declare these as None now and fill them during startup
# They are global so every endpoint function can access them
# Think of these as empty shelves — startup puts the models on them

embeddings=None
vectorstore=None
tokenizer=None
model=None

# ── STARTUP EVENT ─────────────────────────────────────────────────────────────

# This function runs ONCE when the server starts, before any requests come in
# We load all heavy models here so they stay in memory permanently
# This means every request reuses already-loaded models — fast responses
# Without this, every single request would reload models — 30+ seconds each time

@app.on_event("startup")
async def load_models():

    # global tells Python to modify the global variables above
    # without this, Python would create new local variables inside this
    # function and the global ones would stay None forever
    global embeddings,vectorstore,tokenizer,model

    print("Server starting-loading models...")

    # Load the same embedding model used to build the FAISS index
    # Must be identical — same model means same vector space means
    # meaningful similarity comparisons between query and stored chunks
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    # Build path to faiss_index/ from project root
    # Goes up two levels: main.py → src/ → project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

    # Load the saved FAISS index we built in vectorstore.py
    # Much faster than rebuilding from scratch every time
    vectorstore=FAISS.load_local(FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
    print(f"FAISS index loaded with {vectorstore.index.ntotal} vectors.")

    # Load flan-t5 — already downloaded so this loads from local cache
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base")
    model=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print("flan-t5 language model loaded.")

    print("All models loaded. Server is ready to accept requests.")

# ── HEALTH ENDPOINT ───────────────────────────────────────────────────────────

# GET /health — responds to simple browser requests
# Standard pattern in every real API — a quick check that the server is alive
# Monitoring tools ping this every few seconds to check the server is up
# Returns a simple JSON: {"status": "ok"}
@app.get("/health")
def health_check():
    return {"status":"ok"}

# ── ASK ENDPOINT ──────────────────────────────────────────────────────────────

# POST /ask — the main endpoint that answers questions
# We use POST not GET because we are sending data (the question) to the server
# FastAPI reads the incoming JSON, validates it against Question schema,
# and passes it to this function as a Question object
@app.post("/ask")
def ask(question_input:Question):
    # RETRIEVE — convert question to vector, find top_k closest chunks in FAISS
    # question_input.question → the actual question string from the request
    # question_input.top_k    → how many chunks to retrieve (default 3)
    results=vectorstore.similarity_search(question_input.question, k=question_input.top_k)

    # FORMAT CONTEXT — join all retrieved chunk texts into one block
    # This is the open book we hand to flan-t5
    context = "\n\n".join([r.page_content for r in results])

    # BUILD PROMPT — structured instruction for flan-t5
    # "based on the context below" grounds the model in our documents
    # "Write at least 3 sentences" encourages longer answers
    prompt=f"""Answer the question based on the context below.Write at least 3 sentences.
    Context:{context}
    Question:{question_input.question}
    Answer:"""

    # TOKENIZE — convert prompt text into numbers the model understands
    # truncation=True and max_length=512 handle flan-t5's input size limit
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=512)

    # GENERATE — run flan-t5 on the prompt, produce answer tokens
    # max_new_tokens=512 allows longer more detailed answers
    outputs=model.generate(**inputs,max_new_tokens=512)

    # DECODE — convert model's number output back to readable English
    # skip_special_tokens=True removes internal model tokens like <pad>
    answer=tokenizer.decode(outputs[0],skip_special_tokens=True)

    # BUILD SOURCES — extract metadata from each retrieved chunk
    # metadata contains the source filename for each chunk
    # This is what lets us say "this answer came from ml_basics.txt"
    sources = [r.metadata for r in results]

    # RETURN RESPONSE — FastAPI automatically converts this dict to JSON
    # Response looks like:
    # {
    #   "answer": "Overfitting occurs when...",
    #   "sources": [{"source": "data/ml_basics.txt"}, ...]
    # }
    return {
        "answer": answer,
        "sources": sources
    }
