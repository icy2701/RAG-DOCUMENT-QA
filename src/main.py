import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

# Global model references — loaded once at startup, reused across requests
embeddings = None
vectorstore = None
tokenizer = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, vectorstore, tokenizer, model

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print(f"Ready — {vectorstore.index.ntotal} vectors loaded.")
    yield


app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)


class Question(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(question_input: Question):
    results = vectorstore.similarity_search(
        question_input.question,
        k=question_input.top_k
    )

    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""Answer the question using only the context below. Write at least 3 sentences.
Context: {context}
Question: {question_input.question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "sources": [r.metadata for r in results]
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    # Sanitise filename to prevent path traversal
    save_path = os.path.join(BASE_DIR, "data", os.path.basename(file.filename))
    content = await file.read()

    with open(save_path, "wb") as f:
        f.write(content)

    loader = TextLoader(save_path, encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    new_chunks = splitter.split_documents(loader.load())

    # Build a small index for the new document and merge into the live index
    new_store = FAISS.from_documents(new_chunks, embeddings)
    vectorstore.merge_from(new_store)
    vectorstore.save_local(FAISS_PATH)

    return {
        "message": f"'{file.filename}' indexed successfully.",
        "chunks_added": len(new_chunks),
        "total_vectors": vectorstore.index.ntotal
    }