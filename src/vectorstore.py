import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import chunks
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed all chunks and store in FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"Index built with {vectorstore.index.ntotal} vectors.")

vectorstore.save_local(FAISS_PATH)
print(f"Index saved to {FAISS_PATH}")

# Sanity check — verify retrieval returns relevant chunks
results = vectorstore.similarity_search("what is machine learning?", k=3)
for i, result in enumerate(results):
    print(f"\nResult {i + 1}: {result.metadata['source']}")
    print(result.page_content)