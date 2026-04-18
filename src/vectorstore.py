# We import the chunks pipeline from our ingest.py file
# This runs ingest.py and gives us the 24 chunks we already built
from ingest import chunks

# HuggingFaceEmbeddings is a wrapper that loads any HuggingFace embedding model
# and gives it a simple .embed() interface that LangChain understands
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS is the vector database — it stores all our vectors and lets us
# search them by similarity (closest meaning) extremely fast
from langchain_community.vectorstores import FAISS

# ── STEP 1: LOAD THE EMBEDDING MODEL

# We load the same model we tested in test_embed.py
# model_name="all-MiniLM-L6-v2" — small, fast, produces 384-number vectors
# First run downloads ~80MB from HuggingFace, then cached locally forever.
print("Loading Embedding Model...")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding Model Loaded.")

# ── STEP 2: EMBED ALL CHUNKS AND BUILD THE FAISS INDEX

# FAISS.from_documents() does two things in one step:
#   1. Takes every chunk and runs it through the embedding model
#      producing a 384-number vector for each of the 24 chunks
#   2. Stores all those vectors inside a FAISS index structure
#      that is optimised for fast similarity search
# This is the most important line in the whole file — it transforms our text chunks into a format that can be quickly searched by meaning.
print("Embedding chunks and building FAISS index...")
vectorstore=FAISS.from_documents(chunks,embeddings)
print(f"FAISS index built with {vectorstore.index.ntotal} vectors.")

# ── STEP 3: SAVE THE INDEX TO DISK 

# Save the FAISS index to a folder called faiss_index/ in the project root
# This creates two files:
#   faiss_index/index.faiss — the actual vectors stored in binary format
#   faiss_index/index.pkl   — the chunk texts and metadata mapped to each vector
# Without saving, everything disappears when the script stops running
vectorstore.save_local("faiss_index")
print("FAISS index saved to 'faiss_index/' folder.")

# ── STEP 4: TEST RETRIEVAL

# similarity_search() takes a plain text question and does the following:
#   1. Converts the question into a 384-number vector using the same model
#   2. Compares that vector against all 24 stored vectors using cosine similarity
#   3. k=3 means "give me the 3 most relevant chunks."
# This is the entire search engine in one line — it finds the most relevant chunks for any question we ask.
print("\nTesting retrieval with sample query: 'what is machine learning?' ")
results=vectorstore.similarity_search("what is machine learning?",k=3)


# Print the 3 retrieved chunks
# We expect all 3 to come from ml_basics.txt since that's the relevant document
print(f"\n--- TOP 3 RESULTS ---")
for i,result in enumerate(results):
    print(f"\n Result: {i+1}")
    print(f"Source: {result.metadata['source']}")
    print(f"Length: {len(result.page_content)} characters.")
    print(f"Content: {result.page_content}")
    print("-" *60)




