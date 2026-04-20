import os
import sys

# os gives us tools to work with file paths and folders
# sys gives us tools to modify Python's behaviour at runtime

# ── FIX PYTHON'S IMPORT PATH ─────────────────────────────────────────────────

# Python needs to know WHERE to look when you say "import something"
# By default it only looks in the current folder you ran the script from
# Our script is inside src/ but we might run it from the project root
# This line adds the src/ folder to Python's search path explicitly
# so it can find ingest.py no matter where you run the script from
# os.path.abspath(__file__) → full path of this file: /Users/.../src/rag_chain.py
# os.path.dirname(...)      → goes up one level:       /Users/.../src/
# sys.path.append(...)      → adds that path to Python's search list
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── IMPORTS ───────────────────────────────────────────────────────────────────

# HuggingFaceEmbeddings is LangChain's wrapper around sentence-transformers
# It gives the embedding model a standard interface LangChain can work with
# We need this to convert both chunks AND questions into vectors
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS is our vector database
# We already built and saved it in vectorstore.py
# Today we just load it from disk instead of rebuilding it
from langchain_community.vectorstores import FAISS

# AutoTokenizer converts text into numbers the model understands
# Think of it as a translator from human language to model language
# AutoModelForSeq2SeqLM is the actual flan-t5 model
# Seq2Seq means "sequence to sequence" — text goes in, text comes out
# torch is PyTorch — the engine that runs the model calculations
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch    

# ── STEP 1: LOAD THE EMBEDDING MODEL ─────────────────────────────────────────

# We load the exact same embedding model we used in vectorstore.py
# This is non-negotiable — you MUST use the same model to build and search the index. Here is why:
# When we built the FAISS index, each chunk was converted to a vector
# using all-MiniLM-L6-v2. Those vectors live in a specific "space"
# defined by that model.
# Same model = same space = meaningful comparison
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# ── STEP 2: LOAD THE SAVED FAISS INDEX ───────────────────────────────────────

# BASE_DIR works backwards from this file's location to find the project root
# os.path.abspath(__file__)         → /Users/.../src/rag_chain.py
# os.path.dirname(...)              → /Users/.../src/
# os.path.dirname(...dirname(...))  → /Users/.../rag-document-qa/   ← project root
# os.path.join(BASE_DIR, "faiss_index") → /Users/.../rag-document-qa/faiss_index
# This means the script finds faiss_index/ correctly no matter where you run it from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

# FAISS.load_local() reads the two files we saved in vectorstore.py:
#   faiss_index/index.faiss → the actual vectors in binary format
#   faiss_index/index.pkl   → the chunk texts and metadata mapped to each vector
# We pass embeddings because FAISS needs the embedding model on standby to convert new queries into vectors when we search
# allow_dangerous_deserialization=True is required because the .pkl file
# uses Python's pickle format — LangChain forces you to explicitly say
# "yes I trust this file" as a safety check. Since we created it ourselves, we do.
print("Loading FAISS index from disk...")
vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"FAISS index loaded with {vectorstore.index.ntotal} vectors.")

# ── STEP 3: LOAD THE LANGUAGE MODEL (flan-t5) ────────────────────────────────

# AutoTokenizer.from_pretrained() downloads and loads the tokenizer for flan-t5
# A tokenizer's job is to split text into tokens (small units) and convert
# them to numbers the model can process
# For example: "hello world" might become [15339, 995] as token IDs
# First run downloads ~300MB, every run after loads from local cache
print("Loading flan-t5 language model (first run downloads ~1GB)...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# AutoModelForSeq2SeqLM loads the actual neural network weights of flan-t5
# ForSeq2SeqLM means this model is designed for sequence-to-sequence tasks
# which is exactly what we need — take a prompt (sequence) in,
# produce an answer (sequence) out
# This is the ~1GB download on first run
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
print("Language model loaded.")

# ── STEP 4: THE ASK FUNCTION ──────────────────────────────────────────────────

# This function is the entire RAG pipeline in one place
# You give it a plain English question
# It returns a plain English answer + the source chunks it used
def ask(question):

    # PART A — RETRIEVE RELEVANT CHUNKS
    # similarity_search does three things internally:
    #   1. Takes the question text
    #   2. Converts it to a 384-number vector using our embedding model
    #   3. Finds the 3 stored vectors in FAISS that are mathematically closest
    # k=3 means return the top 3 most relevant chunks
    # The result is a list of 3 Document objects, each with page_content and metadata
    relevant_chunks = vectorstore.similarity_search(question, k=3)

    # PART B — FORMAT THE CONTEXT
    # We take the text content of each of the 3 chunks
    # chunk.page_content is the actual text string of that chunk
    # We join them together with double newlines as separators
    # The result is one big block of text containing all 3 relevant passages
    # This is the "open book" we hand to the language model
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # PART C — BUILD THE PROMPT
    # A prompt template is a structured instruction we give the language model
    # We explicitly tell it three things:
    #   1. Your job: answer based on the context below
    #   2. Here is the context (the 3 retrieved chunks)
    #   3. Here is the question
    # The "Answer:" at the end is a signal — flan-t5 sees this and knows
    # it should start generating the answer from that point
    # This structure is important — without "based on the context below"
    # flan-t5 might ignore the context and answer from its training memory
    prompt = f"""Answer the question based on the context below.Write at least 3 sentences.
Context: {context}
Question: {question}
Answer:"""

    # PART D — TOKENIZE THE PROMPT
    # tokenizer() converts our prompt string into numbers the model understands
    # return_tensors="pt" means return PyTorch tensors (the format the model needs)
    # truncation=True means if the prompt is too long, cut it down
    # max_length=512 means the prompt can be at most 512 tokens long
    # flan-t5-base has a limit of 512 input tokens — if we exceed it, it breaks
    # truncation=True handles this gracefully by cutting the excess
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # PART E — GENERATE THE ANSWER
    # model.generate() runs the actual neural network forward pass
    # **inputs unpacks the tokenized prompt dictionary into the function
    # max_new_tokens=200 limits the answer to 200 new tokens (~150 words)
    # Without this limit the model might ramble on forever
    # The result is a tensor of token ID numbers representing the answer
    outputs = model.generate(**inputs, max_new_tokens=512)

    # PART F — DECODE THE ANSWER
    # The model produced numbers — we need to convert them back to text
    # outputs[0] gets the first (and only) generated sequence
    # skip_special_tokens=True removes internal model tokens like <pad> and </s>
    # that would appear as garbage in the output if left in
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return both the answer text and the source chunks
    # We return sources so we can show the user WHERE the answer came from
    return answer, relevant_chunks


# ── STEP 5: TEST WITH 5 QUESTIONS ────────────────────────────────────────────

# We deliberately chose questions that cover different documents:
# Q1, Q2 → should pull from ml_basics.txt
# Q3     → should pull from python_programming.txt
# Q4     → should pull from rptu_specialisations.txt
# Q5     → should pull from BOTH python and ml documents
# If sources match expectations, our retrieval is working correctly
questions = [
    "What is overfitting in machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the main data types in Python?",
    "What specialisations does RPTU offer in computer science?",
    "How is Python used in machine learning?"
]

print("\n" + "=" * 60)
print("RAG SYSTEM TEST — 5 QUESTIONS")
print("=" * 60)

# We loop through each question, call ask(), and print the results
# enumerate() gives us both the index number i and the question text
# so we can label them Q1, Q2, Q3 etc.
for i, question in enumerate(questions):
    print(f"\nQ{i+1}: {question}")
    answer, sources = ask(question)
    print(f"Answer: {answer}")
    print(f"Sources used:")
    # Print which file each of the 3 retrieved chunks came from
    for s in sources:
        print(f"  - {s.metadata['source']}")
    print("-" * 60)