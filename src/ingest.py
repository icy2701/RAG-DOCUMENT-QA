# DirectoryLoader scans an entire folder and loads all files that match a pattern
# TextLoader is the specific loader for .txt files — it tells DirectoryLoader how to actually read each file it finds
from langchain_community.document_loaders import DirectoryLoader,TextLoader

# RecursiveCharacterTextSplitter is the tool that cuts documents into chunks
# 'Recursive' means it tries to split at natural boundaries first (paragraphs,then sentences, then words) before 
# splitting mid-sentence as a last resort.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── STEP 1: LOAD DOCUMENTS 

# We create a loader that points at our data/ folder
# glob="*.txt" means "only load files that end in .txt" — ignore everything else
# loader_cls=TextLoader tells it to use TextLoader to read each file
# encoding="utf-8" ensures special characters are read correctly
loader = DirectoryLoader(
    'data/',
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding":"utf-8"}
)

# .load() actually reads all the files and returns a list of Document objects
# Each Document object contains two things:
#   - page_content: the actual text of the file as a string
#   - metadata: info about the file, like its filename and path
docs=loader.load()

# Print how many documents were loaded — should be 3
print(f"Loaded {len(docs)} documents.")

# Print the source filename of each document so we can see what was loaded
for doc in docs:
    print(f" - {doc.metadata['source']}")

# ── STEP 2: SPLIT INTO CHUNKS 


# We create a splitter with two key settings:
#   chunk_size=500    → each chunk will be at most 500 characters long
#   chunk_overlap=50  → each chunk shares 50 characters with the next chunk
#                       so important sentences at boundaries aren't lost
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

# .split_documents() takes our list of Documents and splits each one into chunks
# It returns a new bigger list — each item is one chunk (still a Document object)
# Each chunk keeps the metadata (filename) from its original document.
chunks=splitter.split_documents(docs)

# Print how many total chunks we got across all 3 documents
print(f"Split into {len(chunks)} chunks.")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks)//len(chunks):.2f} characters.")

# ── STEP 3: INSPECT THE FIRST 3 CHUNKS

# We print the first 3 chunks so we can visually verify the splitting worked
# We should see:
#   - Each chunk is a small piece of one of our documents
#   - The text makes sense and isn't cut in a weird place
#   - The source metadata shows which file the chunk came from
print("\n --- FIRST 3 CHUNKS ---")
for i,chunk in enumerate(chunks[:3]):
    print(f"\n Chunk{i+1}:")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Length: {len(chunk.page_content)} characters")
    print(f"Content: {chunk.page_content}")
    print("-" *60)

