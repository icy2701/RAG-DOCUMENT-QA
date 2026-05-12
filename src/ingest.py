import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

loader = DirectoryLoader(
    DATA_DIR,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Split into 500-char chunks with 50-char overlap so sentences
# at boundaries aren't lost between chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")