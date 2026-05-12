import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Must use the same embedding model used to build the index
vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def ask(question: str) -> tuple:
    results = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([chunk.page_content for chunk in results])

    prompt = f"""Answer the question using only the context below. Write at least 3 sentences.
Context: {context}
Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, results


if __name__ == "__main__":
    questions = [
        "What is overfitting in machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What are the main data types in Python?",
        "What specialisations does RPTU offer in computer science?",
        "How is Python used in machine learning?",
    ]

    for i, question in enumerate(questions):
        print(f"\nQ{i + 1}: {question}")
        answer, sources = ask(question)
        print(f"Answer: {answer}")
        for s in sources:
            print(f"  Source: {s.metadata['source']}")
        print("-" * 60)