from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
result = model.encode("hello world")

print(result.shape)  # expected: (384,)