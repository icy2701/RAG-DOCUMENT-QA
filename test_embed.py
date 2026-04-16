from sentence_transformers import SentenceTransformer

m= SentenceTransformer('all-MiniLM-L6-v2')
result=m.encode("Hello World")
print(result.shape)