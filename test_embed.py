from sentence_transformers import SentenceTransformer

# We are loading a pre-trained AI model called 'all-MiniLM-L6-v2'
# 'Pre-trained' means someone already trained this model on billions of sentences
# so it already understands what words and sentences mean
# The first time you run this line, Python downloads the model (~80MB) from the internet
# Every time after that, it loads from your computer — no internet needed
# We store the loaded model in the variable 'm' so we can use it below
m = SentenceTransformer('all-MiniLM-L6-v2')

# We feed the text "hello world" into the model using .encode()
# The model reads the text and converts it into a list of 384 numbers
# This list of numbers is called an 'embedding' or a 'vector'
# These 384 numbers together represent the MEANING of the text mathematically
# Similar sentences will produce similar numbers — that's the magic of embeddings
result = m.encode("hello world")

# .shape tells us the dimensions of the result
# Since result is a list of 384 numbers, this will print (384,)
# The (384,) means it is a 1-dimensional array with 384 values
# This is our confirmation that the embedding technology works on our machine
# The entire RAG search system depends on this working, so we verify it first
print(result.shape)