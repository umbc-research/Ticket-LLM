from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load the existing DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./ticket_db", embedding_function=embeddings)

# Ask how many items are in it
print(f"Total chunks in database: {db._collection.count()}")
