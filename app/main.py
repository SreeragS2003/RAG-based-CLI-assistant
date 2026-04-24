from app.vector_store import VectorStore
from pathlib import Path

store = VectorStore() #Initialize empty vector store

BASE_DIR = Path(__file__).resolve().parent #Get the directory of the current file (main.py) and resolve it to an absolute path
file_path = BASE_DIR / "data" / "sample.txt" #Construct the full path to the sample.txt file located in the data subdirectory of the base directory

with open(file_path, "r") as f:
    texts = f.readlines() #Read the contents of the sample.txt file into a list of lines
texts = [t.strip() for t in texts if t.strip()] #Remove leading/trailing whitespace and filter out empty lines

store.add_texts(texts) #Add the texts to the vector store, which will generate embeddings and build the FAISS index

query = input("Ask: ")

results = store.search(query) #Search the vector store for the most similar texts to the query, which will generate an embedding for the query and use FAISS to find the closest matches

print("\nTop matches:")
for r in results:
    print("-", r)