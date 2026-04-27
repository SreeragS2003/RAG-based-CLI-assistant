import faiss #Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors
import numpy as np
import json
import os
from app.embeddings import get_embedding

class VectorStore:
    def __init__(self):
        self.texts = [] #Store the original texts for retrieval later
        self.vectors = None #Store the corresponding vector embeddings, can be deleted once the FAISS index is built, as FAISS will handle the vectors internally
        self.index = None #FAISS index for efficient similarity search

    def save(self, path="app/storage"): #Making the FAISS index and the corresponding metadata (original texts) persistent by saving them to disk, so that they can be loaded later without needing to recompute embeddings and rebuild the index from scratch
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")

        # Save metadata (texts)
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(self.texts, f)
    
    def load(self, path="app/storage"): #Function for checking if a saved FAISS index and its corresponding metadata exist on disk, and if they do, load them into memory so that the vector store can be used for similarity search without needing to recompute everything from scratch. This is useful for persisting the state of the vector store across different runs of the program.
        if not os.path.exists(f"{path}/index.faiss"):
            return False

        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/metadata.json", "r") as f:
            self.texts = json.load(f)

        return True

    def add_texts(self, texts, source = "unknown"):
        embeddings = [get_embedding(t) for t in texts] #Get vector embeddings for each text in the input list

        self.texts = [
            {"content": t, "source": source, "id": i}
            for i, t in enumerate(texts)
        ] #Metadata for each text chunk, including the original content, the source (which can be useful for tracing back where the information came from), and a unique ID for each chunk (which can be used for retrieval and reference later on)

        self.vectors = np.array(embeddings).astype("float32") # Convert to numpy array and ensure it's float32 for faiss

        dim = self.vectors.shape[1] # Get the dimensionality of the embeddings
        self.index = faiss.IndexFlatL2(dim) # Create a FAISS index for L2 distance, flat means brute-force search, suitable for small datasets
        self.index.add(self.vectors)

    def search(self, query, k=2):
        query_vec = np.array([get_embedding(query)]).astype("float32") # Get the embedding for the query and convert to numpy array
        _, indices = self.index.search(query_vec, k) # Search the index for the k nearest neighbors of the query vector

        return [self.texts[i]["content"] for i in indices[0]]