import faiss #Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors
import numpy as np
from app.embeddings import get_embedding

class VectorStore:
    def __init__(self):
        self.texts = [] #Store the original texts for retrieval later
        self.vectors = None #Store the corresponding vector embeddings, can be deleted once the FAISS index is built, as FAISS will handle the vectors internally
        self.index = None #FAISS index for efficient similarity search

    def add_texts(self, texts):
        embeddings = [get_embedding(t) for t in texts] #Get vector embeddings for each text in the input list
        print(f"Generated embeddings for {len(texts)} texts.")
        self.texts.extend(texts) # Add new texts to the existing lists

        self.vectors = np.array(embeddings).astype("float32") # Convert to numpy array and ensure it's float32 for faiss

        dim = self.vectors.shape[1] # Get the dimensionality of the embeddings
        self.index = faiss.IndexFlatL2(dim) # Create a FAISS index for L2 distance, flat means brute-force search, suitable for small datasets
        print(f"Created FAISS index {self.index} with dimension {dim}.")
        self.index.add(self.vectors)

    def search(self, query, k=2):
        query_vec = np.array([get_embedding(query)]).astype("float32") # Get the embedding for the query and convert to numpy array
        print(f"Generated embedding for query: '{query}'.")
        distances, indices = self.index.search(query_vec, k) # Search the index for the k nearest neighbors of the query vector
        print(f"Search results - Distances: {distances}, Indices: {indices}")

        return [self.texts[i] for i in indices[0]]