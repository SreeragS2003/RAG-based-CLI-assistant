import chromadb
import os
from app.embeddings import get_embedding
from app.rerank import rerank

class VectorStore:
    def __init__(self, path="app/storage"):
        # PersistentClient automatically saves to disk and loads on startup
        # No manual save/load needed like FAISS — Chroma handles persistence automatically
        self.client = chromadb.PersistentClient(path=path)

        # Collection is equivalent to a FAISS index — it stores vectors and metadata together
        # get_or_create means it loads existing collection if it exists, or creates a new one
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # cosine similarity instead of L2 distance
        )

    def save(self):
        # No-op — Chroma PersistentClient saves automatically on every add/update
        # Kept for backward compatibility with existing lifespan code in api.py
        pass

    def load(self):
        # Check if collection already has documents — if yes, skip re-indexing
        # Equivalent to checking if index.faiss exists in the old FAISS setup
        return self.collection.count() > 0

    def add_texts(self, texts, metadata):
        # Get vector embeddings for each text chunk
        embeddings = [get_embedding(t) for t in texts]

        # Chroma stores texts, embeddings, and metadata together in one call
        # Unlike FAISS which needed separate metadata.json for original texts
        self.collection.add(
            ids=[m["chunk_id"] for m in metadata],        # unique ID per chunk
            embeddings=embeddings,                          # vector embeddings
            documents=[m["content"] for m in metadata],    # original text content
            metadatas=[{"source": m["source"]} for m in metadata]  # source PDF name
        )

    def hybrid_search(self, query, k=3):
        # Get query embedding for semantic similarity search
        query_embedding = get_embedding(query)

        # Query top 10 candidates for re-ranking — same strategy as before
        # Chroma returns documents, embeddings, and distances in one call
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        # Reconstruct candidates in the same format as before for rerank compatibility
        candidates = [
            {
                "content": doc,
                "source": meta["source"],
                "chunk_id": str(i)
            }
            for i, (doc, meta) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0]
            ))
        ]

        # Distances from Chroma (cosine) — lower is more similar, same as FAISS L2
        dists = results["distances"][0]

        # Re-rank candidates based on semantic similarity and keyword overlap
        return rerank(query, candidates, dists)[:k]