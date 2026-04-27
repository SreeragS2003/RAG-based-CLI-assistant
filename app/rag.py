from app.vector_store import VectorStore
from app.llm import ask_llm

class RAG:
    def __init__(self, store: VectorStore):
        self.store = store

    def search(self, query):
        # 1. Retrieve relevant chunks from the in-memory vector store (which is FAISS index)
        results = self.store.hybrid_search(query, k=2)

        # 2. Extract content
        context = "\n".join([r["content"] for r in results])

        # 3. Extract sources
        sources = list(set([r["source"] for r in results]))

        return {
            "context": context,
            "sources": sources
        }
