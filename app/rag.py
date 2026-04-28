from app.vector_store import VectorStore

rag_cache = {} #Simple in-memory cache for RAG results, mapping query to retrieved context and sources. This can help speed up responses for repeated queries by avoiding redundant retrievals from the vector store.
class RAG:
    def __init__(self, store: VectorStore):
        self.store = store

    def search(self, query):
        if query in rag_cache:
            return rag_cache[query]
        # 1. Retrieve relevant chunks from the in-memory vector store (which is FAISS index)
        results = self.store.hybrid_search(query, k=2)
        
        # 2. Extract content
        context = "\n".join([r["content"] for r in results])

        # 3. Extract sources
        sources = list(set([r["source"] for r in results]))

        rag_cache[query] = {"context": context, "sources": sources}  # Cache the result

        return {
            "context": context,
            "sources": sources
        }
