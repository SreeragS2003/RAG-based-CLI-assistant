from app.vector_store import VectorStore
from app.llm import ask_llm

class RAG:
    def __init__(self, store: VectorStore):
        self.store = store

    def answer(self, query):
        # 1. Retrieve relevant chunks from the in-memory vector store (which is FAISS index)
        chunks = self.store.search(query, k=2)

        # 2. Build context from the retrieved chunks
        context = "\n".join(chunks)

        # 3. Create prompt having both the context and the query. The prompt will be sent to the LLM to generate an answer. We can use a system prompt to instruct the LLM on how to use the context to answer the question.
        prompt = f"""
        Context:
        {context}

        Question:
        {query}
        """

        # 4. Call LLM
        response = ask_llm(prompt)

        return response