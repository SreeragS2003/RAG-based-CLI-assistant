from app.vector_store import VectorStore
from app.llm import ask_llm

class RAG:
    def __init__(self, store: VectorStore):
        self.store = store

    def answer(self, query):
        # 1. Retrieve relevant chunks from the in-memory vector store (which is FAISS index)
        results = self.store.hybrid_search(query, k=2)
        print(f"Top {len(results)} retrieved chunks:") #Print the number of chunks retrieved from the vector store based on the query, which can help in understanding how much context is being provided to the LLM for answer generation.
        for r in results:
            print("Metadata:", r["source"], "Content:", r["content"][:100]) #Print the source metadata and the first 100 characters of the content for each retrieved chunk, which can help in debugging and understanding what information is being fed into the LLM for generating the answer.

        # Extract content
        context = "\n".join([r["content"] for r in results])

        # 3. Create prompt having both the context and the query. The prompt will be sent to the LLM to generate an answer. We can use a system prompt to instruct the LLM on how to use the context to answer the question.
        prompt = f"""
        Context:
        {context}

        Question:
        {query}
        """

        # 4. Call LLM
        response = ask_llm(prompt)

        # Collect sources
        sources = list(set([r["source"] for r in results]))

        return response, sources