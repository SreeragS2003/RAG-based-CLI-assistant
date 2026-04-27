from app.rag import RAG
def search_docs(query, store):
    rag = RAG(store) #Initialize RAG with the vector store
    return rag.search(query)


def calculator(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"
    

TOOLS = {
    "search_docs": {
        "description": "Search internal documents for information",
        "function": search_docs
    },
    "calculator": {
        "description": "Perform mathematical calculations",
        "function": calculator
    }
}