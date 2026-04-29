from app.rag import RAG
import math
from langchain.tools import tool

def create_search_tool(store):

    @tool
    async def search_docs(query: str) -> str:
        """Use this to search internal documents for factual information."""
        rag = RAG(store)
        result = await rag.search(query)
        return f"{result['context']}\nSources: {result['sources']}"  # string should be returned

    return search_docs

@tool
def calculator(expr: str):
    """Use this for mathematical calculations like 47/5, sqrt(16), etc."""
    try:
        allowed = {
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "pow": pow
        }

        return str(eval(expr, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {str(e)}"