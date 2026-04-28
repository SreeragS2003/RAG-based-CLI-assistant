from app.rag import RAG
import math

def search_docs(query, store):
    rag = RAG(store) #Initialize RAG with the vector store
    return rag.search(query)

def calculator(expr: str):
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

def execute_tool(action, action_input, store=None):
    # Execute tool
        tool_fn = TOOLS[action]["function"]

        try:
            if action == "search_docs": #Decide which tool to use based on llm response
                return tool_fn(action_input, store)
            return tool_fn(action_input)
        except Exception as e:
            return f"Tool error: {str(e)}"


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