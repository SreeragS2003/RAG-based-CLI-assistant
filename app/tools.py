from app.rag import RAG
import math
import inspect

async def search_docs(query, store):
    rag = RAG(store) #Initialize RAG with the vector store
    return await rag.search(query)

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

async def execute_tool(action, action_input, store=None):
    # Execute tool
    tool = TOOLS[action]
    tool_fn = tool["function"]

    try:
        args = (action_input, store) if tool.get("needs_store") else (action_input,)

        if inspect.iscoroutinefunction(tool_fn):
            return await tool_fn(*args)
        else:
            return tool_fn(*args)
    except Exception as e:
        return f"Tool error: {str(e)}"


TOOLS = {
    "search_docs": {
        "description": "Search internal documents for information",
        "function": search_docs,
        "needs_store" : True #Indicates that this tool needs access to the vector store, which will be passed in when executing the tool. This allows us to manage dependencies and ensure that the necessary resources are available when the tool is called.
    },
    "calculator": {
        "description": "Perform mathematical calculations",
        "function": calculator,
        "needs_store" : False #Indicates that this tool does not need access to the vector store, so we won't pass it in when executing the tool. This allows us to manage dependencies and ensure that we only provide the necessary resources to each tool based on its requirements.
    }
}