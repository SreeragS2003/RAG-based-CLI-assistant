from langchain.tools import tool
from web_agent.web_search import web_search
@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current or external information.
    Use this when the answer is not in internal documents.
    """
    return web_search(query)