from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class WebAgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    search_results: str
    final_answer: str