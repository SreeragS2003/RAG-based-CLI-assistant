from langchain.agents import create_agent
from app.agent import llm
from web_agent.tools import web_search_tool
from web_agent.web_agent_state import WebAgentState
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

def initialize_web_agent():

    graph = StateGraph(WebAgentState)

    graph.add_node("decide", decide_node)
    graph.add_node("search", search_node)
    graph.add_node("answer_with_search", answer_with_search)
    graph.add_node("answer_direct", answer_direct)

    # Entry
    graph.set_entry_point("decide")

    # Conditional routing
    def route(state):
        return "search" if state.get("decision") == "YES" else "answer_direct"

    graph.add_conditional_edges(
        "decide",
        route,
        {
            "search": "search",
            "answer_direct": "answer_direct"
        }
    )

    graph.add_edge("search", "answer_with_search")
    graph.add_edge("answer_with_search", END)
    graph.add_edge("answer_direct", END)

    return graph.compile()

async def decide_node(state: WebAgentState):
    query = state["query"]

    prompt = f"""
    Decide whether this query requires web search.

    Query: {query}

    Rules:
    - If it's current, factual, or unknown → YES
    - Otherwise → NO

    Answer ONLY: YES or NO
    """

    response = await llm.ainvoke(prompt)
    decision = response.content.strip().upper()

    return {"decision": decision}

async def search_node(state: WebAgentState):
    query = state["query"]

    results = await web_search_tool.ainvoke(query)

    return {"search_results": results}

async def answer_with_search(state: WebAgentState):
    query = state["query"]
    context = state["search_results"]

    prompt = f"""
    Use the following web results to answer:

    {context}

    Question: {query}

    Rules:
    - Be accurate
    - Do not hallucinate
    - Summarize clearly
    """

    response = await llm.ainvoke(prompt)

    return {"final_answer": response.content}

async def answer_direct(state: WebAgentState):
    query = state["query"]

    response = await llm.ainvoke(query)

    return {"final_answer": response.content}