from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Annotated
import operator
import os
import httpx

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b",
    temperature=0.3,
    max_retries=1,
    timeout=30,
    http_async_client=httpx.AsyncClient(verify=False),
    http_client=httpx.Client(verify=False),
)

system_prompt = """
You are an intelligent assistant.

Rules:
- Use search_docs for factual/document-based questions
- Use calculator for math
- Do not guess when tools can be used
- Avoid repeating the same tool multiple times
"""

# State — this is passed between every node in the graph
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # operator.add means messages are appended, not overwritten

def initialize_agent(tools):
    # Bind tools to LLM so it knows what's available
    llm_with_tools = llm.bind_tools(tools) #it converts your @tool functions into a JSON schema and sends it with every request so the model knows it can call them.

    # Node 1 — LLM decides what to do
    def call_llm(state: AgentState):
        messages = state["messages"]

        # Inject system prompt if not already present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = llm_with_tools.invoke(input=messages)
        return {"messages": [response]}

    # Node 2 — Execute the tool the LLM chose
    tool_node = ToolNode(tools)

    # Edge condition — should we call a tool or stop?
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"  # go to tool node

        return END  # stop, return final answer

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm") #Setting the entry point which will be used by agent.ainvoke

    # After LLM — decide whether to call tools or end
    graph.add_conditional_edges("llm", should_continue, {
        "tools": "tools",
        END: END
    })

    # After tools — always go back to LLM
    graph.add_edge("tools", "llm")

    return graph.compile()