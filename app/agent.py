from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
import httpx

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"), 
    model="qwen/qwen3-32b",  # specifically fine-tuned for tool use,
    temperature=0.3,
    max_retries=1,
    timeout=6,
    http_async_client=httpx.AsyncClient(verify=False)
)

system_prompt = """
You are an intelligent assistant.

Rules:
- Use search_docs for factual/document-based questions
- Use calculator for math
- Do not guess when tools can be used
- Avoid repeating the same tool multiple times
"""

def initialize_agent(tools):
    agent = create_agent(
        tools=tools,
        model=llm,
        system_prompt=system_prompt
    )
    return agent