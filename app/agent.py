from app.llm import ask_llm
from app.tools import TOOLS
import json
AGENT_PROMPT = """
You are an intelligent assistant with access to tools.

Tools:
- search_docs: use for questions about documents
- calculator: use for math

Rules:
- Decide which tool to use
- Respond in JSON:

{
  "action": "tool_name" or "final",
  "input": "...",
  "answer": "..."
}
"""
def run_agent(query, store):
    prompt = AGENT_PROMPT + f"\nUser: {query}"

    response = ask_llm(prompt)
    
    data = json.loads(response)

    if data["action"] == "final":
        return data["answer"]

    tool_name = data["action"]
    tool_input = data["input"]

    tool_fn = TOOLS[tool_name]["function"]

    # execute tool
    if tool_name == "search_docs":
        result = tool_fn(tool_input, store)
    else:
        result = tool_fn(tool_input)

    # feed result back to LLM
    follow_up = f"""
Tool result:
{result}

Now give final answer.
"""

    final = ask_llm(follow_up)
    return final