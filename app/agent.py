from app.llm import ask_llm
from app.tools import TOOLS
AGENT_PROMPT = """
You are an intelligent agent.

You can:
- search_docs
- calculator

Rules:
- Use tools only if needed
- If you already have enough information, STOP and return final answer
- Do NOT call tools repeatedly for the same query

Format:

Thought: ...
Action: tool name OR "final"
Input: ...

When done:
Action: final
Answer: ...
"""

MAX_STEPS = 5

def run_agent(query, store, memory=None):
    if memory is None:
        memory = []
    history = "\n".join(memory)

    for step in range(MAX_STEPS):
        prompt = f"""
            {AGENT_PROMPT}

            Previous steps:
            {history}

            User: {query}
        """

        response = ask_llm(prompt)

        # simple parsing
        lines = response.split("\n")

        action = None
        action_input = ""
        answer = ""

        for line in lines:
            line = line.strip()
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Input:"):
                action_input = line.replace("Input:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
        
        # If final answer
        if action == "final":
            memory.append(f"User: {query}")
            memory.append(f"Agent: {answer}")
            return answer

        if action not in TOOLS:
            observation = f"Error: Unknown tool '{action}'"
            
            history += f"""
                {response}
                Observation: {observation}
            """
            continue  # let LLM try again

        # Execute tool
        tool_fn = TOOLS[action]["function"]

        if action == "search_docs": #Decide which tool to use based on llm response.
            result = tool_fn(action_input, store)
        else:
            result = tool_fn(action_input)

        # Append observation to history (for memory)
        history += f"""
        {response}
        Observation: {result}
        """

    return "Max steps reached. Could not complete."