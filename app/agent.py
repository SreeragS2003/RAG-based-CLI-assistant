from app.llm import ask_llm
from app.tools import TOOLS
import json
AGENT_PROMPT = """
You are an intelligent agent.

You can:
- search_docs
- calculator

You must follow this format:

Thought: what you are thinking
Action: tool name OR "final"
Input: input to tool

When you have the final answer:
Action: final
Answer: ...

Be concise and logical.
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
            print(f"Parsing line: {line}") #Debugging statement to show each line being parsed, which can help identify any formatting issues in the LLM's response that might be causing problems with extracting the action and input correctly.
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