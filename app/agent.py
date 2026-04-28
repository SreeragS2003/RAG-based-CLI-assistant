from utils.safe_llm import safe_llm_call
from app.llm import ask_llm
from app.tools import TOOLS, execute_tool
from utils.json_parse_helper import parse_json
from utils.logger import logger
AGENT_PROMPT = """
You are an intelligent agent.

You can:
- search_docs
- calculator

Rules:
- Use tools only if needed
- If you already have enough information, STOP and return final answer
- Do NOT call tools repeatedly for the same query

STRICT RULES:
- Always respond in valid JSON
- Do NOT include any extra text
- Do NOT include markdown

Format:
{
  "action": "tool_name OR final",
  "input": "input for the tool",
  "answer": "final answer (only if action is final)"
}
"""
def build_prompt(query, history):
    return f"""
    {AGENT_PROMPT}

    Previous steps:
    {history}

    User: {query}
    """

MAX_STEPS = 5

async def run_agent(query, store, memory=None):
    if memory is None: #fallback to in-memory memory if no memory instance is provided (for backward compatibility and testing)
        from app.memory_store import Memory
        memory = Memory()

    history = memory.get()

    for step in range(MAX_STEPS):
        prompt = build_prompt(query, history) #Construct the prompt for the LLM by combining the agent prompt, the history of previous interactions (which includes the user's query and the assistant's responses), and the current user query. This prompt provides the LLM with all the necessary context to determine the next action to take.

        response = await safe_llm_call(lambda: ask_llm(prompt)) #Call the LLM with the constructed prompt, wrapped in the safe_llm_call function to handle retries and transient errors.

        if not response.strip():
            history += "\nEmpty LLM response"
            continue

        try:
            data = parse_json(response) #Parse the LLM response as JSON, which should contain the action, input, and answer (if final). If the response is not valid JSON, we catch the exception and append an error message to the history, allowing the LLM to try again with a clearer prompt.
        except Exception:
            # handle invalid JSON
            history += f"\nInvalid JSON response: {response}"
            continue

        action = (data.get("action") or "").strip().lower() #Extract the action from the parsed JSON, ensuring it's a string and normalizing it to lowercase for consistent processing. If the action is missing or not a string, we default to an empty string to avoid errors in subsequent checks.
        action_input = data.get("input", "")
        answer = data.get("answer", "")

        logger.info(f"[STEP {step}] Action: {action} -> Action Input : {action_input}")

        # If final answer
        if action == "final":
            memory.add(query, answer) #Add the query and final answer to memory for future context
            for word in answer.split():
                yield word + " " #Word level streaming of the final answer back to the client, allowing for a more responsive user experience where the client can start receiving parts of the answer immediately rather than waiting for the entire response to be generated before sending it back.
            #Since yield is used, run_agent function (it is a generator function now) will pause here and return control to the caller, allowing it to process the answer as it is generated. Once the entire answer has been yielded, the function will exit.
            return

        if action != "final" and action not in TOOLS:
            observation = f"Error: Unknown tool '{action}'"

            history += f"""
            LLM Output: {response}
            Observation: {observation}
            """
            continue #Let LLM try again

        if not action_input:
            history += "\nMissing input"
            continue

        # Execute tool
        result = execute_tool(action, action_input, store)

        # Append observation to history (for memory)
        history += f"""
        LLM Output: {response}
        Observation: {result}
        """

    yield "Max steps reached. Could not complete."