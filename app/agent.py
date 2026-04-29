from utils.safe_llm import safe_llm_call
from app.llm import ask_llm
from app.tools import TOOLS, execute_tool
from utils.json_parse_helper import parse_json
from utils.logger import logger
AGENT_PROMPT = """
You are an intelligent agent.

Available tools:
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
- After calling search_docs ONCE and receiving results:
   DO NOT call search_docs again
- If question contains factual knowledge (e.g. "what is", "explain", "define"):
   ALWAYS use search_docs FIRST
- If question contains math:
   use calculator
- If BOTH are present:
   FIRST use search_docs
   THEN use calculator
   THEN give final answer

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
    if memory is None:
        from app.memory_store import Memory
        memory = Memory()

    past_history = memory.get()
    called_tools = set()  # track all tools called, not just last

    messages = [
        {"role": "system", "content": AGENT_PROMPT},
    ]

    # Inject past memory as context
    if past_history:
        messages.append({"role": "user", "content": f"Previous context:\n{past_history}"})
        messages.append({"role": "assistant", "content": "Understood."})

    messages.append({"role": "user", "content": query})

    for step in range(MAX_STEPS):
        response = await safe_llm_call(lambda: ask_llm(messages))  # pass messages, not prompt string

        if not response.strip():
            messages.append({"role": "user", "content": "Your response was empty. Try again."})
            continue

        try:
            data = parse_json(response)
        except Exception:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Invalid JSON. Respond only in the specified JSON format."})
            continue

        action = (data.get("action") or "").strip().lower()
        action_input = data.get("input", "")
        answer = data.get("answer", "")

        logger.info(f"[STEP {step}] Action: {action} -> Input: {action_input}")

        messages.append({"role": "assistant", "content": response})  # always add model output to history

        if action == "final":
            memory.add(query, answer)
            for word in answer.split():
                yield word + " "
            return

        if action not in TOOLS:
            messages.append({"role": "user", "content": f"Unknown tool '{action}'. Use only: {list(TOOLS.keys())}"})
            continue

        if not action_input:
            messages.append({"role": "user", "content": "Missing input for tool. Please provide input."})
            continue

        # Hard block on repeated tool calls
        if action in called_tools:
            messages.append({"role": "user", "content": f"You already called '{action}'. Do NOT call it again. Use the previous result and give your final answer."})
            continue

        called_tools.add(action)
        result = await execute_tool(action, action_input, store)

        messages.append({"role": "user", "content": f"Tool result for {action}:\n{result}\n\nNow continue. Do NOT call {action} again."})

    yield "Max steps reached. Could not complete."