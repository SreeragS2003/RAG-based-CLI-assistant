import os
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import asyncio
import httpx

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1",
    timeout=5.0,
    max_retries=1,
    http_client=httpx.Client(verify=False)
)

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Rules:
- Use the provided context to answer
- If answer not in context, say "I don't know"
- Be concise
"""
async def groq_call(messages):
    return await asyncio.to_thread(groq_sync_call, messages)


def groq_sync_call(messages):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content

async def ask_llm(messages):
    # Normalize to messages list if a plain string is passed
    if isinstance(messages, str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": messages}
        ]

    # For Gemini fallback, flatten messages to a single string
    def messages_to_text(messages):
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        )

    # Groq — natively supports messages format
    try:
        response = await asyncio.wait_for(
            groq_call(messages),
            timeout=30
        )
        return response

    except Exception as e:
        print(f"[Groq Failed] -> {e}")

    # Gemini fallback — doesn't support messages format natively, so flatten
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=messages_to_text(messages),
            config={"temperature": 0.3}
        )
        return response.text

    except Exception as e:
        print(f"[Gemini Failed] -> {e}")
        return "Error: All LLM providers failed."

async def ask_llm_stream(prompt): #Not used right now
    response = await gemini_client.aio.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config={
            "temperature": 0.3
        }
    )

    async for chunk in response:
        if chunk.text:
            yield chunk.text