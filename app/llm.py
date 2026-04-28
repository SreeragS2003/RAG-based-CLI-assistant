import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Rules:
- Use the provided context to answer
- If answer not in context, say "I don't know"
- Be concise
"""

async def ask_llm(prompt):
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config={
            "temperature": 0.3
        }
    )

    return response.text

async def ask_llm_stream(prompt): #Not used right now
    response = await client.aio.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config={
            "temperature": 0.3
        }
    )

    async for chunk in response:
        if chunk.text:
            yield chunk.text