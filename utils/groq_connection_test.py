import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    http_client=httpx.Client(verify=False)
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(response.choices[0].message.content)