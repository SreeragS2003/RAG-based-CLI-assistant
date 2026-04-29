from google import genai
import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in client.models.list():
    print(model.name)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    http_client=httpx.Client(verify=False)
)

models = groq_client.models.list()

for model in models.data:
    print(model.id)