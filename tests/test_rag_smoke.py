# tests/test_rag_smoke.py — fast CI version
import pytest
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
import httpx

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/free",
            http_client=httpx.Client(verify=False),
            http_async_client=httpx.AsyncClient(verify=False),
        )

    def load_model(self): return self.model
    def generate(self, prompt): return self.model.invoke(prompt).content
    async def a_generate(self, prompt): return (await self.model.ainvoke(prompt)).content
    def get_model_name(self): return "openai/gpt-oss-20b"

judge_llm = OpenRouterLLM()

# Only one critical test case for CI speed
def test_basic_rag_quality():
    from app.vector_store import VectorStore
    from app.rag import RAG

    store = VectorStore()
    if not store.load():
        pytest.skip("No index found")

    rag = RAG(store)
    result = asyncio.get_event_loop().run_until_complete(
        rag.search("What is Amazon Aurora?")
    )

    test_case = LLMTestCase(
        input="What is Amazon Aurora?",
        actual_output=result["context"],
        retrieval_context=result.get("contexts", [result["context"]]),
    )

    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.6, model=judge_llm, include_reason=True)
    ])