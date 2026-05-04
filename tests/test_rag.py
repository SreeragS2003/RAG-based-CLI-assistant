import pytest
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
import httpx

# ── Custom LLM wrapper so DeepEval uses OpenRouter instead of OpenAI ──────
class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/free",
            http_client=httpx.Client(verify=False),
            http_async_client=httpx.AsyncClient(verify=False),
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "openrouter/free"

judge_llm = OpenRouterLLM()

# ── Test data — representative questions about your PDFs ──────────────────
# Add more test cases covering different query types
TEST_CASES = [
    {
        "input": "What is Amazon Aurora?",
        "expected_output": "Amazon Aurora is a cloud-native relational database service by AWS",
        "context": [
            "Aurora is a cloud-native relational database service provided by AWS, designed for high availability, scalability, and performance.",
            "Aurora distributes storage across 3 Availability Zones with 6 copies of data.",
            "Aurora supports MySQL and PostgreSQL wire protocol compatibility."
        ]
    },
    {
        "input": "How does Aurora handle high availability?",
        "expected_output": "Aurora handles high availability by storing 6 copies of data across 3 Availability Zones",
        "context": [
            "Aurora survives AZ failures via quorum-based reads/writes — 4/6 copies for writes, 3/6 for reads.",
            "Aurora distributes storage across 3 Availability Zones with 6 copies of data.",
        ]
    },
]

# ── Helper — run RAG pipeline and get actual answer + retrieved contexts ──
async def run_rag_pipeline(query: str):
    """Run the actual RAG pipeline and return answer + contexts"""
    from app.vector_store import VectorStore
    from app.rag import RAG

    store = VectorStore()
    if not store.load():
        pytest.skip("No vector store found — run indexing first")

    rag = RAG(store)
    result = await rag.search(query)

    # contexts are the retrieved chunks
    contexts = result.get("contexts", [result.get("context", "")])
    return result["context"], contexts


# ── Test 1: Answer Relevancy ──────────────────────────────────────────────
# Does the answer actually address the question?
@pytest.mark.parametrize("case", TEST_CASES)
def test_answer_relevancy(case):
    context, contexts = asyncio.get_event_loop().run_until_complete(
        run_rag_pipeline(case["input"])
    )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=context,
        expected_output=case["expected_output"],
        retrieval_context=contexts,
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge_llm,
        include_reason=True
    )

    assert_test(test_case, [metric])


# ── Test 2: Faithfulness ──────────────────────────────────────────────────
# Is the answer grounded in the retrieved context? (no hallucination)
@pytest.mark.parametrize("case", TEST_CASES)
def test_faithfulness(case):
    context, contexts = asyncio.get_event_loop().run_until_complete(
        run_rag_pipeline(case["input"])
    )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=context,
        retrieval_context=contexts,
    )

    metric = FaithfulnessMetric(
        threshold=0.7,
        model=judge_llm,
        include_reason=True
    )

    assert_test(test_case, [metric])


# ── Test 3: Hallucination ─────────────────────────────────────────────────
# Does the answer contain claims not in context?
@pytest.mark.parametrize("case", TEST_CASES)
def test_no_hallucination(case):
    context, contexts = asyncio.get_event_loop().run_until_complete(
        run_rag_pipeline(case["input"])
    )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=context,
        context=contexts,  # hallucination uses 'context' not 'retrieval_context'
    )

    metric = HallucinationMetric(
        threshold=0.4,  # score below 0.4 means low hallucination
        model=judge_llm,
        include_reason=True
    )

    assert_test(test_case, [metric])


# ── Test 4: Contextual Precision ──────────────────────────────────────────
# Were the most relevant chunks ranked highest?
@pytest.mark.parametrize("case", TEST_CASES)
def test_contextual_precision(case):
    context, contexts = asyncio.get_event_loop().run_until_complete(
        run_rag_pipeline(case["input"])
    )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=context,
        expected_output=case["expected_output"],
        retrieval_context=contexts,
    )

    metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=judge_llm,
        include_reason=True
    )

    assert_test(test_case, [metric])