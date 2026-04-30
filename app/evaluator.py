from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset
import asyncio

async def evaluate_response(query, answer, contexts):
    # Build a single-row dataset — RAGAS expects a dataset format
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],  # list of retrieved chunks as strings
    }

    dataset = Dataset.from_dict(data)

    # Run evaluation in a thread to avoid blocking the event loop
    result = await asyncio.to_thread(
        evaluate,
        dataset=dataset,
        metrics=[
            faithfulness,        # is the answer grounded in the context?
            answer_relevancy,    # is the answer relevant to the question?
            context_precision,   # are the retrieved chunks actually useful?
        ]
    )

    return {
        "faithfulness": round(result["faithfulness"], 3),
        "answer_relevancy": round(result["answer_relevancy"], 3),
        "context_precision": round(result["context_precision"], 3),
    }