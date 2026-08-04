"""
Continuous RAG evaluation using Ragas.

Metrics computed per query:
  - faithfulness         : is the answer factually grounded in retrieved context?
                            (this is the core hallucination check)
  - answer_relevancy     : does the answer actually address the question?
  - context_precision    : are the top-ranked retrieved chunks the relevant ones?
  - context_recall       : did retrieval surface enough of the needed info?
                            (requires a ground_truth reference answer)

Run standalone:
    python evaluation/evaluate.py --testset evaluation/testset.json

Or import `run_evaluation()` from the Streamlit app for an on-demand
"Evaluate this answer" button.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag_pipeline import RAGPipeline, RAGResponse


def build_ragas_dataset(
    questions: List[str],
    responses: List[RAGResponse],
    ground_truths: Optional[List[str]] = None,
) -> Dataset:
    records = {
        "question": questions,
        "answer": [r.answer for r in responses],
        "contexts": [[d.page_content for d in r.retrieved_docs] for r in responses],
    }
    if ground_truths:
        records["ground_truth"] = ground_truths
    return Dataset.from_dict(records)


def run_evaluation(
    pipeline: RAGPipeline,
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
) -> dict:
    """
    Runs the pipeline over `questions`, then scores the results with Ragas.
    Returns a dict of metric_name -> average_score (0-1) plus the per-row
    DataFrame for drill-down in the UI.
    """
    responses = [pipeline.query(q) for q in questions]
    dataset = build_ragas_dataset(questions, responses, ground_truths)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths:
        metrics.append(context_recall)  # needs ground_truth column

    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    summary = {m.name: round(float(df[m.name].mean()), 3) for m in metrics}
    return {"summary": summary, "detail": df, "responses": responses}


def _cli():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on a test set")
    parser.add_argument(
        "--testset",
        default=os.path.join(os.path.dirname(__file__), "testset.json"),
        help="JSON file: [{'question': ..., 'ground_truth': ...}, ...]",
    )
    args = parser.parse_args()

    with open(args.testset) as f:
        testset = json.load(f)

    questions = [row["question"] for row in testset]
    ground_truths = [row.get("ground_truth") for row in testset]
    has_gt = all(ground_truths)

    from src.vector_store import VectorStoreManager
    from src.hybrid_search import HybridRetriever

    store = VectorStoreManager()
    store.load_or_create()
    hybrid = HybridRetriever(store)
    # NOTE: sparse (BM25) index must be built from the same chunks used at
    # ingestion time. For a standalone eval run, re-ingest here if needed:
    #   from src.ingestion import ingest_pipeline
    #   chunks = ingest_pipeline(["data/sample_contract.pdf"])
    #   hybrid.build_sparse_index(chunks)

    pipeline = RAGPipeline(hybrid)

    results = run_evaluation(pipeline, questions, ground_truths if has_gt else None)
    print("\n=== Ragas Evaluation Summary ===")
    for metric, score in results["summary"].items():
        print(f"  {metric:>20}: {score}")
    results["detail"].to_csv("evaluation/last_run_detail.csv", index=False)
    print("\nPer-question detail saved to evaluation/last_run_detail.csv")


if __name__ == "__main__":
    _cli()
