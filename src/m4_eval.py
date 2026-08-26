from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not questions or not answers:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import math

        answer_relevancy.strictness = 1

        llm = None
        embeddings = None
        if OPENAI_API_KEY:
            llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        eval_kwargs = {"metrics": [faithfulness, answer_relevancy, context_precision, context_recall]}
        if llm is not None:
            eval_kwargs["llm"] = llm
        if embeddings is not None:
            eval_kwargs["embeddings"] = embeddings

        result = evaluate(dataset, **eval_kwargs)
        df = result.to_pandas()

        def _clean_float(v):
            if v is None:
                return 0.0
            try:
                fv = float(v)
                return 0.0 if math.isnan(fv) else fv
            except (ValueError, TypeError):
                return 0.0

        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_clean_float(row.get("faithfulness")),
                answer_relevancy=_clean_float(row.get("answer_relevancy")),
                context_precision=_clean_float(row.get("context_precision")),
                context_recall=_clean_float(row.get("context_recall")),
            ))

        return {
            "faithfulness": _clean_float(result.get("faithfulness", 0.0)),
            "answer_relevancy": _clean_float(result.get("answer_relevancy", 0.0)),
            "context_precision": _clean_float(result.get("context_precision", 0.0)),
            "context_recall": _clean_float(result.get("context_recall", 0.0)),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_items = []
    for r in eval_results:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(scores.values()) / len(scores)
        worst_metric = min(scores.keys(), key=lambda k: scores[k])
        worst_score = scores[worst_metric]
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown error", "Inspect pipeline"))
        scored_items.append({
            "question": r.question,
            "expected": r.ground_truth,
            "got": r.answer,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")

    if "/" not in path and "\\" not in path:
        os.makedirs("reports", exist_ok=True)
        rep_path = os.path.join("reports", path)
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report also saved to {rep_path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
