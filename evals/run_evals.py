from deepeval import dataset, evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
import anthropic
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from chatbot import ask


class AnthropicJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._client = anthropic.Anthropic()
        self._async_client = anthropic.AsyncAnthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        kwargs = {"system": "Respond only with valid JSON, no other text."} if schema else {}
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text

    async def a_generate(self, prompt: str, schema=None) -> str:
        kwargs = {"system": "Respond only with valid JSON, no other text."} if schema else {}
        response = await self._async_client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text

    def get_model_name(self) -> str:
        return self.model


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "golden_dataset.json"
RESULTS_DIR = BASE_DIR / "results"
PIPELINE_CACHE_PATH = BASE_DIR / "pipeline_cache.json"


##########################################################################
#                               METRICS
##########################################################################

JUDGE_MODEL = AnthropicJudge("claude-haiku-4-5-20251001")

# We check for relevantness, faithfulness, contextual recall, and hallucination
metrics = [
    AnswerRelevancyMetric(      # Do we answer the question that was asked?
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True,
    ),
    FaithfulnessMetric(         # Is the answer supported by the retrieved source?
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True,
    ),
    ContextualRecallMetric(     # Did we retrieve relevant chunks to answer this question
        # If it's low but the answer is correct, the issue is with chunking, not generation
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True,
    ),
]


def load_golden_dataset() -> list[dict]:
    """We load our golden dataset"""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


##########################################################################
#                            TEST CASES
##########################################################################


def build_test_cases(dataset: list[dict]) -> list[LLMTestCase]:
    """
    The golden dataset contains 20 Q&A pairs.
    For each pair, we call chatbot.ask() to get the pipeline's actual answer.
    Then we create an LLMTestCase containing question, answer, expected answer, and retrieved context.
    Results are cached to pipeline_cache.json — delete it to force a fresh run.
    """
    if PIPELINE_CACHE_PATH.exists():
        print("  Loading pipeline outputs from cache (delete pipeline_cache.json to re-run)\n")
        with open(PIPELINE_CACHE_PATH, "r", encoding="utf-8") as f:
            return [LLMTestCase(**item) for item in json.load(f)]

    test_cases = []
    cache_data = []

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["expected_answer"]

        print(f"  [{i + 1}/{len(dataset)}] Running pipeline for: {question}")
        result = ask(question)

        fields = dict(
            input=question,
            actual_output=result["answer"],
            expected_output=expected_answer,
            retrieval_context=result["context_used"],
            context=result["context_used"],
        )
        test_cases.append(LLMTestCase(**fields))
        cache_data.append(fields)

    with open(PIPELINE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    return test_cases


def save_summary(test_cases: list[LLMTestCase]) -> None:
    """
    Extract scores from evaluated test cases and save the summary into a JSON file.
    DeepEval attaches metric results to each test case after evaluate() runs.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    metric_names = [
        "Answer Relevancy",
        "Faithfulness",
        "Contextual Recall",
    ]

    # Collect per-question results
    questions_results = []
    for case in test_cases:
        q_result = {"question": case.input, "metrics": {}}
        for metric_result in case.metrics_data:
            q_result["metrics"][metric_result.name] = {
                "score": round(metric_result.score, 3) if metric_result.score is not None else None,
                "passed": metric_result.success,
                "reason": metric_result.reason
            }
        questions_results.append(q_result)

    # Compute per-metric averages and pass rates
    aggregates = {}
    for name in metric_names:
        scores = [
            q["metrics"][name]["score"]
            for q in questions_results
            if name in q["metrics"]
        ]
        passing = [s for s in scores if s is not None]
        aggregates[name] = {
            "mean_score": round(sum(passing) / len(passing), 3) if passing else None,
            "pass_rate": round(
                sum(1 for q in questions_results
                    if q["metrics"].get(name, {}).get("passed", False))
                / len(questions_results), 3
            )
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_cases": len(test_cases),
        "aggregates": aggregates,
        "questions": questions_results
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved → {out_path}")


def print_summary(summary: dict) -> None:
    """Print a nice summary of results to the console
    Because I like neat tables :)"""
    print("\n" + "=" * 60)
    print("EVAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Timestamp:   {summary['timestamp']}")
    print(f"Test cases:  {summary['total_cases']}")
    print()

    for metric_name, agg in summary["aggregates"].items():
        short_name = metric_name
        mean = agg["mean_score"]
        rate = agg["pass_rate"]
        bar = "█" * int((mean or 0) * 20)
        mean_str = f"{mean:.3f}" if mean is not None else "  N/A"
        print(f"  {short_name:<22} {bar:<20} {mean_str}  ({rate:.0%} pass rate)")

    print("=" * 60)


##########################################################################
#                            MAIN FUNCTION
##########################################################################

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first 5 test cases (for CI smoke tests)"
    )
    args = parser.parse_args()

    print("── Loading golden dataset ──")
    dataset = load_golden_dataset()

    if args.smoke:
      dataset = dataset[:5]
      print(f"  Smoke mode: running {len(dataset)} of {len(load_golden_dataset())} cases\n")
    else:
      print(f"  {len(dataset)} test cases loaded\n")

    print("── Running pipeline for each question ──")
    test_cases = build_test_cases(dataset)

    print("\n── Running DeepEval metrics ──")
    eval_result = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    print("\n── Saving results ──")
    RESULTS_DIR.mkdir(exist_ok=True)
    metric_names = [
        "Answer Relevancy",
        "Faithfulness",
        "Contextual Recall",
    ]
    questions_results = []
    for result, case in zip(eval_result.test_results, test_cases):
        q_result = {"question": case.input, "metrics": {}}
        for metric_result in (result.metrics_data or []):
            q_result["metrics"][metric_result.name] = {
                "score": round(metric_result.score, 3) if metric_result.score is not None else None,
                "passed": metric_result.success,
                "reason": metric_result.reason
            }
        questions_results.append(q_result)

    aggregates = {}
    for name in metric_names:
        scores = [
            q["metrics"][name]["score"]
            for q in questions_results
            if name in q["metrics"]
        ]
        passing = [s for s in scores if s is not None]
        aggregates[name] = {
            "mean_score": round(sum(passing) / len(passing), 3) if passing else None,
            "pass_rate": round(
                sum(1 for q in questions_results
                    if q["metrics"].get(name, {}).get("passed", False))
                / len(questions_results), 3
            )
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_cases": len(test_cases),
        "aggregates": aggregates,
        "questions": questions_results
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved → {out_path}")
    print_summary(summary)


    # GitHub Actions only knows a step failed if the process exits with a non-zero code, so
    # after saving summary, we check whether any metric failed threshold
    any_failed = any(
        not q["metrics"].get(name, {}).get("passed", True)
        for q in questions_results
        for name in metric_names
    )

    if any_failed:
        print("\n✗ One or more metrics failed threshold. See summary for details.")
        sys.exit(1)    # non-zero exit code to fail the GitHub Actions step in case of failure
    else:
        print("\n✓ All metrics passed threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
