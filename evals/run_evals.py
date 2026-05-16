from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import OpenAI, AsyncOpenAI

import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from chatbot import ask


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "qwen2.5:14b"):
        self.model = model
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self._async_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if schema else {}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if schema else {}
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "golden_dataset.json"
RESULTS_DIR = BASE_DIR / "results"
PIPELINE_CACHE_PATH = BASE_DIR / "pipeline_cache.json"


##########################################################################
#                               METRICS
##########################################################################

JUDGE_MODEL = OllamaJudge("qwen2.5:14b")

# We check for relevantness, faithfulness, contextual recall, and hallucination
metrics = [
    AnswerRelevancyMetric(      # Do we answer the question that was asked?
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=False,
    ),
    FaithfulnessMetric(         # Is the answer supported by the retrieved source?
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=False,
    ),
    ContextualRecallMetric(     # Did we retrieve relevant chunks to answer this question
        # If it's low but the answer is correct, the issue is with chunking, not generation
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=False,
    ),
    HallucinationMetric(
        threshold=0.3,         # Does the context contradict the answer? We want this to be low
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=False,
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
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualRecallMetric",
        "HallucinationMetric",
    ]

    # Collect per-question results
    questions_results = []
    for case in test_cases:
        q_result = {"question": case.input, "metrics": {}}
        for metric_result in case.metrics_data:
            q_result["metrics"][metric_result.name] = {
                "score": round(metric_result.score, 3),
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
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
        short_name = metric_name.replace("Metric", "")
        mean = agg["mean_score"]
        rate = agg["pass_rate"]
        bar = "█" * int((mean or 0) * 20)
        print(f"  {short_name:<22} {bar:<20} {mean:.3f}  ({rate:.0%} pass rate)")

    print("=" * 60)


##########################################################################
#                            MAIN FUNCTION
##########################################################################

def main():
    print("── Loading golden dataset ──")
    dataset = load_golden_dataset()
    print(f"  {len(dataset)} test cases loaded\n")

    print("── Running pipeline for each question ──")
    test_cases = build_test_cases(dataset)

    print("\n── Running DeepEval metrics ──")
    evaluate(test_cases, metrics, async_config=AsyncConfig(run_async=False))

    print("\n── Saving results ──")
    # Rebuild summary from evaluated test cases
    # (evaluate() mutates the test_case objects in place with metric results)
    RESULTS_DIR.mkdir(exist_ok=True)
    metric_names = [
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualRecallMetric",
        "HallucinationMetric",
    ]
    questions_results = []
    for case in test_cases:
        q_result = {"question": case.input, "metrics": {}}
        for metric_result in case.metrics_data:
            q_result["metrics"][metric_result.name] = {
                "score": round(metric_result.score, 3),
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_cases": len(test_cases),
        "aggregates": aggregates,
        "questions": questions_results
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved → {out_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
