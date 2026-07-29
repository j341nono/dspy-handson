from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Literal, Sequence

import dspy
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. DSPy Signature / Module
# -----------------------------------------------------------------------------

DifficultyLabel = Literal["L0", "L1", "L2"]
ALLOWED_LABELS = ("L0", "L1", "L2")


class ClassifySentenceDifficulty(dspy.Signature):
    """Assign exactly one label, L0, L1, or L2, to the English sentence."""

    sentence: str = dspy.InputField(desc="English sentence to classify")
    level: DifficultyLabel = dspy.OutputField(
        desc="Exactly one label from L0, L1, or L2"
    )


class SentenceDifficultyProgram(dspy.Module):
    """A one-predictor DSPy program that GEPA can optimize."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = dspy.Predict(ClassifySentenceDifficulty)

    def forward(self, sentence: str) -> dspy.Prediction:
        return self.classifier(sentence=sentence)


# -----------------------------------------------------------------------------
# 2. Dataset
#
# The initial Signature intentionally does not explain what L0/L1/L2 mean.
# GEPA receives the hidden rubric through textual feedback from the metric.
# -----------------------------------------------------------------------------


def make_example(sentence: str, level: DifficultyLabel, diagnosis: str) -> dspy.Example:
    return dspy.Example(
        sentence=sentence,
        level=level,
        diagnosis=diagnosis,
    ).with_inputs("sentence")


def build_datasets() -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    easy_reason = (
        "L0 means easy: mostly common vocabulary, one short independent clause, "
        "and little or no syntactic embedding."
    )
    medium_reason = (
        "L1 means medium: usually one subordinate or relative clause, moderate abstraction, "
        "or a moderately complex relation between ideas."
    )
    hard_reason = (
        "L2 means hard: multiple or deeply embedded clauses, long-distance dependencies, "
        "dense nominalization, technical vocabulary, or highly abstract reasoning."
    )

    trainset = [
        # L0: easy
        make_example("The cat is sleeping on the sofa.", "L0", easy_reason),
        make_example("Mia opened the window.", "L0", easy_reason),
        make_example("We walk to school every morning.", "L0", easy_reason),
        make_example("Please put the book on the desk.", "L0", easy_reason),
        make_example("My brother made soup for dinner.", "L0", easy_reason),
        make_example("The train arrived ten minutes late.", "L0", easy_reason),
        # L1: medium
        make_example(
            "Although it was raining, the team continued the match.",
            "L1",
            medium_reason,
        ),
        make_example(
            "The museum, which opened last year, attracts many visitors.",
            "L1",
            medium_reason,
        ),
        make_example(
            "If you finish the report today, we can review it tomorrow.",
            "L1",
            medium_reason,
        ),
        make_example(
            "The teacher explained why the experiment had failed.",
            "L1",
            medium_reason,
        ),
        make_example(
            "People often choose public transport because parking is expensive.",
            "L1",
            medium_reason,
        ),
        make_example(
            "The company plans to reduce waste by reusing packaging.",
            "L1",
            medium_reason,
        ),
        # L2: hard
        make_example(
            "The committee's recommendation, which was formulated after months of deliberation, "
            "was rejected on procedural grounds.",
            "L2",
            hard_reason,
        ),
        make_example(
            "Because the variables interact nonlinearly, interpreting the model's coefficients "
            "without additional assumptions can be misleading.",
            "L2",
            hard_reason,
        ),
        make_example(
            "The legislation authorizes regulators to impose sanctions whenever an institution "
            "fails to demonstrate adequate risk controls.",
            "L2",
            hard_reason,
        ),
        make_example(
            "Although the theory appears internally consistent, its explanatory power diminishes "
            "when the underlying distribution shifts.",
            "L2",
            hard_reason,
        ),
        make_example(
            "The proposal presupposes that participants can distinguish correlation from causation "
            "despite incomplete observational evidence.",
            "L2",
            hard_reason,
        ),
        make_example(
            "By foregrounding epistemic uncertainty, the author challenges conventional accounts "
            "of scientific objectivity.",
            "L2",
            hard_reason,
        ),
    ]

    valset = [
        make_example("The children played in the park.", "L0", easy_reason),
        make_example("I forgot my umbrella at home.", "L0", easy_reason),
        make_example("This shop closes at six.", "L0", easy_reason),
        make_example(
            "After the meeting ended, I wrote down the main decisions.",
            "L1",
            medium_reason,
        ),
        make_example(
            "The novel describes a family that moves to a remote island.",
            "L1",
            medium_reason,
        ),
        make_example(
            "Many students find the course useful even though it is demanding.",
            "L1",
            medium_reason,
        ),
        make_example(
            "The seemingly minor amendment has ramifications that extend beyond the jurisdiction "
            "in which it was enacted.",
            "L2",
            hard_reason,
        ),
        make_example(
            "What renders the argument difficult to evaluate is the absence of a clearly specified "
            "counterfactual baseline.",
            "L2",
            hard_reason,
        ),
        make_example(
            "The report attributes the discrepancy to measurement artifacts whose effects "
            "accumulate across successive processing stages.",
            "L2",
            hard_reason,
        ),
    ]

    # The test set is never passed to GEPA.
    testset = [
        make_example("She sent me a short message.", "L0", easy_reason),
        make_example("The dog followed the boy.", "L0", easy_reason),
        make_example("Our room has two windows.", "L0", easy_reason),
        make_example(
            "The city introduced a program to help residents save energy.",
            "L1",
            medium_reason,
        ),
        make_example(
            "While the device is affordable, its battery life is limited.",
            "L1",
            medium_reason,
        ),
        make_example(
            "The article compares two methods for learning vocabulary.",
            "L1",
            medium_reason,
        ),
        make_example(
            "Notwithstanding its empirical advantages, the framework remains vulnerable to "
            "identifiability problems and sampling bias.",
            "L2",
            hard_reason,
        ),
        make_example(
            "The extent to which lexical sophistication predicts comprehension depends on how "
            "syntactic complexity is operationalized.",
            "L2",
            hard_reason,
        ),
        make_example(
            "Had the intervention been implemented earlier, the observed decline might have been "
            "partially mitigated.",
            "L2",
            hard_reason,
        ),
    ]

    return trainset, valset, testset


# -----------------------------------------------------------------------------
# 3. GEPA feedback metric
#
# Current DSPy GEPA expects five compatible arguments. The returned Prediction
# contains both a scalar score and textual feedback used for reflection.
# -----------------------------------------------------------------------------


def normalize_label(value: object) -> str:
    match = re.search(r"\bL[012]\b", str(value).upper())
    return match.group(0) if match else "INVALID"


def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> dspy.Prediction:
    del trace, pred_trace  # Not needed in this one-predictor exercise.

    gold_label = normalize_label(gold.level)
    predicted_label = normalize_label(getattr(pred, "level", ""))
    predictor = pred_name or "program"

    if predicted_label == gold_label:
        return dspy.Prediction(
            score=1.0,
            feedback=(
                f"Correct classification by {predictor}: {gold_label}. "
                f"Relevant rubric: {gold.diagnosis}"
            ),
        )

    if predicted_label == "INVALID":
        error = (
            f"The output was not one of {ALLOWED_LABELS}. Return exactly one valid label."
        )
    else:
        error = f"Predicted {predicted_label}, but the correct label is {gold_label}."

    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"{error} Sentence: {gold.sentence!r} "
            f"Use this hidden rubric to improve the instruction: {gold.diagnosis}"
        ),
    )


# -----------------------------------------------------------------------------
# 4. Evaluation and inspection utilities
# -----------------------------------------------------------------------------


def evaluate_program(
    program: dspy.Module,
    dataset: Sequence[dspy.Example],
    title: str,
) -> float:
    correct = 0
    print(f"\n=== {title} ===")

    for index, example in enumerate(dataset, start=1):
        try:
            prediction = program(sentence=example.sentence)
            predicted_label = normalize_label(getattr(prediction, "level", ""))
            error_text = ""
        except Exception as exc:  # Keep the exercise readable if one call fails.
            predicted_label = "ERROR"
            error_text = f" ({type(exc).__name__}: {exc})"

        is_correct = predicted_label == example.level
        correct += int(is_correct)
        mark = "OK" if is_correct else "NG"
        print(
            f"[{index:02d}] {mark}  gold={example.level:<2} pred={predicted_label:<7} "
            f"sentence={example.sentence}{error_text}"
        )

    accuracy = correct / len(dataset)
    print(f"Accuracy: {correct}/{len(dataset)} = {accuracy:.3f}")
    return accuracy


def print_instructions(program: dspy.Module, title: str) -> None:
    print(f"\n=== {title} ===")
    for name, predictor in program.named_predictors():
        instructions = getattr(predictor.signature, "instructions", "<unavailable>")
        print(f"\nPredictor: {name}\n{instructions}")


def print_gepa_stats(program: dspy.Module) -> None:
    details = getattr(program, "detailed_results", None)
    if details is None:
        return

    scores = getattr(details, "val_aggregate_scores", None)
    total_calls = getattr(details, "total_metric_calls", None)

    print("\n=== GEPA statistics ===")
    if scores:
        print(f"Candidates evaluated: {len(scores)}")
        print(f"Best validation aggregate score: {max(scores):.3f}")
    if total_calls is not None:
        print(f"Total metric calls: {total_calls}")


# -----------------------------------------------------------------------------
# 5. Main hand-on flow
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSPy + GEPA hands-on: sentence difficulty classification"
    )
    parser.add_argument(
        "--budget",
        choices=("light", "medium", "heavy"),
        default=os.getenv("GEPA_BUDGET", "light"),
        help="GEPA optimization budget (default: light)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("GEPA_THREADS", "2")),
        help="Parallel LM calls; lower this if rate-limited (default: 2)",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Show the latest task-LM request after the run",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill in the model name."
        )
    return value


def main() -> None:
    load_dotenv()
    args = parse_args()

    task_model_name = require_env("TASK_MODEL")
    reflection_model_name = os.getenv("REFLECTION_MODEL", "").strip() or task_model_name

    # Provider credentials are read by LiteLLM/DSPy from standard environment
    # variables such as OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY.
    task_lm = dspy.LM(
        task_model_name,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.0,
        max_tokens=200,
    )
    reflection_lm = dspy.LM(
        reflection_model_name,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=1.0,
        max_tokens=8000,
    )
    dspy.configure(lm=task_lm)

    trainset, valset, testset = build_datasets()
    student = SentenceDifficultyProgram()

    print("Task model:", task_model_name)
    print("Reflection model:", reflection_model_name)
    print(f"Data: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    # A. Baseline: the opaque labels have not yet been explained.
    print_instructions(student, "Initial instruction")
    evaluate_program(student, valset, "Baseline validation")
    baseline_test = evaluate_program(student, testset, "Baseline test")

    # B. GEPA compile: trainset produces reflective updates; valset selects prompts.
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    optimizer = dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        auto=args.budget,
        num_threads=args.threads,
        track_stats=True,
        log_dir=str(artifacts_dir / "gepa_logs"),
        seed=0,
    )

    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
    )

    # C. Inspect and evaluate the optimized program.
    print_instructions(optimized, "GEPA-optimized instruction")
    evaluate_program(optimized, valset, "Optimized validation")
    optimized_test = evaluate_program(optimized, testset, "Optimized test")
    print_gepa_stats(optimized)

    output_path = artifacts_dir / "sentence_difficulty_gepa.json"
    optimized.save(str(output_path))

    print("\n=== Summary ===")
    print(f"Baseline test accuracy : {baseline_test:.3f}")
    print(f"Optimized test accuracy: {optimized_test:.3f}")
    print(f"Saved optimized program: {output_path}")

    # D. Try the optimized program on unseen sentences.
    new_sentences = [
        "The baby is asleep.",
        "Because the road was closed, we took a different route.",
        "The interpretation hinges on assumptions whose validity cannot be independently verified.",
    ]
    print("\n=== New sentences ===")
    for sentence in new_sentences:
        result = optimized(sentence=sentence)
        print(f"{normalize_label(result.level):<2}  {sentence}")

    if args.show_history:
        task_lm.inspect_history(n=1)


if __name__ == "__main__":
    main()


