import json
from pathlib import Path

import joblib
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .deferral import (
    calculate_deferral_metrics,
    make_deferral_decisions,
)

from .simulated_expert import (
    CLASS_NAMES,
    predict_expert_label,
)

from .train_baseline import combine_text


BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = (
    BASE_DIR
    / "learning_to_defer"
    / "artifacts"
)

BASELINE_MODEL_PATH = (
    ARTIFACT_DIR
    / "baseline_pipeline.joblib"
)

MARGIN = 0.0
RANDOM_STATE = 42


def create_expert_predictions(texts, true_labels):
    return np.array(
        [
            predict_expert_label(text, true_label)
            for text, true_label in zip(
                texts,
                true_labels,
            )
        ]
    )


def calculate_per_class_results(
    true_labels,
    ai_predictions,
    expert_predictions,
    team_predictions,
    defer_decisions,
):
    results = []

    for label, class_name in enumerate(CLASS_NAMES):
        mask = true_labels == label

        results.append(
            {
                "label": label,
                "class_name": class_name,
                "test_samples": int(mask.sum()),
                "ai_accuracy": float(
                    accuracy_score(
                        true_labels[mask],
                        ai_predictions[mask],
                    )
                ),
                "expert_accuracy": float(
                    accuracy_score(
                        true_labels[mask],
                        expert_predictions[mask],
                    )
                ),
                "team_accuracy": float(
                    accuracy_score(
                        true_labels[mask],
                        team_predictions[mask],
                    )
                ),
                "deferral_rate": float(
                    defer_decisions[mask].mean()
                ),
            }
        )

    return results


def main():
    # Imported here rather than at module scope: `datasets` is only needed
    # to regenerate the artifacts, so the app, its tests and a plain
    # `pip install -r requirements.txt` do not require it.
    from datasets import load_dataset

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The Task 1 baseline model was not found. "
            "Run the Task 1 training command first."
        )

    print("Loading the Task 1 baseline model...")

    baseline_model = joblib.load(
        BASELINE_MODEL_PATH
    )

    print("Loading the AG News dataset...")

    dataset = load_dataset("fancyzhx/ag_news")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_texts = [
        combine_text(article)
        for article in train_dataset
    ]

    test_texts = [
        combine_text(article)
        for article in test_dataset
    ]

    train_labels = np.array(
        train_dataset["label"]
    )

    test_labels = np.array(
        test_dataset["label"]
    )

    print(
        "Generating simulated expert responses "
        "for the training data..."
    )

    train_expert_predictions = (
        create_expert_predictions(
            train_texts,
            train_labels,
        )
    )

    # The competence target is 1 when the expert
    # was correct and 0 when the expert was wrong.
    expert_correct_train = (
        train_expert_predictions
        == train_labels
    ).astype(int)

    print("Generating AI probabilities...")

    train_ai_probabilities = (
        baseline_model.predict_proba(
            train_texts
        )
    )

    test_ai_probabilities = (
        baseline_model.predict_proba(
            test_texts
        )
    )

    train_ai_confidences = (
        train_ai_probabilities.max(axis=1)
    )

    test_ai_confidences = (
        test_ai_probabilities.max(axis=1)
    )

    test_ai_predictions = (
        test_ai_probabilities.argmax(axis=1)
    )

    print(
        "Training the expert competence model..."
    )

    # The competence model learns:
    #
    # P(expert is correct | AI category probabilities)
    #
    # The four AI probabilities indirectly describe
    # which part of the news space the article belongs to.
    competence_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    competence_model.fit(
        train_ai_probabilities,
        expert_correct_train,
    )

    test_expert_competence = (
        competence_model.predict_proba(
            test_ai_probabilities
        )[:, 1]
    )

    print(
        "Generating simulated expert responses "
        "for the test data..."
    )

    test_expert_predictions = (
        create_expert_predictions(
            test_texts,
            test_labels,
        )
    )

    print("Making deferral decisions...")

    defer_decisions = (
        make_deferral_decisions(
            ai_confidences=test_ai_confidences,
            expert_competence_probabilities=(
                test_expert_competence
            ),
            margin=MARGIN,
        )
    )

    team_predictions = np.where(
        defer_decisions,
        test_expert_predictions,
        test_ai_predictions,
    )

    metrics = calculate_deferral_metrics(
        true_labels=test_labels,
        ai_predictions=test_ai_predictions,
        expert_predictions=(
            test_expert_predictions
        ),
        defer_decisions=defer_decisions,
    )

    metrics["margin"] = MARGIN
    metrics["test_samples"] = len(test_labels)

    metrics["competence_model"] = (
        "Logistic Regression using AI "
        "category probabilities"
    )

    metrics["per_class"] = (
        calculate_per_class_results(
            true_labels=test_labels,
            ai_predictions=test_ai_predictions,
            expert_predictions=(
                test_expert_predictions
            ),
            team_predictions=team_predictions,
            defer_decisions=defer_decisions,
        )
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        ARTIFACT_DIR
        / "deferral_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=2,
        )

    joblib.dump(
        competence_model,
        ARTIFACT_DIR
        / "competence_model_full.joblib",
    )

    joblib.dump(
        {
            "true_labels": test_labels,
            "ai_predictions":
                test_ai_predictions,
            "ai_confidences":
                test_ai_confidences,
            "expert_predictions":
                test_expert_predictions,
            "expert_competence":
                test_expert_competence,
            "defer_decisions":
                defer_decisions,
            "team_predictions":
                team_predictions,
        },
        ARTIFACT_DIR
        / "deferral_test_results.joblib",
    )

    print()
    print(
        f"AI-only accuracy: "
        f"{metrics['ai_accuracy']:.2%}"
    )

    print(
        f"Expert-only accuracy: "
        f"{metrics['expert_accuracy']:.2%}"
    )

    print(
        f"Team accuracy: "
        f"{metrics['team_accuracy']:.2%}"
    )

    print(
        f"Deferral rate: "
        f"{metrics['deferral_rate']:.2%}"
    )

    print(
        f"Deferred accuracy: "
        f"{metrics['deferred_accuracy']:.2%}"
    )

    print(
        f"Non-deferred accuracy: "
        f"{metrics['non_deferred_accuracy']:.2%}"
    )

    print(
        f"Beneficial deferrals: "
        f"{metrics['beneficial_deferrals']}"
    )

    print(
        f"Harmful deferrals: "
        f"{metrics['harmful_deferrals']}"
    )

    print(
        f"Oracle upper-bound accuracy: "
        f"{metrics['oracle_accuracy']:.2%}"
    )

    print()
    print(
        f"Results saved in: {ARTIFACT_DIR}"
    )


if __name__ == "__main__":
    main()