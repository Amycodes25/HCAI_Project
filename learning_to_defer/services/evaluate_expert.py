import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .simulated_expert import (
    CLASS_NAMES,
    DEFAULT_PROFILE,
    predict_expert_label,
)

from .train_baseline import combine_text


BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = (
    BASE_DIR
    / "learning_to_defer"
    / "artifacts"
)


def main():
    # Imported here rather than at module scope: `datasets` is only needed
    # to regenerate the artifacts, so the app, its tests and a plain
    # `pip install -r requirements.txt` do not require it.
    from datasets import load_dataset

    print("Loading the official AG News test split...")

    test_dataset = load_dataset(
        "fancyzhx/ag_news",
        split="test",
    )

    texts = [
        combine_text(article)
        for article in test_dataset
    ]

    true_labels = list(test_dataset["label"])

    print(
        f"Querying the simulated expert for "
        f"{len(texts):,} test articles..."
    )

    expert_predictions = [
        predict_expert_label(text, true_label)
        for text, true_label in zip(
            texts,
            true_labels,
        )
    ]

    overall_accuracy = accuracy_score(
        true_labels,
        expert_predictions,
    )

    report = classification_report(
        true_labels,
        expert_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_results = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_indices = [
            index
            for index, value in enumerate(true_labels)
            if value == label
        ]

        correct_predictions = sum(
            expert_predictions[index]
            == true_labels[index]
            for index in class_indices
        )

        measured_accuracy = (
            correct_predictions / len(class_indices)
        )

        per_class_results.append(
            {
                "label": label,
                "class_name": class_name,
                "designed_accuracy":
                    DEFAULT_PROFILE.accuracies[label],
                "measured_accuracy":
                    round(float(measured_accuracy), 6),
                "test_samples": len(class_indices),
            }
        )

    metrics = {
        "expert_name": DEFAULT_PROFILE.name,
        "description": (
            "An imperfect Business specialist with "
            "class-dependent competence."
        ),
        "overall_accuracy":
            round(float(overall_accuracy), 6),
        "test_samples": len(texts),
        "per_class": per_class_results,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            true_labels,
            expert_predictions,
        ).tolist(),
        "seed": DEFAULT_PROFILE.seed,
        "strength": "Business",
        "weakness": "Sports",
    }

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        ARTIFACT_DIR
        / "expert_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=2,
        )

    predictions_path = (
        ARTIFACT_DIR
        / "expert_test_predictions.joblib"
    )

    joblib.dump(
        {
            "true_labels": true_labels,
            "expert_predictions": expert_predictions,
        },
        predictions_path,
    )

    print()
    print(f"Expert accuracy: {overall_accuracy:.2%}")

    for result in per_class_results:
        print(
            f"{result['class_name']}: "
            f"{result['measured_accuracy']:.2%}"
        )

    print()
    print(f"Results saved in: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()