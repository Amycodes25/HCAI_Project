import json
from pathlib import Path

import joblib
import numpy as np

from datasets import load_dataset

from .simulated_expert import CLASS_NAMES
from .train_baseline import combine_text


BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = (
    BASE_DIR
    / "learning_to_defer"
    / "artifacts"
)

POOL_SIZE = 20


def main():
    baseline_path = (
        ARTIFACT_DIR
        / "baseline_pipeline.joblib"
    )

    competence_model_path = (
        ARTIFACT_DIR
        / "competence_model_active_2000.joblib"
    )

    query_indices_path = (
        ARTIFACT_DIR
        / "active_query_indices_2000.joblib"
    )

    required_files = [
        baseline_path,
        competence_model_path,
        query_indices_path,
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required artifact not found: "
                f"{required_file}"
            )

    print("Loading trained models...")

    baseline_model = joblib.load(
        baseline_path
    )

    competence_model = joblib.load(
        competence_model_path
    )

    active_query_indices = joblib.load(
        query_indices_path
    )

    print("Loading AG News training data...")

    train_dataset = load_dataset(
        "fancyzhx/ag_news",
        split="train",
    )

    # The final selected articles come from the
    # active-learning part of the query sequence.
    selected_indices = np.asarray(
        active_query_indices[-POOL_SIZE:],
        dtype=int,
    )

    pool = []

    print("Preparing human-expert articles...")

    for position, dataset_index in enumerate(
        selected_indices
    ):
        article = train_dataset[
            int(dataset_index)
        ]

        text = combine_text(article)
        true_label = int(article["label"])

        ai_probabilities = (
            baseline_model.predict_proba(
                [text]
            )[0]
        )

        ai_prediction = int(
            ai_probabilities.argmax()
        )

        ai_confidence = float(
            ai_probabilities.max()
        )

        estimated_expert_competence = float(
            competence_model.predict_proba(
                ai_probabilities.reshape(1, -1)
            )[0, 1]
        )

        pool.append(
            {
                "pool_position": position,
                "dataset_index": int(
                    dataset_index
                ),
                "text": text,

                # These values remain hidden until
                # the human submits an answer.
                "true_label": true_label,
                "true_category":
                    CLASS_NAMES[true_label],

                "ai_prediction": ai_prediction,
                "ai_category":
                    CLASS_NAMES[ai_prediction],
                "ai_confidence":
                    ai_confidence,

                "estimated_expert_competence":
                    estimated_expert_competence,

                "boundary_distance": float(
                    abs(
                        estimated_expert_competence
                        - ai_confidence
                    )
                ),
            }
        )

    output_path = (
        ARTIFACT_DIR
        / "human_expert_pool.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "pool_size": len(pool),
                "selection_method": (
                    "Articles selected near the "
                    "AI-expert deferral boundary"
                ),
                "articles": pool,
            },
            output_file,
            indent=2,
        )

    print(
        f"Prepared {len(pool)} articles."
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()