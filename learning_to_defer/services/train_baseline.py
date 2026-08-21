"""Train and evaluate the Task 1 AG News baseline.

Run from the repository root:
    python -m learning_to_defer.services.train_baseline
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "learning_to_defer" / "artifacts"
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
RANDOM_STATE = 42

def combine_text(example):
    text = str(example.get("text", "")).strip()

    if text:
        return text

    title = str(example.get("title", "")).strip()
    description = str(example.get("description", "")).strip()

    return f"{title} {description}".strip()

def build_pipeline(min_df=2):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    # Imported here rather than at module scope: `datasets` is only needed
    # to regenerate the artifacts, so the app, its tests and a plain
    # `pip install -r requirements.txt` do not require it.
    from datasets import load_dataset

    print("Loading the official AG News train and test splits...")
    dataset = load_dataset("fancyzhx/ag_news")
    train_split = dataset["train"]
    test_split = dataset["test"]

    x_train = [combine_text(row) for row in train_split]
    y_train = list(train_split["label"])
    x_test = [combine_text(row) for row in test_split]
    y_test = list(test_split["label"])

    pipeline = build_pipeline()
    print(f"Training on all {len(x_train):,} labelled training articles...")
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "AG News",
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "test_accuracy": round(float(accuracy), 6),
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "random_state": RANDOM_STATE,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_DIR / "baseline_pipeline.joblib")
    with (ARTIFACT_DIR / "baseline_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Artifacts saved in {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
