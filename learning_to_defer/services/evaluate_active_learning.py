import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.metrics import brier_score_loss, roc_auc_score

from .active_learning import (
    create_initial_query_set,
    select_active_queries,
    select_random_queries,
    train_competence_model,
)
from .deferral import calculate_deferral_metrics, make_deferral_decisions
from .simulated_expert import predict_expert_label
from .train_baseline import combine_text


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "learning_to_defer" / "artifacts"
BASELINE_MODEL_PATH = ARTIFACT_DIR / "baseline_pipeline.joblib"

BUDGETS = [50, 100, 250, 500, 1000, 2000]
RANDOM_STATES = [42, 52, 62, 72, 82]
INITIAL_BUDGET = 40
DEFERRAL_MARGIN = 0.0


def query_expert(texts, true_labels, query_indices):
    """Query the expert only for the selected training examples."""
    correctness = []

    for index in query_indices:
        expert_prediction = predict_expert_label(
            texts[index],
            true_labels[index],
        )
        correctness.append(int(expert_prediction == true_labels[index]))

    return np.asarray(correctness, dtype=int)


def generate_test_expert_predictions(texts, true_labels):
    """Generate expert predictions only for final test evaluation."""
    return np.asarray(
        [
            predict_expert_label(text, true_label)
            for text, true_label in zip(texts, true_labels)
        ],
        dtype=int,
    )


def evaluate_strategy(
    strategy,
    random_state,
    train_texts,
    train_labels,
    train_ai_probabilities,
    train_ai_confidences,
    train_ai_predictions,
    test_labels,
    test_ai_probabilities,
    test_ai_confidences,
    test_ai_predictions,
    test_expert_predictions,
):
    total_training_samples = len(train_labels)

    queried_indices = create_initial_query_set(
        ai_predictions=train_ai_predictions,
        initial_budget=INITIAL_BUDGET,
        random_state=random_state,
    )

    # A value of -1 means that the expert was not queried.
    observed_expert_correctness = np.full(
        total_training_samples,
        -1,
        dtype=int,
    )
    observed_expert_correctness[queried_indices] = query_expert(
        texts=train_texts,
        true_labels=train_labels,
        query_indices=queried_indices,
    )

    strategy_results = []
    final_model = None

    for budget in BUDGETS:
        while len(queried_indices) < budget:
            competence_model = train_competence_model(
                ai_probabilities=train_ai_probabilities,
                expert_correctness=observed_expert_correctness,
                queried_indices=queried_indices,
                random_state=random_state,
            )

            number_to_query = budget - len(queried_indices)

            if strategy == "active":
                new_indices = select_active_queries(
                    competence_model=competence_model,
                    ai_probabilities=train_ai_probabilities,
                    ai_confidences=train_ai_confidences,
                    already_queried=queried_indices,
                    number_to_query=number_to_query,
                )
            elif strategy == "random":
                new_indices = select_random_queries(
                    total_samples=total_training_samples,
                    already_queried=queried_indices,
                    number_to_query=number_to_query,
                    random_state=random_state + budget,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            observed_expert_correctness[new_indices] = query_expert(
                texts=train_texts,
                true_labels=train_labels,
                query_indices=new_indices,
            )
            queried_indices = np.concatenate([queried_indices, new_indices])

        final_model = train_competence_model(
            ai_probabilities=train_ai_probabilities,
            expert_correctness=observed_expert_correctness,
            queried_indices=queried_indices,
            random_state=random_state,
        )

        estimated_test_competence = final_model.predict_proba(
            test_ai_probabilities
        )[:, 1]

        defer_decisions = make_deferral_decisions(
            ai_confidences=test_ai_confidences,
            expert_competence_probabilities=estimated_test_competence,
            margin=DEFERRAL_MARGIN,
        )

        metrics = calculate_deferral_metrics(
            true_labels=test_labels,
            ai_predictions=test_ai_predictions,
            expert_predictions=test_expert_predictions,
            defer_decisions=defer_decisions,
        )

        # The true competence target is 1 when the expert is correct.
        # Test labels are used here only for final evaluation.
        test_expert_correctness = (
            test_expert_predictions == test_labels
        ).astype(int)

        metrics["competence_roc_auc"] = float(
            roc_auc_score(
                test_expert_correctness,
                estimated_test_competence,
            )
        )
        metrics["competence_brier_score"] = float(
            brier_score_loss(
                test_expert_correctness,
                estimated_test_competence,
            )
        )

        metrics.update(
            {
                "strategy": strategy,
                "budget": int(budget),
                "random_state": int(random_state),
                "queried_examples": int(len(queried_indices)),
            }
        )
        strategy_results.append(metrics)

    return strategy_results, final_model, queried_indices


def aggregate_results(all_results):
    """Calculate means and standard deviations across five runs."""
    summary = []
    metric_names = [
        "team_accuracy",
        "deferral_rate",
        "deferred_accuracy",
        "non_deferred_accuracy",
        "beneficial_deferrals",
        "harmful_deferrals",
        "beneficial_deferral_rate",
        "harmful_deferral_rate",
        "competence_roc_auc",
        "competence_brier_score",
    ]

    for strategy in ["random", "active"]:
        for budget in BUDGETS:
            matching_results = [
                result
                for result in all_results
                if result["strategy"] == strategy
                and result["budget"] == budget
            ]

            row = {
                "strategy": strategy,
                "budget": budget,
                "repetitions": len(matching_results),
            }

            for metric_name in metric_names:
                values = np.asarray(
                    [result[metric_name] for result in matching_results],
                    dtype=float,
                )
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(
                    values.std(ddof=1) if len(values) > 1 else 0.0
                )

            summary.append(row)

    return summary


def main():
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Task 1 model not found. Run the baseline training first."
        )

    print("Loading the Task 1 model...")
    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    print("Loading AG News...")
    dataset = load_dataset("fancyzhx/ag_news")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_texts = [combine_text(article) for article in train_dataset]
    test_texts = [combine_text(article) for article in test_dataset]
    train_labels = np.asarray(train_dataset["label"], dtype=int)
    test_labels = np.asarray(test_dataset["label"], dtype=int)

    print("Generating AI probabilities...")
    train_ai_probabilities = baseline_model.predict_proba(train_texts)
    test_ai_probabilities = baseline_model.predict_proba(test_texts)
    train_ai_confidences = train_ai_probabilities.max(axis=1)
    test_ai_confidences = test_ai_probabilities.max(axis=1)
    train_ai_predictions = train_ai_probabilities.argmax(axis=1)
    test_ai_predictions = test_ai_probabilities.argmax(axis=1)

    test_expert_predictions = generate_test_expert_predictions(
        texts=test_texts,
        true_labels=test_labels,
    )

    all_results = []
    saved_active_model = None
    saved_active_indices = None

    for strategy in ["random", "active"]:
        print(f"\nEvaluating {strategy} querying...")

        for random_state in RANDOM_STATES:
            print(f"  Random state: {random_state}")
            strategy_results, final_model, queried_indices = evaluate_strategy(
                strategy=strategy,
                random_state=random_state,
                train_texts=train_texts,
                train_labels=train_labels,
                train_ai_probabilities=train_ai_probabilities,
                train_ai_confidences=train_ai_confidences,
                train_ai_predictions=train_ai_predictions,
                test_labels=test_labels,
                test_ai_probabilities=test_ai_probabilities,
                test_ai_confidences=test_ai_confidences,
                test_ai_predictions=test_ai_predictions,
                test_expert_predictions=test_expert_predictions,
            )
            all_results.extend(strategy_results)

            # Save the first active run as the reproducible demonstration model.
            if strategy == "active" and random_state == RANDOM_STATES[0]:
                saved_active_model = final_model
                saved_active_indices = queried_indices

    summary = aggregate_results(all_results)
    output = {
        "budgets": BUDGETS,
        "random_states": RANDOM_STATES,
        "initial_budget": INITIAL_BUDGET,
        "deferral_margin": DEFERRAL_MARGIN,
        "number_of_training_articles": len(train_labels),
        "number_of_test_articles": len(test_labels),
        "all_runs": all_results,
        "summary": summary,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_DIR / "active_learning_metrics.json").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(output, metrics_file, indent=2)

    joblib.dump(
        saved_active_model,
        ARTIFACT_DIR / "competence_model_active_2000.joblib",
    )
    joblib.dump(
        saved_active_indices,
        ARTIFACT_DIR / "active_query_indices_2000.joblib",
    )

    print("\nSummary across five runs")
    print("=" * 110)

    for row in summary:
        print(
            f"{row['strategy'].upper():6} | "
            f"Budget {row['budget']:4} | "
            f"Team {row['team_accuracy_mean']:.2%} "
            f"± {row['team_accuracy_std']:.2%} | "
            f"Deferral {row['deferral_rate_mean']:.2%} | "
            f"AUC {row['competence_roc_auc_mean']:.3f} | "
            f"Brier {row['competence_brier_score_mean']:.3f} | "
            f"Useful {row['beneficial_deferral_rate_mean']:.2%}"
        )

    print(f"\nResults saved in: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
