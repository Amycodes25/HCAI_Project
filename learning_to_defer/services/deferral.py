import numpy as np


def make_deferral_decisions(
    ai_confidences,
    expert_competence_probabilities,
    margin=0.0,
):
    """
    Defer when the estimated expert competence is greater
    than the AI confidence by at least the chosen margin.
    """

    ai_confidences = np.asarray(ai_confidences)

    expert_competence_probabilities = np.asarray(
        expert_competence_probabilities
    )

    return (
        expert_competence_probabilities
        > ai_confidences + margin
    )


def create_team_predictions(
    ai_predictions,
    expert_predictions,
    defer_decisions,
):
    """
    Use the expert prediction for deferred articles.
    Otherwise, use the AI prediction.
    """

    ai_predictions = np.asarray(ai_predictions)
    expert_predictions = np.asarray(expert_predictions)
    defer_decisions = np.asarray(defer_decisions)

    return np.where(
        defer_decisions,
        expert_predictions,
        ai_predictions,
    )


def calculate_deferral_metrics(
    true_labels,
    ai_predictions,
    expert_predictions,
    defer_decisions,
):
    true_labels = np.asarray(true_labels)
    ai_predictions = np.asarray(ai_predictions)
    expert_predictions = np.asarray(expert_predictions)
    defer_decisions = np.asarray(defer_decisions)

    team_predictions = create_team_predictions(
        ai_predictions,
        expert_predictions,
        defer_decisions,
    )

    ai_correct = ai_predictions == true_labels
    expert_correct = expert_predictions == true_labels
    team_correct = team_predictions == true_labels

    deferred_count = int(defer_decisions.sum())
    non_deferred_count = int(
        (~defer_decisions).sum()
    )

    beneficial_deferrals = (
        defer_decisions
        & expert_correct
        & ~ai_correct
    )

    harmful_deferrals = (
        defer_decisions
        & ai_correct
        & ~expert_correct
    )

    oracle_predictions_correct = (
        ai_correct | expert_correct
    )

    metrics = {
        "ai_accuracy": float(ai_correct.mean()),
        "expert_accuracy": float(expert_correct.mean()),
        "team_accuracy": float(team_correct.mean()),
        "deferral_rate": float(defer_decisions.mean()),
        "coverage": float((~defer_decisions).mean()),
        "deferred_count": deferred_count,
        "non_deferred_count": non_deferred_count,
        "beneficial_deferrals": int(
            beneficial_deferrals.sum()
        ),
        "harmful_deferrals": int(
            harmful_deferrals.sum()
        ),
        "oracle_accuracy": float(
            oracle_predictions_correct.mean()
        ),
    }

    if deferred_count > 0:
        metrics["deferred_accuracy"] = float(
            expert_correct[defer_decisions].mean()
        )

        metrics["beneficial_deferral_rate"] = float(
            beneficial_deferrals.sum()
            / deferred_count
        )

        metrics["harmful_deferral_rate"] = float(
            harmful_deferrals.sum()
            / deferred_count
        )
    else:
        metrics["deferred_accuracy"] = 0.0
        metrics["beneficial_deferral_rate"] = 0.0
        metrics["harmful_deferral_rate"] = 0.0

    if non_deferred_count > 0:
        metrics["non_deferred_accuracy"] = float(
            ai_correct[~defer_decisions].mean()
        )
    else:
        metrics["non_deferred_accuracy"] = 0.0

    return metrics