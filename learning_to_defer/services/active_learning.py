import numpy as np

from sklearn.linear_model import LogisticRegression


def create_initial_query_set(
    ai_predictions,
    initial_budget,
    random_state,
):
    """
    Create a diverse initial query set.

    The initial set contains examples from all four
    AI-predicted categories.
    """

    ai_predictions = np.asarray(
        ai_predictions
    )

    random_generator = np.random.default_rng(
        random_state
    )

    selected_indices = []

    number_of_classes = len(
        np.unique(ai_predictions)
    )

    queries_per_class = (
        initial_budget // number_of_classes
    )

    for class_label in range(number_of_classes):
        class_indices = np.where(
            ai_predictions == class_label
        )[0]

        chosen = random_generator.choice(
            class_indices,
            size=queries_per_class,
            replace=False,
        )

        selected_indices.extend(
            chosen.tolist()
        )

    remaining_queries = (
        initial_budget
        - len(selected_indices)
    )

    if remaining_queries > 0:
        all_indices = np.arange(
            len(ai_predictions)
        )

        available_indices = np.setdiff1d(
            all_indices,
            np.array(selected_indices),
        )

        additional_indices = (
            random_generator.choice(
                available_indices,
                size=remaining_queries,
                replace=False,
            )
        )

        selected_indices.extend(
            additional_indices.tolist()
        )

    return np.array(
        selected_indices,
        dtype=int,
    )


def train_competence_model(
    ai_probabilities,
    expert_correctness,
    queried_indices,
    random_state,
):
    """
    Train using only the examples for which the
    expert was queried.
    """

    queried_features = (
        ai_probabilities[queried_indices]
    )

    queried_targets = (
        expert_correctness[queried_indices]
    )

    if len(np.unique(queried_targets)) < 2:
        raise ValueError(
            "The queried examples contain only one "
            "competence label. More initial examples "
            "are needed."
        )

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(
        queried_features,
        queried_targets,
    )

    return model


def select_random_queries(
    total_samples,
    already_queried,
    number_to_query,
    random_state,
):
    """
    Random-querying baseline.
    """

    random_generator = np.random.default_rng(
        random_state
    )

    all_indices = np.arange(total_samples)

    available_indices = np.setdiff1d(
        all_indices,
        already_queried,
    )

    return random_generator.choice(
        available_indices,
        size=number_to_query,
        replace=False,
    )


def select_active_queries(
    competence_model,
    ai_probabilities,
    ai_confidences,
    already_queried,
    number_to_query,
):
    """
    Select articles closest to the deferral boundary.

    A small difference between expert competence and
    AI confidence means the system is uncertain about
    who should answer.
    """

    total_samples = len(
        ai_probabilities
    )

    all_indices = np.arange(
        total_samples
    )

    available_indices = np.setdiff1d(
        all_indices,
        already_queried,
    )

    available_probabilities = (
        ai_probabilities[available_indices]
    )

    estimated_expert_competence = (
        competence_model.predict_proba(
            available_probabilities
        )[:, 1]
    )

    available_ai_confidences = (
        ai_confidences[available_indices]
    )

    boundary_distance = np.abs(
        estimated_expert_competence
        - available_ai_confidences
    )

    selected_positions = np.argsort(
        boundary_distance
    )[:number_to_query]

    return available_indices[
        selected_positions
    ]