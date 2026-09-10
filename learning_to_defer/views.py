import html
import json
import random
import re
import time
import uuid
from pathlib import Path

import joblib
from django.conf import settings
from django.shortcuts import redirect, render
from django.utils import timezone

from .forms import ArticleClassificationForm, HumanExpertLabelForm


ARTIFACT_DIR = Path(settings.BASE_DIR) / "learning_to_defer" / "artifacts"
CLASSIFY_CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
USER_STUDY_LENGTH = 10
USER_STUDY_RESULTS_DIR = ARTIFACT_DIR / "user_study_results"
HUMAN_COMPETENCE_PRIOR_ALPHA = 2.0
HUMAN_COMPETENCE_PRIOR_BETA = 2.0
_MODEL_CACHE = {}


def _load_classification_models():
    """Load the Task 1 and Task 3 models once and keep them in memory."""
    if "baseline" not in _MODEL_CACHE:
        _MODEL_CACHE["baseline"] = joblib.load(
            ARTIFACT_DIR / "baseline_pipeline.joblib"
        )
        _MODEL_CACHE["competence"] = joblib.load(
            ARTIFACT_DIR / "competence_model_full.joblib"
        )

    return _MODEL_CACHE["baseline"], _MODEL_CACHE["competence"]
def _clean_article_text(text):
    cleaned_text = str(text)

    # Some articles contain HTML entities encoded multiple times.
    for _ in range(3):
        decoded_text = html.unescape(cleaned_text)

        if decoded_text == cleaned_text:
            break

        cleaned_text = decoded_text

    # Remove old HTML tags.
    cleaned_text = re.sub(
        r"<[^>]+>",
        " ",
        cleaned_text,
    )

    # Remove unnecessary backslashes before punctuation.
    cleaned_text = re.sub(
        r"\\([#$%&])",
        r"\1",
        cleaned_text,
    )

    # Repair malformed apostrophe entities.
    cleaned_text = cleaned_text.replace(
        "&#39;",
        "'",
    ).replace(
        "#39;",
        "'",
    )

    # Normalize whitespace.
    return " ".join(cleaned_text.split())

def _load_metrics():
    metrics_path = ARTIFACT_DIR / "baseline_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def overview(request):
    return render(request, "learning_to_defer/overview.html")


def baseline(request):
    metrics = _load_metrics()
    if metrics:
        metrics["test_accuracy_percent"] = metrics["test_accuracy"] * 100
    return render(
        request,
        "learning_to_defer/baseline.html",
        {"metrics": metrics},
    )
def simulated_expert(request):
    metrics_path = (
        ARTIFACT_DIR
        / "expert_metrics.json"
    )

    metrics = None

    if metrics_path.exists():
        with metrics_path.open(
            encoding="utf-8"
        ) as metrics_file:
            metrics = json.load(metrics_file)

        metrics["overall_accuracy_percent"] = (
            metrics["overall_accuracy"] * 100
        )

        for category in metrics["per_class"]:
            category["designed_accuracy_percent"] = (
                category["designed_accuracy"] * 100
            )

            category["measured_accuracy_percent"] = (
                category["measured_accuracy"] * 100
            )

    return render(
        request,
        "learning_to_defer/simulated_expert.html",
        {
            "metrics": metrics,
        },
    )
def learning_to_defer(request):
    metrics_path = (
        ARTIFACT_DIR
        / "deferral_metrics.json"
    )

    metrics = None

    if metrics_path.exists():
        with metrics_path.open(
            encoding="utf-8"
        ) as metrics_file:
            metrics = json.load(metrics_file)

        percentage_fields = [
            "ai_accuracy",
            "expert_accuracy",
            "team_accuracy",
            "deferral_rate",
            "coverage",
            "deferred_accuracy",
            "non_deferred_accuracy",
            "beneficial_deferral_rate",
            "harmful_deferral_rate",
            "oracle_accuracy",
        ]

        for field in percentage_fields:
            metrics[f"{field}_percent"] = (
                metrics[field] * 100
            )

        metrics["accuracy_improvement_percent"] = (
            metrics["team_accuracy_percent"]
            - metrics["ai_accuracy_percent"]
        )

        metrics["net_benefit"] = (
            metrics["beneficial_deferrals"]
            - metrics["harmful_deferrals"]
        )

        for category in metrics["per_class"]:
            category["ai_accuracy_percent"] = (
                category["ai_accuracy"] * 100
            )

            category["expert_accuracy_percent"] = (
                category["expert_accuracy"] * 100
            )

            category["team_accuracy_percent"] = (
                category["team_accuracy"] * 100
            )

            category["deferral_rate_percent"] = (
                category["deferral_rate"] * 100
            )

    return render(
        request,
        "learning_to_defer/learning_to_defer.html",
        {
            "metrics": metrics,
        },
    )
def active_learning(request):
    metrics_path = (
        ARTIFACT_DIR
        / "active_learning_metrics.json"
    )

    metrics = None
    comparison_rows = []
    full_supervision_accuracy = None

    if metrics_path.exists():
        with metrics_path.open(
            encoding="utf-8"
        ) as metrics_file:
            metrics = json.load(metrics_file)

        summary_lookup = {
            (
                result["strategy"],
                result["budget"],
            ): result
            for result in metrics["summary"]
        }

        for budget in metrics["budgets"]:
            random_result = summary_lookup[
                ("random", budget)
            ]

            active_result = summary_lookup[
                ("active", budget)
            ]

            comparison_rows.append(
                {
                    "budget": budget,

                    "random_accuracy": (
                        random_result[
                            "team_accuracy_mean"
                        ] * 100
                    ),

                    "random_accuracy_std": (
                        random_result[
                            "team_accuracy_std"
                        ] * 100
                    ),

                    "active_accuracy": (
                        active_result[
                            "team_accuracy_mean"
                        ] * 100
                    ),

                    "active_accuracy_std": (
                        active_result[
                            "team_accuracy_std"
                        ] * 100
                    ),

                    "active_advantage": (
                        active_result[
                            "team_accuracy_mean"
                        ]
                        - random_result[
                            "team_accuracy_mean"
                        ]
                    ) * 100,

                    "random_deferral": (
                        random_result[
                            "deferral_rate_mean"
                        ] * 100
                    ),

                    "active_deferral": (
                        active_result[
                            "deferral_rate_mean"
                        ] * 100
                    ),

                    "random_auc": (
                        random_result[
                            "competence_roc_auc_mean"
                        ]
                    ),

                    "active_auc": (
                        active_result[
                            "competence_roc_auc_mean"
                        ]
                    ),

                    "random_brier": (
                        random_result[
                            "competence_brier_score_mean"
                        ]
                    ),

                    "active_brier": (
                        active_result[
                            "competence_brier_score_mean"
                        ]
                    ),

                    "random_useful_rate": (
                        random_result[
                            "beneficial_deferral_rate_mean"
                        ] * 100
                    ),

                    "active_useful_rate": (
                        active_result[
                            "beneficial_deferral_rate_mean"
                        ] * 100
                    ),
                }
            )

        task_3_path = (
            ARTIFACT_DIR
            / "deferral_metrics.json"
        )

        if task_3_path.exists():
            with task_3_path.open(
                encoding="utf-8"
            ) as task_3_file:
                task_3_metrics = json.load(
                    task_3_file
                )

            full_supervision_accuracy = (
                task_3_metrics["team_accuracy"]
                * 100
            )

    context = {
        "metrics": metrics,
        "comparison_rows": comparison_rows,
        "full_supervision_accuracy":
            full_supervision_accuracy,
        "query_efficiency": 5,
    }

    return render(
        request,
        "learning_to_defer/active_learning.html",
        context,
    )
def _initial_human_competence():
    """
    Create the initial participant specific competence model.

    Each category starts with a Beta(2, 2) prior,
    corresponding to an initial estimate of 0.50.
    """

    state = {}

    for category in CLASSIFY_CLASS_NAMES:
        state[category] = {
            "alpha": HUMAN_COMPETENCE_PRIOR_ALPHA,
            "beta": HUMAN_COMPETENCE_PRIOR_BETA,
            "correct": 0,
            "incorrect": 0,
            "estimate": 0.50,
        }

    return state


def _update_human_competence(
    competence_state,
    true_label,
    is_correct,
):
    """
    Update competence for the true article category.

    Correct response:
        alpha += 1

    Incorrect response:
        beta += 1

    Estimated competence:
        alpha / (alpha + beta)
    """

    if not 0 <= true_label < len(CLASSIFY_CLASS_NAMES):
        return competence_state, None, None, None

    category = CLASSIFY_CLASS_NAMES[true_label]
    category_state = competence_state[category]

    before = float(category_state["estimate"])

    if is_correct:
        category_state["alpha"] += 1.0
        category_state["correct"] += 1
    else:
        category_state["beta"] += 1.0
        category_state["incorrect"] += 1

    alpha = category_state["alpha"]
    beta = category_state["beta"]
    after = alpha / (alpha + beta)

    category_state["estimate"] = round(after, 4)
    competence_state[category] = category_state

    return (
        competence_state,
        category,
        round(before, 4),
        round(after, 4),
    )


def _beta_variance(alpha, beta):
    """
    Posterior variance of a Beta(alpha, beta) distribution.

    Larger variance means that we know less about the participant's
    competence for that category.
    """

    total = alpha + beta

    if total <= 0:
        return 0.0

    return (
        alpha * beta
        / ((total ** 2) * (total + 1.0))
    )


def _select_next_user_study_article(
    candidate_articles,
    selected_indices,
    competence_state,
):
    """
    Select the next article adaptively from the unused candidate pool.

    The score combines four signals:

    1. Human-competence uncertainty:
       Prefer categories for which the participant's competence is still
       uncertain.

    2. Human-AI decision-boundary closeness:
       Prefer examples where the learned human competence is close to the
       AI confidence, because these are informative for defer/answer
       decisions.

    3. AI uncertainty:
       Give some preference to articles on which the AI is less certain.

    4. Coverage bonus:
       Encourage observing categories for which no human response has yet
       been collected.

    The participant never sees these scores.
    """

    selected_set = set(selected_indices)

    remaining_indices = [
        index
        for index in range(len(candidate_articles))
        if index not in selected_set
    ]

    if not remaining_indices:
        return None, None

    prior_variance = _beta_variance(
        HUMAN_COMPETENCE_PRIOR_ALPHA,
        HUMAN_COMPETENCE_PRIOR_BETA,
    )

    best_index = None
    best_metadata = None
    best_score = float("-inf")

    for index in remaining_indices:
        article = candidate_articles[index]

        try:
            ai_prediction = int(
                article.get("ai_prediction", 0)
            )
        except (TypeError, ValueError):
            ai_prediction = 0

        if not 0 <= ai_prediction < len(CLASSIFY_CLASS_NAMES):
            ai_prediction = 0

        predicted_category = CLASSIFY_CLASS_NAMES[
            ai_prediction
        ]

        category_state = competence_state.get(
            predicted_category,
            {
                "alpha": HUMAN_COMPETENCE_PRIOR_ALPHA,
                "beta": HUMAN_COMPETENCE_PRIOR_BETA,
                "correct": 0,
                "incorrect": 0,
                "estimate": 0.50,
            },
        )

        alpha = float(
            category_state.get(
                "alpha",
                HUMAN_COMPETENCE_PRIOR_ALPHA,
            )
        )

        beta = float(
            category_state.get(
                "beta",
                HUMAN_COMPETENCE_PRIOR_BETA,
            )
        )

        human_estimate = float(
            category_state.get(
                "estimate",
                alpha / (alpha + beta),
            )
        )

        observations = int(
            category_state.get("correct", 0)
        ) + int(
            category_state.get("incorrect", 0)
        )

        competence_variance = _beta_variance(
            alpha,
            beta,
        )

        if prior_variance > 0:
            competence_uncertainty = min(
                competence_variance / prior_variance,
                1.0,
            )
        else:
            competence_uncertainty = 0.0

        try:
            ai_confidence = float(
                article.get("ai_confidence", 0.0)
            )
        except (TypeError, ValueError):
            ai_confidence = 0.0

        ai_confidence = max(
            0.0,
            min(ai_confidence, 1.0),
        )

        boundary_closeness = max(
            0.0,
            1.0 - abs(
                human_estimate - ai_confidence
            ),
        )

        ai_uncertainty = (
            1.0 - ai_confidence
        )

        coverage_bonus = (
            1.0
            if observations == 0
            else 0.0
        )

        # Weighted adaptive query score.
        score = (
            0.45 * competence_uncertainty
            + 0.30 * boundary_closeness
            + 0.15 * ai_uncertainty
            + 0.10 * coverage_bonus
        )

        metadata = {
            "selection_reason":
                "adaptive_human_competence",
            "selection_score": round(score, 6),
            "selection_predicted_category":
                predicted_category,
            "selection_human_estimate":
                round(human_estimate, 4),
            "selection_human_uncertainty":
                round(competence_uncertainty, 4),
            "selection_boundary_closeness":
                round(boundary_closeness, 4),
            "selection_ai_uncertainty":
                round(ai_uncertainty, 4),
            "selection_coverage_bonus":
                round(coverage_bonus, 4),
        }

        if score > best_score:
            best_score = score
            best_index = index
            best_metadata = metadata

    return best_index, best_metadata


def _save_user_study_results(
    request,
    candidate_articles,
    completed=False,
):
    """
    Save the current participant study state to a JSON file.

    One file is used per anonymous participant/session.
    It is updated after every submitted answer so partial studies
    are not lost if the browser closes early.
    """

    study_id = request.session.get(
        "user_study_id"
    )

    if not study_id:
        return

    USER_STUDY_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "study_id": study_id,
        "started_at": request.session.get(
            "user_study_started_at"
        ),
        "updated_at": timezone.now().isoformat(),
        "completed": completed,
        "candidate_pool_size":
            len(candidate_articles),
        "study_length": min(
            USER_STUDY_LENGTH,
            len(candidate_articles),
        ),
        "selection_strategy":
            "adaptive_human_competence",
        "selected_article_indices":
            request.session.get(
                "user_study_article_indices",
                [],
            ),
        "selection_metadata":
            request.session.get(
                "user_study_selection_metadata",
                [],
            ),
        "responses": request.session.get(
            "user_study_answers",
            [],
        ),
        "human_competence":
            request.session.get(
                "user_study_competence",
                {},
            ),
    }

    output_path = (
        USER_STUDY_RESULTS_DIR
        / f"{study_id}.json"
    )

    temporary_path = output_path.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as result_file:
        json.dump(
            results,
            result_file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(output_path)


def user_study_intro(request):
    return render(
        request,
        "learning_to_defer/user_study_intro.html",
    )


def start_user_study(request):
    """
    Start a new anonymous participant study session.

    The first article is an initial seed. Every later article is selected
    after the previous human response using the participant-specific
    competence model.
    """

    # Clear state belonging to an earlier study run.
    session_keys = [
        "user_study_position",
        "user_study_answers",
        "user_study_article_indices",
        "user_study_selection_metadata",
        "user_study_competence",
        "user_study_id",
        "user_study_started_at",
        "user_study_article_started_at",
    ]

    for key in session_keys:
        request.session.pop(key, None)

    pool_path = (
        ARTIFACT_DIR
        / "human_expert_pool.json"
    )

    if not pool_path.exists():
        return redirect(
            "learning_to_defer:user_study"
        )

    with pool_path.open(
        encoding="utf-8"
    ) as pool_file:
        pool_data = json.load(pool_file)

    candidate_articles = pool_data["articles"]

    if not candidate_articles:
        return redirect(
            "learning_to_defer:user_study"
        )

    total_articles = min(
        USER_STUDY_LENGTH,
        len(candidate_articles),
    )

    # The first item is a neutral seed because there is no participant
    # response yet from which to adapt.
    first_index = random.randrange(
        len(candidate_articles)
    )

    initial_article = candidate_articles[
        first_index
    ]

    initial_metadata = {
        "selection_reason": "initial_seed",
        "selection_score": None,
        "selection_predicted_category":
            initial_article.get("ai_category"),
        "selection_human_estimate": 0.50,
        "selection_human_uncertainty": 1.0,
        "selection_boundary_closeness": None,
        "selection_ai_uncertainty": (
            round(
                1.0
                - float(
                    initial_article.get(
                        "ai_confidence",
                        0.0,
                    )
                ),
                4,
            )
        ),
        "selection_coverage_bonus": 1.0,
    }

    study_id = uuid.uuid4().hex

    request.session[
        "user_study_id"
    ] = study_id

    request.session[
        "user_study_article_indices"
    ] = [first_index]

    request.session[
        "user_study_selection_metadata"
    ] = [initial_metadata]

    request.session[
        "user_study_position"
    ] = 0

    request.session[
        "user_study_answers"
    ] = []

    request.session[
        "user_study_competence"
    ] = _initial_human_competence()

    request.session[
        "user_study_started_at"
    ] = timezone.now().isoformat()

    request.session[
        "user_study_article_started_at"
    ] = time.time()

    request.session.modified = True

    _save_user_study_results(
        request,
        candidate_articles,
        completed=False,
    )

    return redirect(
        "learning_to_defer:user_study"
    )


def user_study(request):
    """
    Participant-facing adaptive user study.

    Article 1 is an initial seed. After each submitted human response,
    the participant-specific competence model is updated and the next
    article is selected adaptively from the unused 20-article pool.

    The participant never sees correctness, ground truth, AI information,
    competence estimates or selection scores.
    """

    pool_path = (
        ARTIFACT_DIR
        / "human_expert_pool.json"
    )

    if not pool_path.exists():
        return render(
            request,
            "learning_to_defer/user_study.html",
            {
                "pool_available": False,
            },
        )

    with pool_path.open(
        encoding="utf-8"
    ) as pool_file:
        pool_data = json.load(pool_file)

    candidate_articles = pool_data["articles"]

    if not candidate_articles:
        return render(
            request,
            "learning_to_defer/user_study.html",
            {
                "pool_available": False,
            },
        )

    total_articles = min(
        USER_STUDY_LENGTH,
        len(candidate_articles),
    )

    selected_indices = request.session.get(
        "user_study_article_indices",
        [],
    )

    selection_metadata = request.session.get(
        "user_study_selection_metadata",
        [],
    )

    if not selected_indices:
        return redirect(
            "learning_to_defer:start_user_study"
        )

    position = request.session.get(
        "user_study_position",
        0,
    )

    answers = request.session.get(
        "user_study_answers",
        [],
    )

    study_id = request.session.get(
        "user_study_id",
        "",
    )

    competence_state = request.session.get(
        "user_study_competence"
    )

    if not competence_state:
        competence_state = (
            _initial_human_competence()
        )
        request.session[
            "user_study_competence"
        ] = competence_state

    # Recovery path: if the previous POST was saved but the next article
    # had not yet been appended, select it here.
    if (
        position < total_articles
        and position >= len(selected_indices)
    ):
        (
            next_index,
            next_metadata,
        ) = _select_next_user_study_article(
            candidate_articles,
            selected_indices,
            competence_state,
        )

        if next_index is not None:
            selected_indices.append(
                next_index
            )
            selection_metadata.append(
                next_metadata
            )

            request.session[
                "user_study_article_indices"
            ] = selected_indices

            request.session[
                "user_study_selection_metadata"
            ] = selection_metadata

            request.session.modified = True

    form = HumanExpertLabelForm(
        request.POST or None
    )

    if (
        request.method == "POST"
        and form.is_valid()
        and position < total_articles
        and position < len(selected_indices)
    ):
        candidate_index = (
            selected_indices[position]
        )

        article = candidate_articles[
            candidate_index
        ]

        current_selection = (
            selection_metadata[position]
            if position < len(
                selection_metadata
            )
            else {}
        )

        selected_label = int(
            form.cleaned_data["label"]
        )

        true_label = int(
            article["true_label"]
        )

        is_correct = (
            selected_label == true_label
        )

        # -------------------------------------------------
        # ONLINE HUMAN COMPETENCE UPDATE
        # -------------------------------------------------

        (
            competence_state,
            competence_category,
            competence_before,
            competence_after,
        ) = _update_human_competence(
            competence_state,
            true_label,
            is_correct,
        )

        request.session[
            "user_study_competence"
        ] = competence_state

        # -------------------------------------------------
        # RESPONSE TIME
        # -------------------------------------------------

        article_started_at = (
            request.session.get(
                "user_study_article_started_at"
            )
        )

        response_time_seconds = None

        if article_started_at is not None:
            response_time_seconds = round(
                max(
                    0,
                    time.time()
                    - article_started_at,
                ),
                2,
            )

        # -------------------------------------------------
        # SAVE CURRENT RESPONSE
        # -------------------------------------------------

        response = {
            "trial_number": position + 1,
            "pool_index": candidate_index,
            "pool_position": article.get(
                "pool_position"
            ),
            "dataset_index": article.get(
                "dataset_index"
            ),

            # Participant response
            "human_label": selected_label,
            "human_category": (
                CLASSIFY_CLASS_NAMES[
                    selected_label
                ]
                if 0 <= selected_label
                < len(CLASSIFY_CLASS_NAMES)
                else str(selected_label)
            ),

            # Hidden ground truth
            "true_label": true_label,
            "true_category": article.get(
                "true_category"
            ),
            "human_correct": is_correct,

            # Online learned competence
            "competence_category":
                competence_category,
            "human_competence_before":
                competence_before,
            "human_competence_after":
                competence_after,

            # Hidden AI information
            "ai_prediction": article.get(
                "ai_prediction"
            ),
            "ai_category": article.get(
                "ai_category"
            ),
            "ai_confidence": article.get(
                "ai_confidence"
            ),
            "estimated_expert_competence":
                article.get(
                    "estimated_expert_competence"
                ),
            "boundary_distance": article.get(
                "boundary_distance"
            ),

            # Why this article was shown
            "selection_reason":
                current_selection.get(
                    "selection_reason"
                ),
            "selection_score":
                current_selection.get(
                    "selection_score"
                ),
            "selection_predicted_category":
                current_selection.get(
                    "selection_predicted_category"
                ),
            "selection_human_uncertainty":
                current_selection.get(
                    "selection_human_uncertainty"
                ),
            "selection_boundary_closeness":
                current_selection.get(
                    "selection_boundary_closeness"
                ),
            "selection_ai_uncertainty":
                current_selection.get(
                    "selection_ai_uncertainty"
                ),

            # Timing
            "response_time_seconds":
                response_time_seconds,
            "submitted_at":
                timezone.now().isoformat(),
        }

        answers.append(response)
        position += 1

        request.session[
            "user_study_answers"
        ] = answers

        request.session[
            "user_study_position"
        ] = position

        # -------------------------------------------------
        # ACTIVE NEXT-ARTICLE SELECTION
        # -------------------------------------------------

        if position < total_articles:
            (
                next_index,
                next_metadata,
            ) = _select_next_user_study_article(
                candidate_articles,
                selected_indices,
                competence_state,
            )

            if next_index is not None:
                selected_indices.append(
                    next_index
                )
                selection_metadata.append(
                    next_metadata
                )

                request.session[
                    "user_study_article_indices"
                ] = selected_indices

                request.session[
                    "user_study_selection_metadata"
                ] = selection_metadata

            request.session[
                "user_study_article_started_at"
            ] = time.time()

        request.session.modified = True

        _save_user_study_results(
            request,
            candidate_articles,
            completed=(
                position >= total_articles
            ),
        )

        return redirect(
            "learning_to_defer:user_study"
        )

    completed = (
        position >= total_articles
    )

    current_article = None

    if (
        not completed
        and position < len(selected_indices)
    ):
        candidate_index = (
            selected_indices[position]
        )

        article = candidate_articles[
            candidate_index
        ]

        current_article = {
            "number": position + 1,
            "total": total_articles,
            "text": _clean_article_text(
                article["text"]
            ),
        }

    context = {
        "pool_available": True,
        "form": form,
        "current_article":
            current_article,
        "completed": completed,
        "answered_count": len(answers),
        "total_articles": total_articles,
        "study_id": (
            study_id[:8].upper()
            if study_id
            else ""
        ),
    }

    return render(
        request,
        "learning_to_defer/user_study.html",
        context,
    )


def human_expert(request):
    pool_path = (
        ARTIFACT_DIR
        / "human_expert_pool.json"
    )

    if not pool_path.exists():
        return render(
            request,
            "learning_to_defer/human_expert.html",
            {
                "pool_available": False,
            },
        )

    with pool_path.open(
        encoding="utf-8"
    ) as pool_file:
        pool_data = json.load(pool_file)

    articles = pool_data["articles"]

    if (
        request.method == "POST"
        and request.POST.get("action") == "reset"
    ):
        request.session.pop(
            "human_expert_position",
            None,
        )

        request.session.pop(
            "human_expert_answers",
            None,
        )

        request.session.pop(
            "human_expert_feedback",
            None,
        )

        return redirect(
            "learning_to_defer:human_expert"
        )

    position = request.session.get(
        "human_expert_position",
        0,
    )

    answers = request.session.get(
        "human_expert_answers",
        [],
    )

    feedback = request.session.get(
        "human_expert_feedback"
    )

    form = HumanExpertLabelForm(
        request.POST or None
    )

    if (
        request.method == "POST"
        and form.is_valid()
        and position < len(articles)
    ):
        article = articles[position]

        selected_label = int(
            form.cleaned_data["label"]
        )

        true_label = int(
            article["true_label"]
        )

        is_correct = (
            selected_label == true_label
        )

        class_names = [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ]

        ai_confidence = (
            article["ai_confidence"]
            * 100
        )

        expert_competence = (
            article[
                "estimated_expert_competence"
            ]
            * 100
        )

        would_defer = (
            article[
                "estimated_expert_competence"
            ]
            > article["ai_confidence"]
        )

        feedback = {
            "is_correct": is_correct,
            "human_category":
                class_names[selected_label],
            "true_category":
                article["true_category"],
            "ai_category":
                article["ai_category"],
            "ai_confidence":
                ai_confidence,
            "expert_competence":
                expert_competence,
            "would_defer":
                would_defer,
        }

        answers.append(
            {
                "position": position,
                "selected_label":
                    selected_label,
                "true_label":
                    true_label,
                "is_correct":
                    is_correct,
            }
        )

        position += 1

        request.session[
            "human_expert_position"
        ] = position

        request.session[
            "human_expert_answers"
        ] = answers

        request.session[
            "human_expert_feedback"
        ] = feedback

        request.session.modified = True

        return redirect(
            "learning_to_defer:human_expert"
        )

    answered_count = len(answers)

    correct_count = sum(
        answer["is_correct"]
        for answer in answers
    )

    accuracy = (
        correct_count / answered_count * 100
        if answered_count > 0
        else 0
    )

    completed = (
        position >= len(articles)
    )

    current_article = None

    if not completed:
        current_article = {
            "number": position + 1,
            "total": len(articles),
            "text":_clean_article_text(articles[position]["text"]),
        }

    context = {
        "pool_available": True,
        "selection_method":
            pool_data["selection_method"],
        "form": form,
        "current_article":
            current_article,
        "feedback": feedback,
        "completed": completed,
        "answered_count":
            answered_count,
        "correct_count":
            correct_count,
        "accuracy": accuracy,
        "progress_percent": (
            answered_count
            / len(articles)
            * 100
        ),
    }

    return render(
        request,
        "learning_to_defer/human_expert.html",
        context,
    )
def classify(request):
    models_available = (
        (ARTIFACT_DIR / "baseline_pipeline.joblib").exists()
        and (ARTIFACT_DIR / "competence_model_full.joblib").exists()
    )

    form = ArticleClassificationForm(request.POST or None)
    result = None

    if (
        models_available
        and request.method == "POST"
        and form.is_valid()
    ):
        baseline_model, competence_model = _load_classification_models()

        text = form.cleaned_data["text"]
        probabilities = baseline_model.predict_proba([text])[0]

        ai_prediction = int(probabilities.argmax())
        ai_confidence = float(probabilities.max())

        estimated_expert_competence = float(
            competence_model.predict_proba(
                probabilities.reshape(1, -1)
            )[0, 1]
        )

        result = {
            "ai_category": CLASSIFY_CLASS_NAMES[ai_prediction],
            "ai_confidence_percent": ai_confidence * 100,
            "expert_competence_percent": (
                estimated_expert_competence * 100
            ),
            "would_defer": (
                estimated_expert_competence > ai_confidence
            ),
            "class_probabilities": [
                {
                    "class_name": class_name,
                    "probability_percent": float(probability) * 100,
                }
                for class_name, probability in zip(
                    CLASSIFY_CLASS_NAMES,
                    probabilities,
                )
            ],
        }

    context = {
        "models_available": models_available,
        "form": form,
        "result": result,
    }

    return render(
        request,
        "learning_to_defer/classify.html",
        context,
    )

