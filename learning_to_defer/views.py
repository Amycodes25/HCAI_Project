import json
from pathlib import Path

from django.conf import settings
from django.shortcuts import redirect, render
import joblib
from .forms import ArticleClassificationForm, HumanExpertLabelForm
import html
import json
import re


ARTIFACT_DIR = Path(settings.BASE_DIR) / "learning_to_defer" / "artifacts"
CLASSIFY_CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

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

