"""Project 4 - Preference elicitation.

Two audiences, deliberately kept apart.

The project pages (overview, features, preference model, study design) are for
the reader marking the work. They use the dashboard shell that Project 3
introduced, and carry its sidebar.

The study pages under /study/ are for a *participant*. They carry no project
navigation at all: someone taking part should see the task, not the coursework
around it. That separation is itself part of the study design -- navigation
back into the explanation would be a confound.

Status
------
This module currently delivers the structure and the participant flow. The
machine learning behind it is deliberately not here yet:

* Task 1, the feature representation, is stubbed in ml/features.py.
* Task 2, the Plackett-Luce likelihood and the estimator for w, is stubbed in
  ml/preference.py.

Until those land, movies are drawn from a small sample file and no preference
vector is estimated. Every page that depends on this says so on screen rather
than showing a placeholder number that might be mistaken for a result.
"""

import uuid

import numpy as np
from django.shortcuts import redirect, render
from django.urls import reverse

from .models import StudySession
from .ml import features, preference

# The two interfaces being compared, per the brief.
PAIRWISE = "pairwise"
RANKING = "ranking"

DESIGN_LABELS = {
    PAIRWISE: "Design 1 - choose between two",
    RANKING: "Design 2 - rank ten",
}

TRIALS_PER_CONDITION = 8
RANKING_SET_SIZE = 10


# --------------------------------------------------------------------------
# Project pages
# --------------------------------------------------------------------------

def _project_context(active):
    return {
        "active_task": active,
        "ml_ready": features.IS_IMPLEMENTED and preference.IS_IMPLEMENTED,
    }


def overview(request):
    return render(request, "project4/overview.html", _project_context("overview"))


def task_features(request):
    names = features.feature_names()
    context = _project_context("features")
    context.update({
        "n_movies": features.movie_count(),
        "n_features": features.n_features(),
        "genres": features.genre_summary(),
        "other_features": names[features.N_GENRES:],
    })
    return render(request, "project4/features.html", context)


def task_preference_model(request):
    context = _project_context("preference_model")
    context["alpha"] = preference.DEFAULT_ALPHA
    context["n_features"] = features.n_features()
    return render(request, "project4/preference_model.html", context)


def task_study_design(request):
    context = _project_context("study_design")
    context.update({
        "trials_per_condition": TRIALS_PER_CONDITION,
        "ranking_set_size": RANKING_SET_SIZE,
    })
    return render(request, "project4/study_design.html", context)


# --------------------------------------------------------------------------
# Participant flow
# --------------------------------------------------------------------------

def _session(request):
    return request.session.setdefault("project4", {})


def _save(request, state):
    request.session["project4"] = state
    request.session.modified = True


def study_consent(request):
    """Nothing is recorded before this form is submitted."""
    if request.method == "POST":
        if not request.POST.get("consent"):
            return render(request, "project4/study_consent.html", {
                "error": "You have to agree before the study can start.",
            })

        # A pseudonymous identifier. No name, no email, nothing that identifies
        # the participant is collected at any point.
        participant = uuid.uuid4().hex[:12]
        session = StudySession.objects.create(
            participant=participant,
            condition_order=_assign_order(),
        )

        state = {
            "participant": participant,
            "order": session.condition_order,
            "stage": 0,
            "trial": 0,
            "responses": [],
        }
        _save(request, state)
        return redirect("project4:study_instructions")

    return render(request, "project4/study_consent.html", {})


def _assign_order():
    """Counterbalance the two designs across participants.

    Within-subjects means everyone sees both interfaces, which makes order a
    confound: whichever comes second benefits from practice and suffers from
    fatigue. Alternating across participants spreads that evenly instead of
    letting it load onto one design.

    The count comes from the database rather than the session. Keeping it in
    the session would reset for every new participant, so everyone would get
    the same order and the counterbalancing would quietly not happen.
    """
    seen = StudySession.objects.count()
    return [PAIRWISE, RANKING] if seen % 2 == 0 else [RANKING, PAIRWISE]


def study_instructions(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    if state["stage"] >= len(state["order"]):
        return redirect("project4:study_questionnaire")

    if request.method == "POST":
        return redirect("project4:study_task")

    design = state["order"][state["stage"]]
    return render(request, "project4/study_instructions.html", {
        "design": design,
        "design_label": DESIGN_LABELS[design],
        "stage": state["stage"] + 1,
        "stages": len(state["order"]),
        "trials": TRIALS_PER_CONDITION,
        "set_size": RANKING_SET_SIZE,
    })


def _read_ordering(request, design):
    """The films this trial produced, best first, as dataset indices.

    Reconstructed from ordinary form fields rather than a value assembled in
    the browser, so the trial behaves identically with JavaScript disabled.
    """
    shown = [
        int(value) for value in request.POST.get("shown", "").split(",") if value.strip()
    ]

    if design == PAIRWISE:
        try:
            chosen = int(request.POST.get("chosen", ""))
        except (TypeError, ValueError):
            return []
        if chosen not in shown:
            return []
        # A choice between two films is a ranking of length two.
        return [chosen] + [index for index in shown if index != chosen]

    positions = []
    for index in shown:
        try:
            positions.append((int(request.POST.get(f"pos_{index}", "")), index))
        except (TypeError, ValueError):
            return []

    # Positions come from number inputs, so a participant can leave two films
    # on the same number. Sorting is stable, so ties keep their display order
    # rather than being discarded.
    return [index for _, index in sorted(positions, key=lambda pair: pair[0])]


def study_task(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    # Reaching this page after the last block -- by refreshing, or by going
    # back -- must move the participant on rather than fall off the end of the
    # condition list.
    if state["stage"] >= len(state["order"]):
        return redirect("project4:study_questionnaire")

    design = state["order"][state["stage"]]

    if request.method == "POST":
        ordering = _read_ordering(request, design)
        state["responses"].append({
            "design": design,
            "trial": state["trial"],
            "ordering": ordering,
            # Response time is measured in the page and posted back; it is one
            # of the study's primary measures, so it is recorded per trial.
            "milliseconds": request.POST.get("elapsed_ms", ""),
        })
        state["trial"] += 1

        if state["trial"] >= TRIALS_PER_CONDITION:
            state["trial"] = 0
            state["stage"] += 1
            _save(request, state)
            if state["stage"] >= len(state["order"]):
                return redirect("project4:study_questionnaire")
            return redirect("project4:study_instructions")

        _save(request, state)
        return redirect("project4:study_task")

    size = 2 if design == PAIRWISE else RANKING_SET_SIZE
    movies = features.sample_movies(size, seed=(state["stage"], state["trial"]))

    return render(request, "project4/study_task.html", {
        "design": design,
        "movies": movies,
        "trial": state["trial"] + 1,
        "trials": TRIALS_PER_CONDITION,
        "progress": round(100 * state["trial"] / TRIALS_PER_CONDITION),
        "stage": state["stage"] + 1,
        "stages": len(state["order"]),
    })


def study_questionnaire(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    if request.method == "POST":
        state["questionnaire"] = {
            key: value for key, value in request.POST.items() if key != "csrfmiddlewaretoken"
        }
        _save(request, state)
        return redirect("project4:study_debrief")

    return render(request, "project4/study_questionnaire.html", {
        "designs": [DESIGN_LABELS[name] for name in state["order"]],
    })


def study_debrief(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    responses = state.get("responses", [])

    # The purpose of the elicitation is to estimate w, so it is estimated here,
    # from every response the participant gave. Both interfaces feed the same
    # estimator: a pairwise choice is a ranking of length two.
    rankings = [
        features.movie_rows(response["ordering"])
        for response in responses
        if len(response.get("ordering", [])) >= 2
    ]
    w = preference.fit(rankings, features.n_features())

    top = []
    if rankings:
        X = features.design_matrix()
        for index in preference.rank_items(X, w)[:5]:
            top.append(features.describe(int(index)))

    weights = sorted(
        zip(features.feature_names(), w),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:6]

    StudySession.objects.filter(participant=state.get("participant")).update(
        completed=True, estimated_w=[float(value) for value in w]
    )

    context = {
        "participant": state.get("participant"),
        "responses": len(responses),
        "recommendations": top,
        "n_movies": features.movie_count(),
        "weights": [(name, f"{value:+.2f}") for name, value in weights],
        "report_url": reverse("project4:index"),
    }
    # The session is the only place anything was kept, and the study is over.
    request.session.pop("project4", None)
    request.session.modified = True
    return render(request, "project4/study_debrief.html", context)
