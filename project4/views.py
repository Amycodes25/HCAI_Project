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
    context = _project_context("features")
    context.update({
        "n_movies": features.movie_count(),
        "feature_names": features.FEATURE_NAMES,
        "is_implemented": features.IS_IMPLEMENTED,
    })
    return render(request, "project4/features.html", context)


def task_preference_model(request):
    context = _project_context("preference_model")
    context["is_implemented"] = preference.IS_IMPLEMENTED
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


def study_task(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    design = state["order"][state["stage"]]

    if request.method == "POST":
        state["responses"].append({
            "design": design,
            "trial": state["trial"],
            "answer": request.POST.get("answer", ""),
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

    StudySession.objects.filter(participant=state.get("participant")).update(completed=True)

    context = {
        "participant": state.get("participant"),
        "responses": len(state.get("responses", [])),
        "report_url": reverse("project4:index"),
    }
    # The session is the only place anything was kept, and the study is over.
    request.session.pop("project4", None)
    request.session.modified = True
    return render(request, "project4/study_debrief.html", context)
