"""Project 4 - Preference elicitation.

Two audiences, deliberately kept apart.

The project pages (overview, features, preference model, study design) are for
the reader marking the work. They use the dashboard shell that Project 3
introduced, and carry its sidebar.

The study pages under /study/ are for a *participant*. They carry no project
navigation at all: someone taking part should see the task, not the coursework
around it. That separation is itself part of the study design -- navigation
back into the explanation would be a confound.

What a participant's data does
------------------------------
Every trial is written to the database as it is answered, not held until the
end: a participant who abandons halfway still leaves the trials they completed,
and an interface that only saved on completion would lose them. The session
cookie carries the flow state (which block, which trial) and nothing else that
matters.

Stored per participant: a random token, the condition order, one row per trial
with its ordering and response time, the closing questionnaire, and the fitted
preference vector. No name, no email, no IP address.
"""

import uuid

import numpy as np
from django.shortcuts import redirect, render
from django.urls import reverse

from .models import StudySession, Trial
from .ml import features, pilot, preference

# The two interfaces being compared, per the brief.
PAIRWISE = "pairwise"
RANKING = "ranking"

DESIGN_LABELS = {
    PAIRWISE: "Design 1 - choose between two",
    RANKING: "Design 2 - rank ten",
}

TRIALS_PER_CONDITION = 8
RANKING_SET_SIZE = 10

# A short block of pairwise comparisons after both elicitation blocks. These are
# never used to fit w. They exist so the study can compute its own primary
# measure: how well a preference estimated from one interface predicts choices
# the estimate has not seen. Without them the design names a measure the
# interface cannot produce.
VALIDATION_TRIALS = 6
VALIDATION = "validation"

# Two unrecorded trials at the start of each elicitation block. The study
# design calls for them so that a participant's first real answer is not also
# their first encounter with the interface, which would load a learning effect
# onto whichever design came first. They are shown, answered and discarded.
PRACTICE_TRIALS = 2

# One instructed-response item inside the validation block. Preference has no
# right answer, so the check cannot ask whether a participant chose correctly;
# instead it asks them to pick the older of two films, which anyone still
# reading the screen can do. The analysis plan excludes participants who fail
# it, and that exclusion is pre-registered rather than decided afterwards.
ATTENTION_CHECK_AT = 3


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
        "validation_trials": VALIDATION_TRIALS,
        "pilot": pilot.cached(),
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
            # The two elicitation blocks, then the held-out validation block.
            "order": list(session.condition_order) + [VALIDATION],
            "stage": 0,
            "trial": 0,
            "practice": 0,
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
        "is_validation": design == VALIDATION,
        "stage": state["stage"] + 1,
        "stages": len(state["order"]),
        "trials": VALIDATION_TRIALS if design == VALIDATION else TRIALS_PER_CONDITION,
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

    if design in (PAIRWISE, VALIDATION):
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


def _in_practice(state, design):
    """Whether this request is still inside the practice run for this block.

    The validation block gets no practice: by then the participant has already
    answered eight pairwise trials, so there is nothing left to learn.
    """
    if design == VALIDATION:
        return False
    return state.get("practice", 0) < PRACTICE_TRIALS


def _read_elapsed(request):
    """The browser-reported thinking time, or None if the page could not send one."""
    try:
        value = int(float(request.POST.get("elapsed_ms", "")))
    except (TypeError, ValueError):
        return None
    # A negative or absurd value means a clock change or a tampered form, not a
    # slow participant. Storing None is more honest than storing nonsense.
    return value if 0 <= value <= 1000 * 60 * 60 else None


def _record_trial(state, design, ordering, elapsed):
    """Persist one answered trial, if its session still exists.

    update_or_create rather than create: a participant who refreshes the POST
    should not produce two rows for the same trial.
    """
    session = StudySession.objects.filter(
        participant=state.get("participant")
    ).first()
    if session is None:
        return

    Trial.objects.update_or_create(
        session=session,
        block=state["stage"],
        index=state["trial"],
        defaults={
            "design": design,
            "ordering": ordering,
            "milliseconds": elapsed,
        },
    )


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
        # Response time is measured in the page and posted back; it is one of
        # the study's measures, so it is recorded per trial.
        elapsed = _read_elapsed(request)

        # Practice answers are shown and thrown away. They exist so the first
        # recorded trial is not also the participant's first attempt.
        if _in_practice(state, design):
            state["practice"] += 1
            _save(request, state)
            return redirect("project4:study_task")

        state["responses"].append({
            "design": design,
            "trial": state["trial"],
            "ordering": ordering,
            "milliseconds": request.POST.get("elapsed_ms", ""),
        })

        # Written now rather than at the debrief. A participant who closes the
        # tab mid-block still leaves everything they answered, and the session
        # cookie stops being the only copy of the study's data.
        _record_trial(state, design, ordering, elapsed)

        state["trial"] += 1

        limit = VALIDATION_TRIALS if design == VALIDATION else TRIALS_PER_CONDITION
        if state["trial"] >= limit:
            state["trial"] = 0
            state["practice"] = 0
            state["stage"] += 1
            _save(request, state)
            if state["stage"] >= len(state["order"]):
                return redirect("project4:study_questionnaire")
            return redirect("project4:study_instructions")

        _save(request, state)
        return redirect("project4:study_task")

    size = RANKING_SET_SIZE if design == RANKING else 2
    practising = _in_practice(state, design)

    # Keyed to the participant as well as the position, so each person sees
    # their own films, as the design says, while staying exactly reproducible
    # from the stored token. Practice draws from a separate part of the seed
    # space so a practice film is not immediately repeated as a real one.
    position = ("practice", state["practice"]) if practising else state["trial"]
    movies = features.sample_movies(
        size, seed=(state["participant"], state["stage"], position)
    )
    limit = VALIDATION_TRIALS if design == VALIDATION else TRIALS_PER_CONDITION
    attention = (design == VALIDATION and state["trial"] == ATTENTION_CHECK_AT)

    return render(request, "project4/study_task.html", {
        "design": design,
        "is_validation": design == VALIDATION,
        "is_practice": practising,
        "is_attention_check": attention,
        "movies": movies,
        "trial": (state["practice"] + 1) if practising else state["trial"] + 1,
        "trials": PRACTICE_TRIALS if practising else limit,
        "progress": 0 if practising else round(100 * state["trial"] / limit),
        "stage": state["stage"] + 1,
        "stages": len(state["order"]),
    })


def study_questionnaire(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    if request.method == "POST":
        answers = {
            key: value for key, value in request.POST.items() if key != "csrfmiddlewaretoken"
        }
        state["questionnaire"] = answers
        StudySession.objects.filter(participant=state.get("participant")).update(
            questionnaire=answers
        )
        _save(request, state)
        return redirect("project4:study_debrief")

    # state["order"] ends with the validation block, which is not one of the
    # two interfaces being compared and has no label. Asking DESIGN_LABELS for
    # it raised KeyError and turned this page into a 500 for every participant
    # who reached it: the questionnaire asks people to compare the interfaces
    # they used, so only those two belong here.
    return render(request, "project4/study_questionnaire.html", {
        "designs": [
            DESIGN_LABELS[name] for name in state["order"] if name in DESIGN_LABELS
        ],
    })


def study_debrief(request):
    state = _session(request)
    if not state:
        return redirect("project4:study_consent")

    responses = state.get("responses", [])

    elicitation = [r for r in responses if r["design"] != VALIDATION]
    validation = [r for r in responses if r["design"] == VALIDATION]

    def rankings_for(rows):
        return [
            features.movie_rows(r["ordering"])
            for r in rows if len(r.get("ordering", [])) >= 2
        ]

    def seconds(rows):
        total = 0.0
        for r in rows:
            try:
                total += float(r.get("milliseconds") or 0) / 1000.0
            except (TypeError, ValueError):
                pass
        return total

    def agreement(w):
        """Share of held-out pairwise choices this estimate predicts correctly.

        This is the study's primary measure. The validation block is never used
        to fit w, so it is a fair test of what an interface actually revealed.
        """
        if not validation:
            return None
        correct = 0
        for r in validation:
            rows = features.movie_rows(r["ordering"])
            correct += int(preference.utilities(rows, w)[0] >= preference.utilities(rows, w)[1])
        return correct / len(validation)

    # One estimate per interface, so the two can be compared on equal terms.
    per_design = []
    for name in state["order"]:
        if name == VALIDATION:
            continue
        rows = [r for r in elicitation if r["design"] == name]
        w_design = preference.fit(rankings_for(rows), features.n_features())
        score = agreement(w_design)
        spent = seconds(rows)
        per_design.append({
            "label": DESIGN_LABELS[name],
            "trials": len(rows),
            "seconds": f"{spent:.0f}",
            "agreement": None if score is None else f"{score:.0%}",
            "per_minute": (None if score is None or spent == 0
                           else f"{score / (spent / 60):.2f}"),
        })

    # And one estimate from everything, for the recommendations shown below.
    w = preference.fit(rankings_for(elicitation), features.n_features())

    top = []
    if elicitation:
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
        "per_design": per_design,
        "validation_trials": len(validation),
        "weights": [(name, f"{value:+.2f}") for name, value in weights],
        "report_url": reverse("project4:index"),
    }
    # The session is the only place anything was kept, and the study is over.
    request.session.pop("project4", None)
    request.session.modified = True
    return render(request, "project4/study_debrief.html", context)
