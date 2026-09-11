"""A pilot on simulated participants, and the power analysis it supports.

Evaluation in human-centric AI is usually done in two steps: simulated users
first, because they give full control over behaviour and real users are costly,
then human users. The study is step two. This is step one.

What it is for
--------------
Two things that cannot be settled by argument.

First, that the estimator recovers a preference vector at all from the number of
interactions the study can afford. If it does not, the study measures nothing
and there is no point recruiting anyone.

Second, how many participants the real study needs. Asserting a round number is
guesswork; deriving one from an effect size measured under the model is not.

What it is not
--------------
It is not evidence about people. Every participant here follows the model
exactly, which is the one thing a real participant certainly does not do. The
pilot therefore establishes that the *instrument* works, not that the hypothesis
is true. If simulated participants showed no difference between the designs,
that would be damning; showing one is necessary, not sufficient.

The comparison is by time, not by trial
---------------------------------------
Comparing the designs per trial would be pointless: a ranking of ten carries up
to log2(10!) = 21.8 bits against a pairwise choice's one, so ranking wins by
construction. The study asks which is the better use of a participant's minute,
so the pilot gives both designs the same time budget and lets each spend it on
as many trials as it can afford.

That requires knowing how long each trial takes, and a simulation cannot know
that. The durations below are assumptions, stated as such, and are exactly what
the human pilot exists to replace. Any conclusion here is conditional on them,
which is why the sensitivity of the result to those durations is reported
alongside it.
"""

import numpy as np

from . import features, preference

# Assumed seconds per trial. Not measured -- see the module docstring. The
# interface records real durations, so a human pilot replaces these directly.
SECONDS_PAIRWISE = 6.0
SECONDS_RANKING = 45.0

# How closely a simulated participant follows their own utility. Plackett-Luce
# with utilities scaled by beta: large beta is a decisive participant, beta -> 0
# is someone answering at random. 3.0 describes someone who mostly follows their
# own taste but is not mechanical about it.
#
# This matters for reading the results. A noisy participant caps how well *any*
# estimate can predict their held-out choices, including their own true w, so
# absolute agreement has to be read against that ceiling rather than against
# 100%. The ceiling is reported alongside the results for exactly that reason.
BETA = 3.0

HELD_OUT_TRIALS = 40
TIME_BUDGETS_MINUTES = (2, 4, 6, 8)


def _sample_ranking(X, w, rng, beta=None):
    """One ranking, drawn from the Plackett-Luce model itself.

    Sampling from the same model the estimator assumes is deliberate. It tests
    whether the estimator can recover w when its assumptions hold, which is the
    question a pilot should answer first.
    """
    # Resolved here rather than as a default argument: a default is bound once
    # at import, so overriding the module-level BETA would silently do nothing.
    beta = BETA if beta is None else beta
    utilities = beta * (X @ w)
    order = []
    remaining = list(range(len(X)))
    while len(remaining) > 1:
        tail = utilities[remaining]
        probabilities = np.exp(tail - tail.max())
        probabilities /= probabilities.sum()
        picked = rng.choice(len(remaining), p=probabilities)
        order.append(remaining.pop(picked))
    order.extend(remaining)
    return X[order]


def _held_out(X, w_true, rng, n=HELD_OUT_TRIALS):
    """Pairwise choices the estimate will be scored on, made by the same user."""
    trials = []
    for _ in range(n):
        pair = X[rng.choice(len(X), 2, replace=False)]
        trials.append(_sample_ranking(pair, w_true, rng))
    return trials


def _agreement(trials, w):
    """Share of held-out choices this estimate predicts correctly."""
    correct = sum(
        int(preference.utilities(trial, w)[0] >= preference.utilities(trial, w)[1])
        for trial in trials
    )
    return correct / len(trials)


def _run_participant(X, rng, budget_seconds):
    """One simulated participant through both designs at a matched time budget."""
    w_true = rng.normal(size=X.shape[1])
    w_true /= np.linalg.norm(w_true)

    validation = _held_out(X, w_true, rng)

    # What the participant's own true preference vector scores on their own
    # held-out choices. No estimate can beat this, because the participant is
    # not perfectly consistent with themselves.
    results = {"ceiling": _agreement(validation, w_true)}

    for design, seconds, set_size in (
        ("pairwise", SECONDS_PAIRWISE, 2),
        ("ranking", SECONDS_RANKING, 10),
    ):
        n_trials = max(1, int(budget_seconds // seconds))
        rankings = [
            _sample_ranking(X[rng.choice(len(X), set_size, replace=False)], w_true, rng)
            for _ in range(n_trials)
        ]
        w_hat = preference.fit(rankings, X.shape[1])
        results[design] = {
            "trials": n_trials,
            "agreement": _agreement(validation, w_hat),
            "cosine": float(
                w_hat @ w_true / (np.linalg.norm(w_hat) * np.linalg.norm(w_true) + 1e-12)
            ),
        }

    return results


def required_sample_size(differences, alpha=0.05, power=0.80):
    """Participants needed to detect this paired difference, at this power.

    A within-subjects design gives each participant both conditions, so the
    relevant test is paired and the relevant effect size is Cohen's dz on the
    per-participant differences.
    """
    from scipy import stats

    differences = np.asarray(differences, dtype=float)
    mean, sd = differences.mean(), differences.std(ddof=1)

    if sd == 0:
        return {"dz": float("inf"), "n": 2, "mean": float(mean), "sd": 0.0}

    dz = mean / sd

    # Iterate rather than use the normal approximation: at small n the t
    # distribution's heavier tails matter, and small n is the interesting case.
    for n in range(2, 2001):
        critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        achieved = stats.nct.sf(critical, df=n - 1, nc=dz * np.sqrt(n))
        if achieved >= power:
            return {"dz": float(dz), "n": n, "mean": float(mean), "sd": float(sd)}

    return {"dz": float(dz), "n": None, "mean": float(mean), "sd": float(sd)}


def run(n_participants=60, seed=20260822):
    """The whole pilot. Returns a dictionary ready to be cached as JSON."""
    X = features.design_matrix()
    rng = np.random.default_rng(seed)

    budgets = []
    for minutes in TIME_BUDGETS_MINUTES:
        seconds = minutes * 60
        rows = [_run_participant(X, rng, seconds) for _ in range(n_participants)]

        pairwise = np.array([r["pairwise"]["agreement"] for r in rows])
        ranking = np.array([r["ranking"]["agreement"] for r in rows])
        ceiling = np.array([r["ceiling"] for r in rows])
        power = required_sample_size(ranking - pairwise)

        budgets.append({
            "minutes": minutes,
            "pairwise_trials": rows[0]["pairwise"]["trials"],
            "ranking_trials": rows[0]["ranking"]["trials"],
            "pairwise_mean": float(pairwise.mean()),
            "pairwise_sd": float(pairwise.std(ddof=1)),
            "ranking_mean": float(ranking.mean()),
            "ranking_sd": float(ranking.std(ddof=1)),
            "ceiling": float(ceiling.mean()),
            "difference": float((ranking - pairwise).mean()),
            "dz": power["dz"],
            "n_required": power["n"],
        })

    return {
        "n_participants": n_participants,
        "beta": BETA,
        "held_out_trials": HELD_OUT_TRIALS,
        "seconds_pairwise": SECONDS_PAIRWISE,
        "seconds_ranking": SECONDS_RANKING,
        "n_features": X.shape[1],
        "n_movies": X.shape[0],
        "budgets": budgets,
    }


def sensitivity(n_participants=30, minutes=6, seed=20260822):
    """How the conclusion moves if ranking is faster or slower than assumed.

    The whole time-normalised comparison rests on an assumed ratio of trial
    durations, so the honest thing is to report how much that assumption is
    carrying.
    """
    global SECONDS_RANKING
    original = SECONDS_RANKING
    X = features.design_matrix()
    rows = []

    try:
        for seconds in (25.0, 35.0, 45.0, 60.0, 90.0):
            SECONDS_RANKING = seconds
            rng = np.random.default_rng(seed)
            results = [_run_participant(X, rng, minutes * 60) for _ in range(n_participants)]
            pairwise = np.array([r["pairwise"]["agreement"] for r in results])
            ranking = np.array([r["ranking"]["agreement"] for r in results])
            rows.append({
                "seconds_ranking": seconds,
                "ranking_trials": results[0]["ranking"]["trials"],
                "difference": float((ranking - pairwise).mean()),
                "favours_ranking": bool((ranking - pairwise).mean() > 0),
            })
    finally:
        SECONDS_RANKING = original

    return {"minutes": minutes, "n_participants": n_participants, "rows": rows}


ARTIFACT = __import__("pathlib").Path(__file__).resolve().parent.parent / "artifacts" / "pilot.json"


def cached():
    """The stored pilot results, or None if it has not been run.

    Pages read this rather than simulating on request: the pilot takes minutes.
    Rebuild with `manage.py run_project4_pilot`.
    """
    import json
    if not ARTIFACT.exists():
        return None
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))
