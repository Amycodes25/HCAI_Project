"""Counterfactual explanations for Project 2, Task 4.

The method from the brief: sample points locally around x, keep those the model
assigns to the desired class, rank them by MAD-weighted L1 distance to x, and
show the best k. If none are found, widen the search and try again.

One thing worth getting right. The obvious implementation perturbs *every*
feature on every draw -- and if the categorical features are re-randomised each
time, no candidate can ever keep the original island or sex. Every counterfactual
then differs in those columns whether or not the change was needed, and the
distance ranking cannot repair it because no sparse candidate is ever generated.
A counterfactual that says "had this penguin been on a different island, been of
the other sex, and been 3 mm longer in the bill" is nearly useless next to one
that says "3 mm longer in the bill".

So each draw perturbs a random *subset* of the features. Sparse candidates are
generated, and the distance then does its job.

Distance follows Wachter et al.: for numeric features the absolute difference
divided by that feature's median absolute deviation, which puts features with
different spreads on comparable footing; for categorical features an indicator
that the value changed at all.
"""

import numpy as np
import pandas as pd

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def mad(X):
    """Median absolute deviation per numeric feature, never zero."""
    values = X[NUMERIC_FEATURES].to_numpy(dtype=float)
    deviations = np.abs(values - np.median(values, axis=0))
    scale = np.median(deviations, axis=0)
    scale[scale == 0] = 1.0
    return dict(zip(NUMERIC_FEATURES, scale))


def distances(candidates, x, mad_scale):
    """MAD-weighted L1 distance from each candidate row to x."""
    total = np.zeros(len(candidates), dtype=float)
    for name in NUMERIC_FEATURES:
        delta = candidates[name].to_numpy(dtype=float) - float(x[name])
        total += np.abs(delta) / mad_scale[name]
    for name in CATEGORICAL_FEATURES:
        total += (candidates[name].to_numpy() != x[name]).astype(float)
    return total


def _sample(x, n, sigma, numeric_sd, categories, rng):
    """n perturbations of x, each touching a random subset of the features."""
    frame = pd.DataFrame([x[FEATURES]] * n).reset_index(drop=True)

    # Which features each draw is allowed to move. Every draw changes at least
    # one, and on average about half, which yields a mix of sparse and broad
    # candidates rather than only broad ones.
    mask = rng.random((n, len(FEATURES))) < 0.5
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[empty, rng.integers(0, len(FEATURES), size=empty.sum())] = True

    for j, name in enumerate(FEATURES):
        touch = mask[:, j]
        if not touch.any():
            continue
        if name in NUMERIC_FEATURES:
            column = frame[name].to_numpy(dtype=float)
            column[touch] += rng.normal(0.0, sigma * numeric_sd[name], touch.sum())
            frame[name] = column
        else:
            options = categories[name]
            frame.loc[touch, name] = rng.choice(options, size=touch.sum())

    return frame


def generate(pipeline, x, target_label, X_train, categories, k=3,
             n=4000, max_rounds=6, seed=0):
    """Best k counterfactuals for x under `pipeline`, or None if none found.

    Returns (frame, distances) where the frame carries the counterfactual rows
    in natural feature values.
    """
    rng = np.random.default_rng(seed)
    numeric_sd = X_train[NUMERIC_FEATURES].std(ddof=0).to_dict()
    mad_scale = mad(X_train)

    lower = X_train[NUMERIC_FEATURES].min()
    upper = X_train[NUMERIC_FEATURES].max()

    found = []
    rounds_used = 0
    for attempt in range(max_rounds):
        rounds_used = attempt + 1
        # Widen both the sample size and the spread on each retry, as the brief
        # suggests, so a hard target eventually becomes reachable.
        sigma = 1.0 + 0.5 * attempt
        candidates = _sample(x, n * (attempt + 1), sigma, numeric_sd, categories, rng)
        candidates[NUMERIC_FEATURES] = candidates[NUMERIC_FEATURES].clip(lower, upper, axis=1)

        hits = candidates[pipeline.predict(candidates) == target_label]
        if len(hits):
            found.append(hits)
        if sum(len(part) for part in found) >= k:
            break

    if not found:
        return None

    pool = pd.concat(found, ignore_index=True).drop_duplicates().reset_index(drop=True)
    scores = distances(pool, x, mad_scale)
    order = np.argsort(scores)[:k]

    result = pool.iloc[order].reset_index(drop=True)
    return result, scores[order], rounds_used


def changed_fields(row, x):
    """Which features a counterfactual actually moved, for display."""
    changes = []
    for name in FEATURES:
        if name in NUMERIC_FEATURES:
            before, after = float(x[name]), float(row[name])
            if abs(after - before) > 1e-9:
                changes.append((name, before, after))
        elif row[name] != x[name]:
            changes.append((name, x[name], row[name]))
    return changes
