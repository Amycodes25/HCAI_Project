"""PDP and ALE for Project 2, Task 5. Both computed here, not taken from a library.

The brief asks which model allows the partial derivatives to be computed exactly
and which forces a discretisation. The answer is implemented rather than only
asserted:

* Multinomial logistic regression is differentiable in closed form. With
  probabilities P and coefficient matrix W, and a numeric feature standardised
  by sigma before it reaches the model,

      dP_k/dx_j = P_k * ( W[k, j] - sum_m P_m * W[m, j] ) / sigma_j

  which is the softmax Jacobian composed with the linear layer and the scaler.
  `_exact_gradient` below is exactly that expression, and the ALE for this model
  integrates it.

* A decision tree predicts a piecewise-constant function. Its derivative is zero
  almost everywhere and undefined on the split points themselves, so there is
  nothing exact to evaluate. ALE falls back to the usual finite difference
  across each bin's edges.

ALE accumulates *local* effects computed only over the points that actually fall
in each bin, so it does not ask the model about feature combinations that never
occur -- which is the well-known failure of PDP under correlated features, and
bill depth and flipper length here are strongly correlated through species.
"""

import numpy as np

from .data import NUMERIC_FEATURES
from .models import LOGREG

EXACT = "exact"
FINITE_DIFFERENCE = "finite-difference"

METHOD_NOTES = {
    EXACT: (
        "Logistic regression is differentiable in closed form, so the accumulated "
        "effect integrates the exact softmax derivative."
    ),
    FINITE_DIFFERENCE: (
        "A decision tree is piecewise constant, so its derivative is zero almost "
        "everywhere and undefined at the splits. The effect uses finite "
        "differences across each bin instead."
    ),
}


def _scaler_scale(pipeline, feature):
    """The standard deviation the pipeline divides `feature` by."""
    scaler = pipeline.named_steps["prep"].named_transformers_["numeric"]
    return float(scaler.scale_[NUMERIC_FEATURES.index(feature)])


def _coefficient_column(pipeline, feature):
    """Coefficients of the transformed column belonging to `feature`."""
    model = pipeline.named_steps["model"]
    return model.coef_[:, NUMERIC_FEATURES.index(feature)]


def _exact_gradient(pipeline, frame, feature):
    """dP_k/dx_j for every row of `frame`, shape (n_rows, n_classes)."""
    probabilities = pipeline.predict_proba(frame)
    weights = _coefficient_column(pipeline, feature)
    sigma = _scaler_scale(pipeline, feature)

    # sum_m P_m W[m, j], one value per row
    weighted_mean = probabilities @ weights
    return probabilities * (weights[None, :] - weighted_mean[:, None]) / sigma


def partial_dependence(pipeline, X, feature, grid_size=50):
    """PDP: average predicted probability with `feature` held at each grid value."""
    values = X[feature].to_numpy(dtype=float)
    grid = np.linspace(values.min(), values.max(), grid_size)

    curves = []
    frame = X.copy()
    for value in grid:
        frame[feature] = value
        curves.append(pipeline.predict_proba(frame).mean(axis=0))

    return grid, np.asarray(curves).T


def accumulated_local_effects(pipeline, X, feature, family, n_bins=20):
    """ALE curve per class, plus which derivative method was used.

    Returns (centres, effects, method) with effects shaped (n_classes, n_bins).
    """
    values = X[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    bin_count = len(edges) - 1

    n_classes = len(pipeline.named_steps["model"].classes_)
    local = np.zeros((n_classes, bin_count))
    counts = np.zeros(bin_count)

    method = EXACT if family == LOGREG else FINITE_DIFFERENCE

    for b in range(bin_count):
        low, high = edges[b], edges[b + 1]
        in_bin = (values >= low) & (values <= high) if b == 0 else (values > low) & (values <= high)
        if not in_bin.any():
            continue
        counts[b] = in_bin.sum()
        rows = X[in_bin]

        if method == EXACT:
            # Integrate the exact derivative across the bin.
            centre = rows.copy()
            centre[feature] = 0.5 * (low + high)
            local[:, b] = _exact_gradient(pipeline, centre, feature).mean(axis=0) * (high - low)
        else:
            lower, upper = rows.copy(), rows.copy()
            lower[feature] = low
            upper[feature] = high
            difference = pipeline.predict_proba(upper) - pipeline.predict_proba(lower)
            local[:, b] = difference.mean(axis=0)

    accumulated = np.cumsum(local, axis=1)

    # Centre so the curve averages to zero over the data distribution.
    weights = counts + 1e-9
    for k in range(n_classes):
        accumulated[k] -= np.average(accumulated[k], weights=weights)

    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, accumulated, method
