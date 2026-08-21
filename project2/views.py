"""Project 2 - Explainability.

One page carrying all five tasks, because the brief requires the counterfactual
and feature-effect regions to stay linked to the model family and the lambda
chosen further up: everything below reads from the same selected model.

Interaction
-----------
State lives entirely in the query string, so any view of the interface can be
linked to and the back button works.

Changing a control does *not* reload the page. Each of the three regions can be
rendered on its own via ?partial=<region>, and the page fetches only the regions
that the changed control actually affects:

    family, lambda  -> all three (the model changed, so everything below it did)
    penguin, target, k -> counterfactuals only
    feature         -> feature effects only

That matters for more than scroll position. A full reload re-renders all three
figures, so changing the feature dropdown at the bottom of the page used to cost
a tree render and a trade-off render it did not need.

Without JavaScript the same controls still work: they belong to a single form
and fall back to an ordinary GET, which returns the whole page.
"""

from django.shortcuts import render

from .ml import artifacts, counterfactuals, effects, models, plots, treeviz
from .ml.data import FEATURE_LABELS, NUMERIC_FEATURES

DEFAULT_LAMBDA = 0.005
MAX_LAMBDA = 0.1
LAMBDA_STEP = 0.001

MODEL = "model"
COUNTERFACTUALS = "counterfactuals"
EFFECTS = "effects"

# The sidebar shows the current selection, so it has to refresh with the rest or
# it quietly reports a model that is no longer on screen.
SIDEBAR = "sidebar"

PARTIALS = {
    MODEL: "project2/_model.html",
    COUNTERFACTUALS: "project2/_counterfactuals.html",
    EFFECTS: "project2/_effects.html",
    SIDEBAR: "project2/_sidebar.html",
}


def _get_float(request, name, default, low, high):
    try:
        value = float(request.GET.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, low), high)


def _get_int(request, name, default, low, high):
    try:
        value = int(request.GET.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, low), high)


def _read_state(request, bundle):
    """The validated control state. Bad input is clamped, never raised."""
    X = bundle["X"]

    family = request.GET.get("family", models.TREE)
    if family not in (models.TREE, models.LOGREG):
        family = models.TREE

    feature = request.GET.get("feature", NUMERIC_FEATURES[0])
    if feature not in NUMERIC_FEATURES:
        feature = NUMERIC_FEATURES[0]

    return {
        "family": family,
        "lam": _get_float(request, "lam", DEFAULT_LAMBDA, 0.0, MAX_LAMBDA),
        "point_index": _get_int(request, "point", 0, 0, len(X) - 1),
        "k": _get_int(request, "k", 3, 1, 5),
        "feature": feature,
    }


def _select(bundle, state):
    candidates = (bundle["trees"] if state["family"] == models.TREE
                  else bundle["logregs"])
    return candidates, models.select_by_lambda(candidates, state["lam"])


def _model_context(bundle, state):
    candidates, selected = _select(bundle, state)
    family = state["family"]

    context = {
        "family_label": models.MODEL_LABELS[family],
        "omega_label": models.OMEGA_LABELS[family],
        "selected": selected,
        "omega_display": (f"{selected['omega']:.2f}" if family == models.LOGREG
                          else str(selected["omega"])),
        "accuracy_pct": f"{selected['accuracy'] * 100:.1f}",
        "n_candidates": len(candidates),
        "tradeoff_plot": plots.tradeoff_figure(candidates, selected, family),
    }

    if family == models.TREE:
        # Drawn as SVG rather than a PNG: themed, selectable, and legible at
        # depth. See ml/treeviz.py.
        svg, legend = treeviz.render(selected["pipeline"], bundle["classes"])
        context["tree_svg"] = svg
        context["tree_legend"] = legend
    else:
        context["model_plot"] = plots.model_figure(selected, bundle["classes"])

    return context


def _counterfactual_context(request, bundle, state):
    X, y = bundle["X"], bundle["y"]
    classes = bundle["classes"]
    _, selected = _select(bundle, state)
    pipeline = selected["pipeline"]

    index = state["point_index"]
    x = X.iloc[index]
    predicted = pipeline.predict(X.iloc[[index]])[0]

    target = request.GET.get("target")
    if target not in classes:
        target = next(c for c in classes if c != predicted)

    context = {
        "points": list(range(len(X))),
        "point_values": [(FEATURE_LABELS[name], x[name]) for name in x.index],
        "point_actual": y.iloc[index],
        "point_predicted": predicted,
        "classes": classes,
        "target": target,
    }

    if target == predicted:
        context["cf_note"] = (
            f"The selected model already predicts {target} for this penguin, "
            "so there is nothing to explain. Choose a different target."
        )
        context["counterfactuals"] = []
        return context

    result = counterfactuals.generate(
        pipeline, x, target, bundle["X_train"], bundle["categories"], k=state["k"]
    )
    if result is None:
        context["cf_note"] = (
            "No counterfactual was found for this target, even after widening the "
            "search. This penguin sits far from that decision region."
        )
        context["counterfactuals"] = []
        return context

    rows, dists, rounds = result
    found = []
    for i in range(len(rows)):
        changes = counterfactuals.changed_fields(rows.iloc[i], x)
        found.append({
            "distance": f"{dists[i]:.2f}",
            "n_changed": len(changes),
            "changes": [
                {
                    "label": FEATURE_LABELS[name],
                    "before": f"{before:.1f}" if isinstance(before, float) else before,
                    "after": f"{after:.1f}" if isinstance(after, float) else after,
                }
                for name, before, after in changes
            ],
        })
    context["counterfactuals"] = found
    context["cf_note"] = (
        f"Ranked by MAD-weighted L1 distance, found in {rounds} "
        f"sampling round{'s' if rounds > 1 else ''}."
    )
    return context


def _effects_context(bundle, state):
    _, selected = _select(bundle, state)
    pipeline = selected["pipeline"]
    feature = state["feature"]

    grid, pdp_values = effects.partial_dependence(pipeline, bundle["X_train"], feature)
    centres, ale_values, method = effects.accumulated_local_effects(
        pipeline, bundle["X_train"], feature, state["family"]
    )
    note = effects.METHOD_NOTES[method]

    return {
        "numeric_features": [(name, FEATURE_LABELS[name]) for name in NUMERIC_FEATURES],
        "feature_label": FEATURE_LABELS[feature],
        "effects_plot": plots.effects_figure(
            grid, pdp_values, centres, ale_values, feature, bundle["classes"], note
        ),
        "derivative_method": method,
        "derivative_note": note,
    }


def index(request):
    bundle = artifacts.load()
    state = _read_state(request, bundle)

    base = dict(state)
    base.update({
        "max_lambda": MAX_LAMBDA,
        "lambda_step": LAMBDA_STEP,
        "train_size": len(bundle["X_train"]),
        "test_size": len(bundle["X_test"]),
    })

    requested = request.GET.get("partial")
    if requested in PARTIALS:
        builder = {
            MODEL: lambda: _model_context(bundle, state),
            COUNTERFACTUALS: lambda: _counterfactual_context(request, bundle, state),
            EFFECTS: lambda: _effects_context(bundle, state),
            # The sidebar reports across all three, so it needs the lot.
            SIDEBAR: lambda: {
                **_model_context(bundle, state),
                **_counterfactual_context(request, bundle, state),
                **_effects_context(bundle, state),
            },
        }[requested]
        context = {**base, **builder()}
        return render(request, PARTIALS[requested], context)

    context = {
        **base,
        **_model_context(bundle, state),
        **_counterfactual_context(request, bundle, state),
        **_effects_context(bundle, state),
    }
    return render(request, "project2/index.html", context)
