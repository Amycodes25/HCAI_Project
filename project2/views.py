"""Project 2 - Explainability.

One page carrying all five tasks, because the brief requires the counterfactual
and feature-effect regions to stay linked to the model family and the lambda
chosen further up: everything below reads from the same selected model.

State lives in the query string rather than the session, so a particular view of
the interface can be linked to, and the back button behaves.
"""

from django.shortcuts import render

from .ml import artifacts, counterfactuals, effects, models, plots
from .ml.data import FEATURE_LABELS, NUMERIC_FEATURES

DEFAULT_LAMBDA = 0.005
MAX_LAMBDA = 0.1
LAMBDA_STEP = 0.001


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


def index(request):
    bundle = artifacts.load()
    X, y = bundle["X"], bundle["y"]
    classes = bundle["classes"]

    family = request.GET.get("family", models.TREE)
    if family not in (models.TREE, models.LOGREG):
        family = models.TREE

    lam = _get_float(request, "lam", DEFAULT_LAMBDA, 0.0, MAX_LAMBDA)
    point_index = _get_int(request, "point", 0, 0, len(X) - 1)
    k = _get_int(request, "k", 3, 1, 5)

    feature = request.GET.get("feature", NUMERIC_FEATURES[0])
    if feature not in NUMERIC_FEATURES:
        feature = NUMERIC_FEATURES[0]

    candidates = bundle["trees"] if family == models.TREE else bundle["logregs"]
    selected = models.select_by_lambda(candidates, lam)
    pipeline = selected["pipeline"]

    # ---- Tasks 1-3: the selected model -----------------------------------
    context = {
        "family": family,
        "family_label": models.MODEL_LABELS[family],
        "omega_label": models.OMEGA_LABELS[family],
        "lam": lam,
        "max_lambda": MAX_LAMBDA,
        "lambda_step": LAMBDA_STEP,
        "selected": selected,
        "omega_display": (f"{selected['omega']:.2f}" if family == models.LOGREG
                          else str(selected["omega"])),
        "accuracy_pct": f"{selected['accuracy'] * 100:.1f}",
        "n_candidates": len(candidates),
        "model_plot": plots.model_figure(selected, classes),
        "tradeoff_plot": plots.tradeoff_figure(candidates, selected, family),
        "train_size": len(bundle["X_train"]),
        "test_size": len(bundle["X_test"]),
    }

    # ---- Task 4: counterfactuals -----------------------------------------
    x = X.iloc[point_index]
    actual = y.iloc[point_index]
    predicted = pipeline.predict(X.iloc[[point_index]])[0]

    target = request.GET.get("target")
    if target not in classes:
        target = next(c for c in classes if c != predicted)

    context.update({
        "points": list(range(len(X))),
        "point_index": point_index,
        "point_values": [(FEATURE_LABELS[name], x[name]) for name in x.index],
        "point_actual": actual,
        "point_predicted": predicted,
        "classes": classes,
        "target": target,
        "k": k,
    })

    if target == predicted:
        context["cf_note"] = (
            f"The selected model already predicts {target} for this penguin, "
            "so there is nothing to explain. Choose a different target."
        )
        context["counterfactuals"] = []
    else:
        result = counterfactuals.generate(
            pipeline, x, target, bundle["X_train"], bundle["categories"], k=k
        )
        if result is None:
            context["cf_note"] = (
                "No counterfactual was found for this target, even after widening "
                "the search. This penguin sits far from that decision region."
            )
            context["counterfactuals"] = []
        else:
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

    # ---- Task 5: feature effects -----------------------------------------
    grid, pdp_values = effects.partial_dependence(pipeline, bundle["X_train"], feature)
    centres, ale_values, method = effects.accumulated_local_effects(
        pipeline, bundle["X_train"], feature, family
    )
    note = effects.METHOD_NOTES[method]

    context.update({
        "numeric_features": [(name, FEATURE_LABELS[name]) for name in NUMERIC_FEATURES],
        "feature": feature,
        "feature_label": FEATURE_LABELS[feature],
        "effects_plot": plots.effects_figure(
            grid, pdp_values, centres, ale_values, feature, classes, note
        ),
        "derivative_method": method,
        "derivative_note": note,
    })

    return render(request, "project2/index.html", context)
