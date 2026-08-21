"""Figures for Project 2, rendered to base64 PNGs.

Django has no direct way to show a matplotlib figure, so the brief suggests
writing it to the media directory and loading the image. We inline the PNG as a
data URI instead: it avoids accumulating one file per request in media/, and it
is the approach Project 1 already uses, so both projects behave the same way.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from .data import FEATURE_LABELS, SPECIES_COLOURS  # noqa: E402
from .models import LOGREG, TREE  # noqa: E402

# Matches the site theme.
SURFACE = "#ffffff"
INK = "#002429"
MUTED = "#58686e"
LINE = "#dfe5ee"
ACCENT = "#62baf9"


def _encode(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.6, linestyle="--", alpha=0.8)
    return ax


def _transformed_names(pipeline):
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def model_figure(entry, classes):
    """The tree itself, or the coefficient matrix for logistic regression."""
    pipeline = entry["pipeline"]

    if entry["family"] == TREE:
        model = pipeline.named_steps["model"]
        leaves = entry["omega"]
        width = max(9.0, min(22.0, leaves * 1.5))
        height = max(4.0, min(11.0, leaves * 0.75))
        fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
        plot_tree(
            model,
            feature_names=[n.split("__", 1)[-1] for n in _transformed_names(pipeline)],
            class_names=list(classes),
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            fontsize=max(6, 11 - leaves // 5),
            ax=ax,
        )
        return _encode(fig)

    coefficients = pipeline.named_steps["model"].coef_
    names = [n.split("__", 1)[-1] for n in _transformed_names(pipeline)]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    _style(ax)
    positions = np.arange(len(names))
    width = 0.26
    for i, species in enumerate(classes):
        ax.bar(positions + i * width, coefficients[i], width,
               label=species, color=SPECIES_COLOURS.get(species, ACCENT),
               edgecolor="none")
    ax.set_xticks(positions + width)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8, color=MUTED)
    ax.axhline(0, color=LINE, linewidth=1)
    ax.set_ylabel("Coefficient", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    return _encode(fig)


def tradeoff_figure(candidates, selected, family):
    """Test accuracy against complexity, with the selected model marked."""
    omegas = [c["omega"] for c in candidates]
    accuracies = [c["accuracy"] for c in candidates]
    order = np.argsort(omegas)

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURFACE)
    _style(ax)
    ax.plot(np.asarray(omegas)[order], np.asarray(accuracies)[order],
            "-o", color=ACCENT, markersize=3.5, linewidth=1.6)
    ax.axvline(selected["omega"], color="#ce9ad5", linestyle="--", linewidth=1.6,
               label=f"selected: {selected['omega']:.2f}"
                     if family == LOGREG else f"selected: {selected['omega']} leaves")
    ax.set_xlabel("Number of leaves" if family == TREE else "‖W‖₁",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("Test accuracy", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    return _encode(fig)


def effects_figure(grid, pdp_values, centres, ale_values, feature, classes, note):
    """PDP and ALE side by side, one curve per species on each."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    label = FEATURE_LABELS[feature]

    for axis in (left, right):
        _style(axis)
        axis.set_xlabel(label, color=MUTED, fontsize=9)

    for i, species in enumerate(classes):
        colour = SPECIES_COLOURS.get(species, ACCENT)
        left.plot(grid, pdp_values[i], linewidth=2, color=colour, label=species)
        right.plot(centres, ale_values[i], linewidth=2, color=colour, label=species)

    left.set_title("Partial dependence", color=INK, fontsize=10, pad=8)
    left.set_ylabel("Mean predicted probability", color=MUTED, fontsize=9)
    left.legend(fontsize=8, frameon=False, labelcolor=INK)

    right.axhline(0, color=LINE, linewidth=1, linestyle="--")
    right.set_title("Accumulated local effects", color=INK, fontsize=10, pad=8)
    right.set_ylabel("Centred effect on probability", color=MUTED, fontsize=9)
    right.legend(fontsize=8, frameon=False, labelcolor=INK)

    fig.text(0.5, -0.04, note, ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    return _encode(fig)
