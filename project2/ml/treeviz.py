"""Render a fitted decision tree as inline SVG.

`sklearn.tree.plot_tree` produces a matplotlib PNG. That has three problems for
this interface:

* It grows very wide with depth, so it either scrolls a long way or shrinks
  until the labels are unreadable.
* It shows the *transformed* feature names, so a one-hot column reads
  `categorical__island_Biscoe <= 0.5`, which is both ugly and backwards -- that
  branch is taken when the penguin is *not* on Biscoe.
* It is a raster image, so it does not match the site's palette, cannot be
  selected or searched, and carries a large base64 payload on every request.

This module walks `tree_` directly and emits SVG: crisp at any zoom, themed with
the site's colours, real selectable text, and a fraction of the bytes. Each node
shows its split in plain language, how many training penguins reach it, and a
stacked bar of the class mixture. Leaves are tinted with the species they
predict.
"""

import html

from .data import SPECIES_COLOURS

NODE_WIDTH = 190
NODE_HEIGHT = 74
H_GAP = 26
V_GAP = 58
PADDING = 16

COLOURS = {
    "ink": "#002429",
    "muted": "#58686e",
    "line": "#cfd9e6",
    "surface": "#ffffff",
    "edge": "#9fb1c4",
}


def _feature_labels(pipeline):
    """Human-readable descriptions of each transformed column.

    Returns a list of (kind, text) where kind is "numeric" or "categorical".
    """
    from .data import FEATURE_LABELS

    labels = []
    for raw in pipeline.named_steps["prep"].get_feature_names_out():
        block, _, name = raw.partition("__")
        if block == "numeric":
            labels.append(("numeric", FEATURE_LABELS.get(name, name)))
        else:
            # e.g. island_Biscoe -> ("Island", "Biscoe")
            for feature in ("island", "sex", "year"):
                prefix = feature + "_"
                if name.startswith(prefix):
                    labels.append(
                        ("categorical",
                         f"{FEATURE_LABELS.get(feature, feature)} is {name[len(prefix):]}")
                    )
                    break
            else:
                labels.append(("categorical", name))
    return labels


def _condition(kind, label, threshold):
    """The question asked at a node, phrased for a reader.

    A one-hot column splits at 0.5, so `<= 0.5` means "does not have this
    value". Rather than print that, the node asks the positive question and the
    branches carry yes and no.
    """
    if kind == "categorical":
        return f"{label}?"
    return f"{label} ≤ {threshold:.1f}"


def _layout(tree, node, depth, cursor, positions):
    """Assign each node an (x, y). Leaves take the next slot; parents centre."""
    left, right = tree.children_left[node], tree.children_right[node]

    if left == -1:
        x = cursor[0] * (NODE_WIDTH + H_GAP)
        cursor[0] += 1
    else:
        _layout(tree, left, depth + 1, cursor, positions)
        _layout(tree, right, depth + 1, cursor, positions)
        x = 0.5 * (positions[left][0] + positions[right][0])

    positions[node] = (x, depth * (NODE_HEIGHT + V_GAP))
    return positions


def _distribution_bar(values, x, y, width):
    """A stacked bar of the class mixture at a node."""
    total = float(values.sum()) or 1.0
    parts = []
    offset = 0.0
    for i, species in enumerate(sorted(SPECIES_COLOURS)):
        share = float(values[i]) / total
        if share <= 0:
            continue
        segment = share * width
        parts.append(
            f'<rect x="{x + offset:.2f}" y="{y}" width="{segment:.2f}" height="6" '
            f'fill="{SPECIES_COLOURS[species]}" rx="3">'
            f'<title>{html.escape(species)}: {share:.0%}</title></rect>'
        )
        offset += segment
    return "".join(parts)


def render(pipeline, classes):
    """SVG markup for the fitted tree inside `pipeline`."""
    model = pipeline.named_steps["model"]
    tree = model.tree_
    labels = _feature_labels(pipeline)
    class_order = list(model.classes_)

    positions = _layout(tree, 0, 0, [0], {})
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    width = max(xs) - min(xs) + NODE_WIDTH + 2 * PADDING
    height = max(ys) + NODE_HEIGHT + 2 * PADDING + 8
    shift = PADDING - min(xs)

    edges, nodes = [], []

    for node, (x, y) in sorted(positions.items()):
        nx, ny = x + shift, y + PADDING
        left, right = tree.children_left[node], tree.children_right[node]
        samples = int(tree.n_node_samples[node])
        values = tree.value[node][0]

        # Edges are drawn first so nodes sit on top of them.
        for child, answer in ((left, "yes"), (right, "no")):
            if child == -1:
                continue
            cx, cy = positions[child]
            cx, cy = cx + shift, cy + PADDING
            start_x, start_y = nx + NODE_WIDTH / 2, ny + NODE_HEIGHT
            end_x, end_y = cx + NODE_WIDTH / 2, cy
            mid_y = (start_y + end_y) / 2
            edges.append(
                f'<path d="M {start_x:.1f} {start_y:.1f} '
                f'C {start_x:.1f} {mid_y:.1f}, {end_x:.1f} {mid_y:.1f}, '
                f'{end_x:.1f} {end_y:.1f}" fill="none" '
                f'stroke="{COLOURS["edge"]}" stroke-width="1.5"/>'
            )
            label_x = (start_x + end_x) / 2
            edges.append(
                f'<text x="{label_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" '
                f'dominant-baseline="middle" font-size="10" font-weight="700" '
                f'fill="{COLOURS["muted"]}">{answer}</text>'
            )

        if left == -1:
            predicted = class_order[int(values.argmax())]
            purity = float(values.max()) / (float(values.sum()) or 1.0)
            tint = SPECIES_COLOURS.get(predicted, COLOURS["surface"])
            nodes.append(
                f'<g><rect x="{nx:.1f}" y="{ny:.1f}" width="{NODE_WIDTH}" '
                f'height="{NODE_HEIGHT}" rx="14" fill="{tint}" fill-opacity="0.20" '
                f'stroke="{tint}" stroke-width="1.5"/>'
                f'<text x="{nx + NODE_WIDTH / 2:.1f}" y="{ny + 26:.1f}" '
                f'text-anchor="middle" font-size="13" font-weight="750" '
                f'fill="{COLOURS["ink"]}">{html.escape(predicted)}</text>'
                f'<text x="{nx + NODE_WIDTH / 2:.1f}" y="{ny + 43:.1f}" '
                f'text-anchor="middle" font-size="10.5" fill="{COLOURS["muted"]}">'
                f'{samples} penguins · {purity:.0%} pure</text>'
                + _distribution_bar(values, nx + 20, ny + 52, NODE_WIDTH - 40)
                + "</g>"
            )
            continue

        kind, label = labels[tree.feature[node]]
        question = _condition(kind, label, tree.threshold[node])
        nodes.append(
            f'<g><rect x="{nx:.1f}" y="{ny:.1f}" width="{NODE_WIDTH}" '
            f'height="{NODE_HEIGHT}" rx="14" fill="{COLOURS["surface"]}" '
            f'stroke="{COLOURS["line"]}" stroke-width="1.5"/>'
            f'<text x="{nx + NODE_WIDTH / 2:.1f}" y="{ny + 27:.1f}" '
            f'text-anchor="middle" font-size="12" font-weight="700" '
            f'fill="{COLOURS["ink"]}">{html.escape(question)}</text>'
            f'<text x="{nx + NODE_WIDTH / 2:.1f}" y="{ny + 43:.1f}" '
            f'text-anchor="middle" font-size="10.5" fill="{COLOURS["muted"]}">'
            f'{samples} penguins</text>'
            + _distribution_bar(values, nx + 20, ny + 52, NODE_WIDTH - 40)
            + "</g>"
        )

    legend = " ".join(
        f'<span class="tree-key"><i style="background:{colour}"></i>{html.escape(name)}</span>'
        for name, colour in sorted(SPECIES_COLOURS.items())
    )

    svg = (
        f'<svg class="tree-svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" '
        f'aria-label="The fitted decision tree" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Inter, Segoe UI, Arial, sans-serif">'
        + "".join(edges) + "".join(nodes) + "</svg>"
    )
    return svg, legend
