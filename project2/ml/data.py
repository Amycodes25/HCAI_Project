"""Palmer Penguins loading and feature definitions for Project 2.

The dataset ships with the app in project2/data/penguins.csv so the site never
needs network access at runtime, matching how Project 3 serves from committed
artifacts.

Features are kept in their natural types in a DataFrame, and encoding happens
inside the model pipeline. That matters here: `island` has three unordered
values, so passing it as an integer 0/1/2 would invent an ordering and make the
logistic-regression coefficient for it meaningless -- in a project about reading
model parameters, that is not a detail we can wave away.

`year` is treated as categorical for the same reason. It records which field
season a penguin was measured in, not a quantity, and nothing sensible is meant
by "half a year later" when generating a counterfactual.
"""

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "penguins.csv"

TARGET = "species"

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_FEATURES = ["island", "sex", "year"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
    "island": "Island",
    "sex": "Sex",
    "year": "Year",
}

SPECIES_COLOURS = {
    "Adelie": "#ce9ad5",
    "Chinstrap": "#62baf9",
    "Gentoo": "#9999d6",
}


def load_penguins():
    """Return the complete cases of the dataset as (X, y).

    Rows with any missing value are dropped, which leaves 333 of 344 penguins.
    """
    frame = pd.read_csv(DATA_FILE)
    frame = frame.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    # year is categorical, but reads from the CSV as an integer.
    frame["year"] = frame["year"].astype(int).astype(str)

    return frame[FEATURES].copy(), frame[TARGET].copy()


def category_values(X):
    """The permitted values of each categorical feature, for perturbation."""
    return {name: sorted(X[name].unique().tolist()) for name in CATEGORICAL_FEATURES}
