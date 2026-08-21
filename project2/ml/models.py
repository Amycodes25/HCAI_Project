"""Model families, complexity measures, and the lambda trade-off for Project 2.

Task 1 fits a decision tree. Task 2 adds a slider over lambda and shows the model
maximising

    acc_test - Omega(f)

with Omega(f) the number of leaves for a tree. Task 3 repeats this for logistic
regression, where we take Omega(f) = ||W||_1, the sum of the absolute values of
all coefficients.

Why that measure: a multinomial logistic regression is interpretable to the
extent that a reader can hold its weights in mind at once. The L1 norm of the
coefficient matrix grows both as more features get non-zero weight and as
individual weights get larger, so it penalises exactly the models that are hard
to read off. Sweeping the inverse regularisation strength C traces out a
sensible range of it, from a near-flat model to one that fits the training data
closely.

Both families are trained inside a Pipeline whose first step encodes the raw
frame, so callers only ever handle natural feature values.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

TREE = "tree"
LOGREG = "logreg"

MODEL_LABELS = {TREE: "Decision tree", LOGREG: "Logistic regression"}

# Omega for each family, and how it is displayed.
OMEGA_LABELS = {TREE: "Leaves", LOGREG: "‖W‖₁"}

LEAF_GRID = list(range(2, 41))
C_GRID = np.logspace(-3, 2, 30)

RANDOM_STATE = 42


def build_preprocessor():
    """Scale the numeric features, one-hot the unordered categorical ones."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _fit(estimator, X_train, y_train):
    pipeline = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    return pipeline.fit(X_train, y_train)


def tree_omega(pipeline):
    """Number of leaves."""
    return int(pipeline.named_steps["model"].get_n_leaves())


def logreg_omega(pipeline):
    """L1 norm of the coefficient matrix."""
    return float(np.abs(pipeline.named_steps["model"].coef_).sum())


def train_trees(X_train, y_train, X_test, y_test):
    """One tree per leaf budget, with its test accuracy and complexity."""
    fitted = []
    for max_leaves in LEAF_GRID:
        pipeline = _fit(
            DecisionTreeClassifier(
                max_leaf_nodes=max_leaves, random_state=RANDOM_STATE
            ),
            X_train,
            y_train,
        )
        fitted.append(
            {
                "family": TREE,
                "pipeline": pipeline,
                "accuracy": float(accuracy_score(y_test, pipeline.predict(X_test))),
                "omega": tree_omega(pipeline),
                "setting": max_leaves,
                "setting_label": f"max_leaf_nodes = {max_leaves}",
            }
        )
    return fitted


def train_logregs(X_train, y_train, X_test, y_test):
    """One logistic regression per value of C.

    Two scikit-learn API notes, both of which bite silently:

    * There is no `multi_class` argument. It was removed in 1.7, and multinomial
      is the default for lbfgs in any case.
    * There is no `penalty` argument either. It was deprecated in 1.8 in favour
      of `l1_ratio`, and passing `penalty="l1"` on a current version is ignored
      in favour of the default ratio -- fitting L2 while appearing to ask for L1.

    So fitting uses the default penalty and varies only C. That is also the
    cleaner reading of the brief, which points out that the lambda on the slider
    is a different thing from the regularisation used during fitting: Omega is
    measured on the coefficients afterwards, whatever penalty produced them.
    """
    fitted = []
    for C in C_GRID:
        pipeline = _fit(
            LogisticRegression(
                C=float(C),
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            X_train,
            y_train,
        )
        fitted.append(
            {
                "family": LOGREG,
                "pipeline": pipeline,
                "accuracy": float(accuracy_score(y_test, pipeline.predict(X_test))),
                "omega": logreg_omega(pipeline),
                "setting": float(C),
                "setting_label": f"C = {C:.4g}",
            }
        )
    return fitted


def select_by_lambda(candidates, lam):
    """The candidate maximising accuracy - lambda * Omega.

    Ties go to the simpler model, which is the point of the exercise.
    """
    return max(
        candidates,
        key=lambda entry: (entry["accuracy"] - lam * entry["omega"], -entry["omega"]),
    )
