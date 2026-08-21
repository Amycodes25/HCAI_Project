"""Build and load the pre-trained Project 2 models.

69 pipelines are fitted here: 39 trees over the leaf grid and 30 logistic
regressions over the C grid. That takes a while, so it happens once and the
result is cached to disk, the same way Project 3 serves from committed
artifacts. A request never trains anything.

Rebuild with:

    python manage.py build_project2
"""

from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .data import category_values, load_penguins
from .models import RANDOM_STATE, train_logregs, train_trees

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "models.joblib"

TEST_SIZE = 0.2

_CACHE = None


def build(verbose=False):
    """Fit every candidate model and write the artifact. Returns the bundle."""
    X, y = load_penguins()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if verbose:
        print(f"training on {len(X_train)} penguins, testing on {len(X_test)}")

    trees = train_trees(X_train, y_train, X_test, y_test)
    if verbose:
        print(f"  {len(trees)} decision trees")

    logregs = train_logregs(X_train, y_train, X_test, y_test)
    if verbose:
        print(f"  {len(logregs)} logistic regressions")

    bundle = {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "categories": category_values(X),
        "classes": sorted(y.unique().tolist()),
        "trees": trees,
        "logregs": logregs,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_FILE, compress=3)
    if verbose:
        size = ARTIFACT_FILE.stat().st_size / 1_000_000
        print(f"wrote {ARTIFACT_FILE} ({size:.1f} MB)")

    return bundle


def load():
    """The cached bundle, building it on first use if the artifact is absent."""
    global _CACHE
    if _CACHE is None:
        if ARTIFACT_FILE.exists():
            _CACHE = joblib.load(ARTIFACT_FILE)
        else:
            _CACHE = build()
    return _CACHE
