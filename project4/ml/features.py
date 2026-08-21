"""Task 1 - feature representation for the movie recommender. NOT YET IMPLEMENTED.

The real work belongs here: load the IMDB 5000 Movie Dataset, choose a feature
vector x for each film, and justify the choice. The utility is linear,
U(x) = w^T x, so the feature design *is* the model -- anything not expressible
as a linear function of x cannot be learned, and a vector too long to identify
from ten interactions cannot be estimated.

Until that lands, the interface is driven by a small sample of films so the
participant flow can be built and tested. `IS_IMPLEMENTED` stays False, and
every page that would otherwise present a result says plainly that the feature
work is outstanding rather than showing a placeholder number.
"""

import csv
import random
from pathlib import Path

IS_IMPLEMENTED = False

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_movies.csv"

# What the feature vector is expected to carry. Deciding this properly, and
# justifying it, is Task 1.
FEATURE_NAMES = [
    "genre (multi-hot)",
    "decade",
    "duration",
    "IMDB score",
    "director standing",
    "content rating",
]

_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        with DATA_FILE.open(encoding="utf-8", newline="") as handle:
            _CACHE = list(csv.DictReader(handle))
    return _CACHE


def movie_count():
    return len(_load())


def sample_movies(n, seed=None):
    """n films for one elicitation trial.

    The brief permits uniform random selection, and notes that an adaptive
    strategy would be an interesting extension. Uniform is what happens here.
    """
    movies = _load()
    # random.Random only accepts scalars, and callers identify a trial by
    # (stage, trial), so the pair is flattened into a string.
    rng = random.Random(str(seed))
    return rng.sample(movies, min(n, len(movies)))
