"""Task 1 - feature representation for the movie recommender.

The brief fixes the utility as U(x) = w^T x and says w must be estimated "from a
limited number of interactions". Those two sentences together decide the design,
and pull in opposite directions.

Because U is linear, anything the representation does not carry cannot be
learned at all: a taste for a particular director exists only if some component
of x expresses it. That argues for more features.

Because w is estimated from roughly ten interactions, every extra component is
another number to identify from the same evidence. A representation with one
column per director would have thousands of components and none of them would be
estimable. That argues for fewer.

So the representation below is deliberately small -- 22 components -- and each
one is chosen because it separates films in a way a person would actually
recognise as taste.

What is included, and why
-------------------------
Genre (multi-hot, 15 columns)
    The strongest and most interpretable signal about taste, and multi-label by
    nature: a film can be both a comedy and a romance, so multi-hot rather than
    one-hot. The 26 genres in the dataset are truncated to the 15 most common;
    the rest appear on so few films that their weights could never be estimated
    from a short session.

Era (1 column, scaled release year)
    People have era preferences, and a single scaled number captures the
    monotone part of that at the cost of one component. One-hot decades would
    cost ten and buy a shape nobody needs.

Runtime (1 column, scaled)
    A real and commonly expressed preference: a ninety-minute film and a
    three-hour film are different propositions on a weeknight.

Critical standing (1 column, scaled IMDB score)
    Separates people who follow acclaim from people who do not.

Popularity (1 column, scaled log vote count)
    Mainstream against obscure. Logged first, because vote counts span five
    orders of magnitude and the raw scale would let a handful of blockbusters
    dominate the inner product.

Production scale (1 column, scaled log budget)
    Blockbuster against small-budget film, which is a distinct axis from
    popularity: expensive films can flop and cheap ones can be seen by everyone.

Director prominence (1 column, scaled log director followers)
    A cheap proxy for the auteur axis that does not require a column per
    director.

Family-friendly (1 column, binary)
    Content rating collapsed to G/PG against everything else. One-hot over the
    dozen ratings in the data would spend a dozen components on a distinction
    that matters mainly at this boundary.

What is excluded, and why
-------------------------
Director and cast identity, plot keywords, and production country are all
high-cardinality. One column each would swamp the budget; the two that carry
usable signal are already represented by director prominence and popularity.

Gross is excluded because it is largely determined by budget and popularity,
which are both present -- a third correlated column adds collinearity, and
collinear features make w harder to identify from few observations, which is
exactly the constraint that matters here.

Standardisation
---------------
Every continuous column is standardised. Since utility is w^T x, the scale of a
column sets the scale of its weight; without standardisation the budget column
(order 1e8) and the genre columns (0 or 1) could not share a sensible prior, and
the regularisation in the estimator would penalise them wildly unevenly.
"""

import functools
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

IS_IMPLEMENTED = True

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "movie_metadata.csv"

N_GENRES = 15

CONTINUOUS_SPECS = [
    ("title_year", "Era", False),
    ("duration", "Runtime", False),
    ("imdb_score", "Critical standing", False),
    ("num_voted_users", "Popularity", True),
    ("budget", "Production scale", True),
    ("director_facebook_likes", "Director prominence", True),
]

FAMILY_RATINGS = {"G", "PG"}


@functools.lru_cache(maxsize=1)
def _build():
    """Load the dataset and compute the design matrix. Cached; runs once."""
    frame = pd.read_csv(DATA_FILE)

    frame = frame.dropna(subset=[column for column, _, _ in CONTINUOUS_SPECS])
    frame = frame.dropna(subset=["movie_title", "genres", "content_rating"])
    frame = frame.drop_duplicates(subset=["movie_title", "title_year"])
    frame = frame.reset_index(drop=True)

    # The dataset stores titles with a trailing non-breaking space.
    frame["movie_title"] = frame["movie_title"].str.replace("\xa0", "", regex=False).str.strip()

    genre_lists = frame["genres"].str.split("|")
    top_genres = (
        genre_lists.explode().value_counts().head(N_GENRES).index.tolist()
    )
    top_genres.sort()

    columns = []
    names = []

    for genre in top_genres:
        columns.append(genre_lists.apply(lambda gs, g=genre: float(g in gs)).to_numpy())
        names.append(genre)

    for column, label, take_log in CONTINUOUS_SPECS:
        values = frame[column].to_numpy(dtype=float)
        if take_log:
            values = np.log10(np.maximum(values, 1.0))
        columns.append(values)
        names.append(label)

    family = frame["content_rating"].isin(FAMILY_RATINGS).to_numpy(dtype=float)
    columns.append(family)
    names.append("Family-friendly")

    X = np.column_stack(columns)

    # Standardise everything except the binary indicators, which are already on
    # a comparable 0/1 scale and are more readable left alone.
    continuous = slice(N_GENRES, N_GENRES + len(CONTINUOUS_SPECS))
    block = X[:, continuous]
    X[:, continuous] = (block - block.mean(axis=0)) / block.std(axis=0)

    return frame, X, names


def feature_names():
    return _build()[2]


def n_features():
    return len(feature_names())


def movie_count():
    return len(_build()[0])


def design_matrix():
    """The full (n_movies, n_features) matrix."""
    return _build()[1]


def movie_rows(indices):
    """Feature rows for the given movie indices."""
    return design_matrix()[list(indices)]


def describe(index):
    """A film as the interface shows it."""
    frame = _build()[0]
    row = frame.iloc[index]
    return {
        "index": int(index),
        "title": row["movie_title"],
        "year": int(row["title_year"]),
        "genres": row["genres"].replace("|", " / "),
        "duration": int(row["duration"]),
        "director": row["director_name"] if pd.notna(row["director_name"]) else "",
        "score": f"{row['imdb_score']:.1f}",
        "rating": row["content_rating"],
    }


def sample_movies(n, seed=None):
    """n films for one trial, drawn uniformly at random.

    The brief permits uniform selection and notes that a more informative or
    adaptive strategy would be an extension; uniform is what is implemented.

    The seed is hashed with blake2b rather than with the built-in hash().
    Python salts string hashing per process, so hash("(0, 1)") differs between
    runs: the films a participant saw would change every time the server was
    restarted, which is not acceptable in an instrument whose results have to
    be reproducible from the stored data. Callers pass the participant token as
    part of the seed, so each person gets their own films and any session can
    be reconstructed exactly from its token.
    """
    digest = hashlib.blake2b(str(seed).encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    chosen = rng.choice(movie_count(), size=n, replace=False)
    return [describe(int(i)) for i in chosen]


def genre_summary():
    """Which genres made the cut, for the write-up."""
    return feature_names()[:N_GENRES]
