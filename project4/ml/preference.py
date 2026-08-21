"""Task 2 - extending Bradley-Terry to rankings, and estimating w.

The model
---------
The brief assumes preferences follow a Bradley-Terry model, which gives the
probability that one item is preferred to another from their utilities:

    P(i > j) = exp(U_i) / (exp(U_i) + exp(U_j)),      U_i = w^T x_i

Design 2 asks a participant to rank ten films, so the model has to describe a
whole ordering rather than one comparison. The extension used here is the
Plackett-Luce model. It treats a ranking as produced sequentially: the
participant picks their favourite from the whole set, then their favourite of
what remains, and so on. Each pick follows the Luce choice rule, so

    P(i_1 > i_2 > ... > i_n) = prod_{k=1}^{n-1}  exp(U_{i_k})
                                                 -----------------------
                                                 sum_{j>=k} exp(U_{i_j})

Why this extension
------------------
Two reasons.

First, it genuinely contains the model the brief starts from. At n = 2 the
product has a single factor, exp(U_1) / (exp(U_1) + exp(U_2)), which is exactly
Bradley-Terry. So the two interfaces are described by one model with one w, and
their results are directly comparable -- which is the entire point of the study.
An extension that did not reduce would be a second model in disguise.

Second, the generative story it encodes is a plausible account of what a person
does when ranking: choose the best, set it aside, choose the best of the rest.

The alternative, and why it was rejected
----------------------------------------
A ranking of ten implies 45 pairwise comparisons, and one could feed all 45 into
Bradley-Terry as if they were separate observations. This is simpler and needs
no new model. It is rejected because those 45 comparisons are not independent:
they come from one ordering produced by one person in one act. Treating them as
independent multiplies the same evidence many times over, so the likelihood is
misspecified and the apparent precision of w is badly overstated -- the
estimator would report confidence it has not earned. In a study whose whole
purpose is to compare how much each interface tells us about w, an estimator
that inflates its own confidence for one of the two designs would decide the
result before any data was collected.

Estimating w
------------
The log-likelihood of a set of observed rankings is concave in w, which makes
the maximum well behaved. It is maximised with a Gaussian prior on w, i.e.

    w_hat = argmax  sum_r log P(ranking_r | w)  -  (alpha/2) ||w||^2

The prior is not decoration. With ten interactions and twenty-two features the
unpenalised maximum is not unique -- any direction the observed films do not
distinguish is unconstrained, and the optimiser will run off along it. The
penalty pins those directions at zero, which is the honest statement that the
data said nothing about them.

Only differences in utility matter, since adding a constant to every U leaves
every probability unchanged. The features are standardised, so this shows up as
w being identified only up to what the observed comparisons actually constrain,
which the prior handles.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

IS_IMPLEMENTED = True

# Strength of the Gaussian prior on w.
DEFAULT_ALPHA = 1.0


def ranking_log_likelihood(w, rankings):
    """Log P(observed rankings | w) under Plackett-Luce.

    `rankings` is a list of (n_items, n_features) arrays, each already ordered
    best first.
    """
    total = 0.0
    for X in rankings:
        utilities = X @ w
        # The k-th factor compares the chosen item against everything still
        # available, so the normaliser runs over the tail of the ordering.
        for k in range(len(utilities) - 1):
            total += utilities[k] - logsumexp(utilities[k:])
    return total


def _objective(w, rankings, alpha):
    """Negative log posterior, and its gradient."""
    value = 0.0
    grad = np.zeros_like(w)

    for X in rankings:
        utilities = X @ w
        for k in range(len(utilities) - 1):
            tail = utilities[k:]
            normaliser = logsumexp(tail)
            value += utilities[k] - normaliser

            # d/dw [ u_k - logsumexp(u_k..u_n) ]
            #     = x_k - sum_j softmax(tail)_j x_j
            weights = np.exp(tail - normaliser)
            grad += X[k] - weights @ X[k:]

    value -= 0.5 * alpha * (w @ w)
    grad -= alpha * w

    return -value, -grad


def fit(rankings, n_features, alpha=DEFAULT_ALPHA):
    """Estimate w from observed rankings.

    A pairwise choice is simply a ranking of length two, so both interfaces feed
    the same estimator and produce comparable estimates.
    """
    rankings = [np.asarray(r, dtype=float) for r in rankings if len(r) >= 2]

    if not rankings:
        return np.zeros(n_features)

    result = minimize(
        _objective,
        x0=np.zeros(n_features),
        args=(rankings, alpha),
        jac=True,
        method="L-BFGS-B",
    )
    return result.x


def utilities(X, w):
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float)


def rank_items(X, w):
    """Indices of the rows of X, best first, under w."""
    return np.argsort(-utilities(X, w))


def pairwise_probability(x_i, x_j, w):
    """P(i preferred to j). The n = 2 case, written out for clarity."""
    difference = float((np.asarray(x_i) - np.asarray(x_j)) @ np.asarray(w))
    return 1.0 / (1.0 + np.exp(-difference))
