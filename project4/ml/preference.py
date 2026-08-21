"""Task 2 - the preference model and its estimator. NOT YET IMPLEMENTED.

What belongs here:

* The Plackett-Luce likelihood, extending Bradley-Terry from a single pairwise
  comparison to a full ranking i1 > i2 > ... > in:

      P(i1 > ... > in | w) = prod_k exp(w^T x_ik) / sum_{j>=k} exp(w^T x_ij)

  It reduces to Bradley-Terry at n = 2, which is what makes it an extension
  rather than a substitution.

* The estimator for w. Regularised maximum likelihood is the floor; a Bayesian
  posterior is what Lecture 9 argues for, since it keeps the uncertainty that a
  point estimate throws away after only a handful of comparisons -- and that
  uncertainty is usable as a study measure in its own right.

Nothing here yet, so the study records responses without estimating anything.
`IS_IMPLEMENTED` stays False until it does.
"""

IS_IMPLEMENTED = False
