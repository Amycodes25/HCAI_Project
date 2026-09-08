"""Persistence for Project 1.

The brief suggests these: "If you decide to implement several learning
algorithms, you might have to define django models for the algorithms and the
variables." Three algorithms are offered, so they are defined here.

They earn their place rather than just answering the hint. Without them a
training run existed only for as long as the response that produced it, so a
user could compare the configurations inside one sweep but never two sweeps
against each other -- and comparing a decision tree against k-nearest
neighbours on the same data is the question the interface exists to help with.
With them, every run is kept against its dataset and the results section can
show what has been tried already.

Only metadata is stored. The uploaded rows stay in the session and are never
written to the database: the app has no need for them after the request, and
keeping someone's data around longer than the task requires would be poor
practice for a course about human-centric systems.
"""

from django.db import models


class Dataset(models.Model):
    """One uploaded CSV, described rather than stored."""

    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    n_rows = models.PositiveIntegerField()
    n_columns = models.PositiveIntegerField()
    column_names = models.JSONField()

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.filename} ({self.n_rows} rows)"


class TrainingRun(models.Model):
    """One sweep: a model family trained across its hyperparameter values."""

    dataset = models.ForeignKey(
        Dataset, on_delete=models.CASCADE, related_name="runs"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    algorithm = models.CharField(max_length=32)
    algorithm_label = models.CharField(max_length=64)
    parameter_name = models.CharField(max_length=64)

    target = models.CharField(max_length=255)
    test_size = models.FloatField()
    score_name = models.CharField(max_length=32)
    score_label = models.CharField(max_length=64)

    # Denormalised so the results table can be listed without walking every
    # ModelResult of every run.
    best_parameter = models.CharField(max_length=64)
    best_score = models.FloatField()

    # best_score is the cross-validated score that chose this configuration;
    # held_out_test_score is that configuration measured once against data it
    # never influenced. Keeping both is the point: a run that only stored the
    # test score would tempt a reader into treating it as having chosen the
    # model, when cross-validation did.
    held_out_test_score = models.FloatField(null=True, blank=True)
    cv_folds = models.PositiveIntegerField(default=0)
    outliers_excluded = models.PositiveIntegerField(default=0)

    training_rows = models.PositiveIntegerField()
    testing_rows = models.PositiveIntegerField()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.algorithm_label} on {self.dataset.filename}"


class ModelResult(models.Model):
    """One configuration within a sweep, and what it scored."""

    run = models.ForeignKey(
        TrainingRun, on_delete=models.CASCADE, related_name="results"
    )
    parameter_value = models.CharField(max_length=64)
    score = models.FloatField()

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"{self.parameter_value}: {self.score}"
