from django.db import models


class StudySession(models.Model):
    """One participant's sitting.

    This exists for two reasons.

    Counterbalancing has to be decided across participants, not within one. An
    earlier version kept the alternation counter in the session, which meant
    every fresh participant started from zero and received the same order --
    counterbalancing that silently never happened, and an order effect loaded
    entirely onto one design. The assignment is therefore derived from a row
    that persists across participants.

    It is also where responses will be written once the study is run. Nothing
    identifying is stored: the participant is a random token, and that token is
    the only handle a participant has if they later ask for their data to be
    erased.
    """

    participant = models.CharField(max_length=32, unique=True)
    condition_order = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    completed = models.BooleanField(default=False)
    estimated_w = models.JSONField(null=True, blank=True)

    # The closing questionnaire: perceived effort per interface, and which one
    # the participant felt described their taste better. Both are named as
    # subjective measures in the study design, so both have to survive the
    # session that collected them.
    questionnaire = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.participant} ({', '.join(self.condition_order)})"


class Trial(models.Model):
    """One answered trial.

    The study design names time per trial as a measure and the held-out block
    as the primary one, and neither is computable from a preference vector
    alone -- they need the individual responses. Rows are written as each trial
    is answered rather than at the end, so an abandoned session still yields
    the trials that were completed.
    """

    session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="trials"
    )
    design = models.CharField(max_length=16)
    block = models.PositiveSmallIntegerField()
    index = models.PositiveSmallIntegerField()

    # Dataset indices of the films shown, best first. A pairwise choice is a
    # ranking of length two, so one field covers both interfaces.
    ordering = models.JSONField()

    # Measured in the browser, so it is the participant's thinking time rather
    # than a round trip. Null when the page could not report it.
    milliseconds = models.PositiveIntegerField(null=True, blank=True)
    answered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["session_id", "block", "index"]
        constraints = [
            models.UniqueConstraint(
                fields=["session", "block", "index"],
                name="one_row_per_trial",
            )
        ]

    def __str__(self):
        return f"{self.session.participant} {self.design} #{self.index}"
