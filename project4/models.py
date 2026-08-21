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

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.participant} ({', '.join(self.condition_order)})"
