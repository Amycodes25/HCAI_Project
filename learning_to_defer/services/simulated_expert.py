import hashlib
from dataclasses import dataclass


CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass(frozen=True)
class ExpertProfile:
    name: str = "Business news specialist"
    seed: int = 42

    world_accuracy: float = 0.86
    sports_accuracy: float = 0.70
    business_accuracy: float = 0.97
    sci_tech_accuracy: float = 0.84

    @property
    def accuracies(self):
        return {
            0: self.world_accuracy,
            1: self.sports_accuracy,
            2: self.business_accuracy,
            3: self.sci_tech_accuracy,
        }


DEFAULT_PROFILE = ExpertProfile()


def stable_random_value(text, salt, seed):
    value = f"{seed}|{salt}|{text}".encode("utf-8")
    digest = hashlib.sha256(value).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return integer / (2**64 - 1)


def predict_expert_label(text, true_label, profile=DEFAULT_PROFILE):
    if true_label not in profile.accuracies:
        raise ValueError(
            "true_label must be one of 0, 1, 2, or 3"
        )

    correctness_value = stable_random_value(
        text,
        "correctness",
        profile.seed,
    )

    required_accuracy = profile.accuracies[true_label]

    if correctness_value < required_accuracy:
        return true_label

    wrong_labels = [
        label
        for label in range(4)
        if label != true_label
    ]

    error_value = stable_random_value(
        text,
        "wrong-label",
        profile.seed,
    )

    error_index = min(
        int(error_value * len(wrong_labels)),
        len(wrong_labels) - 1,
    )

    return wrong_labels[error_index]