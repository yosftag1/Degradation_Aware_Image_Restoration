"""Losses package."""

from .loss_functions import (
    PerceptualLoss,
    FrequencyLoss,
    RestorationLoss,
)

__all__ = [
    "PerceptualLoss",
    "FrequencyLoss",
    "RestorationLoss",
]
