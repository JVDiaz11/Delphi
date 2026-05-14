from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base training loop interface."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError
