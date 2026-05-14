from __future__ import annotations

from abc import ABC

from torch import nn


class BaseModel(nn.Module, ABC):
    """Base model that reports parameter counts."""

    def __str__(self) -> str:
        num_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return f"{self.__class__.__name__}(parameters={num_params:,}, trainable={trainable_params:,})"
