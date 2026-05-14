"""
template/my_project/model.py
----------------------------
TODO: Define your model here.

1. Subclass ``base.BaseModel`` (which extends ``nn.Module``).
2. Implement ``forward(self, x: Tensor) -> Tensor``.
3. Optionally implement ``architecture_summary() -> dict`` to expose
   layer info to the model registry.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from base import BaseModel


class MyModel(BaseModel):
    """TODO: Replace with your own architecture."""

    def __init__(self, *, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_features, hidden_size), nn.ReLU(inplace=True)])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def architecture_summary(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
