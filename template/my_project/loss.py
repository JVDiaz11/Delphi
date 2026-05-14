"""
template/my_project/loss.py
----------------------------
TODO: Define your loss function here.

The loss is passed to ``core.Trainer`` as the ``criterion``
argument.  It must be an ``nn.Module`` called as::

    loss = criterion(logits, targets)

If you need an *auxiliary* loss (weighted regularisation term, etc.) use
``extra_loss_fn`` in ``trainer_factory.py`` instead.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor


class MyLoss(nn.Module):
    """TODO: Replace with your own loss function."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Example: treat as a classification problem
        return nn.functional.cross_entropy(logits, targets)
