"""
template/my_project/metrics.py
--------------------------------
TODO: Define your evaluation metrics here.

Each metric is a plain callable with signature::

    fn(logits: Tensor, targets: Tensor) -> float

Collect them in ``metric_fns`` and pass the dict to
``core.Trainer``.  The keys become the metric names in the
JSONL history, the GUI progress stream, and the status panel.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """TODO: Replace with a metric that makes sense for your task."""
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


# -----------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------

#: Pass this dict as ``metric_fns`` to ``core.Trainer``.
metric_fns: dict[str, Callable[[Tensor, Tensor], float]] = {
    "accuracy": accuracy,
    # TODO: add more metrics here, e.g.:
    # "f1_score": my_f1,
}
