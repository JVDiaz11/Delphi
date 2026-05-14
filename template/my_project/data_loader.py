"""
template/my_project/data_loader.py
-----------------------------------
TODO: Define your data loading here.

1. Create a ``torch.utils.data.Dataset`` subclass that loads your data.
2. Subclass ``base.BaseDataLoader`` to plug in the dataset.
   ``BaseDataLoader`` handles train/validation split automatically.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from base import BaseDataLoader


class MyDataset(Dataset):
    """TODO: Replace with your own dataset."""

    def __init__(self, path: str) -> None:
        # TODO: load your data from ``path``
        # Example: self.data, self.labels = load_csv(path)
        self.data: list[Tensor] = []
        self.labels: list[Tensor] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.data[index], self.labels[index]


class MyDataLoader(BaseDataLoader):
    """TODO: Wire ``MyDataset`` into the Delphi base loader."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: float = 0.1,
        num_workers: int = 4,
    ) -> None:
        dataset = MyDataset(data_path)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
        )
