from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        validation_split: float,
        num_workers: int,
        **kwargs: Any,
    ) -> None:
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.valid_indices: list[int] = []

        self.sampler, self.valid_sampler = self._split_sampler(validation_split)

        init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.sampler is None and shuffle,
            "num_workers": num_workers,
            "pin_memory": kwargs.pop("pin_memory", torch.cuda.is_available()),
            **kwargs,
        }
        if self.sampler is not None:
            init_kwargs["sampler"] = self.sampler

        super().__init__(**init_kwargs)

    def _split_sampler(self, split: float):
        if split <= 0.0:
            return None, None

        valid_len = int(self.n_samples * split)
        if valid_len <= 0 or valid_len >= self.n_samples:
            return None, None

        indices = torch.randperm(self.n_samples).tolist()
        self.valid_indices = indices[:valid_len]
        train_indices = indices[valid_len:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(self.valid_indices)
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            num_workers=getattr(self, "num_workers", 0),
            pin_memory=getattr(self, "pin_memory", torch.cuda.is_available()),
        )
