from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch


def ensure_dir(dirname: str | Path) -> Path:
    path = Path(dirname)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(content: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def prepare_device(n_gpu_use: int) -> tuple[torch.device, list[int]]:
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu

    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys: str) -> None:
        self._data = defaultdict(float)
        self._counts = defaultdict(int)
        self._keys = keys

    def reset(self) -> None:
        self._data.clear()
        self._counts.clear()

    def update(self, key: str, value: float, n: int = 1) -> None:
        self._data[key] += value * n
        self._counts[key] += n

    def avg(self, key: str) -> float:
        count = self._counts.get(key, 0)
        if count == 0:
            return 0.0
        return self._data[key] / count

    def result(self) -> dict[str, float]:
        keys = self._keys if self._keys else self._data.keys()
        return {key: self.avg(key) for key in keys}
