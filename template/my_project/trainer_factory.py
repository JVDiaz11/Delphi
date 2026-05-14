"""
template/my_project/trainer_factory.py
----------------------------------------
TODO: Wire your model, data loader, loss, and metrics into
``core.Trainer`` and run training.

This is the entry point for the training script.  Call it directly or
have your ``train.py`` import ``build_trainer`` and invoke ``trainer.fit()``.

Notes
~~~~~
* ``data.path`` can be an absolute path outside Delphi.
* ``Trainer`` supports custom hooks (``batch_unpack_fn``, ``forward_fn``,
  ``criterion_fn``) for non-standard batch/model signatures such as
  transformer-style dict inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from datetime import datetime

import torch

from core.trainer import Trainer
from core.registry import make_entry, register_model
from utils import prepare_device

from .model import MyModel
from .data_loader import MyDataLoader
from .loss import MyLoss
from .metrics import metric_fns


def normalize_model_config(model_cfg: dict) -> dict:
    """Normalize common alternative model parameter names.

    Extend this mapping per project so users can keep familiar JSON keys.
    """
    aliases = {
        "vocab_size": "num_embeddings",
        "n_embd": "d_model",
        "num_heads": "n_heads",
    }

    resolved = dict(model_cfg)
    for source_key, target_key in aliases.items():
        if source_key in resolved and target_key not in resolved:
            resolved[target_key] = resolved.pop(source_key)
    return resolved


def _move_to_device(item: Any, device: torch.device) -> Any:
    if torch.is_tensor(item):
        return item.to(device)
    if isinstance(item, dict):
        return {key: _move_to_device(value, device) for key, value in item.items()}
    if isinstance(item, tuple):
        return tuple(_move_to_device(value, device) for value in item)
    if isinstance(item, list):
        return [_move_to_device(value, device) for value in item]
    return item


def _split_batch(batch: Any, device: torch.device) -> tuple[Any, Any]:
    batch = _move_to_device(batch, device)
    if isinstance(batch, dict):
        inputs = batch.get("inputs") or batch.get("input") or batch.get("x") or batch.get("features")
        targets = batch.get("targets") or batch.get("target") or batch.get("y") or batch.get("labels")
        if inputs is None:
            inputs = batch
        return inputs, targets
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None
    return batch, None


def _run_model(model: torch.nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def build_epoch_inference_fn(
    *,
    valid_loader: Any,
    max_points: int,
) -> Callable[[torch.nn.Module, int, torch.device], dict[str, Any] | None]:
    """Build a generic epoch-end inference callback.

    Projects should replace or extend this function for domain-specific output.
    """

    def epoch_inference_fn(model: torch.nn.Module, epoch: int, device: torch.device) -> dict[str, Any] | None:
        first_batch = next(iter(valid_loader), None)
        if first_batch is None:
            return {
                "text": "No validation batch available for epoch inference.",
                "status": "skipped",
            }

        inputs, targets = _split_batch(first_batch, device)
        model.eval()
        with torch.no_grad():
            logits = _run_model(model, inputs)

        payload: dict[str, Any] = {
            "text": "Epoch inference completed.",
            "result": f"logits_shape={list(logits.shape)}",
        }

        pred_series: list[float] | None = None
        if torch.is_tensor(logits):
            pred = logits.detach().flatten()
            pred_series = [float(v) for v in pred[:max_points].cpu().tolist()]

        target_series: list[float] | None = None
        if torch.is_tensor(targets):
            tgt = targets.detach().flatten()
            target_series = [float(v) for v in tgt[:max_points].cpu().tolist()]

        series: dict[str, list[float]] = {}
        if target_series:
            series["target"] = target_series
        if pred_series:
            series["prediction"] = pred_series
        if series:
            payload["series"] = series

        return payload

    return epoch_inference_fn


def build_trainer(config: dict) -> tuple[Trainer, dict]:
    """Instantiate and return a ready-to-train ``Trainer``.

    Parameters
    ----------
    config:
        Parsed config dict (see ``configs/config.json``).

    Returns
    -------
    trainer, config
        The trainer and the resolved config (useful for passing to
        ``trainer.fit()`` and the registry).
    """
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    # Device
    device_str = config.get("trainer", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Data
    data_cfg = config.get("data", {})
    loader = MyDataLoader(
        data_path=data_cfg.get("path", "data/"),
        batch_size=int(data_cfg.get("batch_size", 64)),
        validation_split=float(data_cfg.get("train_split", 0.1)),
        num_workers=int(data_cfg.get("num_workers", 4)),
    )
    valid_loader = loader.split_validation()
    if valid_loader is None:
        raise ValueError("validation_split produced no validation set")

    # Model — TODO: forward the right kwargs from config["model"]
    model_cfg_raw = config.get("model", {})
    model_cfg = normalize_model_config({k: v for k, v in model_cfg_raw.items() if not k.startswith("_")})
    model = MyModel(**model_cfg)
    model = model.to(device)

    # Optimiser
    opt_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
    )

    # Trainer
    trainer_cfg = config.get("trainer", {})
    inference_cfg = config.get("inference", {})
    inference_every_epochs = int(inference_cfg.get("every_epochs", 0))
    inference_max_points = int(inference_cfg.get("max_points", 128))

    trainer = Trainer(
        model=model,
        criterion=MyLoss(),
        optimizer=optimizer,
        device=device,
        train_loader=loader,
        valid_loader=valid_loader,
        metric_fns=metric_fns,
        grad_clip=float(trainer_cfg.get("grad_clip", 1.0)),
        inference_every_epochs=inference_every_epochs,
        epoch_inference_fn=build_epoch_inference_fn(
            valid_loader=valid_loader,
            max_points=inference_max_points,
        ) if inference_every_epochs > 0 else None,
        # TODO: pass extra_loss_fn here if you have an auxiliary loss
    )

    return trainer, config


def train(config_path: str | Path) -> None:
    """Load config and run the full training loop."""
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    trainer, config = build_trainer(config)
    trainer_cfg = config.get("trainer", {})
    project_root = Path(__file__).resolve().parents[2]
    save_dir = Path(str(trainer_cfg.get("save_dir", "runs")))
    if not save_dir.is_absolute():
        save_dir = project_root / save_dir
    model_name = str(config.get("name", "experiment"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / model_name / timestamp

    state = trainer.fit(
        epochs=int(trainer_cfg.get("epochs", 50)),
        run_dir=run_dir,
        config=config,
        save_best_only=bool(trainer_cfg.get("save_best_only", True)),
        log_every_epochs=int(trainer_cfg.get("log_every_epochs", 5)),
        log_every_steps=int(trainer_cfg.get("log_every_steps", 100)),
    )

    # Register run
    register_model(make_entry(
        name=f"{config.get('name', 'experiment')}/{run_dir.name}",
        run_dir=run_dir,
        config_path=Path(config_path).resolve(),
        model_type=config.get("extra", {}).get("model_type", "default"),
        epochs=state.epoch,
        best_val_loss=state.best_val_loss if state.best_val_loss < float("inf") else None,
        architecture=trainer.model.architecture_summary() if hasattr(trainer.model, "architecture_summary") else {},
        metadata=config.get("extra", {}),
    ))


if __name__ == "__main__":
    import sys
    train(sys.argv[1] if len(sys.argv) > 1 else "configs/config.json")
