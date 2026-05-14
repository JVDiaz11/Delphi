"""
core/trainer.py
---------------
Generic training loop that can be dropped into any PyTorch project.

Key design decisions
~~~~~~~~~~~~~~~~~~~~
* All domain-specific logic is injected at construction time via callables.
* ``metric_fns``  — dict of ``{name: fn(logits, targets) -> float}``
  evaluated on every batch; the keys drive both the JSONL history and the
  GUI progress stream — no hardcoded metric names anywhere.
* ``extra_loss_fn`` — optional callable that receives ``(logits, targets)``
  and returns a scalar ``Tensor`` added on top of the main criterion.  This
  replaces the old ``closeness_weight`` / ``pow_weight`` parameters.
* Checkpoint saving, JSONL history and ``TrainerState`` are infrastructure
  concerns and stay in this class unchanged.

Usage (minimal example)
~~~~~~~~~~~~~~~~~~~~~~~
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(model.parameters(), lr=1e-3),
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        metric_fns={"accuracy": accuracy_fn},
    )
    trainer.fit(epochs=50, run_dir=Path("runs/exp1"), config=cfg_dict)
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from base.base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class TrainerState:
    epoch: int = 0
    best_val_loss: float = float("inf")


# ---------------------------------------------------------------------------
# Generic Trainer
# ---------------------------------------------------------------------------

class Trainer(BaseTrainer):
    """Domain-agnostic training loop.

    Parameters
    ----------
    model:
        Any ``nn.Module``.
    criterion:
        Primary loss module called as ``criterion(logits, targets)``.
    optimizer:
        Any ``torch.optim.Optimizer``.
    device:
        Target device.
    train_loader / valid_loader:
        Standard ``DataLoader`` instances.
    metric_fns:
        Optional dict mapping metric name → callable
        ``(logits: Tensor, targets: Tensor) -> float``.
        The measured values are streamed via JSON progress events and stored
        in the JSONL history.  If ``None``, only ``loss`` is tracked.
    grad_clip:
        Gradient norm clipping threshold.  ``0.0`` disables clipping.
    extra_loss_fn:
        Optional callable ``(logits: Tensor, targets: Tensor) -> Tensor``
        returning an *unweighted* scalar tensor added to the criterion loss.
        Use a closure to bake in your weighting factor, e.g.::

            extra_loss_fn=lambda logits, tgt: 0.3 * my_aux_loss(logits, tgt)
    batch_unpack_fn:
        Optional callable ``(batch, device) -> (inputs, targets)``.
        Use this for datasets that do not return ``(inputs, targets)`` tuples.
    forward_fn:
        Optional callable ``(model, inputs, batch) -> logits``.
        Useful for complex models that consume structured inputs.
    criterion_fn:
        Optional callable ``(criterion, logits, targets, batch) -> Tensor``
        that computes the primary loss.
    inference_every_epochs:
        Run epoch-end inference every N epochs. ``0`` disables it.
    epoch_inference_fn:
        Optional callable ``(model, epoch, device) -> dict | None``.
        Returned dict is emitted as ``{"type": "epoch_inference", ...}``
        telemetry for the GUI.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        metric_fns: dict[str, Callable[[Tensor, Tensor], float]] | None = None,
        grad_clip: float = 0.0,
        extra_loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        batch_unpack_fn: Callable[[Any, torch.device], tuple[Any, Any]] | None = None,
        forward_fn: Callable[[nn.Module, Any, Any], Tensor] | None = None,
        criterion_fn: Callable[[nn.Module, Tensor, Any, Any], Tensor] | None = None,
        inference_every_epochs: int = 0,
        epoch_inference_fn: Callable[[nn.Module, int, torch.device], dict[str, Any] | None] | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metric_fns: dict[str, Callable[[Tensor, Tensor], float]] = metric_fns or {}
        self.grad_clip = float(grad_clip)
        self.extra_loss_fn = extra_loss_fn
        self.batch_unpack_fn = batch_unpack_fn
        self.forward_fn = forward_fn
        self.criterion_fn = criterion_fn
        self.inference_every_epochs = max(0, int(inference_every_epochs))
        self.epoch_inference_fn = epoch_inference_fn
        self.state = TrainerState()
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_to_device(self, item: Any) -> Any:
        if torch.is_tensor(item):
            return item.to(self.device)
        if isinstance(item, dict):
            return {key: self._move_to_device(value) for key, value in item.items()}
        if isinstance(item, tuple):
            return tuple(self._move_to_device(value) for value in item)
        if isinstance(item, list):
            return [self._move_to_device(value) for value in item]
        return item

    def _default_unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        batch = self._move_to_device(batch)

        if isinstance(batch, dict):
            inputs = (
                batch.get("inputs")
                or batch.get("input")
                or batch.get("x")
                or batch.get("features")
            )
            targets = (
                batch.get("targets")
                or batch.get("target")
                or batch.get("y")
                or batch.get("labels")
            )
            if inputs is None:
                inputs = batch
            return inputs, targets

        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            if len(batch) == 1:
                return batch[0], None

        if torch.is_tensor(batch):
            return batch, None

        return batch, None

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        if self.batch_unpack_fn is not None:
            return self.batch_unpack_fn(batch, self.device)
        return self._default_unpack_batch(batch)

    def _forward(self, inputs: Any, batch: Any) -> Tensor:
        if self.forward_fn is not None:
            return self.forward_fn(self.model, inputs, batch)

        if isinstance(inputs, dict):
            return self.model(**inputs)
        if isinstance(inputs, (tuple, list)):
            return self.model(*inputs)
        return self.model(inputs)

    @staticmethod
    def _supports_signature(fn: Callable[..., Any], *args: Any) -> bool:
        try:
            signature = inspect.signature(fn)
            signature.bind(*args)
            return True
        except (TypeError, ValueError):
            return False

    def _call_first_supported(self, fn: Callable[..., Any], candidates: list[tuple[Any, ...]]) -> Any:
        for args in candidates:
            if self._supports_signature(fn, *args):
                return fn(*args)

        for args in candidates:
            try:
                return fn(*args)
            except TypeError:
                continue

        raise TypeError(f"No compatible call signature for {getattr(fn, '__name__', 'callable')}")

    def _compute_loss(
        self, logits: Tensor, targets: Any, batch: Any
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(total_loss, criterion_loss, extra_loss)``."""
        if self.criterion_fn is not None:
            criterion_loss = self.criterion_fn(self.criterion, logits, targets, batch)
        elif targets is None:
            criterion_loss = self._call_first_supported(
                self.criterion,
                [(logits,), (logits, targets)],
            )
        else:
            criterion_loss = self._call_first_supported(
                self.criterion,
                [(logits, targets), (logits,)],
            )

        if self.extra_loss_fn is not None:
            extra = self._call_first_supported(
                self.extra_loss_fn,
                [(logits, targets, batch), (logits, targets), (logits,)],
            )
        else:
            extra = torch.zeros(1, device=self.device)
        return criterion_loss + extra, criterion_loss, extra

    def _eval_metrics(
        self, logits: Tensor, targets: Any, batch: Any
    ) -> dict[str, float]:
        with torch.no_grad():
            result: dict[str, float] = {}
            for name, fn in self.metric_fns.items():
                value = self._call_first_supported(
                    fn,
                    [(logits, targets, batch), (logits, targets), (logits,)],
                )
                result[name] = float(value)
            return result

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_criterion_loss = 0.0
        total_extra_loss = 0.0
        metric_totals: dict[str, float] = {name: 0.0 for name in self.metric_fns}

        total_batches = max(1, len(self.train_loader))
        log_every_steps: int = max(1, int(getattr(self, "_log_every_steps", 100)))
        run_dir: str = getattr(self, "_run_dir", "")

        for batch_index, batch in enumerate(self.train_loader):
            inputs, targets = self._unpack_batch(batch)
            step = batch_index + 1
            log_this_step = (
                step == 1
                or step % log_every_steps == 0
                or step == total_batches
            )

            self.optimizer.zero_grad(set_to_none=True)

            # Optional activation capture for models that support it
            captured: dict[str, Tensor] | None = None
            if log_this_step and hasattr(self.model, "forward_with_profiles") and torch.is_tensor(inputs):
                logits, captured = self.model.forward_with_profiles(
                    inputs, retain_activation_grads=True
                )
            else:
                logits = self._forward(inputs, batch)

            loss, criterion_loss, extra_loss = self._compute_loss(logits, targets, batch)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            total_criterion_loss += criterion_loss.item()
            total_extra_loss += extra_loss.item()
            batch_metrics = self._eval_metrics(logits.detach(), targets, batch)
            for name, value in batch_metrics.items():
                metric_totals[name] += value

            if log_this_step:
                activations = None
                gradients = None
                if captured is not None and hasattr(self.model, "activation_profile_from_tensors"):
                    activations = self.model.activation_profile_from_tensors(captured)
                    if hasattr(self.model, "gradient_profile_from_tensors"):
                        gradients = self.model.gradient_profile_from_tensors(captured)
                elif hasattr(self.model, "activation_profile") and torch.is_tensor(inputs):
                    with torch.no_grad():
                        activations = self.model.activation_profile(inputs.detach())

                progress_payload: dict = {
                    "type": "epoch_progress",
                    "phase": "train",
                    "epoch": int(self.state.epoch),
                    "batch": step,
                    "batches": total_batches,
                    "progress": step / total_batches,
                    "run_dir": run_dir,
                    "loss": total_loss / step,
                    "criterion_loss": total_criterion_loss / step,
                }
                for name in self.metric_fns:
                    progress_payload[name] = metric_totals[name] / step
                if activations:
                    progress_payload["activations"] = activations
                if gradients:
                    progress_payload["gradients"] = gradients
                print(json.dumps(progress_payload, ensure_ascii=False), flush=True)

        result: dict = {
            "loss": total_loss / total_batches,
            "criterion_loss": total_criterion_loss / total_batches,
            "extra_loss": total_extra_loss / total_batches,
        }
        for name in self.metric_fns:
            result[name] = metric_totals[name] / total_batches

        first_batch = next(iter(self.train_loader), None)
        if first_batch is not None and hasattr(self.model, "activation_profile"):
            first_inputs, _ = self._unpack_batch(first_batch)
            if not torch.is_tensor(first_inputs):
                return result
            with torch.no_grad():
                result["activations"] = self.model.activation_profile(first_inputs)

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_criterion_loss = 0.0
        total_extra_loss = 0.0
        metric_totals: dict[str, float] = {name: 0.0 for name in self.metric_fns}

        first_inputs: Tensor | None = None
        for batch_index, batch in enumerate(self.valid_loader):
            inputs, targets = self._unpack_batch(batch)
            if batch_index == 0 and torch.is_tensor(inputs):
                first_inputs = inputs.detach().clone()

            logits = self._forward(inputs, batch)
            loss, criterion_loss, extra_loss = self._compute_loss(logits, targets, batch)
            total_loss += loss.item()
            total_criterion_loss += criterion_loss.item()
            total_extra_loss += extra_loss.item()
            for name, value in self._eval_metrics(logits, targets, batch).items():
                metric_totals[name] += value

        batches = max(1, len(self.valid_loader))
        result: dict = {
            "loss": total_loss / batches,
            "criterion_loss": total_criterion_loss / batches,
            "extra_loss": total_extra_loss / batches,
        }
        for name in self.metric_fns:
            result[name] = metric_totals[name] / batches

        if first_inputs is not None and hasattr(self.model, "activation_profile"):
            result["activations"] = self.model.activation_profile(first_inputs)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, path: str | Path, *, epoch: int, config: dict, valid_metrics: dict
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "valid_metrics": valid_metrics,
            "config": config,
        }
        torch.save(checkpoint, Path(path))

    def _append_history(self, path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        *,
        epochs: int,
        run_dir: Path,
        config: dict,
        save_best_only: bool = True,
        log_every_epochs: int = 5,
        log_every_steps: int = 100,
    ) -> TrainerState:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        history_path = run_dir / "metrics_history.jsonl"
        log_every_epochs = max(1, int(log_every_epochs))
        self._log_every_steps = max(1, int(log_every_steps))
        self._run_dir = str(run_dir)

        print(f"Training started (logging every {log_every_epochs} epochs)", flush=True)
        training_start = time.perf_counter()
        start_epoch = int(self.state.epoch)

        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            epoch_start = time.perf_counter()
            self.state.epoch = epoch
            train_metrics = self.train_epoch()
            valid_metrics = self.evaluate()
            epoch_seconds = time.perf_counter() - epoch_start
            elapsed_seconds = time.perf_counter() - training_start

            epoch_payload: dict = {
                "type": "epoch_metrics",
                "epoch": epoch,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "run_dir": str(run_dir),
                "train": train_metrics,
                "valid": valid_metrics,
            }
            self.history.append(epoch_payload)
            self._append_history(history_path, epoch_payload)
            print(json.dumps(epoch_payload, ensure_ascii=False), flush=True)

            if (
                self.epoch_inference_fn is not None
                and self.inference_every_epochs > 0
                and epoch % self.inference_every_epochs == 0
            ):
                try:
                    inference_payload = self.epoch_inference_fn(self.model, epoch, self.device)
                except Exception as exc:
                    inference_payload = {
                        "status": "error",
                        "text": f"epoch_inference_fn failed: {exc}",
                    }

                if inference_payload is not None:
                    telemetry = {
                        "type": "epoch_inference",
                        "epoch": epoch,
                        "run_dir": str(run_dir),
                        **inference_payload,
                    }
                    print(json.dumps(telemetry, ensure_ascii=False), flush=True)

            if epoch % log_every_epochs == 0 or epoch == epochs:
                metric_str = "  ".join(
                    f"train_{k}={train_metrics[k]:.4f}"
                    for k in self.metric_fns
                    if k in train_metrics
                )
                val_metric_str = "  ".join(
                    f"val_{k}={valid_metrics[k]:.4f}"
                    for k in self.metric_fns
                    if k in valid_metrics
                )
                print(
                    f"Epoch {epoch:03d}: "
                    f"train_loss={train_metrics['loss']:.5f}  "
                    f"val_loss={valid_metrics['loss']:.5f}  "
                    f"{metric_str}  {val_metric_str}",
                    flush=True,
                )

            if not save_best_only:
                self.save_checkpoint(
                    run_dir / "checkpoint_last.pt",
                    epoch=epoch, config=config, valid_metrics=valid_metrics,
                )
            if valid_metrics["loss"] < self.state.best_val_loss:
                self.state.best_val_loss = valid_metrics["loss"]
                self.save_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    epoch=epoch, config=config, valid_metrics=valid_metrics,
                )

        if save_best_only and not (run_dir / "checkpoint_last.pt").exists():
            self.save_checkpoint(
                run_dir / "checkpoint_last.pt",
                epoch=self.state.epoch, config=config, valid_metrics=valid_metrics,  # type: ignore[possibly-undefined]
            )

        return self.state


__all__ = ["Trainer", "TrainerState"]
