from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch

from core.component_loader import clean_config_dict, load_symbol, resolve_mapping
from core.trainer import Trainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def _load_components(config: dict[str, Any]) -> dict[str, Any]:
    components = config.get("components")
    if not isinstance(components, dict):
        raise ValueError("Config must include a 'components' object for component_factory")
    return components


def _build_data(config: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    builder_path = str(components.get("data_builder", "")).strip()
    if not builder_path:
        raise ValueError("components.data_builder is required")

    builder = load_symbol(builder_path)
    result = builder(config)
    if not isinstance(result, dict):
        raise ValueError("data_builder(config) must return a dict")

    train_loader = result.get("train_loader")
    valid_loader = result.get("valid_loader")
    if train_loader is None or valid_loader is None:
        raise ValueError("data_builder result must include train_loader and valid_loader")
    return result


def _instantiate_model(config: dict[str, Any], components: dict[str, Any], model_overrides: dict[str, Any]) -> torch.nn.Module:
    model_path = str(components.get("model_class", "")).strip()
    if not model_path:
        raise ValueError("components.model_class is required")

    model_cls = load_symbol(model_path)
    model_cfg = clean_config_dict(config.get("model"))
    model_cfg.update(model_overrides)

    extra_model_kwargs = resolve_mapping(config, components.get("model_init"))
    model_cfg.update(extra_model_kwargs)

    model = model_cls(**model_cfg)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_class must instantiate a torch.nn.Module")
    return model


def _instantiate_criterion(config: dict[str, Any], components: dict[str, Any]) -> Any:
    criterion_path = str(components.get("criterion_class", "")).strip()
    if not criterion_path:
        raise ValueError("components.criterion_class is required")

    criterion_cls = load_symbol(criterion_path)
    criterion_kwargs = resolve_mapping(config, components.get("criterion_init"))
    if not criterion_kwargs:
        criterion_kwargs = clean_config_dict(config.get("loss"))
    return criterion_cls(**criterion_kwargs)


def _instantiate_optimizer(config: dict[str, Any], components: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_path = str(components.get("optimizer_class", "torch.optim.AdamW")).strip()
    optimizer_cls = load_symbol(optimizer_path)

    optimizer_kwargs = clean_config_dict(config.get("optimizer"))
    optimizer_kwargs.update(resolve_mapping(config, components.get("optimizer_init")))

    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_class must instantiate a torch.optim.Optimizer")
    return optimizer


def _load_metric_fns(components: dict[str, Any]) -> dict[str, Any]:
    metric_cfg = components.get("metric_fns")
    if not isinstance(metric_cfg, dict):
        return {}

    resolved: dict[str, Any] = {}
    for name, path in metric_cfg.items():
        if isinstance(path, str) and path.strip():
            resolved[name] = load_symbol(path)
    return resolved


def _load_optional_hook(components: dict[str, Any], key: str) -> Any:
    path = components.get(key)
    if not isinstance(path, str) or not path.strip():
        return None
    return load_symbol(path)


def _build_trainer_instance(
    *,
    config: dict[str, Any],
    components: dict[str, Any],
    model: torch.nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader: Any,
    valid_loader: Any,
) -> Any:
    trainer_path = str(components.get("trainer_class", "core.trainer.Trainer")).strip()
    trainer_cls = load_symbol(trainer_path)

    trainer_kwargs = resolve_mapping(config, components.get("trainer_init"))

    metric_fns = _load_metric_fns(components)
    if metric_fns:
        trainer_kwargs.setdefault("metric_fns", metric_fns)

    batch_unpack_fn = _load_optional_hook(components, "batch_unpack_fn")
    forward_fn = _load_optional_hook(components, "forward_fn")
    criterion_fn = _load_optional_hook(components, "criterion_fn")
    extra_loss_fn = _load_optional_hook(components, "extra_loss_fn")

    if batch_unpack_fn is not None:
        trainer_kwargs.setdefault("batch_unpack_fn", batch_unpack_fn)
    if forward_fn is not None:
        trainer_kwargs.setdefault("forward_fn", forward_fn)
    if criterion_fn is not None:
        trainer_kwargs.setdefault("criterion_fn", criterion_fn)
    if extra_loss_fn is not None:
        trainer_kwargs.setdefault("extra_loss_fn", extra_loss_fn)

    if trainer_cls is Trainer:
        trainer_kwargs.setdefault("grad_clip", float(config.get("trainer", {}).get("grad_clip", 0.0)))

    trainer = trainer_cls(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        **trainer_kwargs,
    )
    return trainer


def build_trainer(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    _set_seed(int(config.get("seed", 42)))
    components = _load_components(config)

    device = _resolve_device(str(config.get("trainer", {}).get("device", "auto")))
    data_result = _build_data(config, components)

    model = _instantiate_model(config, components, data_result.get("model_overrides", {})).to(device)
    criterion = _instantiate_criterion(config, components)
    optimizer = _instantiate_optimizer(config, components, model)

    trainer = _build_trainer_instance(
        config=config,
        components=components,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=data_result["train_loader"],
        valid_loader=data_result["valid_loader"],
    )

    if isinstance(data_result.get("config_updates"), dict):
        config.update(data_result["config_updates"])

    return trainer, config


def run_inference(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    input_text: str,
    device: torch.device,
    args: Any,
) -> Any:
    components = _load_components(config)
    inference_path = str(components.get("inference_fn", "")).strip()
    if not inference_path:
        raise ValueError("components.inference_fn is required for component_factory.run_inference")

    inference_fn = load_symbol(inference_path)
    return inference_fn(
        config=config,
        checkpoint_path=checkpoint_path,
        input_text=input_text,
        device=device,
        args=args,
    )
