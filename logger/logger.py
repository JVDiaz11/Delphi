from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path


def _default_config(log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(log_dir / "train.log"),
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"],
        },
    }


def setup_logging(log_dir: str | Path, config_path: str | Path | None = None) -> None:
    log_dir_path = Path(log_dir)
    if config_path is not None:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            handlers = config.get("handlers", {})
            for handler_cfg in handlers.values():
                if isinstance(handler_cfg, dict) and handler_cfg.get("filename"):
                    file_name = Path(handler_cfg["filename"]).name
                    handler_cfg["filename"] = str(log_dir_path / file_name)
            logging.config.dictConfig(config)
            return

    logging.config.dictConfig(_default_config(log_dir_path))
