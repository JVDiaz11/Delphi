from __future__ import annotations

from pathlib import Path

from gui.training_GUI import GUIConfig, main


def build_config() -> GUIConfig:
    project_root = Path(__file__).resolve().parent
    default_config = project_root / "configs" / "default.json"

    return GUIConfig(
        title="Delphi",
        train_script="models/Rouge/train.py",
        infer_script="models/Rouge/infer.py",
        default_config=str(default_config),
        runs_dir="runs",
    )


if __name__ == "__main__":
    main(config=build_config())
