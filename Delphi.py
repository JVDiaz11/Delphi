from __future__ import annotations

from pathlib import Path

from gui.training_GUI import GUIConfig, main


def build_config() -> GUIConfig:
    project_root = Path(__file__).resolve().parent
    default_config = project_root / "configs" / "default.json"

    return GUIConfig(
        title="Delphi",
        default_config=str(default_config),
        runs_dir="runs",
    )


if __name__ == "__main__":
    main(config=build_config())
