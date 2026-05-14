# Delphi config examples

These files are starter configs you can copy and modify per project.

## Available examples

- `MLP_example.json` — multilayer perceptron architecture starter
- `MLP_component_example.json` — minimal component-factory MLP starter (class/function paths in JSON)
- `CNN_example.json` — convolutional neural network architecture starter
- `transformer_example.json` — transformer architecture starter

## How to use

1. Copy one file as your project config.
2. Edit `data.path` to your dataset location (absolute paths are supported and recommended if data is outside Delphi).
3. Edit `model` keys to match your model constructor kwargs.
4. Update `extra.model_type` and metadata so Delphi can filter and label runs in the GUI.
5. Set `extra.project_folder` (recommended) to route GUI actions and local project wiring from `models/<ProjectName>`.
6. Optionally set `extra.project_name` or explicit `extra.train_entrypoint` / `extra.infer_entrypoint` overrides.
7. Optionally set `extra.trainer_factory_path` only when factory is outside the project folder.
8. If you want class/function-path wiring, set `extra.trainer_factory_path = core/component_factory.py` and define `components` in JSON.
9. Configure `inference.every_epochs` (for example `5`) if you want epoch-end inference updates in the GUI Inference tab.
10. Run through `Delphi.py` and select your config file.

## Project folder structure

Recommended layout for each project:

- `models/<ProjectName>/train.py`
- `models/<ProjectName>/infer.py` (or `inference.py` with an `infer.py` wrapper)
- project-specific factory module referenced by `extra.trainer_factory_path`

This lets one Delphi installation manage multiple projects by switching config files.

All examples include `_comment` fields that indicate exactly where to modify values and for what purpose.
