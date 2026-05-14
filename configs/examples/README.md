# Delphi config examples

These files are starter configs you can copy and modify per project.

## Available examples

- `MLP_example.json` — multilayer perceptron architecture starter
- `CNN_example.json` — convolutional neural network architecture starter
- `transformer_example.json` — transformer architecture starter

## How to use

1. Copy one file as your project config.
2. Edit `data.path` to your dataset location (absolute paths are supported and recommended if data is outside Delphi).
3. Edit `model` keys to match your model constructor kwargs.
4. Update `extra.model_type` and metadata so Delphi can filter and label runs in the GUI.
5. Configure `inference.every_epochs` (for example `5`) if you want epoch-end inference updates in the GUI Inference tab.
6. Run through `Delphi.py` and select your config file.

All examples include `_comment` fields that indicate exactly where to modify values and for what purpose.
