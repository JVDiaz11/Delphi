# Inference Payload Schema (epoch-end telemetry)

Delphi supports project-defined inference updates during training via telemetry events of type `epoch_inference`.

## When it runs

Configured in JSON:

- `inference.every_epochs = 0` → disabled
- `inference.every_epochs = N` → run every `N` epochs

## Event shape

The trainer emits this base shape:

```json
{
  "type": "epoch_inference",
  "epoch": 5,
  "run_dir": "...",
  "...": "project-defined fields"
}
```

Your `epoch_inference_fn` should return a dict containing any of the recommended fields below.

## Recommended fields

- `text` (string): short human-readable update shown in the Inference tab log.
- `result` (string | number): primary outcome summary.
- `confidence` (number in `[0,1]`): optional confidence bar update.
- `series` (object): optional plot data for real-time comparison.

`series` format:

```json
{
  "series": {
    "target": [0.1, 0.2, 0.3],
    "prediction": [0.12, 0.19, 0.29]
  }
}
```

## Example payloads

### 1) Text-only / sequence-style

```json
{
  "text": "Validation sample decoded",
  "result": "pred=1,2,3,5,8 | target=1,2,3,5,13"
}
```

### 2) Signal comparison (plot)

```json
{
  "text": "Signal reconstruction preview",
  "result": "mse=0.0041",
  "series": {
    "original": [0.0, 0.2, 0.4, 0.2, 0.0],
    "reconstructed": [0.01, 0.19, 0.39, 0.21, 0.02]
  }
}
```

### 3) Classification-style

```json
{
  "text": "Epoch inference sample",
  "result": "pred=cat | target=dog",
  "confidence": 0.73
}
```

## Implementation point

Use `build_epoch_inference_fn()` in `template/my_project/trainer_factory.py` as your starting point and customize per project.

## Design guidance

- Keep payloads lightweight (small arrays, concise text) for smooth GUI updates.
- Use `series` only when plotting helps interpretability.
- Put task-specific metadata in config `extra`; keep telemetry focused on inference output.
