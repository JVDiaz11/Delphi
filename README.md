# Delphi

A generic, reusable PyTorch training framework extracted from the Delphi
hash-inference project. Use the Delphi launcher GUI to train and visualize
PyTorch workflows from a central tool, without copying Delphi into each
target project folder.

---

## Package layout

```
base/
    base_model.py          ← domain-free BaseModel (nn.Module subclass)
    base_trainer.py        ← abstract BaseTrainer interface
    base_data_loader.py    ← BaseDataLoader with auto train/val split
core/
    trainer.py             ← generic Trainer with injectable metrics
    registry.py            ← run catalogue (models_index.json)
logger/
    logger.py              ← setup_logging() helper
utils/
    util.py                ← ensure_dir, read/write_json, MetricTracker
gui/
    training_GUI.py        ← dynamic-metric Tkinter training GUI
    visualizer.py          ← run history visualizer
configs/
    default.json           ← annotated generic config skeleton
template/
    my_project/            ← copy-and-fill starter scaffold
```

---

## Quick start

### 1. Activate environment

```bash
conda activate crypto
```

### 2. Launch Delphi GUI

```bash
python Delphi.py
```

### 3. Copy the template (optional)

```bash
cp -r template/my_project my_new_project
```

### 4. Fill in the four plug-in points

| File | What to change |
|------|---------------|
| `model.py` | Subclass `BaseModel`, implement `forward()` |
| `data_loader.py` | Subclass `BaseDataLoader`, supply a `Dataset` |
| `loss.py` | Define your primary `nn.Module` loss |
| `metrics.py` | Define `metric_fns: dict[str, Callable]` |

### 5. Run training

```bash
python -m my_new_project.trainer_factory my_new_project/config.json
```

Or open the GUI:

```bash
python Delphi.py
```

---

## Delphi JSON standard

Delphi expects a JSON config with this top-level structure:

```json
{
    "name": "run_name",
    "seed": 42,
    "data": { ... },
    "model": { ... },
    "optimizer": { ... },
    "trainer": { ... },
    "extra": { ... }
}
```

### Required keys and conventions

- `name`: run label used for run directory naming and registry display.
- `data.batch_size`: integer `> 0`.
- `data.train_split`: float in `(0, 1)` used as validation split by `BaseDataLoader`.
- `data.path`: path to your dataset source.
- `trainer.epochs`: integer `> 0`.
- `trainer.device`: `"auto"`, `"cpu"`, or a CUDA device string such as `"cuda"`.
- `trainer.save_dir`: base folder where run artifacts are written.

### Model section convention

- Every key in `model` is forwarded into your model constructor (`MyModel(**model_cfg)`).
- Keep only constructor kwargs in `model`; place project metadata in `extra`.

### Architecture flexibility

Delphi supports any PyTorch architecture as long as your project wiring (`trainer_factory.py`) provides compatible model/data/loss logic. The generic trainer now supports:

- tuple/list batches (classic `(inputs, targets)`)
- dict batches (`inputs`/`targets`, `x`/`y`, `features`/`labels`, etc.)
- custom batch unpacking via `batch_unpack_fn`
- custom forward logic via `forward_fn`
- custom loss invocation via `criterion_fn`

This allows DNNs, CNNs, Transformers, and other custom architectures to share the same Delphi training/GUI pipeline.

### GUI/CLI override convention

The GUI launches train/infer scripts with:

- required: `-c <config.json>`
- optional train overrides: `--data`, `--epochs`, `--batch-size`, `--device`, `--resume-checkpoint`
- optional infer overrides: `--input`, `--checkpoint`

Your training/inference entry scripts should accept these flags if you want full GUI compatibility.

### Real-time epoch inference (flexible)

Delphi can run project-defined inference at epoch end and stream it live to the Inference tab.

- Config key: `inference.every_epochs`
    - `0` = disabled
    - `N > 0` = run epoch inference every N epochs (for example `5`)
- Optional key: `inference.max_points` controls how many points are sent for series preview.

Telemetry payload is generic (`type = "epoch_inference"`) so each project can define its own output:

- text-only inference (`text`, `result`)
- numeric confidence (`confidence`)
- optional signal/series comparison plot (`series: {name: [values...]}`)

Payload format and copy-ready examples are documented in [configs/inference_payload_schema.md](configs/inference_payload_schema.md).

### Registry and run conventions

- Keep `name` stable and meaningful (`project_variant`, `experiment_07`, etc.).
- Keep model-family tags in `extra.model_type` for filtering in the GUI.
- Put all project/domain-specific fields under `extra` (they are persisted as registry metadata).

### Where trained models are saved

Default behavior is:

- `trainer.save_dir` = `runs`
- model folder = `<save_dir>/<name>`
- training run folder = `<save_dir>/<name>/<YYYYMMDD_HHMMSS>`

So from Delphi root, outputs are stored in:

- [runs](runs)
- [runs/models_index.json](runs/models_index.json) (registry)
- [runs/<model_name>/<timestamp>/checkpoint_best.pt](runs)
- [runs/<model_name>/<timestamp>/checkpoint_last.pt](runs)
- [runs/<model_name>/<timestamp>/metrics_history.jsonl](runs)

This matches your target structure where [runs](runs) holds folders for different trained models.

### Starting point

- Use [configs/default.json](configs/default.json) as the commented schema.
- Use [template/my_project/config.json](template/my_project/config.json) as a concrete example.
- Use architecture-specific examples in [configs/examples](configs/examples):
    - [configs/examples/MLP_example.json](configs/examples/MLP_example.json)
    - [configs/examples/CNN_example.json](configs/examples/CNN_example.json)
    - [configs/examples/transformer_example.json](configs/examples/transformer_example.json)

All examples include `_comment` fields that explicitly mark what to edit and why.

---

## Trainer API

```python
from core import Trainer
from pathlib import Path

trainer = Trainer(
    model=model,
    criterion=my_loss,
    optimizer=optimizer,
    device=device,
    train_loader=train_loader,
    valid_loader=valid_loader,
    metric_fns={               # any callables → float
        "accuracy": accuracy_fn,
        "f1": f1_fn,
    },
    grad_clip=1.0,
    extra_loss_fn=lambda logits, tgt: 0.3 * aux_loss(logits, tgt),
)

state = trainer.fit(
    epochs=50,
    run_dir=Path("runs/exp1"),
    config=cfg_dict,
    save_best_only=True,
)
```

All metric keys flow automatically through the JSON progress stream,
JSONL history, GUI plots, and status panel — no hardcoding required.

---

## GUI

```python
from gui.training_GUI import TrainingGUI, GUIConfig
import tkinter as tk

cfg = GUIConfig(
    title="MyProject Training GUI",
    train_script="train.py",
    default_config="configs/config.json",
    runs_dir="runs",
)
root = tk.Tk()
TrainingGUI(root, config=cfg)
root.mainloop()
```

Metric plots are created automatically from the first `epoch_metrics`
JSON event — keys containing `"loss"` go to the Loss plot, all others
go to the Metrics plot.

### Architecture visualization graph (optional)

For clearer architecture rendering (including explicit node-to-node connections), provide a `graph` object inside `model` config:

```json
{
    "model": {
        "graph": {
            "stages": [
                {"id": "input", "label": "Input", "nodes": 8, "activation_key": "input"},
                {"id": "hidden", "label": "Hidden", "nodes": 8, "activation_key": "hidden"},
                {"id": "output", "label": "Output", "nodes": 4, "activation_key": "output"}
            ],
            "connections": [
                {"from": "input", "to": "hidden", "mode": "paired"},
                {"from": "hidden", "to": "output", "edges": [[0, 0], [1, 0], [2, 1], [3, 2]]}
            ]
        }
    }
}
```

Supported connection options:

- `mode: "paired"` (default sampled one-to-one)
- `mode: "attention"`
- `mode: "all_to_all"`
- `mode: "conv1d"` with optional `kernel_size`, `stride`
- `edges: [...]` explicit indexed links for exact neuron/node mapping

Visualizer interaction controls:

- `Node detail` selector (16..256) controls how many nodes are rendered per stage.
- Mouse wheel zooms the architecture map.
- Left-click + drag pans the architecture map.
- Double-click resets zoom/pan.

---

## Registry

```python
from core.registry import make_entry, register_model

register_model(make_entry(
    name="my_experiment/run_001",
    run_dir=Path("runs/my_experiment/run_001"),
    config_path=Path("configs/config.json"),
    model_type="classifier_v1",
    epochs=50,
    best_val_loss=0.032,
    metadata={
        # project-specific fields go here — no schema changes needed
        "token_mode": "hex",
        "custom_flag": True,
    },
))
```

---

## Release

This repository is Delphi `v0.1`: a standalone PyTorch-focused training,
inference, and visualization toolkit with a central GUI launcher.
