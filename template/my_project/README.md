# my_project — Delphi starter template

This directory is the scaffold for a new project built on **Delphi**.
Follow the four steps below to go from template to a working training pipeline.

---

## Step 1 — Define your model (`model.py`)

Subclass `base.BaseModel` (which extends `torch.nn.Module`) and implement `forward()`:

```python
class MyModel(BaseModel):
    def __init__(self, *, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.net = ...

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
```

Wire the constructor kwargs to `config["model"]` in `trainer_factory.py`.

---

## Step 2 — Define your data loader (`data_loader.py`)

1. Create a `torch.utils.data.Dataset` that loads your raw data.
2. Subclass `base.BaseDataLoader` — it handles the train/validation split automatically via `split_validation()`.

```python
class MyDataLoader(BaseDataLoader):
    def __init__(self, data_path, batch_size, ...):
        dataset = MyDataset(data_path)
        super().__init__(dataset=dataset, batch_size=batch_size, ...)
```

---

## Step 3 — Define metrics (`metrics.py`)

Each metric is a plain callable `fn(logits, targets) -> float`.  Put them in the `metric_fns` dict:

```python
metric_fns = {
    "accuracy": accuracy,
    "f1": my_f1,
}
```

The keys become metric names in the JSONL history, the GUI plots, and the status panel — no hardcoding required.

---

## Step 4 — Wire everything and run

`trainer_factory.py` already assembles model → data → loss → metrics → `Trainer`.
Edit it to forward the right config keys, then run training:

```bash
# From the project root:
python -m my_project.trainer_factory configs/config.json

# Or via the GUI:
python Delphi.py
```

For the GUI, launch it with a custom `GUIConfig`:

```python
from gui.training_GUI import TrainingGUI, GUIConfig
import tkinter as tk

cfg = GUIConfig(title="MyProject", train_script="train.py")
root = tk.Tk()
TrainingGUI(root, config=cfg)
root.mainloop()
```

---

## Plug-in points summary

| File | What to change |
|------|---------------|
| `model.py` | Your architecture |
| `data_loader.py` | Your dataset + loader |
| `loss.py` | Your primary loss module |
| `metrics.py` | Your evaluation metrics |
| `trainer_factory.py` | Wire config keys to constructors |
| `config.json` | Hyperparameters and paths |

Project-specific config keys that don't belong to the generic schema go under `"extra"` in `config.json` and are passed through to your factory unchanged.

---

## JSON conventions (Delphi)

When creating new config files, follow these rules:

- Keep top-level keys: `name`, `seed`, `data`, `model`, `optimizer`, `trainer`, `extra`.
- Put model constructor kwargs only in `model` (they are forwarded directly as `**kwargs`).
- `trainer_factory.py` includes `normalize_model_config()` so common alternative keys can be aliased (for example `vocab_size -> num_embeddings`).
- Keep custom/domain metadata in `extra` (for registry and filtering).
- Use `trainer.device` as `auto`, `cpu`, or CUDA device strings.
- Use positive integers for `data.batch_size` and `trainer.epochs`.
- Keep `data.train_split` in `(0, 1)`.

### Epoch-end inference updates

- Configure cadence with `inference.every_epochs` in config.
- `0` disables automatic epoch inference; positive values run every N epochs.
- `trainer_factory.py` provides `build_epoch_inference_fn()` as a generic default and you can customize it per project.
- The GUI Inference tab renders both text and optional `series` payloads in real time.

See [configs/inference_payload_schema.md](../../configs/inference_payload_schema.md) for recommended payload fields and examples.

### Dataset path convention

- `data.path` is project-specific and can point anywhere on disk.
- Prefer absolute paths when datasets are not stored inside the Delphi repository.
- Keep dataset location in each project config (do not hardcode shared local paths in code).

### Architecture map clarity

- You can provide `model.graph` in config to control Visualizer architecture rendering.
- Add explicit `connections[].edges` when you want exact node-to-node lines.
- Without `graph`, Delphi falls back to a default sequential stage view.

### Output folders

- `trainer.save_dir` controls where runs are written.
- If `save_dir` is relative (for example `"runs"`), it is resolved under Delphi project root.
- Each run is saved as `<save_dir>/<name>/<YYYYMMDD_HHMMSS>` and includes checkpoints + history.

Expected layout:

- [runs](../../runs)
- [runs/models_index.json](../../runs/models_index.json)
- [runs/<model_name>/<timestamp>/checkpoint_best.pt](../../runs)
- [runs/<model_name>/<timestamp>/checkpoint_last.pt](../../runs)
- [runs/<model_name>/<timestamp>/metrics_history.jsonl](../../runs)

For a fully commented schema, start from [configs/default.json](../../configs/default.json).

You can also copy and adapt architecture-specific examples from [configs/examples](../../configs/examples).
