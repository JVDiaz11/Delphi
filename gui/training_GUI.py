"""
gui/training_GUI.py
-------------------
Generic training GUI that works with any project built on Delphi.

Key design changes vs. Delphi's training_GUI.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``GUIConfig`` dataclass – supplies window title, script paths, default
  config path, runs directory, etc.  Pass a customised instance to
  ``TrainingGUI`` to brand the GUI for your project.
* Dynamic metric plots – on the first ``epoch_metrics`` JSON payload the
  GUI reads whatever keys are present in ``train`` / ``valid`` dicts and
  builds two plots automatically:
    - "Loss curves" — all keys whose name contains ``"loss"``
    - "Metrics"     — all other numeric keys
  No hardcoded metric names anywhere.
* Dynamic progress text – the MessageWindow progress line lists all metric
  values from the live JSON stream dynamically.
* Registry integration – uses ``core.registry`` instead of a
  project-specific module.
* Generalised inference tab – public hash / guided-search fields removed;
  a generic ``Input`` text field + ``Extra args`` field are provided
  instead.  Consumer projects can subclass ``TrainingGUI`` and override
  ``_build_inference_tab`` to add domain-specific controls.

Usage
~~~~~
    from gui.training_GUI import TrainingGUI, GUIConfig
    import tkinter as tk

    cfg = GUIConfig(title="MyProject Training GUI", train_script="train.py")
    root = tk.Tk()
    TrainingGUI(root, config=cfg)
    root.mainloop()
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import tkinter as tk

# ---------------------------------------------------------------------------
# Colour / font constants
# ---------------------------------------------------------------------------

BG = "#0d1117"
BG2 = "#161b22"
BG3 = "#21262d"
BORDER = "#30363d"
FG = "#c9d1d9"
FG_DIM = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#f0883e"
PURPLE = "#bc8cff"

FONT_MONO = ("Courier New", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_NORM = ("Segoe UI", 9)
FONT_SM = ("Segoe UI", 8)


# ---------------------------------------------------------------------------
# GUI configuration
# ---------------------------------------------------------------------------

@dataclass
class GUIConfig:
    """Project-specific labels and paths for the training GUI.

    All fields have sensible defaults so Delphi can be used without any
    customisation.
    """
    title: str = "Delphi Training GUI"
    train_script: str = "train.py"
    infer_script: str = "infer.py"
    train_scripts_script: str = "scripts/train.py"
    infer_scripts_script: str = "scripts/infer.py"
    default_config: str = "configs/default.json"
    default_data: str = ""
    runs_dir: str = "runs"
    registry_filename: str = "models_index.json"
    window_size: str = "1400x900"


# ---------------------------------------------------------------------------
# Internal GUI state
# ---------------------------------------------------------------------------

@dataclass
class GUIState:
    training_running: bool = False
    inference_running: bool = False
    last_predicted: str = "—"
    last_confidence: float = 0.0
    last_result: str = "—"


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

class MetricPlot(ttk.Frame):
    """Canvas-based line chart for a set of named series."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        y_label: str,
        *,
        height: int = 220,
    ) -> None:
        super().__init__(parent)
        self.configure(style="TFrame")
        self.title = title
        self.y_label = y_label
        self.height = height
        self.canvas = tk.Canvas(
            self, width=540, height=height,
            bg=BG2, highlightthickness=1, highlightbackground=BORDER,
        )
        self.canvas.pack(fill="both", expand=True)
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._colors = {"train": GREEN, "val": BLUE, "aux": ORANGE}
        self.after(50, self._redraw)

    def set_series(self, series: dict[str, list[tuple[int, float]]]) -> None:
        self._series = {key: value[:] for key, value in series.items()}
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        w = int(self.canvas.winfo_width() or 540)
        h = int(self.canvas.winfo_height() or self.height)
        pad_l, pad_r, pad_t, pad_b = 54, 18, 28, 34
        plot_w = max(1, w - pad_l - pad_r)
        plot_h = max(1, h - pad_t - pad_b)

        self.canvas.create_text(10, 10, anchor="nw", fill=FG, font=FONT_BOLD, text=self.title)
        self.canvas.create_text(10, 26, anchor="nw", fill=FG_DIM, font=FONT_SM, text=self.y_label)
        self.canvas.create_line(pad_l, pad_t, pad_l, pad_t + plot_h, fill=BORDER)
        self.canvas.create_line(pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h, fill=BORDER)

        all_points = [pt for s in self._series.values() for pt in s]
        if not all_points:
            self.canvas.create_text(w // 2, h // 2, fill=FG_DIM, font=FONT_NORM, text="No data yet")
            return

        epochs = [x for x, _ in all_points]
        values = [y for _, y in all_points]
        x_min, x_max = min(epochs), max(epochs)
        y_min, y_max = min(values), max(values)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        def to_xy(epoch: int, value: float) -> tuple[float, float]:
            x = pad_l + ((epoch - x_min) / max(1, x_max - x_min)) * plot_w
            y = pad_t + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h
            return x, y

        for tick in range(5):
            frac = tick / 4 if 4 else 0
            y = pad_t + frac * plot_h
            val = y_max - frac * (y_max - y_min)
            self.canvas.create_line(pad_l - 4, y, pad_l, y, fill=BORDER)
            self.canvas.create_text(6, y, anchor="w", fill=FG_DIM, font=FONT_SM, text=f"{val:.3f}")

        self.canvas.create_text(
            w - 6, h - 12, anchor="e", fill=FG_DIM, font=FONT_SM,
            text=f"epochs {x_min}..{x_max}",
        )

        for name, series in self._series.items():
            if len(series) < 2:
                continue
            coords: list[float] = []
            for epoch, value in series:
                x, y = to_xy(epoch, value)
                coords.extend([x, y])
            color = self._colors.get(name, PURPLE)
            self.canvas.create_line(*coords, fill=color, width=2, smooth=True)
            last_epoch, last_value = series[-1]
            x, y = to_xy(last_epoch, last_value)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
            self.canvas.create_text(x + 6, y, anchor="w", fill=color, font=FONT_SM, text=name)

        self.after(250, self._redraw)


class MessageWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("Training Messages")
        self.configure(bg=BG)
        self.geometry("980x620")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Training / inference messages", font=FONT_BOLD).grid(row=0, column=0, sticky="w")
        self.text = scrolledtext.ScrolledText(
            frame, bg=BG2, fg=FG, insertbackground=FG, font=FONT_MONO, wrap="word",
        )
        self.text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.text.configure(state="disabled")
        self._log_lines: list[str] = []
        self._progress_text = "Progress: waiting for training to start"
        self._refresh_text()

    def _refresh_text(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        for line in self._log_lines:
            self.text.insert(tk.END, line + "\n")
        self.text.insert(tk.END, self._progress_text + "\n")
        self.text.see(tk.END)
        self.text.configure(state="disabled")

    def append(self, line: str) -> None:
        self._log_lines.append(line)
        self._refresh_text()

    def set_progress(self, text: str) -> None:
        self._progress_text = text
        self._refresh_text()

    def clear(self) -> None:
        self._log_lines.clear()
        self._refresh_text()


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class TrainingGUI:
    """Generic training GUI.

    Parameters
    ----------
    root:
        The root ``tk.Tk`` window.
    config:
        Optional ``GUIConfig`` instance; defaults are used if omitted.
    """

    def __init__(self, root: tk.Tk, *, config: GUIConfig | None = None) -> None:
        self.root = root
        self.cfg = config or GUIConfig()

        self.root.title(self.cfg.title)
        self.root.configure(bg=BG)
        self.root.geometry(self.cfg.window_size)

        self.state = GUIState()
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.queue: "queue.Queue[str]" = queue.Queue()

        # Dynamic metric series — populated on first epoch_metrics payload
        # {"train_loss": [(epoch, value), ...], "val_loss": [...], ...}
        self._loss_series: dict[str, list[tuple[int, float]]] = {}
        self._metric_series: dict[str, list[tuple[int, float]]] = {}
        self._plots_initialised = False

        self.visualizer_window = None
        self.message_window = None
        self._last_valid_loss: float | None = None

        self._setup_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_queue)
        self.root.after(300, self._refresh_model_list)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, fieldbackground=BG3)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=BG2)
        style.configure("TLabel", background=BG, foreground=FG, font=FONT_NORM)
        style.configure("Dim.TLabel", background=BG, foreground=FG_DIM, font=FONT_SM)
        style.configure("TButton", background=BG3, foreground=FG, font=FONT_NORM)
        style.configure("Start.TButton", background=GREEN, foreground=BG, font=FONT_BOLD)
        style.configure("Stop.TButton", background=RED, foreground=BG, font=FONT_BOLD)
        style.configure("Viz.TButton", background=BLUE, foreground=BG, font=FONT_BOLD)
        style.configure("TNotebook", background=BG, bordercolor=BORDER)
        style.configure("TNotebook.Tab", background=BG3, foreground=FG_DIM, padding=[10, 4], font=FONT_NORM)
        style.map("TNotebook.Tab", background=[("selected", BG2)], foreground=[("selected", FG)])
        style.configure("TEntry", fieldbackground=BG3, foreground=FG, insertcolor=FG, font=FONT_MONO)
        style.configure("TSpinbox", fieldbackground=BG3, foreground=FG, font=FONT_NORM)
        style.configure("TCombobox", fieldbackground=BG3, foreground=FG, font=FONT_NORM)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG3), ("!disabled", BG3)],
            foreground=[("readonly", FG), ("!disabled", FG)],
            selectbackground=[("readonly", BG3)],
            selectforeground=[("readonly", FG)],
            background=[("readonly", BG3), ("active", BG3)],
            lightcolor=[("readonly", BORDER)],
            darkcolor=[("readonly", BORDER)],
            bordercolor=[("readonly", BORDER)],
            arrowcolor=[("readonly", FG), ("!disabled", FG)],
        )
        style.configure("TProgressbar", background=BLUE, troughcolor=BG3)
        self.root.option_add("*TCombobox*Listbox*Background", BG3)
        self.root.option_add("*TCombobox*Listbox*Foreground", FG)
        self.root.option_add("*TCombobox*Listbox*selectBackground", BLUE)
        self.root.option_add("*TCombobox*Listbox*selectForeground", FG)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=2)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        nb = ttk.Notebook(left)
        nb.grid(row=0, column=0, sticky="nsew")
        self.train_tab = ttk.Frame(nb, padding=10)
        self.infer_tab = ttk.Frame(nb, padding=10)
        nb.add(self.train_tab, text="  Training  ")
        nb.add(self.infer_tab, text="  Inference  ")
        self._build_training_tab(self.train_tab)
        self._build_inference_tab(self.infer_tab)

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_status_panel(right)
        self._build_plots(right)
        self._build_log_panel(right)

    def _build_training_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        self.config_path_var = tk.StringVar(value=self.cfg.default_config)
        self.train_resume_model_var = tk.StringVar(value="")
        self.train_resume_mode_var = tk.StringVar(value="last")
        self.data_path_var = tk.StringVar(value=self.cfg.default_data)
        self.epochs_var = tk.StringVar(value="")
        self.batch_size_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="auto")
        self.save_best_only_var = tk.BooleanVar(value=True)
        self._training_visible_entries: list[dict[str, Any]] = []

        row = 0
        ttk.Label(parent, text="Config file:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.config_path_var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse", command=self._browse_config).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(parent, text="Resume model:").grid(row=row, column=0, sticky="w", pady=4)
        resume_frame = ttk.Frame(parent)
        resume_frame.grid(row=row, column=1, columnspan=2, sticky="ew", padx=4)
        resume_frame.columnconfigure(0, weight=1)
        self.train_resume_combo = ttk.Combobox(
            resume_frame, textvariable=self.train_resume_model_var, state="readonly"
        )
        self.train_resume_combo.grid(row=0, column=0, sticky="ew")
        self.train_resume_combo.bind("<<ComboboxSelected>>", self._on_train_resume_model_selected)
        ttk.Button(resume_frame, text="↻ Refresh", command=self._refresh_model_list).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(resume_frame, text="Discover", command=self._discover_models).grid(row=0, column=2, padx=(4, 0))

        row += 1
        ttk.Label(parent, text="Resume from:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            parent, textvariable=self.train_resume_mode_var,
            values=["last", "best"], state="readonly", width=12,
        ).grid(row=row, column=1, sticky="w", padx=4)

        row += 1
        ttk.Label(parent, text="Data path:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.data_path_var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse", command=self._browse_data).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(parent, text="Epoch override:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.epochs_var, width=14).grid(row=row, column=1, sticky="w", padx=4)

        row += 1
        ttk.Label(parent, text="Batch size override:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.batch_size_var, width=14).grid(row=row, column=1, sticky="w", padx=4)

        row += 1
        ttk.Label(parent, text="Device:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            parent, textvariable=self.device_var,
            values=["auto", "cuda", "cpu"], state="readonly", width=12,
        ).grid(row=row, column=1, sticky="w", padx=4)

        row += 1
        ttk.Checkbutton(parent, text="Save best only", variable=self.save_best_only_var).grid(
            row=row, column=0, sticky="w", pady=4
        )

        row += 1
        btns = ttk.Frame(parent)
        btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(btns, text="Start Training", style="Start.TButton", command=self._start_training).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop", style="Stop.TButton", command=self._stop_process).pack(side="left")
        ttk.Button(btns, text="Clear Plots", style="Viz.TButton", command=self._clear_plots).pack(side="left", padx=8)
        ttk.Button(btns, text="Open Messages", style="Viz.TButton", command=self._open_message_window).pack(side="left", padx=8)

        self.training_status_var = tk.StringVar(value="Idle")
        self.training_status_label = ttk.Label(parent, textvariable=self.training_status_var, style="Dim.TLabel")
        self.training_status_label.grid(row=row + 1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.training_progress = ttk.Progressbar(parent, mode="indeterminate")
        self.training_progress.grid(row=row + 2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        # Scripts workflow shortcuts
        scripts_card = ttk.LabelFrame(parent, text=" Scripts Workflow ", padding=8)
        scripts_card.grid(row=row + 3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        scripts_card.columnconfigure(0, weight=1)
        scripts_btns = ttk.Frame(scripts_card)
        scripts_btns.grid(row=0, column=0, sticky="ew")
        ttk.Button(
            scripts_btns, text=f"Run {self.cfg.train_scripts_script}",
            style="Viz.TButton", command=self._start_training_scripts,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            scripts_btns, text=f"Run {self.cfg.infer_scripts_script}",
            style="Viz.TButton", command=self._run_inference_scripts,
        ).pack(side="left")
        ttk.Label(
            scripts_card, text="Inference uses the Input and Config file from the Inference tab.",
            style="Dim.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _build_inference_tab(self, parent: ttk.Frame) -> None:
        """Generic inference tab.  Override in a subclass for project-specific controls."""
        parent.columnconfigure(1, weight=1)

        self.infer_config_var = tk.StringVar(value=self.cfg.default_config)
        self.infer_checkpoint_var = tk.StringVar(value="")
        self.infer_input_var = tk.StringVar(value="")
        self.infer_extra_args_var = tk.StringVar(value="")
        self.model_picker_var = tk.StringVar(value="")
        self.model_type_filter_var = tk.StringVar(value="all")
        self._registry_entries: list[dict[str, Any]] = []
        self._visible_model_entries: list[dict[str, Any]] = []

        row = 0
        ttk.Label(parent, text="Model:").grid(row=row, column=0, sticky="w", pady=4)
        model_frame = ttk.Frame(parent)
        model_frame.grid(row=row, column=1, columnspan=2, sticky="ew", padx=4)
        model_frame.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_picker_var, state="readonly")
        self.model_combo.grid(row=0, column=0, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(model_frame, text="↻ Refresh", command=self._refresh_model_list).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(model_frame, text="Discover", command=self._discover_models).grid(row=0, column=2, padx=(4, 0))

        row += 1
        ttk.Label(parent, text="Model type:").grid(row=row, column=0, sticky="w", pady=4)
        self.model_type_combo = ttk.Combobox(
            parent, textvariable=self.model_type_filter_var,
            values=["all"], state="readonly", width=24,
        )
        self.model_type_combo.grid(row=row, column=1, sticky="w", padx=4)
        self.model_type_combo.bind("<<ComboboxSelected>>", self._on_model_type_filter_changed)

        row += 1
        ttk.Label(parent, text="Config file:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.infer_config_var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_config(target="infer")).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(parent, text="Checkpoint (optional):").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.infer_checkpoint_var).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse", command=self._browse_checkpoint).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(parent, text="Input:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.infer_input_var, width=72).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=4
        )

        row += 1
        ttk.Label(parent, text="Extra CLI args:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.infer_extra_args_var, width=72).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=4
        )
        ttk.Label(
            parent,
            text='Space-separated flags passed directly to the infer script, e.g. "--mode guided --budget 500"',
            style="Dim.TLabel",
        ).grid(row=row + 1, column=1, columnspan=2, sticky="w", padx=4)

        row += 2
        btns = ttk.Frame(parent)
        btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(btns, text="Run Inference", style="Viz.TButton", command=self._run_inference).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop", style="Stop.TButton", command=self._stop_process).pack(side="left")

        self.infer_result_var = tk.StringVar(value="Result: —")
        self.infer_conf_var = tk.StringVar(value="Confidence: 0.0000")
        ttk.Label(parent, textvariable=self.infer_result_var, font=FONT_BOLD).grid(
            row=row + 1, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        ttk.Label(parent, textvariable=self.infer_conf_var, style="Dim.TLabel").grid(
            row=row + 2, column=0, columnspan=3, sticky="w"
        )
        self.confidence_bar = ttk.Progressbar(parent, mode="determinate", maximum=1.0)
        self.confidence_bar.grid(row=row + 3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ttk.Label(parent, text="Live epoch inference", font=FONT_BOLD).grid(
            row=row + 4, column=0, columnspan=3, sticky="w", pady=(12, 0)
        )
        self.infer_live_text = scrolledtext.ScrolledText(
            parent, height=8, bg=BG2, fg=FG, insertbackground=FG, font=FONT_MONO, wrap="word",
        )
        self.infer_live_text.grid(row=row + 5, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.infer_live_text.configure(state="disabled")

        self.infer_signal_canvas = tk.Canvas(
            parent,
            height=180,
            bg=BG2,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.infer_signal_canvas.grid(row=row + 6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._inference_series: dict[str, list[float]] = {}

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Idle")
        self.device_status_var = tk.StringVar(value="Device: —")
        self.dataset_status_var = tk.StringVar(value="Dataset: —")
        self.run_status_var = tk.StringVar(value="Run: —")
        self.last_epoch_var = tk.StringVar(value="Epoch: —")
        self.last_loss_var = tk.StringVar(value="Loss: —")
        self.last_metrics_var = tk.StringVar(value="Metrics: —")

        ttk.Label(card, text="Status", font=FONT_BOLD).grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=self.status_var).grid(row=0, column=1, sticky="e")
        ttk.Label(card, textvariable=self.device_status_var, style="Dim.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(card, textvariable=self.dataset_status_var, style="Dim.TLabel").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(card, textvariable=self.run_status_var, style="Dim.TLabel").grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(card, textvariable=self.last_epoch_var, style="Dim.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(card, textvariable=self.last_loss_var, style="Dim.TLabel").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(card, textvariable=self.last_metrics_var, style="Dim.TLabel").grid(row=6, column=0, columnspan=2, sticky="w")

    def _build_plots(self, parent: ttk.Frame) -> None:
        plots_frame = ttk.Frame(parent)
        plots_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        plots_frame.columnconfigure(0, weight=1)
        plots_frame.rowconfigure(0, weight=1)
        plots_frame.rowconfigure(1, weight=1)
        self._plots_frame = plots_frame

        # Placeholder plots — replaced dynamically on first epoch_metrics event
        self.loss_plot = MetricPlot(plots_frame, "Loss curves", "Lower is better")
        self.loss_plot.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.metric_plot = MetricPlot(plots_frame, "Metrics", "Higher is better")
        self.metric_plot.grid(row=1, column=0, sticky="nsew")

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        log_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        log_card.grid(row=2, column=0, sticky="nsew")
        log_card.rowconfigure(1, weight=1)
        log_card.columnconfigure(0, weight=1)

        ttk.Label(log_card, text="Log", font=FONT_BOLD).grid(row=0, column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(
            log_card, height=16, bg=BG2, fg=FG, insertbackground=FG, font=FONT_MONO, wrap="word",
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_config(self, target: str = "train") -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        if target == "infer":
            self.infer_config_var.set(path)
        else:
            self.config_path_var.set(path)

    def _browse_data(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.json *.txt"), ("All files", "*.*")])
        if path:
            self.data_path_var.set(path)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.infer_checkpoint_var.set(path)

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _build_train_cmd(self, script: str) -> list[str] | None:
        selected_entry = self._selected_training_resume_entry()
        if selected_entry is not None:
            config = Path(str(selected_entry.get("config_path", "")).strip() or self.config_path_var.get().strip())
        else:
            config = Path(self.config_path_var.get().strip())
        if not config.exists():
            messagebox.showerror("Missing config", f"Config file not found: {config}")
            return None

        cmd = [sys.executable, "-u", script, "-c", str(config)]
        if self.data_path_var.get().strip():
            cmd.extend(["--data", self.data_path_var.get().strip()])
        if self.epochs_var.get().strip():
            cmd.extend(["--epochs", self.epochs_var.get().strip()])
        if self.batch_size_var.get().strip():
            cmd.extend(["--batch-size", self.batch_size_var.get().strip()])
        if self.device_var.get().strip():
            cmd.extend(["--device", self.device_var.get().strip()])

        resume_checkpoint = self._resolve_training_resume_checkpoint(selected_entry)
        if selected_entry is not None and not resume_checkpoint:
            return None
        if resume_checkpoint:
            cmd.extend(["--resume-checkpoint", resume_checkpoint])
        return cmd

    def _start_training(self) -> None:
        if self.process is not None:
            messagebox.showwarning("Training running", "A process is already running.")
            return
        cmd = self._build_train_cmd(self.cfg.train_script)
        if cmd:
            self._launch_process(cmd, mode="train")

    def _start_training_scripts(self) -> None:
        if self.process is not None:
            messagebox.showwarning("Training running", "A process is already running.")
            return
        cmd = self._build_train_cmd(self.cfg.train_scripts_script)
        if cmd:
            self._launch_process(cmd, mode="train")

    def _run_inference(self) -> None:
        if self.process is not None:
            messagebox.showwarning("Process running", "Please stop the current process first.")
            return
        config = Path(self.infer_config_var.get().strip())
        if not config.exists():
            messagebox.showerror("Missing config", f"Config file not found: {config}")
            return
        cmd = [sys.executable, "-u", self.cfg.infer_script, "-c", str(config)]
        inp = self.infer_input_var.get().strip()
        if inp:
            cmd.extend(["--input", inp])
        checkpoint = self.infer_checkpoint_var.get().strip()
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        extra = self.infer_extra_args_var.get().strip()
        if extra:
            cmd.extend(extra.split())
        self._launch_process(cmd, mode="infer")

    def _run_inference_scripts(self) -> None:
        if self.process is not None:
            messagebox.showwarning("Process running", "Please stop the current process first.")
            return
        config = Path(self.infer_config_var.get().strip())
        if not config.exists():
            messagebox.showerror("Missing config", f"Config file not found: {config}")
            return
        cmd = [sys.executable, "-u", self.cfg.infer_scripts_script, "-c", str(config)]
        inp = self.infer_input_var.get().strip()
        if inp:
            cmd.extend(["--input", inp])
        checkpoint = self.infer_checkpoint_var.get().strip()
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        extra = self.infer_extra_args_var.get().strip()
        if extra:
            cmd.extend(extra.split())
        self._launch_process(cmd, mode="infer")

    def _launch_process(self, cmd: list[str], *, mode: str) -> None:
        self._clear_log()
        self.status_var.set(f"{mode.capitalize()} running")
        self.training_status_var.set(f"{mode.capitalize()} running")
        self.run_status_var.set("Running")
        self.device_status_var.set(f"Device: {self.device_var.get().strip() or '—'}")
        self.training_progress.start(10)
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),  # project root
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        if self.message_window is None or not self.message_window.winfo_exists():
            self._open_message_window()
        else:
            self.message_window.clear()

    def _reader_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.queue.put(line.rstrip("\n"))
        self.queue.put("__PROCESS_DONE__")

    def _drain_queue(self) -> None:
        try:
            while True:
                line = self.queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    self._finish_process()
                    continue
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("type") == "epoch_progress":
                        self._handle_telemetry(payload)
                        continue
                self._append_log(line)
                self._parse_line(line)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _parse_line(self, line: str) -> None:
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                self._handle_telemetry(payload)
                return

        # Generic well-known prefix lines
        if line.startswith("Training on device:"):
            self.device_status_var.set(line)
        elif line.startswith("Dataset size:"):
            self.dataset_status_var.set(line)
            self.training_status_var.set("Training data loaded")
        elif line.startswith("Run directory:"):
            self.run_status_var.set(line)
        elif line.startswith("GPU:"):
            self.device_status_var.set(line)
        elif line.startswith("Predicted:") or line.startswith("Predicted initial:"):
            value = line.split(":", 1)[1].strip()
            self.state.last_predicted = value
            self.infer_result_var.set(f"Result: {value}")
        elif line.startswith("Average confidence score:"):
            try:
                value = float(line.split(":", 1)[1].strip())
            except ValueError:
                value = 0.0
            self.state.last_confidence = value
            self.infer_conf_var.set(f"Confidence: {value:.4f}")
            self.confidence_bar["value"] = max(0.0, min(1.0, value))

    def _handle_telemetry(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "model_registered":
            self.root.after(500, self._refresh_model_list)
        elif kind == "epoch_inference":
            self._handle_epoch_inference(payload)
        elif kind == "epoch_metrics":
            valid = payload.get("valid") if isinstance(payload.get("valid"), dict) else {}
            train = payload.get("train") if isinstance(payload.get("train"), dict) else {}
            epoch = int(payload.get("epoch", 0))
            train_loss = float(train.get("loss", 0.0))
            val_loss = float(valid.get("loss", 0.0))
            self._last_valid_loss = val_loss

            self._update_dynamic_plots(epoch, train, valid)

            self.last_epoch_var.set(f"Epoch: {epoch}")
            self.last_loss_var.set(
                f"Loss: train {train_loss:.5f} | val {val_loss:.5f}"
            )
            # Build a compact metric summary from whatever keys are present
            metric_parts = []
            for key in train:
                if key not in ("loss", "criterion_loss", "extra_loss", "activations"):
                    t_val = float(train.get(key, 0.0))
                    v_val = float(valid.get(key, 0.0))
                    metric_parts.append(f"{key}: t={t_val:.4f} v={v_val:.4f}")
            self.last_metrics_var.set("  |  ".join(metric_parts) or "—")

            if self.message_window is not None and self.message_window.winfo_exists():
                self.message_window.set_progress(
                    f"Progress: epoch {epoch} complete | "
                    f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
                )

        elif kind == "epoch_progress":
            epoch = int(payload.get("epoch", 0))
            batch = int(payload.get("batch", 0))
            batches = max(1, int(payload.get("batches", 1)))
            progress = float(payload.get("progress", 0.0))
            loss = float(payload.get("loss", 0.0))

            # Build dynamic progress text from all metric keys in payload
            extra_parts = []
            skip_keys = {"type", "phase", "epoch", "batch", "batches", "progress",
                         "run_dir", "loss", "criterion_loss", "activations", "gradients"}
            for key, val in payload.items():
                if key not in skip_keys and isinstance(val, (int, float)):
                    extra_parts.append(f"{key}={float(val):.4f}")

            progress_text = (
                f"Progress: epoch {epoch} | batch {batch}/{batches} | "
                f"{progress * 100:.1f}% | loss={loss:.5f}"
            )
            if extra_parts:
                progress_text += "  " + "  ".join(extra_parts)
            if self._last_valid_loss is not None:
                progress_text += f" | val_loss={self._last_valid_loss:.5f}"

            if self.message_window is not None and self.message_window.winfo_exists():
                self.message_window.set_progress(progress_text)

        if self.visualizer_window is not None:
            try:
                self.visualizer_window.set_live_state(payload)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Epoch inference telemetry
    # ------------------------------------------------------------------

    def _append_inference_live_text(self, text: str) -> None:
        self.infer_live_text.configure(state="normal")
        self.infer_live_text.insert(tk.END, text + "\n")
        self.infer_live_text.see(tk.END)
        self.infer_live_text.configure(state="disabled")

    def _set_inference_series(self, series: dict[str, list[float]]) -> None:
        self._inference_series = {
            str(name): [float(v) for v in values if isinstance(v, (int, float))]
            for name, values in series.items()
            if isinstance(values, list)
        }
        self._redraw_inference_canvas()

    def _redraw_inference_canvas(self) -> None:
        canvas = self.infer_signal_canvas
        canvas.delete("all")
        w = int(canvas.winfo_width() or 560)
        h = int(canvas.winfo_height() or 180)
        pad_l, pad_r, pad_t, pad_b = 40, 16, 16, 26
        plot_w = max(1, w - pad_l - pad_r)
        plot_h = max(1, h - pad_t - pad_b)

        canvas.create_line(pad_l, pad_t, pad_l, pad_t + plot_h, fill=BORDER)
        canvas.create_line(pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h, fill=BORDER)

        if not self._inference_series:
            canvas.create_text(w // 2, h // 2, fill=FG_DIM, font=FONT_NORM, text="No inference series")
            return

        all_values = [v for values in self._inference_series.values() for v in values]
        if not all_values:
            canvas.create_text(w // 2, h // 2, fill=FG_DIM, font=FONT_NORM, text="No inference series")
            return

        y_min, y_max = min(all_values), max(all_values)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        max_len = max(len(values) for values in self._inference_series.values())
        colors = [GREEN, BLUE, ORANGE, PURPLE, RED]

        def to_xy(index: int, value: float) -> tuple[float, float]:
            x = pad_l + (index / max(1, max_len - 1)) * plot_w
            y = pad_t + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h
            return x, y

        for i, (name, values) in enumerate(self._inference_series.items()):
            if len(values) < 2:
                continue
            coords: list[float] = []
            for idx, value in enumerate(values):
                x, y = to_xy(idx, float(value))
                coords.extend([x, y])
            color = colors[i % len(colors)]
            canvas.create_line(*coords, fill=color, width=2)
            lx, ly = to_xy(len(values) - 1, float(values[-1]))
            canvas.create_text(lx + 6, ly, anchor="w", fill=color, font=FONT_SM, text=name)

    def _handle_epoch_inference(self, payload: dict[str, Any]) -> None:
        epoch = int(payload.get("epoch", 0))
        text = str(payload.get("text") or payload.get("message") or "").strip()
        result = payload.get("result")
        confidence = payload.get("confidence")
        series = payload.get("series")

        if result is not None:
            self.infer_result_var.set(f"Result: {result}")

        if isinstance(confidence, (int, float)):
            c = float(confidence)
            self.infer_conf_var.set(f"Confidence: {c:.4f}")
            self.confidence_bar["value"] = max(0.0, min(1.0, c))

        if isinstance(series, dict):
            self._set_inference_series(series)

        line = f"[epoch {epoch}]"
        if text:
            line += f" {text}"
        elif result is not None:
            line += f" result={result}"
        self._append_inference_live_text(line)

    # ------------------------------------------------------------------
    # Dynamic plot management
    # ------------------------------------------------------------------

    def _update_dynamic_plots(
        self,
        epoch: int,
        train: dict,
        valid: dict,
    ) -> None:
        """Populate loss and metric series from whatever keys are in the payload."""
        _skip = {"activations"}
        for key, train_val in train.items():
            if key in _skip or not isinstance(train_val, (int, float)):
                continue
            val_val = valid.get(key)
            series_key_train = f"train_{key}"
            series_key_val = f"val_{key}"

            is_loss = "loss" in key
            series_dict = self._loss_series if is_loss else self._metric_series

            series_dict.setdefault(series_key_train, []).append((epoch, float(train_val)))
            if isinstance(val_val, (int, float)):
                series_dict.setdefault(series_key_val, []).append((epoch, float(val_val)))

        self.loss_plot.set_series(self._loss_series)
        self.metric_plot.set_series(self._metric_series)

    # ------------------------------------------------------------------
    # Plot / log clearing
    # ------------------------------------------------------------------

    def _clear_plots(self) -> None:
        self._loss_series = {}
        self._metric_series = {}
        self.loss_plot.set_series({})
        self.metric_plot.set_series({})
        self.last_epoch_var.set("Epoch: —")
        self.last_loss_var.set("Loss: —")
        self.last_metrics_var.set("Metrics: —")
        self.infer_result_var.set("Result: —")
        self.infer_conf_var.set("Confidence: 0.0000")
        self.confidence_bar["value"] = 0
        self._inference_series = {}
        self._redraw_inference_canvas()
        self.infer_live_text.configure(state="normal")
        self.infer_live_text.delete("1.0", tk.END)
        self.infer_live_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        if self.message_window is not None and self.message_window.winfo_exists():
            self.message_window.append(text)

    # ------------------------------------------------------------------
    # Process helpers
    # ------------------------------------------------------------------

    def _finish_process(self) -> None:
        if self.process is not None:
            code = self.process.poll()
            self._append_log(f"[process] exited with code {code}")
        self.process = None
        self.reader_thread = None
        self.training_progress.stop()
        self.status_var.set("Idle")
        self.training_status_var.set("Idle")
        self.run_status_var.set("Completed")

    def _stop_process(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
        except OSError:
            pass
        self._append_log("[process] terminate requested")

    # ------------------------------------------------------------------
    # Message window
    # ------------------------------------------------------------------

    def _open_message_window(self) -> None:
        if self.message_window is None or not self.message_window.winfo_exists():
            self.message_window = MessageWindow(self.root)
        else:
            try:
                self.message_window.lift()
                self.message_window.focus_force()
            except tk.TclError:
                self.message_window = MessageWindow(self.root)

    # ------------------------------------------------------------------
    # Registry / model list
    # ------------------------------------------------------------------

    def _registry_path(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / self.cfg.runs_dir / self.cfg.registry_filename

    def _refresh_model_list(self) -> None:
        reg_path = self._registry_path()
        if not reg_path.exists():
            empty: list[str] = ["(no models indexed yet)"]
            self._registry_entries = []
            self._visible_model_entries = []
            self._training_visible_entries = []
            self.model_combo["values"] = empty
            self.model_type_combo["values"] = ["all"]
            self.model_type_filter_var.set("all")
            self.train_resume_combo["values"] = ["(none)"]
            self.train_resume_model_var.set("")
            return
        try:
            with reg_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._registry_entries = data if isinstance(data, list) else []
        except Exception as exc:
            messagebox.showerror("Registry error", f"Could not read {reg_path.name}:\n{exc}")
            self._registry_entries = []

        model_types = sorted({str(e.get("model_type", "unknown")).lower() for e in self._registry_entries})
        self.model_type_combo["values"] = ["all", *model_types]
        if self.model_type_filter_var.get() not in self.model_type_combo["values"]:
            self.model_type_filter_var.set("all")

        self._training_visible_entries = list(self._registry_entries)
        train_labels = [e.get("name", str(i)) for i, e in enumerate(self._training_visible_entries)]
        self.train_resume_combo["values"] = ["(none)", *train_labels] if train_labels else ["(none)"]
        if self.train_resume_model_var.get() not in self.train_resume_combo["values"]:
            self.train_resume_model_var.set("(none)")

        self._apply_model_filter_and_select_latest()

    def _selected_training_resume_entry(self) -> dict[str, Any] | None:
        name = self.train_resume_model_var.get().strip()
        if not name or name in ("(none)", ""):
            return None
        return next(
            (e for e in self._training_visible_entries if e.get("name") == name), None
        )

    def _resolve_training_resume_checkpoint(self, entry: dict[str, Any] | None) -> str | None:
        if entry is None:
            return None
        mode = self.train_resume_mode_var.get().strip().lower()
        if mode == "best":
            checkpoint = str(entry.get("checkpoint_best") or "").strip()
            if not checkpoint:
                checkpoint = str(entry.get("checkpoint_last") or "").strip()
        else:
            checkpoint = str(entry.get("checkpoint_last") or "").strip()
            if not checkpoint:
                checkpoint = str(entry.get("checkpoint_best") or "").strip()
        if not checkpoint:
            messagebox.showerror(
                "Resume error",
                "Selected model has no checkpoint_last.pt or checkpoint_best.pt registered.",
            )
            return None
        if not Path(checkpoint).exists():
            messagebox.showerror("Resume error", f"Resume checkpoint not found: {checkpoint}")
            return None
        return checkpoint

    def _on_train_resume_model_selected(self, _event: object = None) -> None:
        entry = self._selected_training_resume_entry()
        if entry is None:
            return
        cfg = str(entry.get("config_path", "")).strip()
        if cfg and Path(cfg).exists():
            self.config_path_var.set(cfg)

    def _apply_model_filter_and_select_latest(self) -> None:
        selected_type = self.model_type_filter_var.get().strip().lower() or "all"
        if selected_type == "all":
            self._visible_model_entries = list(self._registry_entries)
        else:
            self._visible_model_entries = [
                e for e in self._registry_entries
                if str(e.get("model_type", "unknown")).lower() == selected_type
            ]
        labels = [e.get("name", str(i)) for i, e in enumerate(self._visible_model_entries)]
        self.model_combo["values"] = labels if labels else ["(no models for selected type)"]
        if labels:
            self.model_combo.current(len(labels) - 1)
            self._on_model_selected()
        else:
            self.model_picker_var.set("")

    def _on_model_type_filter_changed(self, _event: object = None) -> None:
        self._apply_model_filter_and_select_latest()

    def _discover_models(self) -> None:
        try:
            from core.registry import discover_runs
        except ImportError:
            messagebox.showerror("Import error", "core.registry not found.")
            return
        runs_root = self._registry_path().parent
        new_entries = discover_runs(runs_root, self._registry_path())
        self._refresh_model_list()
        msg = f"Discovered {len(new_entries)} new run(s)." if new_entries else "No new runs found."
        messagebox.showinfo("Discover models", msg)

    def _on_model_selected(self, _event: object = None) -> None:
        name = self.model_picker_var.get()
        entry = next((e for e in self._visible_model_entries if e.get("name") == name), None)
        if entry is None:
            return
        ckpt = entry.get("checkpoint_best") or entry.get("checkpoint_last") or ""
        self.infer_checkpoint_var.set(ckpt)
        cfg = entry.get("config_path", "")
        if cfg and Path(cfg).exists():
            self.infer_config_var.set(cfg)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._stop_process()
        if self.visualizer_window is not None:
            try:
                self.visualizer_window.destroy()
            except tk.TclError:
                pass
        if self.message_window is not None:
            try:
                self.message_window.destroy()
            except tk.TclError:
                pass
        self.root.after(150, self.root.destroy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config: GUIConfig | None = None) -> None:
    root = tk.Tk()
    TrainingGUI(root, config=config)
    root.mainloop()


if __name__ == "__main__":
    main()
