from __future__ import annotations

from datetime import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox

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
YELLOW = "#f2cc60"
CYAN = "#39c5cf"

FONT_MONO = ("Courier New", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_NORM = ("Segoe UI", 9)
FONT_SM = ("Segoe UI", 8)

RUN_COLORS = [GREEN, BLUE, ORANGE, PURPLE, CYAN, YELLOW, RED]


@dataclass
class RunRecord:
    run_dir: Path
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    last_activations: dict[str, list[float]] = field(default_factory=dict)
    last_gradients: dict[str, list[float]] = field(default_factory=dict)
    live_state: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.name}  •  {self.run_dir.name}"


class SimpleGraph(ttk.Frame):
    def __init__(self, parent: tk.Widget, title: str) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG2, highlightthickness=1, highlightbackground=BORDER)
        self.canvas.pack(fill="both", expand=True)
        self.series: dict[str, list[tuple[float, float]]] = {}
        self.labels: dict[str, str] = {}
        self.palette = RUN_COLORS[:]
        self.selected_point: tuple[str, float, float] | None = None
        self.scale_mode: str = "linear"  # "linear" or "log10"
        self.x_axis_mode: str = "epoch"  # "epoch" or "time"
        self.fixed_epoch_range: tuple[float, float] | None = None
        self.on_x_axis_mode_changed: callable | None = None
        self.custom_y_range: tuple[float, float] | None = None
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self._redraw_id = None
        self.after(50, self.redraw)

    def set_series(self, series: dict[str, list[tuple[float, float]]], labels: dict[str, str] | None = None) -> None:
        self.series = {key: value[:] for key, value in series.items()}
        self.labels = labels or {}
        self.redraw()

    def _on_click(self, event: tk.Event) -> None:
        best: tuple[str, int, float, float] | None = None
        bbox = self._plot_bbox()
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        points = self._scaled_points()
        for name, pts in points.items():
            for idx, (x, y, epoch, value) in enumerate(pts):
                dist = (x - event.x) ** 2 + (y - event.y) ** 2
                if best is None or dist < best[3]:
                    best = (name, epoch, value, dist)
        if best is not None:
            self.selected_point = (best[0], float(best[1]), best[2])
            self.redraw()

    def _on_right_click(self, event: tk.Event) -> None:
        bbox = self._plot_bbox()
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox

        if not (x0 - 50 <= event.x <= x1 and y0 <= event.y <= y1):
            return

        menu = tk.Menu(self.canvas, tearoff=False, bg=BG3, fg=FG, activebackground=BLUE, activeforeground=FG)
        axis_label = "Switch X-axis to training time" if self.x_axis_mode == "epoch" else "Switch X-axis to epochs"
        menu.add_command(label=axis_label, command=self._toggle_x_axis_mode)
        menu.add_separator()
        menu.add_command(label="Toggle Log10 Scale", command=self._toggle_scale)
        menu.add_command(label="Set Y Range...", command=self._set_y_range)
        menu.add_command(label="Reset Y Range", command=self._reset_y_range)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.after_idle(menu.destroy)

    def _toggle_x_axis_mode(self) -> None:
        self.x_axis_mode = "time" if self.x_axis_mode == "epoch" else "epoch"
        if self.on_x_axis_mode_changed is not None:
            try:
                self.on_x_axis_mode_changed(self.x_axis_mode)
            except Exception:
                pass
        self.redraw()

    def _toggle_scale(self) -> None:
        self.scale_mode = "log10" if self.scale_mode == "linear" else "linear"
        self.redraw()

    def _set_y_range(self) -> None:
        import tkinter.simpledialog as simpledialog
        dialog = tk.Toplevel(self)
        dialog.title("Set Y Range")
        dialog.configure(bg=BG2)
        
        # Get current range
        if self.custom_y_range:
            y_min, y_max = self.custom_y_range
        else:
            all_points = [pt for pts in self.series.values() for pt in pts]
            if all_points:
                values = [y for _, y in all_points]
                y_min, y_max = min(values), max(values)
            else:
                y_min, y_max = 0, 1
        
        # Min label and entry
        tk.Label(dialog, text="Min value:", bg=BG2, fg=FG).grid(row=0, column=0, padx=5, pady=5)
        min_var = tk.StringVar(value=f"{y_min:.6f}")
        min_entry = tk.Entry(dialog, textvariable=min_var, width=20, bg=BG3, fg=FG, insertbackground=FG)
        min_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Max label and entry
        tk.Label(dialog, text="Max value:", bg=BG2, fg=FG).grid(row=1, column=0, padx=5, pady=5)
        max_var = tk.StringVar(value=f"{y_max:.6f}")
        max_entry = tk.Entry(dialog, textvariable=max_var, width=20, bg=BG3, fg=FG, insertbackground=FG)
        max_entry.grid(row=1, column=1, padx=5, pady=5)
        
        def apply_range() -> None:
            try:
                new_min = float(min_var.get())
                new_max = float(max_var.get())
                if new_min >= new_max:
                    tk.messagebox.showerror("Invalid Range", "Min must be less than Max", parent=dialog)
                    return
                self.custom_y_range = (new_min, new_max)
                self.redraw()
                dialog.destroy()
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter valid numbers", parent=dialog)
        
        def cancel() -> None:
            dialog.destroy()
        
        tk.Button(dialog, text="Apply", command=apply_range, bg=BLUE, fg=FG, padx=10).grid(row=2, column=0, padx=5, pady=10)
        tk.Button(dialog, text="Cancel", command=cancel, bg=BG3, fg=FG, padx=10).grid(row=2, column=1, padx=5, pady=10)
        
        dialog.transient(self)
        dialog.grab_set()
        min_entry.focus_set()
        min_entry.select_range(0, tk.END)

    def _reset_y_range(self) -> None:
        self.custom_y_range = None
        self.redraw()

    def _plot_bbox(self) -> tuple[int, int, int, int] | None:
        w = int(self.canvas.winfo_width() or 800)
        h = int(self.canvas.winfo_height() or 260)
        if w <= 0 or h <= 0:
            return None
        return (54, 14, w - 18, h - 36)

    def _scaled_points(self) -> dict[str, list[tuple[float, float, float, float]]]:
        import math
        bbox = self._plot_bbox()
        if bbox is None:
            return {}
        x0, y0, x1, y1 = bbox
        plot_w = max(1, x1 - x0)
        plot_h = max(1, y1 - y0)
        all_points = [pt for pts in self.series.values() for pt in pts]
        if not all_points:
            return {}
        epochs = [x for x, _ in all_points]
        values = [y for _, y in all_points]
        x_min, x_max = min(epochs), max(epochs)
        if self.x_axis_mode == "epoch" and self.fixed_epoch_range is not None:
            fx_min, fx_max = self.fixed_epoch_range
            x_min = min(x_min, fx_min)
            x_max = max(x_max, fx_max)
        
        # Use custom range if set, otherwise auto-calculate
        if self.custom_y_range is not None:
            y_min, y_max = self.custom_y_range
        else:
            y_min, y_max = min(values), max(values)
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0

        scaled: dict[str, list[tuple[float, float, float, float]]] = {}
        for name, pts in self.series.items():
            out: list[tuple[float, float, float, float]] = []
            for epoch, value in pts:
                x = x0 + ((epoch - x_min) / max(1, x_max - x_min)) * plot_w
                
                # Apply log scale if selected
                if self.scale_mode == "log10" and value > 0:
                    log_value = math.log10(value)
                    log_y_min = math.log10(y_min) if y_min > 0 else 0
                    log_y_max = math.log10(y_max) if y_max > 0 else 1
                    normalized = (log_value - log_y_min) / max(0.001, log_y_max - log_y_min)
                else:
                    normalized = (value - y_min) / max(0.001, y_max - y_min)
                
                y = y0 + (1.0 - normalized) * plot_h
                out.append((x, y, epoch, value))
            scaled[name] = out
        return scaled

    def redraw(self) -> None:
        import math
        self.canvas.delete("all")
        w = int(self.canvas.winfo_width() or 800)
        h = int(self.canvas.winfo_height() or 260)
        bbox = self._plot_bbox()
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        plot_w = max(1, x1 - x0)
        plot_h = max(1, y1 - y0)

        self.canvas.create_line(x0, y0, x0, y1, fill=BORDER)
        self.canvas.create_line(x0, y1, x1, y1, fill=BORDER)

        scaled = self._scaled_points()
        if not scaled:
            self.canvas.create_text(w // 2, h // 2, fill=FG_DIM, font=FONT_NORM, text="No metrics loaded")
            return

        all_points = [pt for pts in scaled.values() for pt in pts]
        epochs = [epoch for _, _, epoch, _ in all_points]
        values = [value for _, _, _, value in all_points]
        x_min, x_max = min(epochs), max(epochs)
        
        # Calculate y range (same logic as _scaled_points)
        if self.custom_y_range is not None:
            y_min, y_max = self.custom_y_range
        else:
            y_min, y_max = min(values), max(values)
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0

        # Draw Y-axis ticks and labels
        for tick in range(5):
            frac = tick / 4 if 4 else 0
            y = y0 + frac * plot_h
            
            if self.scale_mode == "log10" and y_min > 0 and y_max > 0:
                log_y_min = math.log10(y_min)
                log_y_max = math.log10(y_max)
                log_val = log_y_max - frac * (log_y_max - log_y_min)
                val = 10 ** log_val
            else:
                val = y_max - frac * (y_max - y_min)
            
            self.canvas.create_line(x0 - 4, y, x0, y, fill=BORDER)
            self.canvas.create_text(6, y, anchor="w", fill=FG_DIM, font=FONT_SM, text=f"{val:.4f}")

        # Draw scale indicator
        scale_label = "log10" if self.scale_mode == "log10" else "linear"
        x_axis_label = "epochs" if self.x_axis_mode == "epoch" else "time(min)"
        self.canvas.create_text(x1, h - 12, anchor="e", fill=FG_DIM, font=FONT_SM, text=f"{x_axis_label} {x_min:.2f}..{x_max:.2f} [{scale_label}]")

        # Draw series lines
        visible_series: list[tuple[str, str]] = []
        for index, (name, pts) in enumerate(scaled.items()):
            if len(pts) < 2:
                continue
            color = self.palette[index % len(self.palette)]
            visible_series.append((name, color))
            coords: list[float] = []
            for x, y, _, _ in pts:
                coords.extend([x, y])
            self.canvas.create_line(*coords, fill=color, width=2, smooth=True)
            last_x, last_y, last_epoch, last_value = pts[-1]
            self.canvas.create_oval(last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill=color, outline="")
            label = self.labels.get(name, name)
            self.canvas.create_text(last_x + 6, last_y, anchor="w", fill=color, font=FONT_SM, text=label)

        # Draw legend in top-right corner
        if visible_series:
            legend_x = x1 - 10
            legend_y = y0 + 10
            for i, (name, color) in enumerate(visible_series):
                label = self.labels.get(name, name)
                item_y = legend_y + i * 18
                self.canvas.create_rectangle(legend_x - 40, item_y - 6, legend_x - 30, item_y + 4, fill=color, outline=color)
                self.canvas.create_text(legend_x - 22, item_y, anchor="w", fill=FG_DIM, font=FONT_SM, text=label)

        if self.selected_point is not None:
            name, epoch, value = self.selected_point
            x_label = "epoch" if self.x_axis_mode == "epoch" else "time(min)"
            self.canvas.create_text(
                x0 + 4,
                y0 + 4,
                anchor="nw",
                fill=FG,
                font=FONT_BOLD,
                text=f"Selected: {name} | {x_label} {epoch:.3f} | {value:.6f}",
            )


class ActivationView(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=BG2, highlightthickness=1, highlightbackground=BORDER)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.model_info: dict[str, Any] = {}
        self.activations: dict[str, list[float]] = {}
        self.gradients: dict[str, list[float]] = {}
        self.mode: str = "activations"
        self.title = "Model architecture"
        self.node_cap: int = 16
        self.node_radius: int = 5
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self._drag_anchor: tuple[float, float] | None = None
        self._redraw_scheduled: bool = False
        self._redraw_interval_ms: int = 33
        self._connection_color_cache: dict[tuple[int, int], str] = {}
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel_linux_up)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel_linux_down)
        self.canvas.bind("<Double-Button-1>", self._reset_view_event)
        self._request_redraw()

    def _request_redraw(self) -> None:
        if self._redraw_scheduled:
            return
        self._redraw_scheduled = True
        self.after(self._redraw_interval_ms, self._perform_redraw)

    def _perform_redraw(self) -> None:
        self._redraw_scheduled = False
        self.redraw()

    def set_model(
        self,
        model_info: dict[str, Any],
        activations: dict[str, list[float]] | None = None,
        gradients: dict[str, list[float]] | None = None,
    ) -> None:
        self.model_info = model_info or {}
        self.activations = activations or {}
        self.gradients = gradients or {}
        self._request_redraw()

    def set_profiles(
        self,
        activations: dict[str, list[float]] | None = None,
        gradients: dict[str, list[float]] | None = None,
    ) -> None:
        if activations is not None:
            self.activations = activations or {}
        if gradients is not None:
            self.gradients = gradients or {}
        self._request_redraw()

    def set_activations(self, activations: dict[str, list[float]] | None) -> None:
        self.activations = activations or {}
        self._request_redraw()

    def set_gradients(self, gradients: dict[str, list[float]] | None) -> None:
        self.gradients = gradients or {}
        self._request_redraw()

    def set_mode(self, mode: str) -> None:
        self.mode = "gradients" if str(mode).lower() == "gradients" else "activations"
        self._request_redraw()

    def set_node_cap(self, node_cap: int) -> None:
        self.node_cap = max(4, int(node_cap))
        self._request_redraw()

    def set_node_radius(self, node_radius: int) -> None:
        self.node_radius = max(2, min(14, int(node_radius)))
        self._request_redraw()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.canvas.yview_moveto(0.0)
        self._request_redraw()

    def _reset_view_event(self, _event: tk.Event | None = None) -> None:
        self.reset_view()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        self.canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def _on_mouse_wheel_linux_up(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(-1, "units")

    def _on_mouse_wheel_linux_down(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(1, "units")

    def _default_graph(self) -> dict[str, Any]:
        seq_len = int(self.model_info.get("seq_len", 64))
        num_layers = int(self.model_info.get("num_layers", 4))
        stages = [
            {"id": "embedding", "label": "Embedding", "nodes": seq_len, "activation_key": "embedding", "kind": "embedding"},
        ]
        for index in range(num_layers):
            stages.append({
                "id": f"encoder_{index}",
                "label": f"Encoder {index + 1}",
                "nodes": seq_len,
                "activation_key": f"encoder_{index}",
                "kind": "hidden",
            })
        stages.extend([
            {"id": "norm", "label": "Norm", "nodes": seq_len, "activation_key": "norm", "kind": "norm"},
            {"id": "logits", "label": "Logits", "nodes": seq_len, "activation_key": "logits", "kind": "output"},
        ])
        connections: list[dict[str, str]] = []
        for index, stage in enumerate(stages[1:], start=1):
            connections.append({"from": stages[index - 1]["id"], "to": stage["id"], "mode": "paired"})
        return {"layout": "columns", "stages": stages, "connections": connections}

    @staticmethod
    def _sample_values_with_indices(
        values: list[float], count: int, *, fallback: float = 0.0
    ) -> tuple[list[float], list[int]]:
        if count <= 0:
            return [], []
        if not values:
            return [fallback] * count, list(range(count))
        if len(values) == count:
            return [float(v) for v in values], list(range(count))
        if len(values) < count:
            padded = [float(v) for v in values]
            padded.extend([padded[-1] if padded else fallback] * (count - len(padded)))
            source_indices = list(range(len(values)))
            if source_indices:
                source_indices.extend([source_indices[-1]] * (count - len(source_indices)))
            else:
                source_indices = list(range(count))
            return padded, source_indices
        source_indices: list[int] = []
        sampled: list[float] = []
        for index in range(count):
            source_index = round(index * (len(values) - 1) / max(1, count - 1))
            sampled.append(float(values[source_index]))
            source_indices.append(int(source_index))
        return sampled, source_indices

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        q = max(0.0, min(1.0, q))
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(len(ordered) - 1, low + 1)
        frac = position - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    @classmethod
    def _robust_bounds(cls, values: list[float], *, low_q: float = 0.10, high_q: float = 0.90) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        low = cls._percentile(values, low_q)
        high = cls._percentile(values, high_q)
        if high <= low:
            minimum = min(values)
            maximum = max(values)
            if maximum <= minimum:
                maximum = minimum + 1e-6
            return minimum, maximum
        return low, high

    def _stage_node_positions(
        self,
        stage: dict[str, Any],
        x: float,
        *,
        top: float,
        bottom: float,
        max_nodes: int,
        node_gap: float,
        overflow_mode: bool,
    ) -> tuple[list[tuple[float, float]], list[float], int, list[int]]:
        activation_key = str(stage.get("activation_key") or stage.get("id") or "")
        source = self.gradients if self.mode == "gradients" else self.activations
        raw_values = source.get(activation_key, [])
        requested_nodes = max(1, int(stage.get("nodes", max_nodes)))
        node_count = min(max_nodes, requested_nodes)
        samples, source_indices = self._sample_values_with_indices(raw_values, node_count)
        usable_h = max(1.0, bottom - top)
        if node_count == 1:
            positions = [(x, top + usable_h / 2)]
        else:
            gap = node_gap if overflow_mode else usable_h / max(1, node_count - 1)
            positions = [(x, top + idx * gap) for idx in range(node_count)]
        return positions, samples, requested_nodes, source_indices

    @staticmethod
    def _nearest_display_index(source_indices: list[int], raw_index: int) -> int:
        if not source_indices:
            return 0
        target = int(raw_index)
        return min(range(len(source_indices)), key=lambda idx: abs(source_indices[idx] - target))

    @staticmethod
    def _iter_explicit_edges(
        edges: list[Any],
    ) -> list[tuple[int, int, float | None]]:
        parsed: list[tuple[int, int, float | None]] = []
        for edge in edges:
            src: Any = None
            tgt: Any = None
            weight: float | None = None
            if isinstance(edge, dict):
                src = edge.get("from")
                tgt = edge.get("to")
                raw_weight = edge.get("weight")
                if isinstance(raw_weight, (int, float)):
                    weight = float(raw_weight)
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                src = edge[0]
                tgt = edge[1]
                if len(edge) >= 3 and isinstance(edge[2], (int, float)):
                    weight = float(edge[2])

            if isinstance(src, int) and isinstance(tgt, int):
                parsed.append((src, tgt, weight))
        return parsed

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple[int, int, int]:
        value = color.lstrip("#")
        if len(value) != 6:
            return (0, 0, 0)
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"#{max(0, min(255, int(r))):02x}{max(0, min(255, int(g))):02x}{max(0, min(255, int(b))):02x}"

    @classmethod
    def _blend_hex(cls, color_a: str, color_b: str, t: float) -> str:
        t = max(0.0, min(1.0, float(t)))
        a = cls._hex_to_rgb(color_a)
        b = cls._hex_to_rgb(color_b)
        blended = (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )
        return cls._rgb_to_hex(blended)

    @classmethod
    def _connection_color_from_strength(cls, strength: float) -> str:
        low_blue = BLUE
        high_red = RED
        return cls._blend_hex(low_blue, high_red, max(0.0, min(1.0, strength)))

    def _draw_gradient_connection(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        strength: float,
        width: int = 1,
    ) -> None:
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.03:
            self.canvas.create_line(x0, y0, x1, y1, fill=BORDER, width=max(1, int(width)))
            return

        signal_color = self._connection_color_from_strength(strength)
        base_color = BORDER
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        segments = max(6, min(14, int(length / 90) + 6))

        strength_bucket = int(round(strength * 100.0))

        for index in range(segments):
            t0 = index / segments
            t1 = (index + 1) / segments
            xa = x0 + (x1 - x0) * t0
            ya = y0 + (y1 - y0) * t0
            xb = x0 + (x1 - x0) * t1
            yb = y0 + (y1 - y0) * t1

            midpoint = (t0 + t1) * 0.5
            envelope = max(0.0, 1.0 - abs((midpoint * 2.0) - 1.0))
            blend_amount = (envelope ** 0.85) * strength

            blend_bucket = int(round(blend_amount * 100.0))
            cache_key = (strength_bucket, blend_bucket)
            color = self._connection_color_cache.get(cache_key)
            if color is None:
                color = self._blend_hex(base_color, signal_color, blend_amount)
                self._connection_color_cache[cache_key] = color

            self.canvas.create_line(xa, ya, xb, yb, fill=color, width=max(1, int(width)))

    def _draw_connections(
        self,
        stage_positions: dict[str, list[tuple[float, float]]],
        connections: list[dict[str, Any]],
        stage_source_indices: dict[str, list[int]],
        stage_samples: dict[str, list[float]],
        *,
        transform: Callable[[float, float], tuple[float, float]],
    ) -> None:
        all_abs_values = [abs(float(v)) for values in stage_samples.values() for v in values]
        if all_abs_values:
            mag_low, mag_high = self._robust_bounds(all_abs_values, low_q=0.10, high_q=0.90)
            if mag_high <= mag_low:
                mag_high = mag_low + 1e-9
        else:
            mag_low, mag_high = 0.0, 1.0

        def edge_strength(
            source_stage: str,
            target_stage: str,
            src_index: int,
            tgt_index: int,
            weight: float | None = None,
        ) -> float:
            source_values = stage_samples.get(source_stage, [])
            target_values = stage_samples.get(target_stage, [])

            src_value = abs(float(source_values[src_index])) if 0 <= src_index < len(source_values) else 0.0
            tgt_value = abs(float(target_values[tgt_index])) if 0 <= tgt_index < len(target_values) else 0.0

            magnitude = 0.5 * (src_value + tgt_value)
            magnitude_norm = (magnitude - mag_low) / max(1e-9, mag_high - mag_low)
            magnitude_norm = max(0.0, min(1.0, magnitude_norm))

            diff = abs(src_value - tgt_value)
            similarity = 1.0 - (diff / max(1e-9, src_value + tgt_value))
            similarity = max(0.0, min(1.0, similarity))

            transmission = 0.65 * magnitude_norm + 0.35 * similarity

            if weight is not None:
                weight_norm = abs(float(weight)) / (1.0 + abs(float(weight)))
                transmission = 0.8 * transmission + 0.2 * weight_norm

            return max(0.0, min(1.0, transmission))

        for connection in connections:
            source_stage = str(connection.get("from", ""))
            target_stage = str(connection.get("to", ""))
            source = stage_positions.get(source_stage, [])
            target = stage_positions.get(target_stage, [])
            if not source or not target:
                continue

            explicit_edges = connection.get("edges")
            if isinstance(explicit_edges, list) and explicit_edges:
                source_indices = stage_source_indices.get(source_stage, list(range(len(source))))
                target_indices = stage_source_indices.get(target_stage, list(range(len(target))))
                for src_raw, tgt_raw, weight in self._iter_explicit_edges(explicit_edges):
                    src_idx = self._nearest_display_index(source_indices, src_raw)
                    tgt_idx = self._nearest_display_index(target_indices, tgt_raw)
                    x0, y0 = source[src_idx]
                    x1, y1 = target[tgt_idx]
                    x0, y0 = transform(x0, y0)
                    x1, y1 = transform(x1, y1)
                    width = 1 if weight is None else min(4, max(1, int(round(abs(weight) * 2))))
                    strength = edge_strength(source_stage, target_stage, src_idx, tgt_idx, weight)
                    self._draw_gradient_connection(x0, y0, x1, y1, strength=strength, width=width)
                continue

            mode = str(connection.get("mode", "paired"))
            if mode == "attention":
                src_indices = sorted({0, len(source) // 2, max(0, len(source) - 1)})
                tgt_indices = sorted({0, len(target) // 2, max(0, len(target) - 1)})
                for src_index in src_indices:
                    for tgt_index in tgt_indices:
                        x0, y0 = source[src_index]
                        x1, y1 = target[tgt_index]
                        x0, y0 = transform(x0, y0)
                        x1, y1 = transform(x1, y1)
                        strength = edge_strength(source_stage, target_stage, src_index, tgt_index)
                        self._draw_gradient_connection(x0, y0, x1, y1, strength=strength, width=1)
            elif mode == "all_to_all":
                max_lines = max(1, int(connection.get("max_lines", 256)))
                src_step = max(1, len(source) // max(1, int(max_lines ** 0.5)))
                tgt_step = max(1, len(target) // max(1, int(max_lines ** 0.5)))
                for src_index in range(0, len(source), src_step):
                    for tgt_index in range(0, len(target), tgt_step):
                        x0, y0 = source[src_index]
                        x1, y1 = target[tgt_index]
                        x0, y0 = transform(x0, y0)
                        x1, y1 = transform(x1, y1)
                        strength = edge_strength(source_stage, target_stage, src_index, tgt_index)
                        self._draw_gradient_connection(x0, y0, x1, y1, strength=strength, width=1)
            elif mode == "conv1d":
                kernel = max(1, int(connection.get("kernel_size", 3)))
                stride = max(1, int(connection.get("stride", 1)))
                for tgt_index in range(0, len(target), stride):
                    left = max(0, tgt_index - kernel // 2)
                    right = min(len(source), left + kernel)
                    for src_index in range(left, right):
                        x0, y0 = source[src_index]
                        x1, y1 = target[tgt_index]
                        x0, y0 = transform(x0, y0)
                        x1, y1 = transform(x1, y1)
                        strength = edge_strength(source_stage, target_stage, src_index, tgt_index)
                        self._draw_gradient_connection(x0, y0, x1, y1, strength=strength, width=1)
            else:
                pairs = max(len(source), len(target))
                for pair_index in range(pairs):
                    src_index = round(pair_index * (len(source) - 1) / max(1, pairs - 1))
                    tgt_index = round(pair_index * (len(target) - 1) / max(1, pairs - 1))
                    x0, y0 = source[src_index]
                    x1, y1 = target[tgt_index]
                    x0, y0 = transform(x0, y0)
                    x1, y1 = transform(x1, y1)
                    strength = edge_strength(source_stage, target_stage, src_index, tgt_index)
                    self._draw_gradient_connection(x0, y0, x1, y1, strength=strength, width=1)

    @staticmethod
    def _color_from_value(value: float, scale: float = 1.0) -> str:
        def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
            t = max(0.0, min(1.0, t))
            r = int(a[0] + (b[0] - a[0]) * t)
            g = int(a[1] + (b[1] - a[1]) * t)
            b_ = int(a[2] + (b[2] - a[2]) * t)
            return f"#{r:02x}{g:02x}{b_:02x}"

        normalized = max(0.0, min(scale, value)) / max(scale, 1e-9)
        clear = (BG2[1:3], BG2[3:5], BG2[5:7])
        clear_rgb = tuple(int(part, 16) for part in clear)
        if normalized <= 0.0:
            return BG2

        # Health-oriented mapping:
        # - low activity        -> dim cyan/teal
        # - optimal band        -> green
        # - saturation (too high)-> orange/red
        underactive_end = 0.20
        optimal_start = 0.20
        optimal_end = 0.75
        saturated_start = 0.85

        underactive_low = clear_rgb
        underactive_high = (57, 197, 207)   # cyan
        optimal_low = (63, 185, 80)         # green
        optimal_high = (109, 214, 123)      # bright green
        warning = (240, 136, 62)            # orange
        saturated = (248, 81, 73)           # red

        if normalized < underactive_end:
            local_t = normalized / max(underactive_end, 1e-9)
            return blend(underactive_low, underactive_high, local_t)

        if optimal_start <= normalized <= optimal_end:
            local_t = (normalized - optimal_start) / max(optimal_end - optimal_start, 1e-9)
            return blend(optimal_low, optimal_high, local_t)

        if normalized < saturated_start:
            local_t = (normalized - optimal_end) / max(saturated_start - optimal_end, 1e-9)
            return blend(optimal_high, warning, local_t)

        local_t = (normalized - saturated_start) / max(1.0 - saturated_start, 1e-9)
        return blend(warning, saturated, local_t)

    def redraw(self) -> None:
        self.canvas.delete("all")
        w = int(self.canvas.winfo_width() or 900)
        h = int(self.canvas.winfo_height() or 420)

        model_info = self.model_info
        if not model_info:
            source = self.activations if self.activations else self.gradients
            if source:
                lengths = [len(values) for values in source.values() if isinstance(values, list) and values]
                seq_len_guess = max(lengths) if lengths else 64
                num_layers_guess = len([name for name in source.keys() if str(name).startswith("encoder_")])
                if num_layers_guess <= 0:
                    num_layers_guess = 4
                emb_values = source.get("embedding") if isinstance(source, dict) else None
                embedding_dim_guess = len(emb_values) if isinstance(emb_values, list) and emb_values else seq_len_guess
                model_info = {
                    "model": "Live",
                    "seq_len": int(seq_len_guess),
                    "num_layers": int(num_layers_guess),
                    "embedding_dim": int(embedding_dim_guess),
                }
            else:
                self.canvas.create_text(w // 2, h // 2, fill=FG_DIM, font=FONT_NORM, text="No model selected")
                return

        seq_len = int(model_info.get("seq_len", 64))
        num_layers = int(model_info.get("num_layers", 4))
        embedding_dim = int(model_info.get("embedding_dim", 32))
        title = f"{model_info.get('model', 'Model')} | seq={seq_len} layers={num_layers} emb={embedding_dim}"
        self.canvas.create_text(10, 12, anchor="nw", fill=FG_DIM, font=FONT_SM, text=title)

        graph = model_info.get("graph") if isinstance(model_info.get("graph"), dict) else self._default_graph()
        stages = graph.get("stages", []) if isinstance(graph, dict) else []
        connections = graph.get("connections", []) if isinstance(graph, dict) else []
        if not stages:
            stages = self._default_graph().get("stages", [])
        if not connections:
            connections = self._default_graph().get("connections", [])

        top = 62
        legend_height = 18
        legend_gap = 30
        viewport_bottom = h - (24 + legend_height + legend_gap)

        label_area = 120
        left = label_area
        right = w - 30
        max_nodes = min(max(4, self.node_cap), seq_len)
        node_radius = int(self.node_radius)

        max_requested_nodes = 1
        for stage in stages:
            try:
                requested = int(stage.get("nodes", max_nodes))
            except Exception:
                requested = max_nodes
            max_requested_nodes = max(max_requested_nodes, min(max_nodes, max(1, requested)))

        base_visible_nodes = 16
        viewport_span = max(1.0, viewport_bottom - top)
        base_gap = viewport_span / max(1, base_visible_nodes - 1)
        node_gap = max(4.0, base_gap * (node_radius / 5.0))
        overflow_mode = max_requested_nodes > base_visible_nodes
        bottom = top + ((max_requested_nodes - 1) * node_gap if overflow_mode else viewport_span)

        stage_positions: dict[str, list[tuple[float, float]]] = {}
        stage_samples: dict[str, list[float]] = {}
        stage_source_indices: dict[str, list[int]] = {}

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        def transform(x: float, y: float) -> tuple[float, float]:
            tx = (x - center_x) * self.zoom + center_x + self.pan_x
            ty = (y - center_y) * self.zoom + center_y + self.pan_y
            return tx, ty

        usable_w = max(1.0, right - left)
        for stage_index, stage in enumerate(stages):
            x = left + (stage_index + 0.5) * (usable_w / max(1, len(stages)))
            positions, samples, requested_nodes, source_indices = self._stage_node_positions(
                stage,
                x,
                top=top,
                bottom=bottom,
                max_nodes=max_nodes,
                node_gap=node_gap,
                overflow_mode=overflow_mode,
            )
            stage_id = str(stage.get("id", f"stage_{stage_index}"))
            stage_positions[stage_id] = positions
            stage_samples[stage_id] = samples
            stage_source_indices[stage_id] = source_indices
            lx, ly = transform(x, top - 26)
            self.canvas.create_text(lx, ly, anchor="s", fill=FG, font=FONT_SM, text=str(stage.get("label", stage_id)))
            if requested_nodes > len(positions):
                sx, sy = transform(x, bottom + 4)
                self.canvas.create_text(sx, sy, anchor="n", fill=FG_DIM, font=FONT_SM, text=f"showing {len(positions)}/{requested_nodes}")

        self._draw_connections(
            stage_positions,
            connections,
            stage_source_indices,
            stage_samples,
            transform=transform,
        )

        for stage_index, stage in enumerate(stages):
            stage_id = str(stage.get("id", f"stage_{stage_index}"))
            positions = stage_positions.get(stage_id, [])
            samples = stage_samples.get(stage_id, [])
            # Per-layer robust normalization (10th..90th percentile) to avoid
            # both global compression and outlier-driven saturation.
            low, high = self._robust_bounds(samples, low_q=0.10, high_q=0.90)
            denom = max(1e-9, high - low)
            for node_index, ((x, y), value) in enumerate(zip(positions, samples)):
                normalized = (float(value) - low) / denom
                normalized = max(0.0, min(1.0, normalized))
                color = self._color_from_value(normalized, scale=1.0)
                outline = BORDER if color == BG2 else ""
                tx, ty = transform(x, y)
                radius = max(2, min(14, int(node_radius * self.zoom)))
                self.canvas.create_oval(tx - radius, ty - radius, tx + radius, ty + radius, fill=color, outline=outline)
                if stage_index == 0 and len(positions) <= 12:
                    self.canvas.create_text(tx - 10, ty, anchor="e", fill=FG_DIM, font=FONT_SM, text=str(node_index))

        legend_left = left
        legend_right = right
        legend_top = bottom + 20
        legend_bottom = legend_top + legend_height
        legend_steps = max(24, int((legend_right - legend_left) / 6))

        if self.mode == "gradients":
            legend_metric = "gradient magnitude"
            legend_low = "low / vanishing"
            legend_mid = "healthy range"
            legend_high = "high / exploding"
        else:
            legend_metric = "activation"
            legend_low = "0.0 / clear"
            legend_mid = "optimal band: green"
            legend_high = "saturated: red"

        self.canvas.create_text(10, legend_top + legend_height / 2, anchor="w", fill=FG_DIM, font=FONT_SM, text=legend_metric)
        for step in range(legend_steps):
            x0 = legend_left + (step / legend_steps) * (legend_right - legend_left)
            x1 = legend_left + ((step + 1) / legend_steps) * (legend_right - legend_left)
            value = step / max(1, legend_steps - 1)
            color = self._color_from_value(value, scale=1.0)
            outline = BORDER if color == BG2 else ""
            self.canvas.create_rectangle(x0, legend_top, x1, legend_bottom, fill=color, outline=outline)

        self.canvas.create_rectangle(legend_left, legend_top, legend_right, legend_bottom, outline=BORDER)
        self.canvas.create_text(legend_left, legend_bottom + 12, anchor="w", fill=FG_DIM, font=FONT_SM, text=legend_low)
        self.canvas.create_text((legend_left + legend_right) / 2, legend_bottom + 12, anchor="center", fill=FG_DIM, font=FONT_SM, text=legend_mid)
        self.canvas.create_text(legend_right, legend_bottom + 12, anchor="e", fill=FG_DIM, font=FONT_SM, text=legend_high)
        self.canvas.create_text(
            right,
            top - 42,
            anchor="e",
            fill=FG_DIM,
            font=FONT_SM,
            text=f"nodes={self.node_cap}  node_size={self.node_radius}  (wheel/scrollbar=scroll, dbl-click=reset)",
        )

        scroll_bottom = max(float(h), float(legend_bottom + 28.0))
        self.canvas.configure(scrollregion=(0.0, 0.0, float(max(w, right + 30)), scroll_bottom))


class VisualizerWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, runs_root: str | Path) -> None:
        super().__init__(master)
        self.title("Delphi Visualizer")
        self.configure(bg=BG)
        self.geometry("1500x980")
        self.runs_root = Path(runs_root)
        self.records: dict[str, RunRecord] = {}
        self.live_record_key: str | None = None
        self._selected_records_cache: list[RunRecord] = []
        self._latest_val_loss: float | None = None
        self._panes: dict[str, dict[str, Any]] = {}
        self._live_graph_name: str = "Live run"

        self._setup_style()
        self._build_ui()
        self.refresh_runs()

    def _setup_style(self) -> None:
        style = ttk.Style(self)
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
        style.configure("TNotebook", background=BG, bordercolor=BORDER)
        style.configure("TNotebook.Tab", background=BG3, foreground=FG_DIM, padding=[10, 4], font=FONT_NORM)
        style.map("TNotebook.Tab", background=[("selected", BG2)], foreground=[("selected", FG)])
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
        self.option_add("*TCombobox*Listbox*Background", BG3)
        self.option_add("*TCombobox*Listbox*Foreground", FG)
        self.option_add("*TCombobox*Listbox*selectBackground", BLUE)
        self.option_add("*TCombobox*Listbox*selectForeground", FG)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer, style="Card.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Loaded models", font=FONT_BOLD).grid(row=0, column=0, sticky="w")
        self.run_list = tk.Listbox(left, selectmode="extended", bg=BG2, fg=FG, selectbackground=BLUE, selectforeground=BG, font=FONT_MONO, exportselection=False, width=42)
        self.run_list.grid(row=1, column=0, sticky="nsew", pady=8)
        self.run_list.bind("<<ListboxSelect>>", lambda _e: self.load_selected_runs())

        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky="ew")
        ttk.Button(buttons, text="Refresh", command=self.refresh_runs).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Load selected", command=self.load_selected_runs).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Clear live", command=self.clear_live_state).pack(side="left")

        self.summary_var = tk.StringVar(value="No model selected")
        ttk.Label(left, textvariable=self.summary_var, style="Dim.TLabel", wraplength=300, justify="left").grid(row=3, column=0, sticky="ew", pady=(10, 0))

        options_card = ttk.LabelFrame(left, text=" Options / Logs ", padding=8)
        options_card.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        options_card.columnconfigure(1, weight=1)

        ttk.Label(options_card, text="Node size:", style="Dim.TLabel").grid(row=0, column=0, sticky="w")
        self.node_size_var = tk.StringVar(value="5")
        node_size = ttk.Combobox(
            options_card,
            textvariable=self.node_size_var,
            values=["2", "3", "4", "5", "6", "7", "8"],
            state="readonly",
            width=14,
        )
        node_size.grid(row=0, column=1, sticky="w")
        node_size.bind("<<ComboboxSelected>>", self._on_node_size_selected)

        ttk.Label(options_card, text="X-axis:", style="Dim.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.graph_x_mode_var = tk.StringVar(value="epoch")
        graph_x_mode = ttk.Combobox(
            options_card,
            textvariable=self.graph_x_mode_var,
            values=["epoch", "time"],
            state="readonly",
            width=14,
        )
        graph_x_mode.grid(row=1, column=1, sticky="w", pady=(6, 0))
        graph_x_mode.bind("<<ComboboxSelected>>", self._on_graph_x_mode_selected)

        ttk.Label(options_card, text="Node detail:", style="Dim.TLabel").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.node_detail_var = tk.StringVar(value="16")
        node_detail = ttk.Combobox(
            options_card,
            textvariable=self.node_detail_var,
            values=["16", "32", "64", "128", "256"],
            state="readonly",
            width=14,
        )
        node_detail.grid(row=2, column=1, sticky="w", pady=(6, 0))
        node_detail.bind("<<ComboboxSelected>>", self._on_node_detail_selected)

        ttk.Button(
            options_card,
            text="Reset architecture view",
            command=self._reset_architecture_view,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Button(
            options_card,
            text="Export layer values log",
            command=self._export_layer_values_log,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.right_panes = ttk.Panedwindow(right, orient="vertical")
        self.right_panes.grid(row=0, column=0, sticky="nsew")

        self._create_dynamic_pane("top", default_kind="activation", weight=3)
        self._create_dynamic_pane("middle", default_kind="losses", weight=2)
        self._create_dynamic_pane("bottom", default_kind="status", weight=1)

    def _create_dynamic_pane(self, slot: str, *, default_kind: str, weight: int) -> None:
        pane_frame = ttk.Frame(self.right_panes, style="Card.TFrame", padding=4)
        pane_frame.columnconfigure(0, weight=1)
        pane_frame.rowconfigure(1, weight=1)

        title_var = tk.StringVar(value="")
        title_label = ttk.Label(pane_frame, textvariable=title_var, font=FONT_BOLD, cursor="hand2")
        title_label.grid(row=0, column=0, sticky="w", padx=(2, 2), pady=(0, 4))
        title_label.bind("<Button-1>", lambda event, pane_slot=slot: self._open_pane_menu(event, pane_slot))

        content = ttk.Frame(pane_frame, style="Card.TFrame")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        self._panes[slot] = {
            "frame": pane_frame,
            "title_var": title_var,
            "content": content,
            "kind": "",
            "widget": None,
            "status_vars": None,
        }
        self.right_panes.add(pane_frame, weight=weight)
        self._set_pane_kind(slot, default_kind)

    def _open_pane_menu(self, event: tk.Event, slot: str) -> None:
        menu = tk.Menu(self, tearoff=False, bg=BG3, fg=FG, activebackground=BLUE, activeforeground=FG)
        options = [
            ("Activation map", "activation"),
            ("Gradient map", "gradient"),
            ("Training / validation losses", "losses"),
            ("Live status", "status"),
        ]
        current_kind = str(self._panes.get(slot, {}).get("kind", ""))
        selected_kind = tk.StringVar(value=current_kind)
        for label, kind in options:
            menu.add_radiobutton(
                label=label,
                value=kind,
                variable=selected_kind,
                command=lambda selected=kind, pane_slot=slot: self._set_pane_kind(pane_slot, selected),
            )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.after_idle(menu.destroy)

    def _set_pane_kind(self, slot: str, kind: str) -> None:
        pane = self._panes.get(slot)
        if pane is None:
            return

        content: ttk.Frame = pane["content"]
        for child in content.winfo_children():
            child.destroy()

        pane["kind"] = kind
        pane["widget"] = None
        pane["status_vars"] = None

        if kind == "activation":
            pane["title_var"].set("Architecture + activation map")
            view = ActivationView(content)
            view.grid(row=0, column=0, sticky="nsew")
            view.set_mode("activations")
            try:
                view.set_node_cap(int(self.node_detail_var.get().strip()))
            except ValueError:
                view.set_node_cap(16)
            try:
                view.set_node_radius(int(self.node_size_var.get().strip()))
            except ValueError:
                view.set_node_radius(5)
            pane["widget"] = view
        elif kind == "gradient":
            pane["title_var"].set("Architecture + gradient map")
            view = ActivationView(content)
            view.grid(row=0, column=0, sticky="nsew")
            view.set_mode("gradients")
            try:
                view.set_node_cap(int(self.node_detail_var.get().strip()))
            except ValueError:
                view.set_node_cap(16)
            try:
                view.set_node_radius(int(self.node_size_var.get().strip()))
            except ValueError:
                view.set_node_radius(5)
            pane["widget"] = view
        elif kind == "losses":
            pane["title_var"].set("Training / validation losses")
            graph = SimpleGraph(content, "Training / validation losses")
            graph.grid(row=0, column=0, sticky="nsew")
            graph.x_axis_mode = self.graph_x_mode_var.get().strip().lower() or "epoch"
            graph.on_x_axis_mode_changed = self._on_graph_x_axis_mode_changed
            pane["widget"] = graph
        else:
            pane["title_var"].set("Live status")
            status = ttk.Frame(content, style="Card.TFrame", padding=8)
            status.grid(row=0, column=0, sticky="nsew")
            status.columnconfigure(0, weight=1)
            status_var = tk.StringVar(value="Waiting for training/inference updates")
            detail_var = tk.StringVar(value="")
            ttk.Label(status, textvariable=status_var).grid(row=0, column=0, sticky="w")
            ttk.Label(status, textvariable=detail_var, style="Dim.TLabel").grid(row=1, column=0, sticky="w")
            pane["widget"] = status
            pane["status_vars"] = (status_var, detail_var)

        self._sync_panes_with_current_state()

    def _activation_views(self) -> list[ActivationView]:
        views: list[ActivationView] = []
        for pane in self._panes.values():
            widget = pane.get("widget")
            if isinstance(widget, ActivationView):
                views.append(widget)
        return views

    def _graphs(self) -> list[SimpleGraph]:
        graphs: list[SimpleGraph] = []
        for pane in self._panes.values():
            widget = pane.get("widget")
            if isinstance(widget, SimpleGraph):
                graphs.append(widget)
        return graphs

    def _status_targets(self) -> list[tuple[tk.StringVar, tk.StringVar]]:
        targets: list[tuple[tk.StringVar, tk.StringVar]] = []
        for pane in self._panes.values():
            status_vars = pane.get("status_vars")
            if isinstance(status_vars, tuple) and len(status_vars) == 2:
                targets.append((status_vars[0], status_vars[1]))
        return targets

    def _set_live_status(self, title: str, detail: str) -> None:
        for status_var, detail_var in self._status_targets():
            status_var.set(title)
            detail_var.set(detail)

    def _infer_model_info_from_profiles(
        self,
        activations: dict[str, Any] | None,
        gradients: dict[str, Any] | None,
    ) -> dict[str, Any]:
        activation_map = activations if isinstance(activations, dict) else {}
        gradient_map = gradients if isinstance(gradients, dict) else {}

        profile_lengths: list[int] = []
        for source in (activation_map, gradient_map):
            for values in source.values():
                if isinstance(values, list) and values:
                    profile_lengths.append(len(values))

        seq_len = max(profile_lengths) if profile_lengths else 64
        encoder_like = [key for key in activation_map.keys() if str(key).startswith("encoder_")]
        num_layers = max(1, len(encoder_like)) if encoder_like else 4
        embedding_values = activation_map.get("embedding")
        embedding_dim = len(embedding_values) if isinstance(embedding_values, list) and embedding_values else seq_len

        return {
            "model": "Live",
            "seq_len": int(seq_len),
            "num_layers": int(num_layers),
            "embedding_dim": int(embedding_dim),
        }

    def _sync_panes_with_current_state(self) -> None:
        selected = self._selected_records_cache
        if selected:
            current_model = selected[0]
            for view in self._activation_views():
                try:
                    view.set_node_cap(int(self.node_detail_var.get().strip()))
                except ValueError:
                    view.set_node_cap(16)
                try:
                    view.set_node_radius(int(self.node_size_var.get().strip()))
                except ValueError:
                    view.set_node_radius(5)
                view.set_model(
                    current_model.config.get("model", {}),
                    current_model.last_activations,
                    current_model.last_gradients,
                )
            self._render_loss_series(selected)
        else:
            for graph in self._graphs():
                graph.set_series({})
                graph.fixed_epoch_range = None

    def refresh_runs(self) -> None:
        self.records.clear()
        self.run_list.delete(0, tk.END)
        if not self.runs_root.exists():
            self.summary_var.set(f"Runs folder not found: {self.runs_root}")
            return

        candidates = sorted(self.runs_root.rglob("metrics_history.jsonl"))
        for history_path in candidates:
            run_dir = history_path.parent
            config_path = run_dir / "param_used.json"
            if not config_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
            name = config.get("name", run_dir.parent.name)
            record = RunRecord(run_dir=run_dir, name=name, config=config)
            try:
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    record.history.append(payload)
                    if isinstance(payload.get("valid"), dict) and payload["valid"].get("activations"):
                        record.last_activations = payload["valid"]["activations"]
                    elif isinstance(payload.get("train"), dict) and payload["train"].get("activations"):
                        record.last_activations = payload["train"]["activations"]
                    if isinstance(payload.get("valid"), dict) and payload["valid"].get("gradients"):
                        record.last_gradients = payload["valid"]["gradients"]
                    elif isinstance(payload.get("train"), dict) and payload["train"].get("gradients"):
                        record.last_gradients = payload["train"]["gradients"]
            except Exception:
                pass
            key = str(run_dir)
            self.records[key] = record
            self.run_list.insert(tk.END, record.display_name)
            self.run_list.itemconfig(self.run_list.size() - 1, fg=FG)

        self.summary_var.set(f"Found {len(self.records)} runs")
        if self.records:
            self.run_list.selection_clear(0, tk.END)
            self.run_list.selection_set(tk.END)
            self.load_selected_runs()

    def _selected_records(self) -> list[RunRecord]:
        selected = self.run_list.curselection()
        values = list(self.records.values())
        return [values[index] for index in selected if index < len(values)]

    def load_selected_runs(self) -> None:
        selected = self._selected_records()
        self._selected_records_cache = selected[:]
        if not selected:
            self.summary_var.set("No model selected")
            for view in self._activation_views():
                view.set_model({}, {}, {})
            for graph in self._graphs():
                graph.set_series({})
                graph.fixed_epoch_range = None
            return

        current_model = selected[0]
        self._render_loss_series(selected)
        for view in self._activation_views():
            try:
                view.set_node_cap(int(self.node_detail_var.get().strip()))
            except ValueError:
                view.set_node_cap(16)
            try:
                view.set_node_radius(int(self.node_size_var.get().strip()))
            except ValueError:
                view.set_node_radius(5)
            view.set_model(
                current_model.config.get("model", {}),
                current_model.last_activations,
                current_model.last_gradients,
            )
        self.summary_var.set(
            f"Selected {len(selected)} model(s). Active: {current_model.name} | {current_model.run_dir.name}"
        )

    def _on_node_detail_selected(self, _event: tk.Event | None = None) -> None:
        try:
            node_cap = int(self.node_detail_var.get().strip())
        except ValueError:
            node_cap = 16
        for view in self._activation_views():
            view.set_node_cap(node_cap)

    def _on_node_size_selected(self, _event: tk.Event | None = None) -> None:
        try:
            node_radius = int(self.node_size_var.get().strip())
        except ValueError:
            node_radius = 5
        for view in self._activation_views():
            view.set_node_radius(node_radius)

    def _reset_architecture_view(self) -> None:
        for view in self._activation_views():
            view.reset_view()

    def _on_graph_x_mode_selected(self, _event: tk.Event | None = None) -> None:
        mode = self.graph_x_mode_var.get().strip().lower()
        if mode not in {"epoch", "time"}:
            mode = "epoch"
        for graph in self._graphs():
            graph.x_axis_mode = mode
        self._on_graph_x_axis_mode_changed(mode)

    def _on_graph_x_axis_mode_changed(self, _mode: str) -> None:
        self.graph_x_mode_var.set(_mode)
        for graph in self._graphs():
            graph.x_axis_mode = _mode
        if self._selected_records_cache:
            self._render_loss_series(self._selected_records_cache)

    def _export_layer_values_log(self) -> None:
        selected = self._selected_records()
        run_dir: Path | None = selected[0].run_dir if selected else None
        output_root = (run_dir if run_dir is not None else self.runs_root).resolve()
        logs_dir = output_root / "activation_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        first_view = next((view for view in self._activation_views() if view.activations or view.gradients), None)
        if first_view is None:
            first_view = next((view for view in self._activation_views()), None)
        activations = (first_view.activations if first_view is not None else {}) or {}
        gradients = (first_view.gradients if first_view is not None else {}) or {}
        if not activations and not gradients:
            messagebox.showwarning("No data", "No activation or gradient values are currently available to export.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = logs_dir / f"layer_values_{stamp}.txt"

        def _stats(values: list[float]) -> tuple[float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0
            minimum = min(values)
            maximum = max(values)
            mean = sum(values) / len(values)
            return minimum, mean, maximum

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("Delphi Layer Values Log\n")
            fh.write(f"timestamp: {datetime.now().isoformat()}\n")
            fh.write(f"node_size: {self.node_size_var.get()}\n")
            fh.write(f"x_axis_mode: {self.graph_x_mode_var.get()}\n")
            fh.write(f"run_dir: {str(run_dir) if run_dir is not None else 'n/a'}\n")
            fh.write("\n")

            fh.write("[ACTIVATIONS]\n")
            for layer_name in sorted(activations.keys()):
                values = [float(v) for v in activations.get(layer_name, [])]
                mn, avg, mx = _stats(values)
                fh.write(f"{layer_name}\n")
                fh.write(f"  count={len(values)} min={mn:.8e} mean={avg:.8e} max={mx:.8e}\n")
                fh.write("  values=" + ", ".join(f"{v:.8e}" for v in values) + "\n")

            fh.write("\n[GRADIENTS]\n")
            for layer_name in sorted(gradients.keys()):
                values = [float(v) for v in gradients.get(layer_name, [])]
                mn, avg, mx = _stats(values)
                fh.write(f"{layer_name}\n")
                fh.write(f"  count={len(values)} min={mn:.8e} mean={avg:.8e} max={mx:.8e}\n")
                fh.write("  values=" + ", ".join(f"{v:.8e}" for v in values) + "\n")

        messagebox.showinfo("Export complete", f"Layer values log written to:\n{out_path}")

    def _render_loss_series(self, selected: list[RunRecord]) -> None:
        labels: dict[str, str] = {}
        loss_series: dict[str, list[tuple[float, float]]] = {}
        fixed_epoch_max = 1.0

        for record in selected:
            configured_epochs = float(record.config.get("trainer", {}).get("epochs", 0) or 0)
            if configured_epochs > 0:
                fixed_epoch_max = max(fixed_epoch_max, configured_epochs)

            train_points: list[tuple[float, float]] = []
            val_points: list[tuple[float, float]] = []
            for payload in record.history:
                epoch = int(payload.get("epoch", 0))
                elapsed_seconds = float(payload.get("elapsed_seconds", 0.0) or 0.0)
                x_value = float(epoch)
                if self.graph_x_mode_var.get().strip().lower() == "time" and elapsed_seconds > 0.0:
                    x_value = elapsed_seconds / 60.0

                valid = payload.get("valid", {}) if isinstance(payload.get("valid"), dict) else {}
                train = payload.get("train", {}) if isinstance(payload.get("train"), dict) else {}
                if train.get("loss") is not None:
                    train_points.append((x_value, float(train["loss"])))
                if valid.get("loss") is not None:
                    val_points.append((x_value, float(valid["loss"])))

            loss_series[f"{record.display_name} • train"] = train_points
            loss_series[f"{record.display_name} • val"] = val_points
            labels[f"{record.display_name} • train"] = f"{record.name} train"
            labels[f"{record.display_name} • val"] = f"{record.name} val"

        for graph in self._graphs():
            graph.fixed_epoch_range = (1.0, fixed_epoch_max) if graph.x_axis_mode == "epoch" else None
            graph.set_series(loss_series, labels)

    def set_live_state(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        kind = payload.get("type")
        if kind == "run_start":
            run_dir = payload.get("run_dir", "")
            self._set_live_status("Training started", str(run_dir))
            model_info = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
            self._live_graph_name = str(payload.get("name") or Path(str(run_dir)).name or "Live run")
            if not self._live_graph_name:
                self._live_graph_name = "Live run"
            for view in self._activation_views():
                view.set_model(model_info, {}, {})
            # Refresh run list so the new training run shows up
            self.refresh_runs()
            return
        if kind == "epoch_progress":
            epoch = int(payload.get("epoch", 0))
            batch = int(payload.get("batch", 0))
            batches = max(1, int(payload.get("batches", 1)))
            progress = float(payload.get("progress", 0.0))
            loss = float(payload.get("loss", 0.0))
            ce_loss = float(payload.get("cross_entropy_loss", 0.0))
            acc = float(payload.get("nibble_accuracy", 0.0))
            close = float(payload.get("inference_closeness", 0.0))
            status_text = f"Epoch {epoch} progress"
            detail = (
                f"{batch}/{batches} ({progress * 100:.1f}%) | loss={loss:.5f} | ce={ce_loss:.5f} | acc={acc:.4f} | close={close:.4f}"
            )
            if self._latest_val_loss is not None:
                detail += f" | val_loss={self._latest_val_loss:.5f}"
            else:
                detail += " | val_loss=pending"
            self._set_live_status(status_text, detail)
            activations = payload.get("activations") or {}
            gradients = payload.get("gradients") or {}
            if activations or gradients:
                for view in self._activation_views():
                    if not view.model_info:
                        view.set_model(self._infer_model_info_from_profiles(activations, gradients), {}, {})
                    view.set_profiles(activations if activations else None, gradients if gradients else None)

            self._update_graph_live(
                self._live_graph_name,
                payload,
                epoch,
                {"loss": loss},
                {},
            )
            return
        if kind == "epoch_metrics":
            epoch = int(payload.get("epoch", 0))
            train = payload.get("train", {})
            valid = payload.get("valid", {})
            run_dir = payload.get("run_dir")
            self._set_live_status(
                f"Epoch {epoch}",
                f"train_loss={train.get('loss', 0):.5f} | val_loss={valid.get('loss', 0):.5f} | run={run_dir or 'live'}",
            )
            self._latest_val_loss = float(valid.get("loss", 0.0))
            activations = valid.get("activations") or train.get("activations") or {}
            gradients = valid.get("gradients") or train.get("gradients") or {}
            if activations or gradients:
                for view in self._activation_views():
                    if not view.model_info:
                        view.set_model(self._infer_model_info_from_profiles(activations, gradients), {}, {})
                    view.set_profiles(activations if activations else None, gradients if gradients else None)
            
            # Auto-select the active run if not already selected
            active_name = None
            if not self.run_list.curselection() and run_dir:
                # Try to find and select the matching run
                for idx, record in enumerate(self.records.values()):
                    if str(record.run_dir) == str(run_dir):
                        self.run_list.selection_set(idx)
                        active_name = record.display_name
                        break
            elif self.run_list.curselection():
                records = self._selected_records()
                if records:
                    active_name = records[0].display_name
            if not active_name:
                active_name = self._live_graph_name
            
            if active_name:
                self._update_graph_live(active_name, payload, epoch, train, valid)
            return
        if kind == "inference_result":
            self._set_live_status(
                "Inference result",
                f"predicted={payload.get('predicted_initial', '—')} | confidence={payload.get('confidence', 0):.4f}",
            )
            return

    def _update_graph_live(self, name: str, payload: dict[str, Any], epoch: int, train: dict[str, Any], valid: dict[str, Any]) -> None:
        for graph in self._graphs():
            current_series = {k: v[:] for k, v in graph.series.items()}
            train_key = f"{name} • train"
            val_key = f"{name} • val"
            if train_key not in current_series:
                current_series[train_key] = []
            if val_key not in current_series:
                current_series[val_key] = []

            x_value = float(epoch)
            elapsed_seconds = float(payload.get("elapsed_seconds", 0.0) or 0.0)
            if graph.x_axis_mode == "time" and elapsed_seconds > 0.0:
                x_value = elapsed_seconds / 60.0

            if train.get("loss") is not None:
                current_series[train_key].append((x_value, float(train.get("loss", 0.0))))
            if valid.get("loss") is not None:
                current_series[val_key].append((x_value, float(valid.get("loss", 0.0))))
            graph.set_series(current_series, graph.labels)

    def clear_live_state(self) -> None:
        self._set_live_status("Waiting for training/inference updates", "")


def open_visualizer(master: tk.Misc, runs_root: str | Path) -> VisualizerWindow:
    return VisualizerWindow(master, runs_root)
