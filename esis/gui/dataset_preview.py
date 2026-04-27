from __future__ import annotations

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np

from esis.datasets import DatasetSample, ensure_dataset_index
from esis.segmentation import available_segmenters, create_segmenter
from esis.utils.config import default_dataset_roots, project_root, runs_root
from esis.utils.io import (
    as_bgr,
    colorize_mask,
    ensure_dir,
    get_video_frame_count,
    image_to_png_base64,
    read_image,
    read_video_frame,
    resize_to_fit,
    write_image,
)


class DatasetPreviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ESIS Dataset Segmentation Viewer")
        self.root.geometry("1800x950")

        self.dataset_var = tk.StringVar(value="endovis17")
        self.sample_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.IntVar(value=0)
        self.segmenter_var = tk.StringVar(value="mask_loader")
        self.backend_status_var = tk.StringVar(value="Backend status: not loaded")

        self.project_root = project_root()
        self.dataset_roots = default_dataset_roots(self.project_root)
        self.dataset_indexes: dict[str, object] = {}
        self.dataset_samples: dict[str, list[DatasetSample]] = {}
        self.selected_sample: DatasetSample | None = None
        self.segmenters: dict[str, object] = {}
        self.segmenter_errors: dict[str, str] = {}
        self.current_raw_photo: tk.PhotoImage | None = None
        self.current_label_photo: tk.PhotoImage | None = None
        self.current_segmentation_photo: tk.PhotoImage | None = None
        self.current_raw_image: np.ndarray | None = None
        self.current_label_image: np.ndarray | None = None
        self.current_segmentation_image: np.ndarray | None = None
        self.current_segmentation_result: object | None = None

        self._build_layout()
        self._load_dataset(self.dataset_var.get())

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(container)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(top_bar, text="Dataset").pack(side=tk.LEFT)
        dataset_box = ttk.Combobox(
            top_bar,
            textvariable=self.dataset_var,
            values=sorted(self.dataset_roots),
            state="readonly",
            width=16,
        )
        dataset_box.pack(side=tk.LEFT, padx=(8, 16))
        dataset_box.bind("<<ComboboxSelected>>", self._on_dataset_changed)
        ttk.Button(top_bar, text="Reload Index", command=self._reload_current_dataset).pack(side=tk.LEFT)
        ttk.Label(top_bar, text="Segmenter").pack(side=tk.LEFT, padx=(16, 4))
        segmenter_box = ttk.Combobox(
            top_bar,
            textvariable=self.segmenter_var,
            values=available_segmenters(),
            state="readonly",
            width=20,
        )
        segmenter_box.pack(side=tk.LEFT)
        segmenter_box.bind("<<ComboboxSelected>>", self._on_segmenter_changed)
        ttk.Button(top_bar, text="Export Overlay", command=self._export_current_overlay).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(top_bar, textvariable=self.status_var).pack(side=tk.RIGHT)

        left_panel = ttk.Frame(container)
        left_panel.grid(row=1, column=0, sticky="nsw", padx=(0, 10))
        left_panel.rowconfigure(1, weight=1)
        ttk.Label(left_panel, text="Samples").grid(row=0, column=0, sticky="w")

        self.sample_listbox = tk.Listbox(left_panel, width=48, exportselection=False)
        self.sample_listbox.grid(row=1, column=0, sticky="nsw")
        self.sample_listbox.bind("<<ListboxSelect>>", self._on_sample_selected)

        list_scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.sample_listbox.yview)
        list_scrollbar.grid(row=1, column=1, sticky="ns")
        self.sample_listbox.configure(yscrollcommand=list_scrollbar.set)

        right_panel = ttk.Frame(container)
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.columnconfigure(1, weight=1)
        right_panel.columnconfigure(2, weight=1)
        right_panel.rowconfigure(3, weight=1)

        ttk.Label(right_panel, text="Selected Sample").grid(row=0, column=0, columnspan=3, sticky="w")
        self.meta_text = tk.Text(right_panel, height=6, wrap="word")
        self.meta_text.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        ttk.Label(right_panel, textvariable=self.backend_status_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 8))

        preview_frame = ttk.Frame(right_panel)
        preview_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(2, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, text="Raw Image").grid(row=0, column=0, sticky="w")
        ttk.Label(preview_frame, text="Label Image").grid(row=0, column=1, sticky="w")
        ttk.Label(preview_frame, text="Segmentation Result").grid(row=0, column=2, sticky="w")

        self.raw_label = ttk.Label(preview_frame, anchor="center", relief="solid")
        self.raw_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        self.label_label = ttk.Label(preview_frame, anchor="center", relief="solid")
        self.label_label.grid(row=1, column=1, sticky="nsew", padx=6)

        self.segmentation_label = ttk.Label(preview_frame, anchor="center", relief="solid")
        self.segmentation_label.grid(row=1, column=2, sticky="nsew", padx=(6, 0))

        slider_frame = ttk.Frame(right_panel)
        slider_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        slider_frame.columnconfigure(1, weight=1)
        ttk.Label(slider_frame, text="Frame Index").grid(row=0, column=0, sticky="w")
        self.frame_scale = tk.Scale(
            slider_frame,
            orient=tk.HORIZONTAL,
            from_=0,
            to=0,
            variable=self.frame_var,
            command=self._on_frame_changed,
            showvalue=True,
        )
        self.frame_scale.grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _on_dataset_changed(self, _event: object | None = None) -> None:
        self._load_dataset(self.dataset_var.get())

    def _reload_current_dataset(self) -> None:
        self._load_dataset(self.dataset_var.get(), rebuild=True)

    def _on_segmenter_changed(self, _event: object | None = None) -> None:
        if self.selected_sample is not None:
            self._set_selected_sample(self.selected_sample)

    def _load_dataset(self, dataset_name: str, rebuild: bool = False) -> None:
        index = ensure_dataset_index(dataset_name, rebuild=rebuild, root=self.project_root)
        samples = sorted(index.samples, key=lambda sample: sample.sample_id.lower())
        self.dataset_indexes[dataset_name] = index
        self.dataset_samples[dataset_name] = samples

        self.sample_listbox.delete(0, tk.END)
        for sample in samples:
            self.sample_listbox.insert(tk.END, sample.sample_id)

        self.status_var.set(f"{dataset_name}: {len(samples)} samples")
        self.selected_sample = None
        self.meta_text.delete("1.0", tk.END)
        self.backend_status_var.set("Backend status: not loaded")
        self._clear_preview()

        if samples:
            self.sample_listbox.selection_set(0)
            self.sample_listbox.activate(0)
            self._set_selected_sample(samples[0])

    def _on_sample_selected(self, _event: object | None = None) -> None:
        selection = self.sample_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        samples = self.dataset_samples.get(self.dataset_var.get(), [])
        if 0 <= index < len(samples):
            self._set_selected_sample(samples[index])

    def _set_selected_sample(self, sample: DatasetSample) -> None:
        self.selected_sample = sample
        self._update_meta(sample)
        if sample.modality == "image_frame":
            self.frame_scale.configure(state=tk.DISABLED, from_=0, to=0)
            self.frame_var.set(0)
            self._render_image_sample(sample)
            return

        if sample.video_path and Path(sample.video_path).exists() and str(sample.video_path).lower().endswith(".avi"):
            frame_count = get_video_frame_count(sample.video_path)
            max_index = max(frame_count - 1, 0)
            self.frame_scale.configure(state=tk.NORMAL, from_=0, to=max_index)
            self.frame_var.set(0)
            self._render_video_frame(sample, 0)
            return

        self.frame_scale.configure(state=tk.DISABLED, from_=0, to=0)
        self.frame_var.set(0)
        self._render_empty("Raw image unavailable", "Label unavailable")

    def _on_frame_changed(self, value: str) -> None:
        if self.selected_sample is None:
            return
        if self.selected_sample.modality == "video_sequence":
            self._render_video_frame(self.selected_sample, int(float(value)))

    def _update_meta(self, sample: DatasetSample) -> None:
        backend_info = self._describe_current_backend()
        lines = [
            f"dataset: {sample.dataset_name}",
            f"sample_id: {sample.sample_id}",
            f"sequence_id: {sample.sequence_id}",
            f"split: {sample.split}",
            f"modality: {sample.modality}",
            f"segmenter: {backend_info['backend_name']}",
            f"backend_status: {backend_info['status']}",
            f"model_name: {backend_info['model_name']}",
            f"checkpoint_path: {backend_info['checkpoint_path']}",
            f"device: {backend_info['device']}",
            f"image_path: {sample.image_path or '-'}",
            f"label_path: {sample.label_path or '-'}",
            f"video_path: {sample.video_path or '-'}",
        ]
        if sample.metadata:
            lines.append(f"metadata: {sample.metadata}")
        if backend_info["error"]:
            lines.append(f"backend_error: {backend_info['error']}")
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert("1.0", "\n".join(lines))
        self.backend_status_var.set(
            f"Backend status: {backend_info['status']} | model={backend_info['model_name']} | checkpoint_loaded={backend_info['checkpoint_loaded']} | device={backend_info['device']}"
        )

    def _render_image_sample(self, sample: DatasetSample) -> None:
        raw = self._load_raw_image(sample)
        label = self._load_label_image(sample)
        segmentation = self._run_segmenter(raw, sample)
        self._set_preview_images(raw, label, segmentation)

    def _render_video_frame(self, sample: DatasetSample, frame_index: int) -> None:
        if not sample.video_path:
            self._render_empty("Video unavailable", "Label unavailable", "Segmentation unavailable")
            return
        raw = read_video_frame(sample.video_path, frame_index)
        label = self._load_frame_label(sample, frame_index)
        segmentation = self._run_segmenter(raw, sample)
        self._set_preview_images(raw, label, segmentation)
        self.status_var.set(f"{sample.sample_id} frame {frame_index}")

    def _load_raw_image(self, sample: DatasetSample) -> np.ndarray | None:
        if sample.image_path and Path(sample.image_path).exists():
            return read_image(sample.image_path)
        return None

    def _load_label_image(self, sample: DatasetSample) -> np.ndarray | None:
        if sample.label_path and Path(sample.label_path).exists():
            return colorize_mask(read_image(sample.label_path))
        return None

    def _load_frame_label(self, sample: DatasetSample, frame_index: int) -> np.ndarray | None:
        if sample.modality != "video_sequence":
            return self._load_label_image(sample)
        return None

    def _run_segmenter(self, raw: np.ndarray | None, sample: DatasetSample) -> np.ndarray | None:
        if raw is None:
            self.current_segmentation_result = None
            return None
        segmenter_name = self.segmenter_var.get()
        try:
            if segmenter_name not in self.segmenters:
                self.segmenters[segmenter_name] = create_segmenter(segmenter_name)
                self.segmenter_errors.pop(segmenter_name, None)
            result = self.segmenters[segmenter_name].segment(raw, sample)
            self.current_segmentation_result = result
            self._update_meta(sample)
        except Exception as exc:  # noqa: BLE001
            self.segmenter_errors[segmenter_name] = str(exc)
            self.current_segmentation_result = None
            self._update_meta(sample)
            return None
        return self._render_segmentation_overlay(raw, result.mask)

    def _render_segmentation_overlay(self, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raw_bgr = as_bgr(raw)
        mask_color = colorize_mask(mask)
        if mask_color.shape[:2] != raw_bgr.shape[:2]:
            mask_color = cv2.resize(
                mask_color,
                (raw_bgr.shape[1], raw_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        overlay = cv2.addWeighted(raw_bgr, 0.65, mask_color, 0.35, 0.0)
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        return overlay

    def _set_preview_images(
        self,
        raw: np.ndarray | None,
        label: np.ndarray | None,
        segmentation: np.ndarray | None,
    ) -> None:
        self.current_raw_image = None if raw is None else raw.copy()
        self.current_label_image = None if label is None else label.copy()
        self.current_segmentation_image = None if segmentation is None else segmentation.copy()
        raw_photo = self._to_photo(raw, fallback_text="Raw image unavailable")
        label_photo = self._to_photo(label, fallback_text="Label unavailable")
        segmentation_photo = self._to_photo(segmentation, fallback_text="Segmentation unavailable")
        self.current_raw_photo = raw_photo
        self.current_label_photo = label_photo
        self.current_segmentation_photo = segmentation_photo
        self.raw_label.configure(image=raw_photo, text="")
        self.label_label.configure(image=label_photo, text="")
        self.segmentation_label.configure(image=segmentation_photo, text="")

    def _render_empty(self, raw_text: str, label_text: str, segmentation_text: str = "Segmentation unavailable") -> None:
        self.current_raw_image = None
        self.current_label_image = None
        self.current_segmentation_image = None
        self.current_segmentation_result = None
        raw_photo = self._to_photo(None, fallback_text=raw_text)
        label_photo = self._to_photo(None, fallback_text=label_text)
        segmentation_photo = self._to_photo(None, fallback_text=segmentation_text)
        self.current_raw_photo = raw_photo
        self.current_label_photo = label_photo
        self.current_segmentation_photo = segmentation_photo
        self.raw_label.configure(image=raw_photo, text="")
        self.label_label.configure(image=label_photo, text="")
        self.segmentation_label.configure(image=segmentation_photo, text="")

    def _clear_preview(self) -> None:
        self._render_empty("No sample selected", "No sample selected", "No sample selected")

    def _describe_current_backend(self) -> dict[str, str]:
        backend_name = self.segmenter_var.get()
        segmenter = self.segmenters.get(backend_name)
        error = self.segmenter_errors.get(backend_name, "")
        checkpoint_loaded = getattr(segmenter, "checkpoint_loaded", None)
        if error:
            status = "load_failed"
        elif segmenter is None:
            status = "not_loaded"
        elif checkpoint_loaded is False:
            status = "checkpoint_missing"
        else:
            status = "loaded"

        model_name = "-"
        checkpoint_path = "-"
        device = "-"
        if segmenter is not None:
            model_name = self._read_backend_attr(segmenter, ("model_name", "name"), default=segmenter.__class__.__name__)
            checkpoint_path = self._read_backend_attr(
                segmenter,
                ("checkpoint_resolution.checkpoint_path", "config.checkpoint_path", "config.model_id"),
                default="-",
            )
            device = self._read_backend_attr(segmenter, ("device",), default="-")
            device = str(device)

        return {
            "backend_name": backend_name,
            "status": status,
            "model_name": str(model_name),
            "checkpoint_path": str(checkpoint_path or "-"),
            "device": str(device),
            "checkpoint_loaded": str(checkpoint_loaded) if checkpoint_loaded is not None else ("True" if segmenter is not None else "False"),
            "error": error,
        }

    def _read_backend_attr(self, obj: object, paths: tuple[str, ...], default: object = "-") -> object:
        for path in paths:
            current = obj
            ok = True
            for part in path.split("."):
                if not hasattr(current, part):
                    ok = False
                    break
                current = getattr(current, part)
            if ok and current is not None:
                return current
        return default

    def _export_current_overlay(self) -> None:
        if self.selected_sample is None or self.current_segmentation_image is None:
            messagebox.showwarning("Export Overlay", "There is no segmentation overlay to export.")
            return

        backend_name = self.segmenter_var.get()
        dataset_name = self.selected_sample.dataset_name
        sample_name = self._safe_name(self.selected_sample.sample_id)
        frame_suffix = ""
        if self.selected_sample.modality == "video_sequence":
            frame_suffix = f"_frame{self.frame_var.get():06d}"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = ensure_dir(
            runs_root(self.project_root) / "gui_exports" / backend_name / dataset_name / f"{sample_name}{frame_suffix}_{stamp}"
        )

        overlay_path = export_dir / "overlay.png"
        mask_path = export_dir / "mask.png"
        meta_path = export_dir / "meta.json"
        write_image(overlay_path, self.current_segmentation_image)

        payload = {
            "dataset_name": dataset_name,
            "backend_name": backend_name,
            "sample": self.selected_sample.to_dict(),
            "frame_index": int(self.frame_var.get()) if self.selected_sample.modality == "video_sequence" else None,
            "backend_info": self._describe_current_backend(),
            "overlay_path": str(overlay_path),
            "mask_path": None,
        }

        if self.current_segmentation_result is not None and hasattr(self.current_segmentation_result, "mask"):
            write_image(mask_path, self.current_segmentation_result.mask)
            payload["mask_path"] = str(mask_path)

        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_var.set(f"Overlay exported: {overlay_path}")
        messagebox.showinfo("Export Overlay", f"Saved overlay to:\n{overlay_path}")

    def _safe_name(self, text: str) -> str:
        return text.replace("/", "__").replace("\\", "__").replace(" ", "_").replace(":", "_")

    def _to_photo(self, image: np.ndarray | None, fallback_text: str) -> tk.PhotoImage:
        if image is None:
            canvas = np.full((540, 720, 3), 40, dtype=np.uint8)
            cv2.putText(canvas, fallback_text, (24, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
            image = canvas
        else:
            image = as_bgr(image)
            image = resize_to_fit(image, max_width=720, max_height=540)
        encoded = image_to_png_base64(image)
        return tk.PhotoImage(data=encoded)


def launch_dataset_preview_app() -> int:
    root = tk.Tk()
    app = DatasetPreviewApp(root)
    root.mainloop()
    return 0
