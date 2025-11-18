from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_class_names(data_yaml: Path) -> Dict[int, str]:
    data = yaml.safe_load(Path(data_yaml).read_text())
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def load_labels(label_path: Path) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            records.append(
                {
                    "class_id": int(float(cls)),
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": float(w),
                    "height": float(h),
                }
            )
    return pd.DataFrame.from_records(records)


def yolo_to_xyxy(df: pd.DataFrame, img_shape: Tuple[int, int, int]) -> pd.DataFrame:
    h, w = img_shape[:2]
    out = df.copy()
    out["x1"] = (df["x_center"] - df["width"] / 2) * w
    out["x2"] = (df["x_center"] + df["width"] / 2) * w
    out["y1"] = (df["y_center"] - df["height"] / 2) * h
    out["y2"] = (df["y_center"] + df["height"] / 2) * h
    return out


def plot_image_with_labels(
    image_path: Path,
    label_path: Path,
    class_names: Dict[int, str],
    focus_class: Optional[int] = None,
    draw_classes: Optional[Iterable[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    df = load_labels(label_path)
    df_xyxy = yolo_to_xyxy(df, img_rgb.shape)

    if ax is None:
        h, w = img_rgb.shape[:2]
        dpi = 100
        fig_w, fig_h = w / dpi, h / dpi
        _, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img_rgb, interpolation="nearest")
    ax.axis("off")

    for _, row in df_xyxy.iterrows():
        cls = int(row["class_id"])
        if draw_classes is not None and cls not in draw_classes:
            continue
        color = "lime" if focus_class is None or cls != focus_class else "red"
        rect = plt.Rectangle(
            (row["x1"], row["y1"]),
            row["x2"] - row["x1"],
            row["y2"] - row["y1"],
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    if title:
        ax.set_title(title)
    return ax


def _file_has_class(label_path: Path, class_id: int) -> bool:
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if cls == class_id:
                    return True
    except OSError:
        return False
    return False


def _resolve_image_path(label_path: Path, image_dir: Path) -> Optional[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for ext in exts:
        candidate = image_dir / (label_path.stem + ext)
        if candidate.exists():
            return candidate
    for candidate in image_dir.glob(f"{label_path.stem}.*"):
        if candidate.is_file():
            return candidate
    return None


def sample_label_files(
    label_dir: Path,
    n: int = 1,
    seed: Optional[int] = None,
    require_class: Optional[int] = None,
    prefix: Optional[str] = None,
) -> List[Path]:
    label_paths = list(Path(label_dir).glob("*.txt"))
    if prefix:
        label_paths = [p for p in label_paths if p.name.startswith(prefix)]
    if require_class is not None:
        label_paths = [p for p in label_paths if _file_has_class(p, require_class)]
    if seed is not None:
        random.seed(seed)
    if n >= len(label_paths):
        return label_paths
    return random.sample(label_paths, n)


def show_samples(
    label_dir: Path,
    image_dir: Path,
    class_names: Dict[int, str],
    n: int = 1,
    seed: Optional[int] = None,
    focus_class: Optional[int] = None,
    draw_classes: Optional[Iterable[int]] = None,
    require_class: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    samples = sample_label_files(
        label_dir, n=n, seed=seed, require_class=require_class, prefix=prefix
    )
    if not samples:
        print(
            f"No label files found in {label_dir} matching class {require_class}"
            if require_class is not None
            else f"No label files found in {label_dir}"
        )
        return

    cols = 1 if len(samples) <= 1 else 3
    rows = max(1, (len(samples) + cols - 1) // cols)
    base_w, base_h = 12, 8 
    
    if len(samples) == 1:
        label_path = samples[0]
        image_path = _resolve_image_path(label_path, image_dir)
        if image_path is None:
            fig, ax = plt.subplots(figsize=(base_w, base_h), dpi=200)
            ax.set_axis_off()
            ax.set_title(f"Missing image for {label_path.name}")
            plt.tight_layout()
            plt.show()
            return
        img = cv2.imread(str(image_path))
        if img is None:
            fig, ax = plt.subplots(figsize=(base_w, base_h), dpi=200)
            ax.set_axis_off()
            ax.set_title(f"Could not read image for {label_path.name}")
            plt.tight_layout()
            plt.show()
            return
        h, w = img.shape[:2]
        dpi = 100
        fig_w, fig_h = w / dpi, h / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        plot_image_with_labels(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
            focus_class=focus_class,
            draw_classes=draw_classes,
            ax=ax,
            title=label_path.stem,
        )
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(
        rows, cols, figsize=(base_w * cols, base_h * rows), dpi=200
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, label_path in zip(axes, samples):
        image_path = _resolve_image_path(label_path, image_dir)
        if image_path is None:
            ax.set_axis_off()
            ax.set_title(f"Missing image for {label_path.name}")
            continue
        plot_image_with_labels(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
            focus_class=focus_class,
            draw_classes=draw_classes,
            ax=ax,
            title=label_path.stem,
        )

    for ax in axes[len(samples) :]:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
