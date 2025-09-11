"""
Plot cosine similarity matrices for vectors CSVs produced by generate_vector_csv.py.

Supports both SHORT (5120 dims) and LONG (20480 dims) vectors. For LONG, can
optionally plot per-chunk heatmaps: [last, exp_977, exp_933, exp_841].

Usage examples (run from repo root):
  python -m src.diagnostics.plot_similarity_matrix \
      --csv vectors_long_mode_centered.csv \
      --mode long \
      --per-chunk true

  python -m src.diagnostics.plot_similarity_matrix \
      --csv vectors_short_mode_raw.csv \
      --mode short

Outputs PNGs under outputs/diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np


def _read_vectors_csv(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        labels = next(reader)
        cols = len(labels)
        rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            vals = [float(x) if x != "" else np.nan for x in row[:cols]]
            rows.append(vals)
    mat = np.array(rows, dtype=np.float32)  # [dims, samples]
    return labels, mat


def _cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for column vectors in X (dims × samples)."""
    # Replace NaNs (if any) with 0 to avoid propagation; vectors should be full
    X = np.nan_to_num(X, nan=0.0)
    # Normalize columns
    norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
    Xn = X / norms
    # Cosine sim = Xn^T Xn
    return Xn.T @ Xn


def _plot_heatmap(M: np.ndarray, labels: List[str], title: str, out_path: str):
    import matplotlib
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.6)))
    im = plt.imshow(M, cmap="viridis", vmin=-1.0, vmax=1.0)

    # Title in black
    plt.title(title, color="black")

    # Colorbar with black text
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label="cosine")
    cbar.ax.yaxis.set_tick_params(color="black")
    for t in cbar.ax.get_yticklabels():
        t.set_color("black")
    cbar.set_label("cosine", color="black")

    # Ticks and tick labels in black
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    ax = plt.gca()
    ax.tick_params(axis="both", colors="black")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color("black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_heatmap_with_numbers(
    M: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    annot: bool = True,
    decimals: int = 2,
):
    import matplotlib
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_w = max(8, len(labels) * 0.6)
    fig_h = max(6, len(labels) * 0.6)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(M, cmap="viridis", vmin=-1.0, vmax=1.0)
    
    # Title in black
    plt.title(title, color="black")

    # Colorbar with black text
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label="cosine")
    cbar.ax.yaxis.set_tick_params(color="black")
    for t in cbar.ax.get_yticklabels():
        t.set_color("black")
    cbar.set_label("cosine", color="black")

    # Ticks and tick labels in black
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    ax = plt.gca()
    ax.tick_params(axis="both", colors="black")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color("black")

    if annot:
        fmt = f"{{:.{decimals}f}}"
        fs = 7 if len(labels) > 16 else 8
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = float(M[i, j])
                # Always use black text for annotations
                plt.text(j, i, fmt.format(val), ha="center", va="center", color="black", fontsize=fs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot cosine similarity matrix from vectors CSV")
    ap.add_argument("--csv", required=True, help="Path to vectors CSV (short or long mode)")
    ap.add_argument("--mode", choices=["short", "long"], required=True, help="Vector mode")
    ap.add_argument("--per-chunk", type=str, default="false", help="For long mode: plot per-chunk heatmaps (true/false)")
    ap.add_argument("--annot", type=str, default="true", help="Overlay numeric values on heatmap (true/false)")
    ap.add_argument("--decimals", type=int, default=2, help="Decimal places for annotations")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "diagnostics"), help="Output directory for PNGs")
    ap.add_argument("--title", default=None, help="Optional plot title override")
    args = ap.parse_args()

    per_chunk = args.per_chunk.lower() in {"1", "true", "yes", "y"}
    annot = args.annot.lower() in {"1", "true", "yes", "y"}

    labels, mat = _read_vectors_csv(args.csv)
    dims, n = mat.shape

    if args.mode == "short":
        if dims != 5120:
            print(f"⚠️ Expected 5120 rows for short mode, found {dims}. Proceeding.")
        M = _cosine_matrix(mat)
        title = args.title or f"Cosine Similarity – SHORT ({os.path.basename(args.csv)})"
        out = os.path.join(args.out_dir, f"similarity_short_{os.path.splitext(os.path.basename(args.csv))[0]}.png")
        _plot_heatmap_with_numbers(M, labels, title, out, annot=annot, decimals=args.decimals)
        print(f"✅ Wrote {out}")
        return

    # long mode
    d_model = 5120
    if dims % d_model != 0:
        raise ValueError(f"Vector dims {dims} not divisible by 5120; not a long-mode CSV")
    if dims != 4 * d_model:
        print(f"⚠️ Expected 20480 rows for long mode, found {dims}. Proceeding.")

    if not per_chunk:
        M = _cosine_matrix(mat)
        title = args.title or f"Cosine Similarity – LONG (all chunks) ({os.path.basename(args.csv)})"
        out = os.path.join(args.out_dir, f"similarity_long_all_{os.path.splitext(os.path.basename(args.csv))[0]}.png")
        _plot_heatmap_with_numbers(M, labels, title, out, annot=annot, decimals=args.decimals)
        print(f"✅ Wrote {out}")
        return

    # Per-chunk heatmaps
    chunks = {
        "last": mat[:d_model, :],
        "exp_977": mat[d_model : 2 * d_model, :],
        "exp_933": mat[2 * d_model : 3 * d_model, :],
        "exp_841": mat[3 * d_model : 4 * d_model, :],
    }
    for name, X in chunks.items():
        M = _cosine_matrix(X)
        title = args.title or f"Cosine Similarity – LONG[{name}] ({os.path.basename(args.csv)})"
        out = os.path.join(args.out_dir, f"similarity_long_{name}_{os.path.splitext(os.path.basename(args.csv))[0]}.png")
        _plot_heatmap_with_numbers(M, labels, title, out, annot=annot, decimals=args.decimals)
        print(f"✅ Wrote {out}")


if __name__ == "__main__":
    main()
