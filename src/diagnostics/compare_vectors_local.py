"""
Compare (raw_long - centered_long) against pooled corpus mean per sample (local only).

Inputs:
  --mean-path: Path to local corpus_mean_*.safetensors
  --raw-csv:   Long-mode vectors CSV (raw) from generate_vector_csv.py
  --centered-csv: Long-mode vectors CSV (centered)

Token lengths (choose one):
  --lengths-csv: CSV with a header matching the vector CSV labels and one row of
                 token lengths (ints) per sample (after reversal; same count either way).
  --text-dir:    Directory containing the 20 standard text samples (defaults to
                 src/training_data/text-samples). Lengths are computed locally via
                 HuggingFace tokenizer EleutherAI/pythia-12b. This matches the
                 extractor behavior (prepend BOS).

Output:
  Writes per-chunk metrics to outputs/diagnostics/compare_report_local.csv

Run examples:
  python -m src.diagnostics.compare_vectors_local \
      --mean-path outputs/corpus_mean_1757478514.safetensors \
      --raw-csv vectors_long_mode_raw.csv \
      --centered-csv vectors_long_mode_centered.csv \
      --text-dir src/training_data/text-samples

  python -m src.diagnostics.compare_vectors_local \
      --mean-path outputs/corpus_mean_1757478514.safetensors \
      --raw-csv vectors_long_mode_raw.csv \
      --centered-csv vectors_long_mode_centered.csv \
      --lengths-csv token_lengths.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.utils.centering import load_corpus_mean
from src.utils.diagnostics import compare_long_raw_centered_against_mean


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


def _read_lengths_csv(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        labels = next(reader)
        vals = next(reader)
    return {lab: int(val) for lab, val in zip(labels, vals)}


def _compute_lengths_from_texts(labels: List[str], text_dir: str) -> Dict[str, int]:
    """Compute token lengths for the 20 standard samples using HF tokenizer.

    This expects labels in the format produced by generate_vector_csv.py:
      [long_1..long_10, short_1..short_10], and text files matching
      {i:02d}_*_long.txt and {i:02d}_*_short.txt inside text_dir.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is required to compute token lengths from texts. "
            "Install it or provide --lengths-csv."
        ) from e

    long_files: List[str | None] = []
    short_files: List[str | None] = []
    for i in range(1, 11):
        lf = sorted(glob.glob(os.path.join(text_dir, f"{i:02d}_*_long.txt")))
        sf = sorted(glob.glob(os.path.join(text_dir, f"{i:02d}_*_short.txt")))
        long_files.append(lf[0] if lf else None)
        short_files.append(sf[0] if sf else None)

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")

    def text_len(path: str | None) -> int:
        if path is None:
            return 0
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # Match extractor behavior: BOS is prepended
        ids = tok(text, add_special_tokens=False).input_ids
        return len(ids) + 1

    # Build map in provided labels order
    lengths: Dict[str, int] = {}
    # long_1..long_10
    for i in range(1, 11):
        lab = f"long_{i}"
        if lab in labels:
            lengths[lab] = text_len(long_files[i - 1])
    # short_1..short_10
    for i in range(1, 11):
        lab = f"short_{i}"
        if lab in labels:
            lengths[lab] = text_len(short_files[i - 1])

    return lengths


def main():
    ap = argparse.ArgumentParser(description="Compare (raw - centered) against pooled corpus mean for long vectors (local)")
    # Accept kebab-case flags, with underscore aliases for backward compatibility
    ap.add_argument("--mean-path", "--mean_path", dest="mean_path", required=True, help="Path to corpus_mean_*.safetensors")
    ap.add_argument("--raw-csv", "--raw_csv", dest="raw_csv", required=True, help="Path to long-mode raw vectors CSV")
    ap.add_argument("--centered-csv", "--centered_csv", dest="centered_csv", required=True, help="Path to long-mode centered vectors CSV")
    ap.add_argument("--lengths-csv", "--lengths_csv", dest="lengths_csv", help="CSV with token lengths per sample (header + 1 row of ints)")
    ap.add_argument("--text-dir", "--text_dir", dest="text_dir", default="src/training_data/text-samples", help="Directory of text samples to infer lengths (if lengths_csv not provided)")
    ap.add_argument("--out", default=os.path.join("outputs", "diagnostics", "compare_report_local.csv"), help="Output CSV path for the report")
    args = ap.parse_args()

    # Load vectors
    labels_raw, mat_raw = _read_vectors_csv(args.raw_csv)
    labels_cen, mat_cen = _read_vectors_csv(args.centered_csv)
    if labels_raw != labels_cen:
        raise ValueError("Raw and centered CSV headers do not match")
    labels = labels_raw

    dims, n = mat_raw.shape
    if dims % 5120 != 0:
        raise ValueError(f"Vector dims {dims} not divisible by 5120; expected long-mode CSV")
    d_model = 5120
    if dims != 4 * d_model:
        print(f"⚠️ Expected 20480 dims, found {dims}. Proceeding; results assume long-mode")

    # Resolve lengths
    if args.lengths_csv:
        lengths_map = _read_lengths_csv(args.lengths_csv)
    else:
        lengths_map = _compute_lengths_from_texts(labels, args.text_dir)

    # Load corpus mean locally
    cm = load_corpus_mean(args.mean_path).to(torch.float32).cpu().numpy()

    # Prepare output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows_out: List[List[str]] = [[
        "sample", "doc_len", "chunk", "cosine", "l2_diff", "||diff||", "||pred||",
    ]]

    # Compare per sample
    for j, lab in enumerate(labels):
        L = int(lengths_map.get(lab, 0))
        if L <= 0:
            print(f"⚠️ Skipping {lab}: missing token length")
            continue
        raw_vec = mat_raw[:, j]
        cen_vec = mat_cen[:, j]
        res = compare_long_raw_centered_against_mean(raw_vec, cen_vec, L, cm)
        for chunk, m in res["chunks"].items():
            rows_out.append([
                lab,
                str(L),
                chunk,
                f"{m['cosine']:.6f}",
                f"{m['l2_diff']:.6f}",
                f"{m['l2_a']:.6f}",
                f"{m['l2_b']:.6f}",
            ])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_out)
    print(f"✅ Wrote report to {args.out}")


if __name__ == "__main__":
    main()
