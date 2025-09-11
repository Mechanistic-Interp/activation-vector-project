"""
Compare long vectors (raw vs centered) against pooled corpus mean per sample.

Inputs:
  --mean_path: Path to local corpus_mean_*.safetensors (downloaded from Modal)
  --raw_csv:  CSV produced by generate_vector_csv.py (long mode, raw)
  --centered_csv: CSV produced by generate_vector_csv.py --center (long mode, centered)

Token length resolution (choose one):
  --lengths_csv: Optional CSV with one header row matching the vector CSV labels and one row
                 of integer token counts per sample (after reversal). If provided, used directly.
  --use_text_samples: If set, resolves lengths by re-processing the standard 20 text samples
                      in src/training_data/text-samples via the deployed extractor.

Outputs:
  - Writes per-chunk metrics to outputs/diagnostics/compare_report.csv
    Columns: sample, doc_len, chunk, cosine, l2_diff, l2_diff_normA, l2_diff_normB

Run:
  modal run src/diagnostics/compare_vectors_with_mean.py \
    --mean_path outputs/corpus_mean_1757478514.safetensors \
    --raw_csv vectors_long_mode_raw.csv \
    --centered_csv vectors_long_mode_centered.csv \
    --use_text_samples
"""

import csv
import os
from typing import Dict, List, Optional

import modal
import numpy as np
import torch

from src.utils.centering import load_corpus_mean
from src.utils.diagnostics import compare_long_raw_centered_against_mean


Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

app = modal.App("compare-vectors-with-mean")


def _read_vectors_csv(path: str) -> (List[str], np.ndarray):
    """Read vector CSV where columns are samples and rows are dims.

    Returns: (labels, matrix) where matrix has shape [dims, samples].
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        labels = next(reader)
        cols = len(labels)
        rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            # pad/truncate to columns
            vals = [float(x) if x != "" else np.nan for x in row[:cols]]
            rows.append(vals)
    mat = np.array(rows, dtype=np.float32)
    return labels, mat


def _read_lengths_csv(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        labels = next(reader)
        vals = next(reader)
    return {lab: int(val) for lab, val in zip(labels, vals)}


def _resolve_lengths_via_text_samples(labels: List[str]) -> Dict[str, int]:
    """Resolve token lengths for the 20 standard samples by reprocessing texts.

    Assumes the label order used by generate_vector_csv.py:
      [long_1..long_10, short_1..short_10]
    and corresponding files in src/training_data/text-samples.
    """
    # Load files
    base = "src/training_data/text-samples"
    long_files = []
    short_files = []
    for i in range(1, 11):
        # We find files by pattern and choose the first match
        import glob

        lf = glob.glob(os.path.join(base, f"{i:02d}_*_long.txt"))
        sf = glob.glob(os.path.join(base, f"{i:02d}_*_short.txt"))
        long_files.append(lf[0] if lf else None)
        short_files.append(sf[0] if sf else None)

    extractor = Pythia12BExtractor()
    lengths: Dict[str, int] = {}

    # long_1..long_10
    for i, fp in enumerate(long_files, 1):
        lab = f"long_{i}"
        if fp is None:
            lengths[lab] = 0
            continue
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        mat = extractor.get_activation_matrix.remote(text=text)
        lengths[lab] = int(mat["shape"][1])

    # short_1..short_10
    for i, fp in enumerate(short_files, 1):
        lab = f"short_{i}"
        if fp is None:
            lengths[lab] = 0
            continue
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        mat = extractor.get_activation_matrix.remote(text=text)
        lengths[lab] = int(mat["shape"][1])

    # Filter to requested labels
    return {lab: lengths.get(lab, 0) for lab in labels}


@app.local_entrypoint()
def main(
    mean_path: str,
    raw_csv: str,
    centered_csv: str,
    lengths_csv: str = None,
    use_text_samples: bool = False,
    report_out: str = os.path.join("outputs", "diagnostics", "compare_report.csv"),
):
    # Read CSVs
    labels_raw, mat_raw = _read_vectors_csv(raw_csv)
    labels_cen, mat_cen = _read_vectors_csv(centered_csv)

    if labels_raw != labels_cen:
        raise ValueError("Raw and centered CSV headers do not match.")
    labels = labels_raw

    dims, n_samples = mat_raw.shape
    if dims % 5120 != 0:
        raise ValueError(f"Vector dims {dims} not divisible by 5120; expected long-mode CSV.")
    d_model = 5120
    if dims != 4 * d_model:
        print(f"⚠️ Expected 20480 dims, found {dims}. Proceeding but results assume long-mode.")

    # Resolve lengths
    if lengths_csv:
        lengths_map = _read_lengths_csv(lengths_csv)
    elif use_text_samples:
        lengths_map = _resolve_lengths_via_text_samples(labels)
    else:
        raise ValueError("Provide --lengths_csv or --use_text_samples to resolve token lengths.")

    # Load corpus mean
    cm_t = load_corpus_mean(mean_path)
    cm_np = cm_t.to(torch.float32).cpu().numpy()

    # Prepare output directory
    os.makedirs(os.path.dirname(report_out), exist_ok=True)

    # Compute per-sample comparisons
    rows_out: List[List[str]] = [[
        "sample", "doc_len", "chunk", "cosine", "l2_diff", "||diff||", "||pred||",
    ]]

    for j, lab in enumerate(labels):
        L = int(lengths_map.get(lab, 0))
        if L <= 0:
            print(f"⚠️ Skipping {lab}: missing token length")
            continue

        raw_vec = mat_raw[:, j]
        cen_vec = mat_cen[:, j]

        res = compare_long_raw_centered_against_mean(raw_vec, cen_vec, L, cm_np)
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

    # Write report
    with open(report_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    print(f"✅ Wrote comparison report to {report_out}")

