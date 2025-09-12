"""
Vector Diagnostics and Visualization

Generates visual diagnostics for activation vectors to validate centering and
long-vector composition. Saves plots under `outputs/diagnostics`.

Usage examples:
  modal run -m src.diagnostics.visualize_vectors --text "Your text" --center --mode long
  modal run -m src.diagnostics.visualize_vectors --file path/to/text.txt --center --mode long

Behavior:
  - Resolves latest corpus_mean_*.safetensors from the mounted Modal volume
    (activation-vector-project) when --center is passed.
  - Extracts raw and centered vectors (short/long) and the activation matrix
    (to determine effective token length) for the given text.
  - For long vectors, splits into 4 chunks: [last, 2.3%, 6.7%, 15.9%].
  - Computes the predicted mean contribution by applying the same pooling
    weights to the corpus mean slice (restricted to the doc token length).
  - Creates plots visualizing norms and alignment between (raw - centered)
    and the predicted mean contribution.
"""

import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import modal
from ..utils.volume_utils import find_latest_corpus_mean_path

# Modal references
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

app = modal.App("vector-diagnostics")

# Mount training data to resolve corpus mean file
training_volume = modal.Volume.from_name(
    "training_data", create_if_missing=True
)


@app.function(volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


@app.function(volumes={"/training_data": training_volume}, timeout=120)
def load_corpus_mean_slice(cm_path: str, length: int) -> List[List[float]]:
    import torch
    from src.utils.centering import load_corpus_mean as _load

    cm = _load(cm_path)
    length = min(length, cm.shape[1])
    cm_slice = cm[:, :length].to(torch.float32).cpu().numpy().tolist()
    return cm_slice


def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except Exception:
        print("⚠️ matplotlib is not available. Install it to produce plots.")
        return False


def _exp_weights(length: int, depth_fraction: float) -> np.ndarray:
    # positions: 0 = last token (reversed order)
    positions = np.arange(length, dtype=np.float32)
    decay_rate = np.log(2.0) / (depth_fraction * float(length))
    w = np.exp(-decay_rate * positions)
    w /= max(w.sum(), 1e-6)
    return w


def _compute_predicted_mean_chunks(
    corpus_mean: np.ndarray,  # [d_model, max_tokens]
    doc_len: int,
) -> Dict[str, np.ndarray]:
    # Slice to effective length
    cm_slice = corpus_mean[:, :doc_len]  # [d_model, L]

    # Weights per long-chunk
    w_last = np.zeros((doc_len,), dtype=np.float32)
    w_last[0] = 1.0
    w_977 = _exp_weights(doc_len, 0.023)
    w_933 = _exp_weights(doc_len, 0.067)
    w_841 = _exp_weights(doc_len, 0.159)

    return {
        "last": cm_slice @ w_last,  # [d_model]
        "exp_977": cm_slice @ w_977,
        "exp_933": cm_slice @ w_933,
        "exp_841": cm_slice @ w_841,
    }


def _split_long_chunks(vec: np.ndarray, d_model: int) -> Dict[str, np.ndarray]:
    assert vec.size == 4 * d_model, f"Expected 4*d_model vector, got {vec.size}"
    return {
        "last": vec[:d_model],
        "exp_977": vec[d_model : 2 * d_model],
        "exp_933": vec[2 * d_model : 3 * d_model],
        "exp_841": vec[3 * d_model : 4 * d_model],
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _plot_diagnostics(
    out_dir: str,
    base_name: str,
    d_model: int,
    raw_long: Optional[np.ndarray],
    centered_long: Optional[np.ndarray],
    predicted_mean_long: Optional[Dict[str, np.ndarray]],
):
    has_mpl = _ensure_matplotlib()
    if not has_mpl:
        return

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: Chunk norms (raw, centered, diff)
    if raw_long is not None and centered_long is not None:
        raw_chunks = _split_long_chunks(raw_long, d_model)
        cen_chunks = _split_long_chunks(centered_long, d_model)
        diff_chunks = {k: raw_chunks[k] - cen_chunks[k] for k in raw_chunks}

        labels = ["last", "exp_977", "exp_933", "exp_841"]
        raw_norms = [np.linalg.norm(raw_chunks[k]) for k in labels]
        cen_norms = [np.linalg.norm(cen_chunks[k]) for k in labels]
        diff_norms = [np.linalg.norm(diff_chunks[k]) for k in labels]

        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(10, 4))
        plt.bar(x - width, raw_norms, width=width, label="raw")
        plt.bar(x, cen_norms, width=width, label="centered")
        plt.bar(x + width, diff_norms, width=width, label="raw - centered")
        plt.xticks(x, labels)
        plt.ylabel("L2 norm")
        plt.title("Long vector chunk norms")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_chunk_norms.png"), dpi=150)
        plt.close()

        # Figure 2: Alignment between diff and predicted mean per chunk
        if predicted_mean_long is not None:
            plt.figure(figsize=(10, 6))
            for i, k in enumerate(labels, 1):
                plt.subplot(2, 2, i)
                diff = diff_chunks[k]
                pm = predicted_mean_long[k]
                cos = _cosine(diff, pm)
                # Plot first 200 dims for a compact overlay
                n = min(200, diff.shape[0])
                plt.plot(diff[:n], label="raw - centered", lw=0.9)
                plt.plot(pm[:n], label="pred mean", lw=0.9)
                plt.title(f"{k} | cos={cos:.3f}")
                plt.grid(True, alpha=0.2)
                if i == 1:
                    plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{base_name}_alignment.png"), dpi=150)
            plt.close()


def _load_text_arg(text: Optional[str], file: Optional[str]) -> str:
    if file:
        with open(file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return text or ""


@app.local_entrypoint()
def main(
    text: str = "The capital of France is Paris.",
    file: str = None,
    mode: str = "long",
    center: bool = True,
):
    """Run diagnostics for a single input text and save plots.

    Args:
        text: direct text input (ignored if file is provided)
        file: path to text file
        mode: "short" or "long" (diagnostics focus on long)
        center: whether to subtract corpus mean (requires volume with mean files)
    """
    # Prepare IO
    content = _load_text_arg(text, file)
    if not content:
        raise ValueError("No text provided (empty). Pass --text or --file.")

    out_dir = os.path.join("outputs", "diagnostics")
    base_name = ("file" if file else "text") + f"_{mode}"

    # Modal handles
    extractor = Pythia12BExtractor()

    # Resolve centering vector if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()
        print(f"📦 Using corpus mean: {centering_vector}")

    # Fetch raw and centered vectors
    print("🔬 Extracting vectors (raw/centered)...")
    raw = extractor.get_activation_vector.remote(
        text=content, pooling_strategy=mode, center=False
    )
    centered = extractor.get_activation_vector.remote(
        text=content,
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )

    d_model = raw["d_model"]
    raw_vec = np.array(raw["vector"], dtype=np.float32)
    centered_vec = np.array(centered["vector"], dtype=np.float32)

    predicted_mean_long = None

    if mode == "long":
        # Need token length to slice corpus mean correctly
        mat_info = extractor.get_activation_matrix.remote(text=content)
        doc_len = mat_info["shape"][1]

        if center and centering_vector is not None:
            cm_slice = load_corpus_mean_slice.remote(centering_vector, doc_len)
            cm_slice_np = np.array(cm_slice, dtype=np.float32)  # [d_model, L]
            predicted_mean_long = _compute_predicted_mean_chunks(
                cm_slice_np, cm_slice_np.shape[1]
            )

    # Produce plots
    if mode == "long":
        _plot_diagnostics(
            out_dir, base_name, d_model, raw_vec, centered_vec, predicted_mean_long
        )

    print(f"✅ Diagnostics saved under: {out_dir}")
