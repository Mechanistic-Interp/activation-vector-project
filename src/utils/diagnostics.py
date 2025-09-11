"""
Diagnostics utilities for validating long-vector centering identity.

Core identity (per document of token length L, after reversal):
  diff_long = raw_long - centered_long = pooled_mean_long
where pooled_mean_long is obtained by applying the same long-mode pooling
curves to the corpus mean matrix slice of length L.

This module provides functions to compute the predicted pooled mean from a
corpus mean matrix and compare it against (raw_long - centered_long).
"""

from typing import Dict, List, Any

import numpy as np


def _exp_weights(length: int, depth_fraction: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)
    decay_rate = np.log(2.0) / (depth_fraction * float(length))
    w = np.exp(-decay_rate * positions)
    s = float(w.sum()) if w.sum() > 0 else 1.0
    return (w / s).astype(np.float32)


def _compute_predicted_mean_chunks(corpus_mean: np.ndarray, doc_len: int) -> Dict[str, np.ndarray]:
    """Compute expected long-mode mean contribution from corpus mean slice.

    Args:
        corpus_mean: [d_model, max_tokens] mean matrix (reversed order)
        doc_len: effective token length for this document

    Returns:
        Dict with keys {"last", "exp_977", "exp_933", "exp_841"} each [d_model]
    """
    cm_slice = corpus_mean[:, :doc_len]
    w_last = np.zeros((doc_len,), dtype=np.float32); w_last[0] = 1.0
    w_977 = _exp_weights(doc_len, 0.023)
    w_933 = _exp_weights(doc_len, 0.067)
    w_841 = _exp_weights(doc_len, 0.159)
    return {
        "last": cm_slice @ w_last,
        "exp_977": cm_slice @ w_977,
        "exp_933": cm_slice @ w_933,
        "exp_841": cm_slice @ w_841,
    }


def _split_long_chunks(vec: np.ndarray, d_model: int) -> Dict[str, np.ndarray]:
    assert vec.size == 4 * d_model, f"Expected 4*d_model, got {vec.size}"
    return {
        "last": vec[:d_model],
        "exp_977": vec[d_model : 2 * d_model],
        "exp_933": vec[2 * d_model : 3 * d_model],
        "exp_841": vec[3 * d_model : 4 * d_model],
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_long_raw_centered_against_mean(
    raw_long: np.ndarray,
    centered_long: np.ndarray,
    doc_len: int,
    corpus_mean: np.ndarray,
) -> Dict[str, Any]:
    """Compare raw_long - centered_long to pooled corpus mean per chunk.

    Args:
        raw_long: [4*d_model] long vector from raw activations
        centered_long: [4*d_model] long vector after centering
        doc_len: effective token length for this document (after reversal)
        corpus_mean: [d_model, max_tokens] mean matrix in reversed order

    Returns:
        Dict with per-chunk cosine and L2 error metrics, plus norms.
    """
    raw = raw_long.astype(np.float32)
    cen = centered_long.astype(np.float32)
    d_model = raw.size // 4

    diff = raw - cen
    diff_chunks = _split_long_chunks(diff, d_model)

    pm = _compute_predicted_mean_chunks(corpus_mean, doc_len)

    results: Dict[str, Any] = {"d_model": d_model, "doc_len": doc_len, "chunks": {}}
    for k in ["last", "exp_977", "exp_933", "exp_841"]:
        a = diff_chunks[k]
        b = pm[k]
        results["chunks"][k] = {
            "cosine": _cosine(a, b),
            "l2_diff": float(np.linalg.norm(a - b)),
            "l2_a": float(np.linalg.norm(a)),
            "l2_b": float(np.linalg.norm(b)),
        }
    return results


def batch_compare_long_raw_centered_against_mean(
    raw_list: List[np.ndarray],
    centered_list: List[np.ndarray],
    lengths: List[int],
    corpus_mean: np.ndarray,
) -> List[Dict[str, Any]]:
    """Batch version: compare multiple documents.

    Args:
        raw_list: list of [4*d_model] raw long vectors
        centered_list: list of [4*d_model] centered long vectors
        lengths: list of effective token lengths per document
        corpus_mean: [d_model, max_tokens]

    Returns:
        List of per-document result dicts.
    """
    assert len(raw_list) == len(centered_list) == len(lengths)
    return [
        compare_long_raw_centered_against_mean(r, c, L, corpus_mean)
        for r, c, L in zip(raw_list, centered_list, lengths)
    ]


__all__ = [
    "compare_long_raw_centered_against_mean",
    "batch_compare_long_raw_centered_against_mean",
]

