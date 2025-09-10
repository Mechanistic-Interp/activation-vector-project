"""
Centering utilities: load corpus mean and subtract with length handling.
"""

from typing import Tuple
import os
import torch
from safetensors.torch import load_file


def load_corpus_mean(file_path: str) -> torch.Tensor:
    """Load corpus mean from safetensors file.

    Args:
        file_path: Path to the corpus mean safetensors file

    Returns:
        Corpus mean tensor [5120, max_tokens]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus mean file not found: {file_path}")

    tensors = load_file(file_path)
    if "corpus_mean" not in tensors:
        raise KeyError(f"File {file_path} does not contain 'corpus_mean' tensor")

    return tensors["corpus_mean"]


def subtract_corpus_mean(activation_matrix: torch.Tensor, corpus_mean: torch.Tensor) -> torch.Tensor:
    """Subtract corpus mean from activation matrix with length handling.

    Implements the centering step:
    - If doc shorter than corpus mean: subtract only up to doc length
    - If doc longer than corpus mean: truncate to corpus mean length, then subtract

    Args:
        activation_matrix: Document activation matrix [5120, doc_tokens]
        corpus_mean: Precomputed corpus mean [5120, max_tokens]

    Returns:
        Centered activation matrix [5120, processed_tokens]
    """
    d_model, doc_tokens = activation_matrix.shape
    corpus_d_model, max_tokens = corpus_mean.shape

    assert d_model == corpus_d_model == 5120, f"d_model mismatch: doc={d_model}, corpus={corpus_d_model}"

    if doc_tokens <= max_tokens:
        corpus_mean_subset = corpus_mean[:, :doc_tokens]
        centered = activation_matrix - corpus_mean_subset
        return centered
    else:
        activation_truncated = activation_matrix[:, :max_tokens]
        centered = activation_truncated - corpus_mean
        return centered

