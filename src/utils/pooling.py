"""
Pooling utilities for token-level activation matrices.

Only short/long strategies are exposed.
Input shape: [d_model, seq_len]
Output shape (short): [d_model]
Output shape (long): [4 * d_model]
"""

from typing import Literal, Optional, Dict, Any

import torch


def _as_float32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        return t.to(torch.float32)
    return t


def last_token_pool_tokens(activation_matrix: torch.Tensor) -> torch.Tensor:
    """Select last token vector in fp32.

    NOTE: activation_matrix is already reversed so that column 0 is the last token.

    activation_matrix: [d_model, seq_len]
    returns: [d_model]
    """
    act_f32 = _as_float32(activation_matrix)
    return act_f32[:, 0]


def exp_weight_841(activation_matrix: torch.Tensor) -> torch.Tensor:
    """ExpWeight at 84.1st percentile: depth ≈ 15.9% of doc length (σ = -1.0)."""
    act_f32 = _as_float32(activation_matrix)
    d_model, seq_len = act_f32.shape
    if seq_len == 1:
        return act_f32.squeeze(1)

    device = act_f32.device
    depth_fraction = 0.159
    decay_rate = torch.log(torch.tensor(2.0, dtype=torch.float32, device=device)) / (
        depth_fraction * float(seq_len)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    weights = torch.exp(-decay_rate * positions)
    weights = weights / weights.sum().clamp_min(1e-6)
    return act_f32 @ weights


def exp_weight_977(activation_matrix: torch.Tensor) -> torch.Tensor:
    """ExpWeight at 97.7th percentile: depth ≈ 2.3% of doc length (σ = -2.0)."""
    act_f32 = _as_float32(activation_matrix)
    d_model, seq_len = act_f32.shape
    if seq_len == 1:
        return act_f32.squeeze(1)

    device = act_f32.device
    depth_fraction = 0.023
    decay_rate = torch.log(torch.tensor(2.0, dtype=torch.float32, device=device)) / (
        depth_fraction * float(seq_len)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    weights = torch.exp(-decay_rate * positions)
    weights = weights / weights.sum().clamp_min(1e-6)
    return act_f32 @ weights


def exp_weight_933(activation_matrix: torch.Tensor) -> torch.Tensor:
    """ExpWeight at 93.3rd percentile: depth ≈ 6.7% of doc length (σ = -1.5)."""
    act_f32 = _as_float32(activation_matrix)
    d_model, seq_len = act_f32.shape
    if seq_len == 1:
        return act_f32.squeeze(1)

    device = act_f32.device
    depth_fraction = 0.067
    decay_rate = torch.log(torch.tensor(2.0, dtype=torch.float32, device=device)) / (
        depth_fraction * float(seq_len)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    weights = torch.exp(-decay_rate * positions)
    weights = weights / weights.sum().clamp_min(1e-6)
    return act_f32 @ weights


def pool_tokens(
    activation_matrix: torch.Tensor,
    *,
    strategy: Literal["short", "long"] = "short",
    params: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Dispatch only the supported strategies: "short" and "long".

    - "short": single d_model vector using ExpWeight-B (≈ 6.7% depth)
    - "long":  concat of [last_token, ExpWeight-A (≈ 2.3%), ExpWeight-B (≈ 6.7%), ExpWeight-C (≈ 15.9%)]
    """
    params = params or {}
    if strategy == "short":
        return exp_weight_933(activation_matrix)
    if strategy == "long":
        last_token = last_token_pool_tokens(activation_matrix)
        exp_977 = exp_weight_977(activation_matrix)
        exp_933 = exp_weight_933(activation_matrix)
        exp_841 = exp_weight_841(activation_matrix)
        return torch.cat([last_token, exp_977, exp_933, exp_841], dim=0)

    raise ValueError(f"Unknown pooling strategy: {strategy}")
