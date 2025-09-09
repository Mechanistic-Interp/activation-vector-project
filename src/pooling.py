"""
Pooling utilities for token-level activation matrices.

All pooling computations are performed in float32 for numerical stability
and returned as float32 tensors on the same device as the inputs.

Input shape convention: [d_model, seq_len]
Output shape: [d_model]
"""

from typing import Literal, Optional, Dict, Any

import torch


def _as_float32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        return t.to(torch.float32)
    return t


def mean_pool_tokens(activation_matrix: torch.Tensor) -> torch.Tensor:
    """Mean over sequence dimension in fp32.

    activation_matrix: [d_model, seq_len]
    returns: [d_model]
    """
    act_f32 = _as_float32(activation_matrix)
    return act_f32.mean(dim=1)


def last_token_pool_tokens(activation_matrix: torch.Tensor) -> torch.Tensor:
    """Select last token vector in fp32.

    activation_matrix: [d_model, seq_len]
    returns: [d_model]
    """
    act_f32 = _as_float32(activation_matrix)
    return act_f32[:, -1]


def exponential_pool_tokens(
    activation_matrix: torch.Tensor,
    *,
    percentile_90_weight: float = 0.5,
    min_effective_length: int = 10,
) -> torch.Tensor:
    """Exponential decay pooling from the sequence end in fp32.

    The decay rate is set so that the weight at the 90th percentile position
    from the end equals `percentile_90_weight`. For very short sequences, we
    clamp the effective length to `min_effective_length` to avoid extreme peaking.

    activation_matrix: [d_model, seq_len]
    returns: [d_model]
    """
    act_f32 = _as_float32(activation_matrix)
    d_model, seq_len = act_f32.shape
    if seq_len == 1:
        return act_f32.squeeze(1)

    device = act_f32.device

    # Effective length to reduce extreme peaking on very short sequences
    seq_len_eff = max(int(seq_len), int(min_effective_length))

    # lambda = -ln(p) / (0.1 * L_eff), where p is target weight at 90th percentile
    p = float(percentile_90_weight)
    p = max(min(p, 0.999), 1e-6)
    decay_rate = torch.log(
        torch.tensor(1.0 / p, dtype=torch.float32, device=device)
    ) / (0.1 * float(seq_len_eff))

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    distances_from_end = (seq_len - 1) - positions

    weights = torch.exp(-decay_rate * distances_from_end)
    denom = weights.sum().clamp_min(1e-6)
    weights = weights / denom

    # Weighted sum: [d_model, seq_len] @ [seq_len] -> [d_model]
    return act_f32 @ weights


def softmax_norm_pool_tokens(
    activation_matrix: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Content-adaptive pooling via softmax over token L2 norms in fp32.

    activation_matrix: [d_model, seq_len]
    returns: [d_model]
    """
    act_f32 = _as_float32(activation_matrix)
    device = act_f32.device
    eps = 1e-6

    # Token magnitudes across d_model
    token_norms = torch.linalg.norm(act_f32, dim=0)
    temp = max(float(temperature), 1e-6)
    logits = token_norms / temp
    # Softmax with numerical stability
    max_logit = logits.max()
    stabilized = logits - max_logit
    exp_scores = torch.exp(stabilized)
    weights = exp_scores / exp_scores.sum().clamp_min(eps)

    return act_f32 @ weights


def pool_tokens(
    activation_matrix: torch.Tensor,
    *,
    strategy: Literal["exp", "mean", "last", "softmax_norm"] = "exp",
    params: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Dispatch to a pooling strategy in fp32.

    Supported strategies:
      - "exp": exponential_pool_tokens
      - "mean": mean_pool_tokens
      - "last": last_token_pool_tokens
      - "softmax_norm": softmax_norm_pool_tokens
    """
    params = params or {}
    if strategy == "exp":
        return exponential_pool_tokens(activation_matrix, **params)
    if strategy == "mean":
        return mean_pool_tokens(activation_matrix)
    if strategy == "last":
        return last_token_pool_tokens(activation_matrix)
    if strategy == "softmax_norm":
        return softmax_norm_pool_tokens(activation_matrix, **params)
    raise ValueError(f"Unknown pooling strategy: {strategy}")


__all__ = [
    "pool_tokens",
    "exponential_pool_tokens",
    "mean_pool_tokens",
    "last_token_pool_tokens",
    "softmax_norm_pool_tokens",
]
