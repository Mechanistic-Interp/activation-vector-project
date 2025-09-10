"""
I/O utilities for saving vectors and metadata.
"""

import json
import os
from typing import Dict, Any

import torch
from safetensors.torch import save_file


def save_vector_to_disk(result: Dict[str, Any], text: str, out_dir: str) -> str:
    """
    Save vector results to disk as safetensors and metadata JSON.

    Args:
        result: Dictionary from get_activation_vector containing vector data
        text: Original input text that generated the vector
        out_dir: Output directory to save files

    Saves:
        - {filename}.safetensors: Vector data in fp16 format
        - {filename}_metadata.json: Metadata about the vector
    """
    os.makedirs(out_dir, exist_ok=True)

    pooling_strategy = result.get("pooling_strategy", "unknown")
    filename = f"activation_vector_{pooling_strategy}_{len(text)}"
    filepath_base = os.path.join(out_dir, filename)

    vector_tensor = torch.tensor(result["vector"], dtype=torch.float16, device="cpu")
    save_file({"vec": vector_tensor}, f"{filepath_base}.safetensors")
    print(f"\n✅ Saved vector to {filepath_base}.safetensors (fp16)")

    metadata = {
        "text": text,
        "pooling_strategy": result.get("pooling_strategy", "unknown"),
        "shape": result["shape"],
        "d_model": result["d_model"],
        "layers_used": result["layers_used"],
        "activation_types": result.get("activation_types", []),
        "centered": result.get("centered", False),
        "text_length": len(text),
        "text_word_count": len(text.split()),
    }

    with open(f"{filepath_base}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {filepath_base}_metadata.json")

    return filepath_base

