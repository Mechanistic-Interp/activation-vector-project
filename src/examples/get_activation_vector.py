"""
Minimal example: call get_activation_vector via Modal.

Usage examples:
  modal run src/examples/get_activation_vector.py --text "Hello world" --mode short
  modal run src/examples/get_activation_vector.py --text "Some longer text..." --mode long
  modal run src/examples/get_activation_vector.py --file src/training_data/text-samples/01_bonded_cats_apartment.txt --mode long --center

Notes:
- When --center is set, the script resolves the latest corpus mean path from the
  mounted training-data volume.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import modal

from src.utils.volume_utils import find_latest_corpus_mean_path


# Reference your deployed extractor class
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

app = modal.App("example-get-activation-vector")

# Mount training data volume to resolve corpus mean path when centering
training_volume = modal.Volume.from_name("training-data-volume", create_if_missing=True)


@app.function(volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


def _load_text(maybe_path: str) -> str:
    if os.path.isfile(maybe_path):
        with open(maybe_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return maybe_path


@app.local_entrypoint()
def main(
    text: str = "The capital of France is Paris.",
    file: Optional[str] = None,
    mode: str = "short",  # "short" (5120) or "long" (20480)
    center: bool = False,
):
    # Prepare input text
    actual_text = _load_text(file) if file else _load_text(text)
    print("📝 Text (preview):", (actual_text[:120] + "...") if len(actual_text) > 120 else actual_text)
    print("🔧 Mode:", mode)
    print("🎯 Centering:", center)

    # Resolve corpus mean path if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()
        print("📦 Using corpus mean:", centering_vector)

    # Call the deployed extractor
    extractor = Pythia12BExtractor()
    result = extractor.get_activation_vector.remote(
        text=actual_text,
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )

    # Display results
    vec = result["vector"]
    shape = result.get("shape", (len(vec),))
    print("✅ Received vector with shape:", shape)
    print("   Pooling strategy:", result.get("pooling_strategy"))
    print("   Layers used:", result.get("layers_used"))
    print("   Centered:", result.get("centered"))

    # Preview a few values
    preview = ", ".join(f"{v:.5f}" for v in vec[:8])
    print("   Values [0:8]:", preview)

    return {
        "shape": shape,
        "centered": result.get("centered"),
        "pooling_strategy": result.get("pooling_strategy"),
    }

