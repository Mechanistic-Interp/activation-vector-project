"""
Minimal example: call get_activation_vector via Modal.

Usage examples:
  modal run -m src.examples.get_activation_vector --text "Hello world" --mode short
  modal run -m src.examples.get_activation_vector --text "Some longer text..." --mode long
  modal run -m src.examples.get_activation_vector --file src/training_data/text-samples/01_bonded_cats_apartment.txt --mode long --center

Save to CSV (two-line format):
  modal run -m src.examples.get_activation_vector \
    --file src/training_data/text-samples/01_bonded_cats_apartment.txt \
    --mode long \
    --center \
    --output-csv outputs/bonded_cats_long.csv

CSV format produced:
- Header: "<file_name>_<mode>" (e.g., "01_bonded_cats_apartment_long")
- Next line: the full vector values as a single row (comma-separated)

Notes:
- When --center is set, the script resolves the latest corpus mean path from the
  mounted training-data volume.
"""

from __future__ import annotations

import os
import csv
from typing import Optional

import modal

from src.utils.volume_utils import find_latest_corpus_mean_path


# Reference your deployed extractor class
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)
# Define image with torch for the pooling utilities
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "safetensors>=0.3.1",
    "numpy>=1.21.0",
    "packaging",
)


app = modal.App("example-get-activation-vector", image=image)

# Mount training data volume to resolve corpus mean path when centering
training_volume = modal.Volume.from_name("training_data", create_if_missing=True)


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
    mode: str = "long",
    center: bool = True,
    output_csv: Optional[str] = None,
):
    # Prepare input text
    actual_text = _load_text(file) if file else _load_text(text)
    print(
        "📝 Text (preview):",
        (actual_text[:120] + "...") if len(actual_text) > 120 else actual_text,
    )

    # Resolve centering vector if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()
        print("📦 Using corpus mean:", centering_vector)
    else:
        print("📦 Centering disabled")

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

    # Optionally save to CSV in the requested format
    if output_csv:
        # Determine header name based on file name (if provided) and mode
        if file and os.path.isfile(file):
            base_name = os.path.splitext(os.path.basename(file))[0]
        else:
            base_name = "input_text"
        header_name = f"{base_name}_{mode}"

        # If output is a directory, create a sensible filename
        out_path = output_csv
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, f"{header_name}.csv")
        elif not out_path.lower().endswith(".csv"):
            # Treat as a directory path that doesn't yet exist
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, f"{header_name}.csv")
        else:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Write header on first line and vector vertically as a single column
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([header_name])
            w.writerows([[v] for v in vec])
        print(f"💾 Saved CSV to: {out_path}")

    return {
        "shape": shape,
        "centered": result.get("centered"),
        "pooling_strategy": result.get("pooling_strategy"),
        "saved_csv": bool(output_csv),
    }
