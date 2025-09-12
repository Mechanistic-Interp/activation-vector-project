"""
Generate Vector Analysis CSVs from Existing Text Samples

Processes 20 text samples (10 long, 10 short) from src/training_data/text-samples/
and generates TWO CSVs with activation vectors — one for "long" mode and one for
"short" mode. Vectors are output so that each vector occupies a column and each
dimension is one row down (i.e., values go down the rows), rather than being
pipe-separated into a single cell.

Usage:
    modal run -m src.generate_vector_csv
"""

import modal
from modal import enable_output
import csv
import json
import glob
import os
from typing import List, Dict, Any, Optional
from .utils.volume_utils import find_latest_corpus_mean_path

# Define image with torch for the pooling utilities
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "safetensors>=0.3.1",
)

# Reference your deployed extractor
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

app = modal.App("generate-vector-csv")

# Mount the training data volume to discover corpus mean files
training_volume = modal.Volume.from_name(
    "training_data", create_if_missing=True
)


@app.function(image=image, volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    """Wrapper to discover latest corpus mean path with volume mounted."""
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


def load_text_samples() -> Dict[str, List[str]]:
    """Load all text samples from the training_data directory."""
    base_path = "src/training_data/text-samples"

    long_texts = []
    short_texts = []

    print("📂 Loading text samples from files...")

    for i in range(1, 11):
        # Find long text file
        long_pattern = f"{base_path}/{i:02d}_*_long.txt"
        long_files = glob.glob(long_pattern)

        if long_files:
            with open(long_files[0], "r", encoding="utf-8") as f:
                long_text = f.read().strip()
                long_texts.append(long_text)
                print(f"   ✅ Loaded long text {i}: {len(long_text)} chars")
        else:
            print(f"   ❌ Long text {i} not found with pattern: {long_pattern}")
            long_texts.append("")

        # Find short text file
        short_pattern = f"{base_path}/{i:02d}_*_short.txt"
        short_files = glob.glob(short_pattern)

        if short_files:
            with open(short_files[0], "r", encoding="utf-8") as f:
                short_text = f.read().strip()
                short_texts.append(short_text)
                print(f"   ✅ Loaded short text {i}: {len(short_text)} chars")
        else:
            print(f"   ❌ Short text {i} not found with pattern: {short_pattern}")
            short_texts.append("")

    return {"long_texts": long_texts, "short_texts": short_texts}


def format_float(x: Optional[float]) -> str:
    """Format float for CSV output with fixed precision; empty if None."""
    if x is None:
        return ""
    return f"{x:.6f}"


def write_mode_matrix_csv(
    filename: str,
    header_labels: List[str],
    vectors: List[Optional[List[float]]],
):
    """Write vectors with columns = samples (20 columns), rows = dimensions.

    - First row is a header with the 20 sample labels.
    - Subsequent rows contain the vector values per dimension.
    - No sample text is included.
    """
    # Determine maximum vector length across available vectors
    max_len = 0
    for v in vectors:
        if v is not None:
            max_len = max(max_len, len(v))

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        # Header row: 20 labels
        writer.writerow(header_labels)

        # Rows: each dimension index
        for i in range(max_len):
            row: List[str] = []
            for v in vectors:
                cell: Optional[float] = None
                if v is not None and i < len(v):
                    cell = v[i]
                row.append(format_float(cell))
            writer.writerow(row)


@app.local_entrypoint()
def main(center: bool = False):
    """Generate two CSVs (long-mode and short-mode) with activation vectors.

    Args:
        center: If True, subtract corpus mean using the latest file in the mounted
                training-data volume (auto-resolved inside the extractor).

    Usage:
        modal run -m src.generate_vector_csv
        modal run -m src.generate_vector_csv --center
    """

    print("🔬 GENERATING VECTOR ANALYSIS CSV FROM EXISTING TEXT SAMPLES")
    print("=" * 80)

    # Load text samples from files
    samples = load_text_samples()
    long_texts = samples["long_texts"]
    short_texts = samples["short_texts"]

    print(f"📊 Loaded {len(long_texts)} long texts and {len(short_texts)} short texts")
    print("⚡ Extracting vectors for both modes and building matrices...")

    # Create extractor instance
    extractor = Pythia12BExtractor()

    # If centering is requested, resolve the latest corpus mean path inside the volume
    centering_vector: Optional[str] = None
    if center:
        with enable_output():
            centering_vector = resolve_latest_corpus_mean_path.remote()
        print(f"📦 Using corpus mean: {centering_vector}")

    # Prepare containers for vectors (preserve order: 10 long samples, then 10 short samples)
    long_mode_vectors: List[Optional[List[float]]] = []
    short_mode_vectors: List[Optional[List[float]]] = []

    # Ordering of vectors: long_1..long_10, short_1..short_10

    # Process long texts first (samples 1-10)
    for i, text in enumerate(long_texts, 1):
        if not text:
            print(f"⚠️  Skipping empty long text {i}")
            long_mode_vectors.append(None)
            short_mode_vectors.append(None)
            continue

        print(f"📝 Processing long sample {i}... ({len(text)} chars)")

        # Extract vectors in both modes
        try:
            # Long mode extraction
            print(f"   🔬 Extracting long mode vector...")
            with enable_output():
                result_long = extractor.get_activation_vector.remote(
                    text=text,
                    pooling_strategy="long",
                    center=center,
                    centering_vector=centering_vector,
                )
            long_mode_vectors.append(result_long["vector"])

            # Short mode extraction
            print(f"   🔬 Extracting short mode vector...")
            with enable_output():
                result_short = extractor.get_activation_vector.remote(
                    text=text,
                    pooling_strategy="short",
                    center=center,
                    centering_vector=centering_vector,
                )
            short_mode_vectors.append(result_short["vector"])

            print(f"   ✅ Long mode: {len(result_long['vector'])} dims")
            print(f"   ✅ Short mode: {len(result_short['vector'])} dims")

        except Exception as e:
            print(f"   ❌ Error processing long sample {i}: {str(e)}")
            long_mode_vectors.append(None)
            short_mode_vectors.append(None)

    # Process short texts (samples 1-10)
    for i, text in enumerate(short_texts, 1):
        if not text:
            print(f"⚠️  Skipping empty short text {i}")
            long_mode_vectors.append(None)
            short_mode_vectors.append(None)
            continue

        print(f"📝 Processing short sample {i}... ({len(text)} chars)")

        # Extract vectors in both modes
        try:
            # Long mode extraction
            print(f"   🔬 Extracting long mode vector...")
            with enable_output():
                result_long = extractor.get_activation_vector.remote(
                    text=text,
                    pooling_strategy="long",
                    center=center,
                    centering_vector=centering_vector,
                )
            long_mode_vectors.append(result_long["vector"])

            # Short mode extraction
            print(f"   🔬 Extracting short mode vector...")
            with enable_output():
                result_short = extractor.get_activation_vector.remote(
                    text=text,
                    pooling_strategy="short",
                    center=center,
                    centering_vector=centering_vector,
                )
            short_mode_vectors.append(result_short["vector"])

            print(f"   ✅ Long mode: {len(result_long['vector'])} dims")
            print(f"   ✅ Short mode: {len(result_short['vector'])} dims")

        except Exception as e:
            print(f"   ❌ Error processing short sample {i}: {str(e)}")
            long_mode_vectors.append(None)
            short_mode_vectors.append(None)

    # Write two CSV files: one for long-mode vectors, one for short-mode vectors
    long_csv = f"vectors_long_mode_{'centered' if center else 'raw'}.csv"
    short_csv = f"vectors_short_mode_{'centered' if center else 'raw'}.csv"

    # Build labels (no text included in CSVs)
    sample_labels = [f"long_{i}" for i in range(1, 11)] + [
        f"short_{i}" for i in range(1, 11)
    ]

    print(f"\n💾 Writing long-mode vectors to {long_csv}...")
    write_mode_matrix_csv(long_csv, sample_labels, long_mode_vectors)

    print(f"💾 Writing short-mode vectors to {short_csv}...")
    write_mode_matrix_csv(short_csv, sample_labels, short_mode_vectors)

    print("✅ CSV files generated successfully:")
    print(f"   - Long mode:  {os.path.abspath(long_csv)}")
    print(f"   - Short mode: {os.path.abspath(short_csv)}")

    return {
        "long_csv": long_csv,
        "short_csv": short_csv,
        "num_samples": len(long_mode_vectors),
        "long_texts": len(long_texts),
        "short_texts": len(short_texts),
        "success": True,
    }
