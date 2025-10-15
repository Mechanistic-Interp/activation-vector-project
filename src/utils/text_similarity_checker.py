"""
Text similarity checker against german chocolate reference.

Accepts a text string, extracts its activation vector, and compares it
to the german chocolate reference vector, printing the cosine similarity.
"""

import modal
from typing import Optional
from src.utils.volume_utils import find_latest_corpus_mean_path
from .compare_to_reference import compare_to_german_chocolate_reference

# Reference your deployed extractor
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

# Define image for the comparison utilities
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0", "safetensors>=0.3.1", "numpy>=1.21.0", "packaging"
)

app = modal.App("text-similarity-checker", image=image)

# Mount training data volume to resolve corpus mean path when centering
training_volume = modal.Volume.from_name(
    "training_data", create_if_missing=True
)


@app.function(volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    """Wrapper to discover latest corpus mean path with volume mounted."""
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


def _load_text(maybe_path: str) -> str:
    """Load text from file path or return text directly (same as get_activation_vector.py)."""
    import os
    if os.path.isfile(maybe_path):
        with open(maybe_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return maybe_path


def check_text_similarity(
    text: str,
    mode: str = "short",
    center: bool = True,
    reference_path: str = "outputs/german_chocolate_average.safetensors"
) -> float:
    """
    Check similarity of text to german chocolate reference vector.
    
    Args:
        text: Text string to analyze
        mode: "short" (5120 dims) or "long" (20480 dims)
        center: Whether to subtract corpus mean
        reference_path: Path to the german chocolate reference safetensors file
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    # Process text the same way as get_activation_vector.py
    actual_text = _load_text(text)
    print(f"🐛 Debug: Processing text length: {len(actual_text)} chars")
    print(f"🐛 Debug: Text preview: {actual_text[:100]}...")
    
    # Resolve centering vector if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()
        print(f"🐛 Debug: Resolved centering vector: {centering_vector}")

    # Call the deployed extractor
    extractor = Pythia12BExtractor()
    print(f"🐛 Debug: Calling extractor with:")
    print(f"  - text length: {len(actual_text)}")
    print(f"  - text hash: {hash(actual_text)}")
    print(f"  - pooling_strategy: {mode}")
    print(f"  - center: {center}")
    print(f"  - centering_vector: {centering_vector}")
    
    result = extractor.get_activation_vector.remote(
        text=actual_text,  # Use processed text
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )
    
    print(f"🐛 Debug: Extractor returned:")
    print(f"  - shape: {result.get('shape')}")
    print(f"  - pooling_strategy: {result.get('pooling_strategy')}")
    print(f"  - layers_used: {result.get('layers_used')}")
    print(f"  - centered: {result.get('centered')}")

    # Extract the vector
    vector = result["vector"]
    
    # Save vector as CSV for debugging
    import csv
    import os
    debug_csv_path = "outputs/debug_activation_vector.csv"
    os.makedirs(os.path.dirname(debug_csv_path), exist_ok=True)
    with open(debug_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for v in vector:
            writer.writerow([float(v)])
    print(f"🐛 Debug: Saved activation vector to {debug_csv_path}")
    print(f"🐛 Debug: Vector shape: {len(vector)}, first 5 values: {vector[:5]}")
    
    # Compare to reference
    similarity = compare_to_german_chocolate_reference(vector, reference_path)
    
    # Print result
    print(f"Cosine similarity to german chocolate reference: {similarity:.6f}")
    
    return similarity


@app.local_entrypoint()
def main(
    text: str = "The capital of France is Paris.",
    file: Optional[str] = None,
    mode: str = "short",
    no_center: bool = False,
    reference_path: str = "outputs/german_chocolate_average.safetensors"
):
    """
    Command line interface for text similarity checking.
    
    Args:
        text: Text to analyze (ignored if file is provided)
        file: Path to text file
        mode: "short" (5120 dims) or "long" (20480 dims)
        no_center: Disable corpus mean centering
        reference_path: Path to reference safetensors file
    """
    # Use file if provided, otherwise use text (same logic as get_activation_vector.py)
    input_text = file if file else text
    
    similarity = check_text_similarity(
        text=input_text,
        mode=mode,
        center=not no_center,
        reference_path=reference_path
    )
    
    return similarity
