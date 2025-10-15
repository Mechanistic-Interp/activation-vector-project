"""
Silent activation vector extraction utility.

This module provides a function to extract activation vectors using the same methodology
as get_activation_vector.py but without any console output or file writing.
Returns only the vector data for programmatic use.
"""

import modal
from typing import List, Optional
from ..utils.volume_utils import find_latest_corpus_mean_path

# Reference your deployed extractor
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

# Define image for volume utilities
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0", "safetensors>=0.3.1", "numpy>=1.21.0", "packaging"
)

app = modal.App("silent-vector-extraction", image=image)


@app.local_entrypoint()
def extract_vector_entrypoint(
    text: Optional[str] = None,
    file: Optional[str] = None,
    mode: str = "long",
    center: bool = True,
) -> List[float]:
    """Modal entrypoint for silent vector extraction."""
    return get_activation_vector_silent(text=text, file=file, mode=mode, center=center)

# Mount training data volume to resolve corpus mean path when centering
training_volume = modal.Volume.from_name(
    "training_data", create_if_missing=True
)


@app.function(volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    """Wrapper to discover latest corpus mean path with volume mounted."""
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


def _load_text(maybe_path: str) -> str:
    """Load text from file path or return text directly (silent version)."""
    try:
        with open(maybe_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        # If file doesn't exist, treat as direct text
        return maybe_path


def get_activation_vector_silent(
    text: Optional[str] = None,
    file: Optional[str] = None,
    mode: str = "long",
    center: bool = True,
) -> List[float]:
    """
    Extract activation vector silently without any output or file writing.
    
    Args:
        text: Direct text input (ignored if file is provided)
        file: Path to text file
        mode: "short" (5120 dims) or "long" (20480 dims)
        center: Whether to subtract corpus mean
        
    Returns:
        List of floats representing the activation vector
        
    Raises:
        ValueError: If no text is provided or if extraction fails
    """
    # Prepare input text
    if file:
        actual_text = _load_text(file)
    elif text:
        actual_text = _load_text(text)
    else:
        raise ValueError("Either 'text' or 'file' parameter must be provided")
    
    if not actual_text.strip():
        raise ValueError("No text content found (empty input)")

    # Resolve centering vector if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()

    # Call the deployed extractor
    extractor = Pythia12BExtractor()
    result = extractor.get_activation_vector.remote(
        text=actual_text,
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )

    # Return only the vector data
    return result["vector"]


def get_activation_vector_from_text(
    text: str,
    mode: str = "long",
    center: bool = True,
) -> List[float]:
    """
    Convenience function to extract activation vector from direct text.
    
    Args:
        text: Text content to process
        mode: "short" (5120 dims) or "long" (20480 dims)
        center: Whether to subtract corpus mean
        
    Returns:
        List of floats representing the activation vector
    """
    return get_activation_vector_silent(text=text, mode=mode, center=center)


def get_activation_vector_from_file(
    file_path: str,
    mode: str = "long",
    center: bool = True,
) -> List[float]:
    """
    Convenience function to extract activation vector from file.
    
    Args:
        file_path: Path to text file
        mode: "short" (5120 dims) or "long" (20480 dims)
        center: Whether to subtract corpus mean
        
    Returns:
        List of floats representing the activation vector
    """
    return get_activation_vector_silent(file=file_path, mode=mode, center=center)
