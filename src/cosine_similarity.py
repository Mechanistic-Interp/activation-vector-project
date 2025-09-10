"""
Cosine Similarity Calculator for Pythia-12B Activation Vectors

This script extracts activation vectors from two pieces of text using your deployed
Pythia-12B model and calculates their cosine similarity.

Usage:
    # Direct text:
    modal run cosine_similarity.py --text1 "First text" --text2 "Second text"
    modal run cosine_similarity.py --text1 "First text" --text2 "Second text" --mode "long"

    # File paths (auto-detected):
    modal run cosine_similarity.py --text1 "document1.txt" --text2 "document2.txt"

    # Explicit file parameters:
    modal run cosine_similarity.py --file1 "doc1.txt" --file2 "doc2.txt"

    # Mixed input types:
    modal run cosine_similarity.py --text1 "Direct text here" --file2 "document.txt"
"""

import modal
import numpy as np
from typing import List
from .utils.volume_utils import find_latest_corpus_mean_path

# Reference your deployed extractor
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)
# Define image with torch for the pooling utilities
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "safetensors>=0.3.1",
)

app = modal.App("cosine-similarity", image=image)

# Mount training data volume to discover corpus mean files when centering
training_volume = modal.Volume.from_name("training-data-volume", create_if_missing=True)


@app.function(volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    """Wrapper to discover latest corpus mean path with volume mounted."""
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1, vec2: Lists of floats representing the vectors

    Returns:
        Float between -1 and 1, where 1 = identical, 0 = orthogonal, -1 = opposite
    """
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)

    # Calculate cosine similarity: (a · b) / (||a|| * ||b||)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def load_text_input(text_input: str) -> str:
    """
    Load text from file path or return text directly.

    Args:
        text_input: Either direct text or path to text file

    Returns:
        Text content to process
    """
    # Check if input looks like a file path
    if text_input.endswith(".txt") or "/" in text_input or "\\" in text_input:
        try:
            with open(text_input, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                raise ValueError(f"File {text_input} is empty")
            print(f"📁 Loaded text from file: {text_input} ({len(content)} chars)")
            return content
        except FileNotFoundError:
            print(f"⚠️ File not found: {text_input}, treating as direct text")
            return text_input
        except Exception as e:
            print(f"⚠️ Error reading file {text_input}: {e}, treating as direct text")
            return text_input
    else:
        # Direct text input
        return text_input


@app.local_entrypoint()
def main(
    text1: str = "The capital of France is Paris.",
    text2: str = "Paris is the capital of France.",
    mode: str = "short",
    file1: str = None,
    file2: str = None,
    center: bool = False,
):
    """
    Calculate cosine similarity between activation vectors of two texts.

    Args:
        text1: First text to compare (or file path if ends with .txt)
        text2: Second text to compare (or file path if ends with .txt)
        mode: "short" (5120 dims) or "long" (20480 dims)
        center: If True, subtract corpus mean using latest file in training-data volume
        file1: Optional explicit file path for first text
        file2: Optional explicit file path for second text

    Usage:
        # Direct text:
        modal run cosine_similarity.py --text1 "Hello world" --text2 "Hi there"

        # File paths:
        modal run cosine_similarity.py --text1 "document1.txt" --text2 "document2.txt"

        # Explicit file parameters:
        modal run cosine_similarity.py --file1 "doc1.txt" --file2 "doc2.txt"

        # Mixed:
        modal run cosine_similarity.py --text1 "Direct text" --file2 "document.txt"
    """

    print("🧮 Pythia-12B Activation Vector Cosine Similarity")
    print("=" * 60)

    # Handle file inputs - explicit file parameters take precedence
    actual_text1 = load_text_input(file1) if file1 else load_text_input(text1)
    actual_text2 = load_text_input(file2) if file2 else load_text_input(text2)

    # Create extractor instance
    extractor = Pythia12BExtractor()

    # Show text previews (truncated for long texts)
    text1_preview = (
        actual_text1[:100] + "..." if len(actual_text1) > 100 else actual_text1
    )
    text2_preview = (
        actual_text2[:100] + "..." if len(actual_text2) > 100 else actual_text2
    )

    print(f"📝 Text 1 ({len(actual_text1)} chars): {text1_preview}")
    print(f"📝 Text 2 ({len(actual_text2)} chars): {text2_preview}")
    print(f"🔧 Mode: {mode}")
    print()

    # Resolve centering vector if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()
        print(f"📦 Using corpus mean: {centering_vector}")

    # Extract vectors for both texts
    print("🔬 Extracting activation vectors...")

    # Get vector for text 1
    print("   Extracting vector 1...")
    result1 = extractor.get_activation_vector.remote(
        text=actual_text1,
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )

    # Get vector for text 2
    print("   Extracting vector 2...")
    result2 = extractor.get_activation_vector.remote(
        text=actual_text2,
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )

    # Verify vectors have same shape
    if result1["shape"] != result2["shape"]:
        raise ValueError(
            f"Vector shapes don't match: {result1['shape']} vs {result2['shape']}"
        )

    print(f"✅ Extracted vectors successfully!")
    print(f"   Vector dimensions: {result1['shape']}")
    print(f"   Centered: {result1.get('centered', 'N/A')}")
    print()

    # Calculate cosine similarity
    print("📊 Calculating cosine similarity...")
    similarity = cosine_similarity(result1["vector"], result2["vector"])

    # Display results
    print("=" * 60)
    print("🎯 COSINE SIMILARITY RESULTS")
    print("=" * 60)
    print(f"Similarity Score: {similarity:.6f}")
    print()

    # Interpretation guide
    if similarity > 0.9:
        interpretation = "Very High - Nearly identical semantic meaning"
    elif similarity > 0.7:
        interpretation = "High - Strong semantic similarity"
    elif similarity > 0.5:
        interpretation = "Moderate - Some semantic similarity"
    elif similarity > 0.3:
        interpretation = "Low - Weak semantic similarity"
    elif similarity > 0.0:
        interpretation = "Very Low - Little semantic similarity"
    elif similarity > -0.3:
        interpretation = "Slightly Negative - Somewhat opposing"
    else:
        interpretation = "Negative - Opposing semantic meaning"

    print(f"Interpretation: {interpretation}")
    print()

    # Technical details
    print("🔧 Technical Details:")
    print(f"   Vector 1 length: {len(result1['vector'])}")
    print(f"   Vector 2 length: {len(result2['vector'])}")
    print(f"   Model layers used: {result1['layers_used']}")
    print(f"   Pooling strategy: {result1['pooling_strategy']}")

    # Vector magnitude info
    vec1_magnitude = np.linalg.norm(result1["vector"])
    vec2_magnitude = np.linalg.norm(result2["vector"])
    print(f"   Vector 1 magnitude: {vec1_magnitude:.3f}")
    print(f"   Vector 2 magnitude: {vec2_magnitude:.3f}")

    print()
    print("💡 Tips:")
    print("   • Scores close to 1.0 indicate very similar semantic meaning")
    print("   • Scores around 0.0 indicate unrelated content")
    print("   • Use 'long' mode for more detailed comparisons (20K dimensions)")

    return {
        "cosine_similarity": similarity,
        "text1": actual_text1,
        "text2": actual_text2,
        "text1_chars": len(actual_text1),
        "text2_chars": len(actual_text2),
        "mode": mode,
        "vector_dimensions": result1["shape"],
        "interpretation": interpretation,
        "used_files": {"file1": file1 is not None, "file2": file2 is not None},
    }
