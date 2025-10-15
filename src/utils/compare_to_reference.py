"""
Simple vector comparison utility.

Compares a given vector to a reference vector stored in safetensors format
and returns the cosine similarity.
"""

import numpy as np
from typing import List
from safetensors.torch import load_file


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
    
    return float(dot_product / (norm_a * norm_b))


def load_reference_vector(safetensors_path: str, tensor_name: str = "german_chocolate_vector") -> List[float]:
    """
    Load reference vector from safetensors file.
    
    Args:
        safetensors_path: Path to the safetensors file
        tensor_name: Name of the tensor in the file
        
    Returns:
        List of floats representing the reference vector
    """
    tensors = load_file(safetensors_path)
    if tensor_name not in tensors:
        available_keys = list(tensors.keys())
        raise KeyError(f"Tensor '{tensor_name}' not found. Available keys: {available_keys}")
    
    tensor = tensors[tensor_name]
    return tensor.cpu().numpy().flatten().tolist()


def compare_to_german_chocolate_reference(
    vector: List[float], 
    reference_path: str = "outputs/german_chocolate_average.safetensors"
) -> float:
    """
    Compare a vector to the german chocolate reference vector.
    
    Args:
        vector: The vector to compare (from get_vector_silent.py)
        reference_path: Path to the german chocolate reference safetensors file
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    reference_vector = load_reference_vector(reference_path, "german_chocolate_vector")
    return cosine_similarity(vector, reference_vector)

