"""
Iterative text generation using Pythia 12B with activation tracking.

This script generates text token-by-token, appending each new token to the context
and generating the next one until reaching max_tokens or an end-of-sequence token.

Also tracks activation vectors and cosine similarity to a reference vector at each step.

Usage:
    # Generate from text string
    modal run -m src.utils.generate_text --text "Once upon a time"
    
    # Generate from file
    modal run -m src.utils.generate_text --file story_prompt.txt
    
    # Custom parameters
    modal run -m src.utils.generate_text --text "The capital of France is" --max-tokens 50 --temperature 0.7
"""

import modal
from typing import Optional, List
import csv
import os

# Reference the deployed Pythia 12B model classes
Pythia12BInference = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BSnapshotInference"
)
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BActivationExtractor"
)

# Define image for utilities
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0", "safetensors>=0.3.1", "numpy>=1.21.0", "packaging"
)

app = modal.App("generate-text-iterative", image=image)

# Mount training data volume for corpus mean
training_volume = modal.Volume.from_name(
    "training_data", create_if_missing=True
)


@app.function(volumes={"/training_data": training_volume}, timeout=60)
def resolve_latest_corpus_mean_path() -> str:
    """Wrapper to discover latest corpus mean path with volume mounted."""
    from src.utils.volume_utils import find_latest_corpus_mean_path
    return find_latest_corpus_mean_path("/training_data/corpus_mean_output")


def _load_text(maybe_path: str) -> str:
    """Load text from file path or return text directly."""
    import os
    if os.path.isfile(maybe_path):
        with open(maybe_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return maybe_path


@app.local_entrypoint()
def main(
    text: str = "Once upon a time",
    file: Optional[str] = None,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    mode: str = "short",
    center: bool = True,
    reference_path: str = "outputs/german_chocolate_average.safetensors",
    output_prefix: str = "outputs/generation",
):
    """
    Generate text iteratively, one token at a time, tracking activations and similarity.
    
    Args:
        text: Input text prompt (ignored if file is provided)
        file: Path to text file
        max_tokens: Maximum number of tokens to generate (default: 200)
        temperature: Sampling temperature (0.0 to 2.0, default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.95)
        top_k: Top-k sampling parameter (default: 50)
        do_sample: Whether to sample or use greedy decoding (default: True)
        mode: Activation vector mode "short" (5120) or "long" (20480)
        center: Whether to apply corpus mean centering
        reference_path: Path to reference safetensors for similarity comparison
        output_prefix: Prefix for output CSV files
    """
    import numpy as np
    from src.utils.compare_to_reference import load_reference_vector, cosine_similarity
    
    # Load input text
    current_text = _load_text(file) if file else _load_text(text)
    
    # Create model instances
    model = Pythia12BInference()
    extractor = Pythia12BExtractor()
    
    # Resolve centering vector if requested
    centering_vector = None
    if center:
        centering_vector = resolve_latest_corpus_mean_path.remote()
    
    # Load reference vector for similarity computation
    reference_vector = load_reference_vector(reference_path, "german_chocolate_vector")
    
    # Get EOS token ID for Pythia
    EOS_TOKEN_ID = 0
    
    # Storage for activations and similarities
    all_activations: List[List[float]] = []  # Will be [tokens+1, 5120]
    all_similarities: List[float] = []        # Will be [tokens+1]
    
    tokens_generated = 0
    
    # Get initial activation (time 0 - before generating any tokens)
    initial_result = extractor.get_activation_vector.remote(
        text=current_text,
        pooling_strategy=mode,
        center=center,
        centering_vector=centering_vector,
    )
    initial_vector = initial_result["vector"]
    initial_similarity = cosine_similarity(initial_vector, reference_vector)
    
    all_activations.append(initial_vector)
    all_similarities.append(initial_similarity)
    
    # Generation loop
    for i in range(max_tokens):
        # Get raw logits from model (top 100)
        # We get logits WITHOUT sampling, then do sampling ourselves
        logits_result = model.generate_single_token.remote(
            text=current_text,
            temperature=1.0,   # Get raw logits (temperature doesn't matter when return_logits=True)
            top_p=1.0,
            top_k=100,         # Get top 100 logits
            do_sample=False,   # Greedy just to get the logits
            return_logits=True,  # Get the raw logits
        )
        
        # Extract top 100 logits
        top_logits_info = logits_result["top_logits"]
        logit_values = np.array(top_logits_info["values"])  # Top 100 logit values
        token_ids = np.array(top_logits_info["indices"])     # Corresponding token IDs
        
        # Apply temperature and softmax locally to get probabilities
        if temperature > 0:
            # Apply temperature scaling
            scaled_logits = logit_values / temperature
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Sample from the distribution or pick argmax
            if do_sample:
                # Random sampling based on probabilities
                selected_idx = np.random.choice(len(token_ids), p=probabilities)
                next_token_id = int(token_ids[selected_idx])
            else:
                # Greedy: pick highest probability (first token after sorting)
                next_token_id = int(token_ids[0])
        else:
            # Temperature = 0 means greedy (pick the top token)
            next_token_id = int(token_ids[0])
        
        # Check for end of sequence
        if next_token_id == EOS_TOKEN_ID:
            break
        
        # Append our selected token to the text using the model's tokenizer
        append_result = model.append_token_to_text.remote(
            text=current_text,
            token_id=next_token_id
        )
        
        next_token = append_result["next_token"]
        current_text = append_result["full_text"]
        tokens_generated += 1
        
        # Extract activation vector for current text
        activation_result = extractor.get_activation_vector.remote(
            text=current_text,
            pooling_strategy=mode,
            center=center,
            centering_vector=centering_vector,
        )
        current_vector = activation_result["vector"]
        
        # Calculate cosine similarity to reference
        similarity = cosine_similarity(current_vector, reference_vector)
        
        # Store results
        all_activations.append(current_vector)
        all_similarities.append(similarity)
        
        # Print only the current text
        print(current_text, flush=True)
    
    # Save activations to CSV [5120, tokens+1] (transposed for standard format)
    activations_path = f"{output_prefix}_activations.csv"
    os.makedirs(os.path.dirname(activations_path) or ".", exist_ok=True)
    
    # Transpose to [5120, tokens+1] format
    activations_array = np.array(all_activations).T  # Shape: [5120, tokens+1]
    
    with open(activations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in activations_array:
            writer.writerow(row.tolist())
    
    # Save similarities to CSV [tokens+1]
    similarities_path = f"{output_prefix}_similarities.csv"
    with open(similarities_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sim in all_similarities:
            writer.writerow([sim])
    
    return {
        "generated_text": current_text,
        "tokens_generated": tokens_generated,
        "stopped_by_eos": tokens_generated < max_tokens,
        "final_length": len(current_text),
        "activations_file": activations_path,
        "similarities_file": similarities_path,
        "activations_shape": f"[5120, {len(all_similarities)}]",
        "similarities_shape": f"[{len(all_similarities)}]",
    }
