"""
Pythia 12B Activation Vector Extraction using TransformerLens on Modal

Extracts activations from layers 25-27, mean pools across layers, reverses token order,
and applies exponential weighting with Normal distribution depths.

Provides two Modal methods:
- get_activation_vector(): Returns final pooled vectors (short/long modes only)
- get_activation_matrix(): Returns raw [5120, tokens] matrix for corpus mean computation

Use via Modal's direct function invocation:
extractor = modal.Cls.from_name("activation-vector-project", "Pythia12BActivationExtractor")()
result = extractor.get_activation_vector.remote(text="Your text", mode="short")
"""

import modal
from typing import Dict, Any, Literal, List, Optional
from src.utils.pooling import pool_tokens
from src.utils.io import save_vector_to_disk
from src.utils.centering import load_corpus_mean, subtract_corpus_mean

# Create Modal app
app = modal.App("pythia-12b-activation-extraction")

# Create a volume for caching model weights (persists across runs)
model_cache = modal.Volume.from_name(
    "pythia-12b-transformer-lens-cache", create_if_missing=True
)

# Reference training data volume for corpus mean access
training_volume = modal.Volume.from_name(
    "activation-vector-project", create_if_missing=True
)

# Define the container image with TransformerLens and dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.1.0",
    "transformer-lens==2.0.0",
    "transformers>=4.37.2",
    "safetensors==0.4.1",
)

# Import heavy dependencies at image level for GPU snapshots
with image.imports():
    import torch
    from transformer_lens import HookedTransformer
    import os


@app.cls(
    image=image,
    gpu="A100-80GB",  # A100 for 12B model - better GPU snapshot compatibility
    memory=65536,  # 64GB RAM
    volumes={
        "/cache": model_cache,
        "/training_data": training_volume,
    },
    timeout=900,  # 15 minute timeout
    scaledown_window=120,  # Keep warm for 2 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots for fast cold starts
    experimental_options={"enable_gpu_snapshot": True},  # Snapshot GPU memory
    # Scaling parameters optimized for sequential processing with multiple containers
    max_containers=10,  # Allow up to 10 A100 containers for parallel document processing
)
# No @modal.concurrent - each container handles one remote call at a time for cache isolation
class Pythia12BActivationExtractor:
    """Modal class for extracting activation vectors from Pythia 12B using TransformerLens"""

    @modal.enter(snap=True)
    def setup(self):
        """Initialize the model on container startup - this will be snapshotted"""
        # Initialize verbose first
        self.verbose = False  # Minimal logging by default

        # Set cache directory for HuggingFace models
        cache_dir = "/cache/huggingface"
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["TORCH_HOME"] = cache_dir

        if self.verbose:
            print("Loading Pythia 12B with TransformerLens for snapshot...")

        # Model configuration
        model_name = "EleutherAI/pythia-12b"

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"Using device: {self.device}")

        # Load model with TransformerLens (prefer no-processing in reduced precision)
        if self.verbose:
            print("Loading Pythia-12b with TransformerLens...")
        if hasattr(HookedTransformer, "from_pretrained_no_processing"):
            # Recommended when using reduced precision
            self.model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
            # Ensure correct device
            self.model.to(self.device)
        else:
            # Fallback: disable processing steps explicitly
            self.model = HookedTransformer.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device=self.device,
                cache_dir=cache_dir,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )

        # Set to eval mode
        self.model.eval()

        # Store model configuration
        self.d_model = self.model.cfg.d_model  # Should be 5120 for Pythia-12b
        self.n_layers = self.model.cfg.n_layers  # Should be 36 for Pythia-12b
        self.target_layers = [25, 26, 27]  # Layers to extract activations from

        if self.verbose:
            print("Model loaded successfully!")
            print(f"Model dimensions: d_model={self.d_model}, n_layers={self.n_layers}")
            print(f"Target layers for extraction: {self.target_layers}")
            print(f"Model device: {self.device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            print("GPU memory snapshot will be created after this method completes.")

    # Helper functions for shared extraction logic

    def _extract_and_prepare_activations(
        self,
        text: str,
        target_layers: Optional[List[int]] = None,
        centering_vector: Optional[str] = None,
        center: bool = False,
    ) -> torch.Tensor:
        """
        Shared helper: Extract blocks 25-27 → Mean pool → Reverse token order → Center

        Args:
            text: Input text to process
            target_layers: Optional layer indices (default [25, 26, 27])
            center: Whether to subtract corpus mean for centering (default True)

        Returns: [d_model, seq_len] activation matrix in float32, ready for pooling
        """
        import torch

        # Resolve layers
        layers = target_layers if target_layers is not None else self.target_layers

        # Tokenize input
        tokens = self.model.to_tokens(text, prepend_bos=True)

        # Build filter for residual stream activations (post only)
        names_filter = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        if self.verbose:
            print(f"Requesting hooks: {names_filter}")

        # Run forward pass with cache to get activations
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(tokens, names_filter=names_filter)

        # Extract activations from layers 25-27
        layer_activations = []

        for layer in layers:
            resid_post_key = f"blocks.{layer}.hook_resid_post"
            if resid_post_key not in cache.keys():
                raise ValueError(f"No activations found for layer {layer}")

            resid_post = cache[resid_post_key]
            if resid_post.shape[0] == 1:
                resid_post = resid_post[0]  # Remove batch dim: [seq_len, d_model]

            layer_activations.append(resid_post)
            if self.verbose:
                print(f"Layer {layer}: resid_post shape {resid_post.shape}")

        # Find minimum sequence length across all layers (handle inconsistent lengths)
        min_seq_len = min(act.shape[0] for act in layer_activations)

        # Trim all activations to minimum length for consistent stacking
        trimmed_activations = [act[:min_seq_len] for act in layer_activations]

        # Stack layers: [num_layers, min_seq_len, d_model]
        stacked_activations = torch.stack(trimmed_activations, dim=0)
        if self.verbose:
            print(
                f"Stacked activations shape: {stacked_activations.shape} (trimmed to min_seq_len={min_seq_len})"
            )

        # Mean pool across layers: [seq_len, d_model]
        mean_pooled = stacked_activations.mean(dim=0)
        if self.verbose:
            print(f"Mean pooled shape: {mean_pooled.shape}")

        # Transpose to [d_model, seq_len] for token-wise pooling
        activation_matrix = mean_pooled.transpose(0, 1)

        # Reverse token order: column 0 = last token, column 1 = second-last, etc.
        activation_matrix = activation_matrix.flip(dims=[1])

        # Center by subtracting corpus mean (if enabled)
        if center:
            try:
                if centering_vector is None:
                    raise ValueError(
                        "center=True requires centering_vector parameter to be provided"
                    )
                corpus_mean = load_corpus_mean(centering_vector)
                corpus_mean = corpus_mean.to(activation_matrix.device)

                activation_matrix = subtract_corpus_mean(activation_matrix, corpus_mean)

            except Exception as e:
                if self.verbose:
                    print(
                        f"Warning: Centering failed ({type(e).__name__}: {str(e)}), continuing without centering..."
                    )
                # Continue without centering if it fails

        # Upcast to fp32 for stable pooling
        activation_matrix = activation_matrix.to(torch.float32)

        if self.verbose:
            print(
                f"Final activation matrix shape (reversed): {activation_matrix.shape}"
            )

        return activation_matrix

    def _apply_pooling(
        self,
        activation_matrix: torch.Tensor,
        pooling_strategy: Literal["short", "long"],
        pooling_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Shared helper: Apply pooling strategies to compute weighted averages

        Args:
            activation_matrix: [d_model, seq_len] matrix ready for pooling
            pooling_strategy: Pooling strategy to use
            pooling_params: Optional parameters for pooling

        Returns: Raw weighted average vector (not normalized)
        """
        # Apply pooling to get raw weighted average
        vector = pool_tokens(
            activation_matrix, strategy=pooling_strategy, params=pooling_params
        )

        return vector

    @modal.method()
    def get_activation_vector(
        self,
        text: str,
        pooling_strategy: Literal["short", "long"] = "short",
        target_layers: Optional[List[int]] = None,
        centering_vector: Optional[str] = None,
        center: bool = False,
        pooling_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract activation vectors from layers 25-27 with mean pooling and token reversal.

        Algorithm:
        1. Extract activations from blocks 25–27 → [3, seq_len, d_model]
        2. Mean pool across 3 layers → [seq_len, d_model] → transpose → [d_model, seq_len]
        3. Reverse token order → [d_model, seq_len] (column 0 = last token)
        4. Apply pooling strategies

        Args:
            text: Input text to process.
            pooling_strategy: One of {"short", "long"}.
                             "short" → [d_model] vector, "long" → [4 * d_model] vector.
            target_layers: Optional layer indices (default [25, 26, 27]).
            centering_vector: Optional path to custom centering safetensors file.
            center: Whether to apply centering (requires centering_vector if True).
            pooling_params: Optional dict of params for the chosen strategy.

        Returns:
            A dictionary with:
            - vector: List[float], length d_model (short) or 4*d_model (long).
            - shape: Vector shape.
            - d_model: Model hidden size.
            - layers_used: The layer indices used.
            - activation_types: ["resid_post"].
            - pooling_strategy: The strategy used.
            - pooling_params: The parameters applied.
            - centered: Whether centering was applied.
        """
        # Validation: centering requires centering_vector
        if center and centering_vector is None:
            raise ValueError(
                "center=True requires centering_vector parameter to be provided"
            )

        # Step 1: Extract and prepare activation matrix using shared helper
        activation_matrix = self._extract_and_prepare_activations(
            text, target_layers, centering_vector, center
        )

        # Step 2: Apply pooling to get raw weighted average using shared helper
        vector = self._apply_pooling(
            activation_matrix, pooling_strategy, pooling_params
        )

        # Step 3: Prepare result
        layers = target_layers if target_layers is not None else self.target_layers
        vector_list = vector.cpu().float().tolist()

        result = {
            "vector": vector_list,
            "shape": list(vector.shape),
            "d_model": self.d_model,
            "layers_used": layers,
            "activation_types": ["resid_post"],
            "pooling_strategy": pooling_strategy,
            "pooling_params": pooling_params or {},
            "centered": center,  # Whether corpus mean centering was applied
        }

        return result

    @modal.method()
    def get_activation_matrix(
        self,
        text: str,
        target_layers: Optional[List[int]] = None,
        centering_vector: Optional[str] = None,
        center: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract intermediate [5120, tokens] activation matrix with optional centering.

        Algorithm:
        1. Extract activations from blocks 25–27 → Mean pool across 3 layers
        2. Reverse token order (column 0 = last token)
        3. Center by subtracting corpus mean (if enabled)
        4. Return raw [5120, tokens] matrix before any pooling strategies

        Args:
            text: Input text to process.
            target_layers: Optional layer indices (default [25, 26, 27]).
            centering_vector: Optional path to custom centering safetensors file.
            center: Whether to apply centering (requires centering_vector if True).

        Returns:
            Dictionary with activation matrix data, or None if text is empty:
            - activation_matrix: List[List[float]] of shape [5120, tokens]
            - shape: [5120, tokens]
            - d_model: 5120
            - layers_used: [25, 26, 27]
            - tokens: Number of tokens in sequence
            - word_count: Actual word count of processed text
            - centered: Whether corpus mean centering was applied
        """
        # Basic text validation
        if not text or not text.strip():
            return None

        # Validation: centering requires centering_vector
        if center and centering_vector is None:
            raise ValueError(
                "center=True requires centering_vector parameter to be provided"
            )

        text = text.strip()
        word_count = len(text.split())

        # Use shared helper to get the prepared activation matrix
        activation_matrix = self._extract_and_prepare_activations(
            text, target_layers, centering_vector, center
        )

        # Resolve layers for return info
        layers = target_layers if target_layers is not None else self.target_layers

        # Convert to Python lists for JSON (store as fp32)
        activation_matrix_list = activation_matrix.cpu().float().tolist()

        result = {
            "activation_matrix": activation_matrix_list,
            "shape": list(activation_matrix.shape),
            "d_model": self.d_model,
            "layers_used": layers,
            "tokens": activation_matrix.shape[1],  # sequence length
            "word_count": word_count,  # Include word count for debugging
            "centered": center,  # Whether corpus mean centering was applied
        }

        return result

    @modal.method()
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        import torch

        return {
            "model_name": "EleutherAI/pythia-12b",
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "target_layers": self.target_layers,
            "vector_dims": {
                "short": self.d_model,
                "long": self.d_model * 4,  # 4 concatenated pooling strategies
            },
            "activation_types": ["resid_post"],
            "pooling_method": "configurable",
            "pooling_options": ["short", "long"],
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "snapshot_enabled": True,
        }


# Local entrypoint for testing
@app.local_entrypoint()
def main(
    text: str = "The capital of France is Paris",
    mode: str = "short",
    target_layers: str = "25,26,27",
    out_dir: str = "outputs",
):
    """
    Test activation vector extraction locally.

    Usage:
        # Test vector extraction
        modal run src/extract_vector.py --text "Your text here" --mode "short"

        # Test matrix extraction (for corpus mean)
        modal run src/extract_vector.py --text "Your text here" --mode "matrix"

        # Deploy to Modal
        modal deploy src/extract_vector.py
    """
    print("\n" + "=" * 60)
    print("Pythia 12B Activation Vector Extraction")
    print("=" * 60 + "\n")

    # Create extractor instance
    extractor = Pythia12BActivationExtractor()

    # Get model info
    print("Model Information:")
    info = extractor.get_model_info.remote()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Parse CLI string into list of ints for layers
    try:
        layers = [int(x.strip()) for x in target_layers.split(",") if x.strip()]
    except Exception:
        layers = [25, 26, 27]

    print(f"Input text: {text}")
    print(f"Mode: {mode}")
    print(f"Target layers: {layers}")

    # Test both methods based on mode
    if mode == "matrix":
        # Test get_activation_matrix method
        print("\n" + "=" * 60)
        print("Testing get_activation_matrix() for corpus mean...")

        result = extractor.get_activation_matrix.remote(
            text=text,
            target_layers=layers,
        )

        print("Matrix Results:")
        print(f"  Matrix shape: {result['shape']}")
        print(f"  d_model: {result['d_model']}")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Layers used: {result['layers_used']}")

        # Don't save the full matrix - too large for demo
        print(f"  Matrix sample [0][:5]: {result['activation_matrix'][0][:5]}")

    else:
        # Test get_activation_vector method
        print("\n" + "=" * 60)
        print("Testing get_activation_vector()...")

        result = extractor.get_activation_vector.remote(
            text=text,
            target_layers=layers,
            pooling_strategy=mode,  # Use mode as pooling strategy (short/long)
        )

        print("Vector Results:")
        print(f"  Vector shape: {result['shape']}")
        print(f"  Pooling strategy: {result['pooling_strategy']}")
        print(f"  Layers used: {result['layers_used']}")
        print(f"  First 10 vector values: {result['vector'][:10]}")

        # Save the vector locally as safetensors (fp16) only
        save_vector_to_disk(result, text, out_dir)
