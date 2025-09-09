"""
Pythia 12B Activation Vector Extraction using TransformerLens on Modal

Extracts residual stream activations (resid_post) with configurable pooling strategies
and returns either a single vector (short) or per-layer matrix plus flattened
vector (long). Default target layers are [25, 26, 27].
"""

import modal
from typing import Dict, Any, Literal, List, Optional
from src.pooling import pool_tokens

# Create Modal app
app = modal.App("pythia-12b-activation-extraction")

# Create a volume for caching model weights (persists across runs)
model_cache = modal.Volume.from_name(
    "pythia-12b-transformer-lens-cache", create_if_missing=True
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
    volumes={"/cache": model_cache},
    timeout=900,  # 15 minute timeout
    scaledown_window=120,  # Keep warm for 2 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots for fast cold starts
    experimental_options={"enable_gpu_snapshot": True},  # Snapshot GPU memory
)
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

    # Pooling moved to src/pooling.py

    @modal.method()
    def get_activation_vector(
        self,
        text: str,
        mode: Literal["short", "long"] = "short",
        target_layers: Optional[List[int]] = None,
        pooling_strategy: Literal["exp", "mean", "last", "softmax_norm"] = "exp",
        pooling_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract activation vectors from selected layers with configurable pooling.

        Args:
            text: Input text to process.
            mode: "short" returns a single d_model vector (mean over layers, L2-normalized).
                  "long" returns per-layer vectors (L2-normalized), plus a flattened vector.
            target_layers: Optional list of layer indices (default [25, 26, 27]).
            pooling_strategy: One of {"exp", "mean", "last", "softmax_norm"}.
            pooling_params: Optional dict of params for the chosen strategy.

        Returns:
            A dictionary with:
            - vector: List[float], length d_model (short) or num_layers*d_model (long).
            - shape: Vector shape.
            - mode: "short" or "long".
            - d_model: Model hidden size.
            - layers_used: The layer indices used.
            - activation_types: ["resid_post"].
            - pooling_method: The strategy used.
            - pooling_params: The parameters applied.
            - num_activations_found: Count of pooled per-layer vectors.
            - matrix (only when mode=="long"): List[List[float]] of shape [num_layers, d_model].
            - matrix_shape (only when mode=="long"): [num_layers, d_model].
        """
        import torch

        # Resolve layers
        layers = target_layers if target_layers else self.target_layers

        # Tokenize input
        tokens = self.model.to_tokens(text, prepend_bos=True)

        # Build filter for residual stream activations (post only)
        names_filter = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        if self.verbose:
            print(f"Requesting hooks: {names_filter}")

        # Run forward pass with cache to get activations
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(tokens, names_filter=names_filter)

        # Extract residual stream activations from target layers (post only)
        activations = []
        for layer in layers:
            layer_activations = []

            # Get resid_post activations if available
            resid_post_key = f"blocks.{layer}.hook_resid_post"
            if resid_post_key in cache.keys():
                resid_post = cache[resid_post_key]
                if resid_post.shape[0] == 1:
                    resid_post = resid_post[0]  # Remove batch dim: [seq_len, d_model]
                # Transpose to [d_model, seq_len] for pooling
                resid_post = resid_post.transpose(0, 1)
                # Upcast to fp32 for stable pooling
                resid_post = resid_post.to(torch.float32)
                # Apply configured pooling to get [d_model] vector
                pooled_post = pool_tokens(
                    resid_post,
                    strategy=pooling_strategy,
                    params=pooling_params,
                )
                layer_activations.append(pooled_post)
                if self.verbose:
                    print(f"Layer {layer}: resid_post pooled {pooled_post.shape}")
            else:
                if self.verbose:
                    print(f"Layer {layer}: resid_post not available")

            # Add available activations to the list
            activations.extend(layer_activations)

        # Stack all activations: [num_activations, d_model]
        # (number depends on which hooks are available)
        if not activations:
            raise ValueError("No activations found for any target layers")

        stacked_activations = torch.stack(activations)
        if self.verbose:
            print(f"Stacked activations shape: {stacked_activations.shape}")

        # Apply mode-specific processing
        eps = 1e-6
        if mode == "short":
            # Mean pool across all available activation vectors to get single d_model vector
            vector = stacked_activations.mean(dim=0)
            # Normalize vector for cosine similarity in fp32 with epsilon
            norm = vector.norm(p=2)
            vector = vector / (norm + eps)
        else:  # mode == "long"
            # Keep as num_activations x d_model matrix, normalize each vector independently
            for i in range(stacked_activations.shape[0]):
                norm_i = stacked_activations[i].norm(p=2)
                stacked_activations[i] = stacked_activations[i] / (norm_i + eps)
            vector = (
                stacked_activations.flatten()
            )  # Flatten to num_activations*d_model vector

        # Convert to Python lists for JSON (store as fp32)
        vector_list = vector.cpu().float().tolist()
        matrix_list = None
        matrix_shape = None
        if mode == "long":
            matrix_list = stacked_activations.cpu().float().tolist()
            matrix_shape = list(stacked_activations.shape)

        # Determine which activation types were actually found
        found_activation_types = ["resid_post"] if len(activations) > 0 else []

        result = {
            "vector": vector_list,
            "shape": list(vector.shape),
            "mode": mode,
            "d_model": self.d_model,
            "layers_used": layers,
            "activation_types": found_activation_types,
            "pooling_method": pooling_strategy,
            "pooling_params": pooling_params or {},
            "num_activations_found": len(activations),
        }

        if mode == "long":
            result["matrix"] = matrix_list
            result["matrix_shape"] = matrix_shape

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
                "long": self.d_model * len(self.target_layers),
            },
            "activation_types": ["resid_post"],
            "pooling_method": "configurable",
            "pooling_options": ["exp", "mean", "last", "softmax_norm"],
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "snapshot_enabled": True,
        }


# Web endpoint for easy API access
@app.function(
    image=image,
    timeout=60,
)
@modal.fastapi_endpoint(method="POST")
def extract_activation_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Web endpoint for activation vector extraction.

    Request body:
    {
        "text": "Your input text",
        "mode": "short" | "long",
        "target_layers": [25, 26, 27],  # optional
        "pooling_strategy": "exp" | "mean" | "last" | "softmax_norm",  # optional
        "pooling_params": { ... }  # optional, e.g. {"min_effective_length": 10}
    }
    """
    # Get or create inference instance
    extractor = Pythia12BActivationExtractor()

    try:
        text = request.get("text", "")
        if not text:
            return {"success": False, "error": "No text provided"}

        result = extractor.get_activation_vector.remote(
            text=text,
            mode=request.get("mode", "short"),
            target_layers=request.get("target_layers"),
            pooling_strategy=request.get("pooling_strategy", "exp"),
            pooling_params=request.get("pooling_params"),
        )
        return {"success": True, **result}

    except Exception as e:
        return {"success": False, "error": str(e)}


# Health check endpoint
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health_check() -> Dict[str, str]:
    """Simple health check endpoint"""
    extractor = Pythia12BActivationExtractor()
    info = extractor.get_model_info.remote()
    return {
        "status": "healthy",
        "model": "pythia-12b-transformer-lens",
        **info,
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
        # Extract single vector
        modal run src/extract_vector.py --text "Your text here"

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

    # Extract single vector
    print(f"Input text: {text}")
    print(f"Mode: {mode}")

    # Parse CLI string into list of ints for layers
    try:
        layers = [int(x) for x in target_layers.split(",") if x.strip() != ""]
    except Exception:
        layers = [25, 26, 27]

    result = extractor.get_activation_vector.remote(
        text=text,
        mode=mode,
        target_layers=layers,
    )

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Vector shape: {result['shape']}")
    print(f"  Mode: {result['mode']}")
    print(f"  Layers used: {result['layers_used']}")
    print(f"  First 10 vector values: {result['vector'][:10]}")

    # Save the vector locally as safetensors (fp16) only
    import json
    import torch as _torch
    from safetensors.torch import save_file as _save_safetensors

    import os as _os

    _os.makedirs(out_dir, exist_ok=True)
    filename = f"activation_vector_{mode}_{len(text)}"
    filepath_base = _os.path.join(out_dir, filename)
    _tensor_fp16 = _torch.tensor(result["vector"], dtype=_torch.float16, device="cpu")
    _save_safetensors({"vec": _tensor_fp16}, f"{filepath_base}.safetensors")
    print(f"\n✅ Saved vector to {filepath_base}.safetensors (fp16)")

    # Save metadata
    metadata = {
        "text": text,
        "mode": result["mode"],
        "shape": result["shape"],
        "d_model": result["d_model"],
        "layers_used": result["layers_used"],
        "activation_types": result.get("activation_types", []),
        "pooling_method": result.get("pooling_method", ""),
    }
    with open(f"{filepath_base}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {filepath_base}_metadata.json")

    print("\n" + "=" * 60)
    print("✨ Note: This uses TransformerLens for mechanistic interpretability")
    print("   Deploy with: modal deploy src/extract_vector.py")
