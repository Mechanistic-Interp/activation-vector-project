"""
Pythia 12B Model Deployment on Modal with GPU Memory Snapshots
Optimized for fast cold starts using GPU memory snapshotting (no web endpoints).
"""

import modal
from modal import enable_output
from typing import Optional, Dict, Any

# Create Modal app
app = modal.App("pythia-12b-snapshot")

# Create a volume for caching model weights (persists across runs)
model_cache = modal.Volume.from_name("pythia-12b-cache", create_if_missing=True)

# Define the container image with necessary dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "numpy<2.0",
    "torch==2.1.0",
    "transformers==4.36.0",
    "accelerate==0.25.0",
    "safetensors==0.4.1",
    "sentencepiece==0.1.99",
    "einops==0.7.0",
)

"""
Import heavy deps at image build/import time for faster cold starts and to
avoid import shadowing issues inside methods.
"""
with image.imports():
    import os
    import torch
    from transformers import GPTNeoXForCausalLM, AutoTokenizer


@app.cls(
    image=image,
    gpu="A100-80GB",  # A100 80GB for 12B model
    memory=65536,  # 64GB RAM
    volumes={"/cache": model_cache},
    timeout=900,  # 15 minute timeout
    scaledown_window=120,  # Keep warm for 2 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots
    experimental_options={"enable_gpu_snapshot": True},  # Snapshot GPU memory
    max_containers=10,  # enable parallel containers if needed
)
class Pythia12BSnapshotInference:
    """Modal class for Pythia 12B model inference with GPU snapshots"""

    @modal.enter(snap=True)
    def setup(self):
        """Initialize the model on container startup - this will be snapshotted"""
        # Set cache directory for HuggingFace models
        cache_dir = "/cache/huggingface"
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        print("Loading Pythia 12B model for snapshot...")

        # Model configuration
        model_name = "EleutherAI/pythia-12b"

        # Prefer A100-friendly matmul behavior when on CUDA (TF32 acceleration)
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Reduce cuDNN autotuning side effects prior to snapshot
                torch.backends.cudnn.benchmark = False
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
        except Exception:
            # Non-fatal; continue with defaults if backend flags aren't available
            pass

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )

        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model to CPU first to avoid Accelerate hooks, then move to GPU.
        # This is more snapshot-friendly than device_map="cuda" in some environments.
        print("Loading model weights to CPU (FP16)...")
        self.model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )

        if torch.cuda.is_available():
            print("Moving model to CUDA device...")
            self.model.to("cuda")
            # Ensure CUDA queue is idle before snapshot
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # Set to eval mode
        self.model.eval()

        print(f"Model loaded and ready for snapshot!")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print("GPU memory snapshot will be created after this method completes.")
        # Ensure no pending GPU work prior to snapshot
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # Flag for deferred compilation post-snapshot
        self._compiled = False

    @modal.method()
    def generate_single_token(
        self,
        text: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate exactly one token given input text.

        Args:
            text: Input text prompt
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            return_logits: Whether to return logits for all tokens

        Returns:
            Dictionary containing:
            - next_token: The generated token (string)
            - next_token_id: The token ID (int)
            - full_text: Input text + generated token
            - logits: (optional) Logits for all vocabulary tokens
        """
        # Optionally compile on first call (after snapshot restore) to avoid snapshot conflicts
        # Enable by setting env var ENABLE_COMPILE_AFTER_SNAPSHOT=1
        import os

        if (
            not getattr(self, "_compiled", False)
            and os.environ.get("ENABLE_COMPILE_AFTER_SNAPSHOT", "0") == "1"
        ):
            if hasattr(torch, "compile") and torch.cuda.is_available():
                try:
                    print("Compiling model (post-snapshot)...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._compiled = True
                except Exception as e:
                    print(f"Compile skipped due to error: {e}")
                    self._compiled = False

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,  # Pythia max context length
        )

        # Move to GPU
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # Generate exactly one token
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,  # Generate exactly 1 token
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 50,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=return_logits,
            )

        # Extract the generated token
        generated_ids = outputs.sequences
        new_token_id = generated_ids[0, -1].item()
        new_token = self.tokenizer.decode([new_token_id], skip_special_tokens=False)

        # Decode full text
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        result = {
            "next_token": new_token,
            "next_token_id": new_token_id,
            "full_text": full_text,
            "input_length": len(input_ids[0]),
        }

        # Add logits if requested
        if return_logits and outputs.scores:
            # Get logits for the generated token
            logits = outputs.scores[0][0].cpu().numpy().tolist()
            result["logits_shape"] = len(logits)
            # Only return top 100 logits to reduce response size
            top_k_logits = torch.topk(outputs.scores[0][0], k=min(100, len(logits)))
            result["top_logits"] = {
                "values": top_k_logits.values.cpu().numpy().tolist(),
                "indices": top_k_logits.indices.cpu().numpy().tolist(),
            }

        return result

    @modal.method()
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "EleutherAI/pythia-12b",
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": 2048,
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "cuda_available": torch.cuda.is_available(),
            "snapshot_enabled": True,
        }

    @modal.method()
    def batch_generate(
        self,
        texts: list[str],
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> list[Dict[str, Any]]:
        """
        Generate single tokens for multiple texts in a batch.

        Args:
            texts: List of input text prompts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            List of result dictionaries for each input
        """
        results = []
        for text in texts:
            result = self.generate_single_token(
                text=text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                return_logits=False,
            )
            results.append(result)
        return results


# Removed web endpoints (FastAPI) to keep deployment lean and GPU-optimized


# Local entrypoint for testing
@app.local_entrypoint()
def main(
    text: str = "I love to eat",
    temperature: float = 0.8,
    do_sample: bool = True,
    batch: bool = False,
):
    """
    Test the Pythia 12B model with GPU snapshots locally.

    Usage:
    modal run -m src.deployed_apps.pythia_12b_modal_snapshot --text "Your input text"
    modal run -m src.deployed_apps.pythia_12b_modal_snapshot --batch  # Test batch mode
    """
    print(f"\n{'=' * 60}")
    print("Pythia 12B Single Token Generation (GPU Snapshot Version)")
    print(f"{'=' * 60}\n")

    # Create inference instance (Modal remote function call)
    inference = Pythia12BSnapshotInference()

    # Get model info
    print("Model Information:")
    with enable_output():
        info = inference.get_model_info.remote()
    for key, value in info.items():
        print(f"  {key}: {value}")

    if batch:
        # Test batch generation
        test_texts = [
            "The meaning of life is",
            "Once upon a time",
            "In the beginning",
            "The capital of France is",
            "Machine learning is",
        ]

        print(f"\nBatch Generation Test ({len(test_texts)} prompts)")
        print("=" * 40)

        with enable_output():
            results = inference.batch_generate.remote(
                texts=test_texts,
                temperature=temperature,
                do_sample=do_sample,
            )

        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"\n{i + 1}. Input: {text}")
            print(f"   Next token: '{result['next_token']}'")
            print(f"   Full: {result['full_text']}")
    else:
        # Single generation
        print(f"\nInput text: {text}")
        print(f"Temperature: {temperature}")
        print(f"Sampling: {do_sample}")

        # Generate single token
        with enable_output():
            result = inference.generate_single_token.remote(
                text=text,
                temperature=temperature,
                do_sample=do_sample,
                return_logits=False,
            )

        print(f"\n{'=' * 60}")
        print("Results:")
        print(f"  Next token: '{result['next_token']}'")
        print(f"  Token ID: {result['next_token_id']}")
        print(f"  Full text: {result['full_text']}")
        print(f"  Input length: {result['input_length']} tokens")

    print(f"{'=' * 60}\n")

    # Performance note
    print("💡 Note: This version uses GPU memory snapshots for faster cold starts.")
    print(
        "   First deployment will be slow, but subsequent starts will be much faster!"
    )
    print("   Deploy with: modal deploy -m src.deployed_apps.pythia_12b_modal_snapshot")
