"""
Pythia 12B Model Deployment on Modal with GPU Memory Snapshots
Optimized for fast cold starts using GPU memory snapshotting
"""

import modal
from typing import Optional, Dict, Any
import json

# Create Modal app
app = modal.App("pythia-12b-snapshot")

# Create a volume for caching model weights (persists across runs)
model_cache = modal.Volume.from_name("pythia-12b-cache", create_if_missing=True)

# Define the container image with necessary dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.1.0",
    "transformers==4.36.0",
    "accelerate==0.25.0",
    "safetensors==0.4.1",
    "sentencepiece==0.1.99",
    "einops==0.7.0",
)


@app.cls(
    image=image,
    gpu="A100-80GB",  # A100 80GB for 12B model
    memory=65536,  # 64GB RAM
    volumes={"/cache": model_cache},
    timeout=900,  # 15 minute timeout
    scaledown_window=120,  # Keep warm for 2 minutes
    enable_memory_snapshot=True,  # Enable memory snapshots
)
class Pythia12BSnapshotInference:
    """Modal class for Pythia 12B model inference with GPU snapshots"""

    @modal.enter(snap=True)
    def setup(self):
        """Initialize the model on container startup - this will be snapshotted"""
        import torch
        from transformers import GPTNeoXForCausalLM, AutoTokenizer
        import os

        # Set cache directory for HuggingFace models
        cache_dir = "/cache/huggingface"
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        print("Loading Pythia 12B model for snapshot...")

        # Model configuration
        model_name = "EleutherAI/pythia-12b"

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

        # Load model with optimizations directly to GPU
        print("Loading model weights to GPU...")
        self.model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map="auto",  # Automatically distribute across available devices
            cache_dir=cache_dir,
            trust_remote_code=False,
        )

        # Move model to CUDA and compile for better performance
        if torch.cuda.is_available():
            print("Compiling model with torch.compile for optimized performance...")
            # Compile the model for faster inference
            self.model = torch.compile(self.model, mode="reduce-overhead")

            # Warm up the model with a dummy forward pass to trigger compilation
            dummy_input = self.tokenizer("Hello", return_tensors="pt")
            dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
            with torch.no_grad():
                _ = self.model.generate(**dummy_input, max_new_tokens=1)
            print("Model compilation complete!")

        # Set to eval mode
        self.model.eval()

        print(f"Model loaded and ready for snapshot!")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print("GPU memory snapshot will be created after this method completes.")

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
        import torch

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
        import torch

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


# Web endpoint for easy API access
@app.function(
    image=image,
    timeout=60,
)
@modal.fastapi_endpoint(method="POST")
def generate_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Web endpoint for single token generation.

    Request body:
    {
        "text": "Your input text",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": true,
        "return_logits": false
    }

    For batch generation:
    {
        "texts": ["text1", "text2", ...],
        "temperature": 1.0,
        ...
    }
    """
    # Check if batch mode
    if "texts" in request:
        texts = request.get("texts", [])
        if not texts:
            return {"error": "No texts provided for batch generation"}

        params = {
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 0.95),
            "top_k": request.get("top_k", 50),
            "do_sample": request.get("do_sample", True),
        }

        # Get or create inference instance
        inference = Pythia12BSnapshotInference()

        try:
            results = inference.batch_generate.remote(texts, **params)
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Single generation mode
    text = request.get("text", "")
    if not text:
        return {"error": "No text provided"}

    # Get generation parameters
    params = {
        "temperature": request.get("temperature", 1.0),
        "top_p": request.get("top_p", 0.95),
        "top_k": request.get("top_k", 50),
        "do_sample": request.get("do_sample", True),
        "return_logits": request.get("return_logits", False),
    }

    # Get or create inference instance
    inference = Pythia12BSnapshotInference()

    # Generate single token
    try:
        result = inference.generate_single_token.remote(text, **params)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Health check endpoint
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health_check() -> Dict[str, str]:
    """Simple health check endpoint"""
    inference = Pythia12BSnapshotInference()
    info = inference.get_model_info.remote()
    return {
        "status": "healthy",
        "model": "pythia-12b",
        "snapshot_enabled": info.get("snapshot_enabled", False),
        "cuda_available": info.get("cuda_available", False),
    }


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
        modal run pythia_12b_modal_snapshot.py --text "Your input text"
        modal run pythia_12b_modal_snapshot.py --batch  # Test batch mode
    """
    print(f"\n{'=' * 60}")
    print("Pythia 12B Single Token Generation (GPU Snapshot Version)")
    print(f"{'=' * 60}\n")

    # Create inference instance (Modal remote function call)
    inference = Pythia12BSnapshotInference()

    # Get model info
    print("Model Information:")
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
    print("   Deploy with: modal deploy pythia_12b_modal_snapshot.py")
