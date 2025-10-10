"""
Example: Generate next token using deployed Pythia 12B model.

This script demonstrates how to call the deployed Pythia12BSnapshotInference
model to generate the next token for given text input.

Usage examples:
  modal run -m src.examples.generate_next_token --text "The capital of France is"
  modal run -m src.examples.generate_next_token --text "Once upon a time" --temperature 0.8
  modal run -m src.examples.generate_next_token --file story.txt --do-sample false
"""

from __future__ import annotations

from typing import Optional

import modal
from modal import enable_output

# Reference the deployed Pythia 12B model class
Pythia12BInference = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BSnapshotInference"
)

app = modal.App("generate-next-token")


def _load_text_from_file(file_path: str) -> str:
    """Load text from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


@app.function()
def generate_next_token(
    text: str,
    temperature: float = 0.01,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    return_logits: bool = False,
):
    model = Pythia12BInference()
    result = model.generate_single_token.remote(
        text=text,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        return_logits=return_logits,
    )
    return result


@app.local_entrypoint()
def main(
    text: Optional[str] = None,
    file: Optional[str] = None,
    temperature: float = 0.01,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    return_logits: bool = False,
):
    with modal.enable_output():
        """
        Generate the next token using the deployed Pythia 12B model.

        Args:
            text: Input text prompt (use either text OR file, not both)
            file: Path to text file (use either text OR file, not both)
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            return_logits: Whether to return top logits for vocabulary
        """
        # Validate input - must provide either text or file, not both
        if text and file:
            raise ValueError("Please provide either --text or --file, not both")
        if not text and not file:
            # Default text if neither provided
            text = "The meaning of life is"
            print("ℹ️  No input provided, using default text")

        print("\n" + "=" * 60)
        print("🤖 Pythia 12B Next Token Generation")
        print("=" * 60 + "\n")

        # Create model instance
        model = Pythia12BInference()

        # Get model info with output enabled to see logs
        print("📊 Getting model information...")
        with enable_output():
            info = model.get_model_info.remote()

        print("\n📋 Model Details:")
        print(f"   Model: {info['model_name']}")
        print(f"   Parameters: {info['num_parameters']:,}")
        print(f"   Vocab size: {info['vocab_size']:,}")
        print(f"   Max context: {info['max_length']} tokens")
        print(
            f"   GPU snapshot: {'✅ Enabled' if info['snapshot_enabled'] else '❌ Disabled'}"
        )

        print("\n" + "=" * 60)
        print("💬 SINGLE TOKEN GENERATION")
        print("=" * 60)

        # Load input text
        if file:
            actual_text = _load_text_from_file(file)
            print(f"\n📁 Loaded text from file: {file}")
        else:
            actual_text = text

        # Show preview for long texts
        text_preview = (
            (actual_text[:200] + "...") if len(actual_text) > 200 else actual_text
        )
        print(f'\n📝 Input text: "{text_preview}"')
        print(f"   Length: {len(actual_text)} characters")

        # Show generation parameters
        print("\n⚙️  Generation settings:")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")
        print(f"   Top-k: {top_k}")
        print(f"   Sampling: {'enabled' if do_sample else 'greedy decoding'}")
        print(f"   Return logits: {'yes' if return_logits else 'no'}")

        print("\n🔮 Generating next token...")

        # Generate next token with output enabled to see logs
        with enable_output():
            result = model.generate_single_token.remote(
                text=actual_text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                return_logits=return_logits,
            )

        # Display results
        print("\n" + "=" * 60)
        print("✅ GENERATION COMPLETE")
        print("=" * 60)

        print(f'\n🎯 Next token: "{result["next_token"]}"')
        print(f"   Token ID: {result['next_token_id']}")
        print(f"   Input tokens: {result['input_length']}")

        # Show complete text (truncated if too long)
        full_text = result["full_text"]
        if len(full_text) > 300:
            full_text_display = full_text[:300] + "..."
        else:
            full_text_display = full_text
        print("\n📄 Complete text:")
        print(f'   "{full_text_display}"')

        # Show top logits if returned
        if "top_logits" in result:
            print("\n📊 Top 10 token predictions:")
            top_logits = result["top_logits"]
            for i in range(min(10, len(top_logits["indices"]))):
                token_id = top_logits["indices"][i]
                logit_value = top_logits["values"][i]
                # Note: We can't decode token IDs without the tokenizer
                print(f"   {i + 1}. Token ID {token_id}: {logit_value:.4f}")

        print("\n" + "=" * 60)
        print("💡 Tips:")
        print("   • Lower temperature (0.1-0.5) = more focused/deterministic")
        print("   • Higher temperature (0.8-1.5) = more creative/random")
        print("   • Use --do-sample false for deterministic output")
        print("   • GPU snapshots make subsequent calls much faster!")
        print("=" * 60 + "\n")

        return {
            "success": True,
            "input_source": "file" if file else "text",
            "sampling_enabled": do_sample,
            "temperature": temperature,
        }
