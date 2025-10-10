"""
Example: Iteratively generate N tokens using the deployed Pythia 12B model.

This script calls the deployed `Pythia12BSnapshotInference` Modal class in a loop
to produce multiple tokens, appending each new token to the input text.

Usage examples:
  modal run -m src.examples.generate_tokens --text "The capital of France is" --num-tokens 10
  modal run -m src.examples.generate_tokens --text "Once upon a time" --num-tokens 50 --temperature 0.8
  modal run -m src.examples.generate_tokens --file story.txt --num-tokens 25 --do-sample false

Notes:
  - By default, the loop stops early if the model emits the EOS token.
    Use --stop-on-eos false to force generation of exactly N tokens.
  - This example calls the 1-token endpoint repeatedly; for long outputs,
    consider adding a multi-token method to the Modal class.
"""

from __future__ import annotations

from typing import Optional

import modal
from modal import enable_output

# Reference the deployed Pythia 12B model class from the aggregated app
Pythia12BInference = modal.Cls.from_name(
    "activation-vector-project", "Pythia12BSnapshotInference"
)

app = modal.App("example-generate-n-tokens")


def _load_text_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


@app.local_entrypoint()
def main(
    text: Optional[str] = None,
    file: Optional[str] = None,
    num_tokens: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    stop_on_eos: bool = True,
    quiet: bool = False,
):
    """
    Iteratively generate multiple tokens from a starting text prompt.

    Args:
        text: Input text prompt (use either text OR file, not both)
        file: Path to a text file (use either text OR file, not both)
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to sample or use greedy decoding
        stop_on_eos: Stop early if the model outputs the EOS token
        quiet: If true, only print the final result (no per-token output)
    """

    # Validate input - must provide either text or file, not both
    if text and file:
        raise ValueError("Please provide either --text or --file, not both")
    if not text and not file:
        text = "The meaning of life is"  # sensible default

    if file:
        base_text = _load_text_from_file(file)
    else:
        base_text = text

    # Construct model client
    model = Pythia12BInference()

    # Fetch model info (for EOS id) with output enabled
    with enable_output():
        info = model.get_model_info.remote()

    eos_token_id = info.get("eos_token_id")  # may be None if not exposed

    # Show header
    if not quiet:
        print("\n" + "=" * 60)
        print("🤖 Pythia 12B Multi-Token Generation")
        print("=" * 60)
        preview = (base_text[:200] + "...") if len(base_text) > 200 else base_text
        print(f"\n📝 Prompt: {preview!r}")
        print(
            f"⚙️  Settings: temp={temperature}, top_p={top_p}, top_k={top_k}, sample={do_sample}"
        )
        print(f"🎯 Target tokens: {num_tokens}  |  Stop on EOS: {stop_on_eos}")
        if eos_token_id is None and stop_on_eos:
            print("⚠️  EOS id not available from server; stop_on_eos will be ignored.")
        print("\n🔁 Generating...")

    # Iterative generation loop
    current_text = base_text
    emitted_tokens = []
    stopped_early = False

    try:
        for i in range(num_tokens):
            with enable_output():
                result = model.generate_single_token.remote(
                    text=current_text,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    return_logits=False,
                )

            tok = result["next_token"]
            tok_id = result["next_token_id"]
            current_text = result["full_text"]
            emitted_tokens.append(tok)

            if not quiet:
                # Print tokens as they arrive (no newline, flush)
                print(tok, end="", flush=True)

            if stop_on_eos and eos_token_id is not None and tok_id == eos_token_id:
                stopped_early = True
                break

    except KeyboardInterrupt:
        if not quiet:
            print("\n⏹️  Interrupted; returning partial result.")

    # Final output
    if not quiet:
        print("\n\n" + "=" * 60)
        print("✅ Done")
        print("=" * 60)
        print(
            f"Tokens generated: {len(emitted_tokens)}{' (early stop on EOS)' if stopped_early else ''}"
        )
        print("\n📄 Final text (truncated preview):")
        display = (
            current_text if len(current_text) <= 600 else current_text[:600] + "..."
        )
        print(display)

    return {
        "success": True,
        "tokens_generated": len(emitted_tokens),
        "stopped_early": stopped_early,
        "final_text": current_text,
    }
