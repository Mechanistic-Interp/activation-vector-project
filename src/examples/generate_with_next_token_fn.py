"""
Iteratively call an existing generate_next_token function for N tokens.

This is a thin CLI wrapper that imports your function and loops it.
It does NOT reimplement generation logic; it simply calls your function
repeatedly, appending each new token to the text.

Assumptions about your function (best-effort heuristics):
  - Module: "generate_next_token" (file generate_next_token.py at repo root)
  - Function: "generate_next_token"
  - Call signature: fn(text: str, **kwargs) -> result
  - Result may be one of:
      * dict with keys: next_token (str), full_text (str, optional), next_token_id (int, optional)
      * tuple/list like (next_token, next_token_id?)
      * str: the next token text

If your function differs, pass a dotted path with --fn, e.g.:
  modal run -m src.examples.generate_with_next_token_fn --fn "mypkg.module:my_func" --text "Hello" --num-tokens 20
or run locally:
  python -m src.examples.generate_with_next_token_fn --text "Hello" --num-tokens 20
"""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Dict, Optional, Tuple


def _load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _import_fn(dotted: str):
    """Import a function from a dotted path like "module.sub:func" or "module.sub.func"."""
    if ":" in dotted:
        mod_name, fn_name = dotted.split(":", 1)
    else:
        # Support module.sub.func
        parts = dotted.split(".")
        mod_name, fn_name = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn


def _coerce_result(
    result: Any,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Extract (next_token, full_text, next_token_id) from a variety of shapes."""
    next_token: Optional[str] = None
    full_text: Optional[str] = None
    next_token_id: Optional[int] = None

    if isinstance(result, dict):
        # Common keys
        next_token = (
            result.get("next_token")
            or result.get("token")
            or result.get("piece")
            or result.get("next")
        )
        full_text = result.get("full_text") or result.get("output") or result.get("text")
        nid = result.get("next_token_id") or result.get("token_id") or result.get("id")
        if isinstance(nid, int):
            next_token_id = nid
    elif isinstance(result, (tuple, list)):
        if len(result) >= 1 and isinstance(result[0], str):
            next_token = result[0]
        if len(result) >= 2 and isinstance(result[1], int):
            next_token_id = result[1]
    elif isinstance(result, str):
        next_token = result

    return next_token, full_text, next_token_id


def main():
    ap = argparse.ArgumentParser(description="Iterative generation wrapper for generate_next_token")
    ap.add_argument("--text", type=str, help="Starting text prompt")
    ap.add_argument("--file", type=str, help="Path to file containing starting text")
    ap.add_argument("--num-tokens", type=int, default=20, help="Number of tokens to generate")
    ap.add_argument("--fn", type=str, default="generate_next_token:generate_next_token", help="Dotted path to function (module:func)")
    ap.add_argument("--fn-kwargs", type=str, default="{}", help="JSON dict of extra kwargs to pass on each call")
    ap.add_argument("--stop-on-eos", type=str, default="true", help="Stop early if EOS encountered (requires id or token string)")
    ap.add_argument("--eos-id", type=int, help="EOS token id for early stop (optional)")
    ap.add_argument("--eos-token", type=str, help="EOS token string for early stop (optional)")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-token output")

    args = ap.parse_args()

    if args.text and args.file:
        raise SystemExit("Provide either --text or --file, not both")
    if not args.text and not args.file:
        args.text = "The meaning of life is"

    start_text = _load_text_from_file(args.file) if args.file else args.text

    try:
        extra_kwargs: Dict[str, Any] = json.loads(args.fn_kwargs)
        if not isinstance(extra_kwargs, dict):
            raise ValueError
    except Exception:
        raise SystemExit("--fn-kwargs must be a JSON object")

    stop_on_eos = str(args.stop_on_eos).lower() in {"1", "true", "yes", "y"}

    fn = _import_fn(args.fn)

    current_text = start_text
    emitted = []
    stopped_early = False

    if not args.quiet:
        print("Generating:")

    for _ in range(args.num_tokens):
        result = fn(current_text, **extra_kwargs)
        next_token, full_text, next_id = _coerce_result(result)

        if next_token is None and full_text is None:
            raise SystemExit("generate_next_token did not return a recognizable result shape")

        # Prefer full_text when available; otherwise append next_token
        if full_text is not None:
            current_text = full_text
        else:
            current_text = (current_text or "") + (next_token or "")

        if next_token is not None:
            emitted.append(next_token)
            if not args.quiet:
                print(next_token, end="", flush=True)

        if stop_on_eos:
            if args.eos_id is not None and next_id is not None and args.eos_id == next_id:
                stopped_early = True
                break
            if args.eos_token is not None and next_token is not None and args.eos_token == next_token:
                stopped_early = True
                break

    if not args.quiet:
        print()
        print("\nDone.")
        print(f"Tokens generated: {len(emitted)}{' (early stop on EOS)' if stopped_early else ''}")
        preview = current_text if len(current_text) <= 800 else current_text[:800] + "..."
        print("\nFinal text (preview):\n" + preview)

    # Return info for programmatic callers
    return {
        "tokens_generated": len(emitted),
        "stopped_early": stopped_early,
        "final_text": current_text,
    }


if __name__ == "__main__":
    main()

