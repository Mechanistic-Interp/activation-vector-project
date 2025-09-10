"""
Volume-related helpers shared across scripts.

These utilities assume the Modal function calling them has the appropriate
volume mounted (e.g., volumes={"/training_data": training_volume}).
"""

import glob
import os
from typing import Optional


def find_latest_corpus_mean_path(base_dir: str = "/training_data/corpus_mean_output") -> str:
    """Return the latest corpus mean safetensors path in a mounted volume.

    Looks for files matching f"{base_dir}/corpus_mean_*.safetensors" and returns
    the one with the greatest numeric timestamp suffix.
    """
    pattern = os.path.join(base_dir, "corpus_mean_*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No corpus mean files found in {base_dir}. Run corpus mean aggregation first."
        )

    def ts(f: str) -> int:
        try:
            base = os.path.basename(f)
            return int(base.split("_")[-1].split(".")[0])
        except Exception:
            return -1

    files.sort(key=ts, reverse=True)
    return files[0]

