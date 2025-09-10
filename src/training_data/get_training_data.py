import modal
import random
import pickle
import os
from typing import List

app = modal.App("training-data-fetcher")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "torch", "huggingface_hub"
)

volume = modal.Volume.from_name("training-data-volume", create_if_missing=True)


def filter_by_word_count(text: str, min_words: int, max_words: int) -> bool:
    """
    Check if text falls within word count range.

    Args:
        text: Text to check
        min_words: Minimum word count (inclusive)
        max_words: Maximum word count (inclusive)

    Returns:
        True if text is within word count range, False otherwise
    """
    if not text or not text.strip():
        return False

    word_count = len(text.split())
    return min_words <= word_count <= max_words


@app.function(
    image=image,
    volumes={"/cache": volume},
    timeout=60 * 10,  # 10 minutes
    memory=2048,  # 2GB RAM
)
def fetch_and_cache_data(
    num_samples: int = 5000,
    seed: int = 42,
    overwrite: bool = False,
    word_count_min: int = 700,
    word_count_max: int = 1500,
) -> List[str]:
    """
    Fetch training data from C4 dataset and cache it with word count filtering.

    Args:
        num_samples: Number of text samples to return (default: 5000)
        seed: Random seed for reproducibility (default: 42)
        overwrite: If True, ignore cache and fetch fresh data (default: False)
        word_count_min: Minimum word count for filtering (default: 700)
        word_count_max: Maximum word count for filtering (default: 1500)

    Returns:
        List of text strings from C4 dataset that meet word count criteria
    """
    from datasets import load_dataset

    os.environ["HF_DATASETS_CACHE"] = "/cache"
    # Include word count range in cache filename for different configurations
    cache_file = "/cache/training_data.pkl"

    # Check cache first (unless overwriting)
    if not overwrite and os.path.exists(cache_file):
        print("Loading cached training data...")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            print(f"Found {len(cached_data)} samples in existing cache")
            # Verify cached data meets current word count criteria
            valid_cached = [
                text
                for text in cached_data
                if filter_by_word_count(text, word_count_min, word_count_max)
            ]
            print(
                f"Found {len(valid_cached)} valid samples from {len(cached_data)} cached (word range: {word_count_min}-{word_count_max})"
            )

            if len(valid_cached) >= num_samples:
                print(
                    f"Using {num_samples} samples from {len(valid_cached)} valid cached"
                )
                random.seed(seed)
                return (
                    random.sample(valid_cached, num_samples)
                    if len(valid_cached) > num_samples
                    else valid_cached
                )
        except Exception as e:
            print(f"Cache error: {e}, fetching fresh data...")
    elif overwrite:
        print(f"Overwrite=True, deleting existing cache if it exists")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Deleted old cache file")

    # Fetch fresh data with adaptive sampling
    print(
        f"Fetching {num_samples} samples from C4 dataset (word range: {word_count_min}-{word_count_max})..."
    )
    try:
        dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
        random.seed(seed)

        # Adaptive sampling - keep fetching until we have enough valid samples
        valid_samples = []
        all_samples_processed = 0
        rejected_too_short = 0
        rejected_word_count = 0

        # Estimate we might need 3-5x more samples due to filtering
        max_to_process = num_samples * 10  # Safety limit

        print("Starting adaptive sampling with word count filtering...")

        for i, example in enumerate(dataset):
            if len(valid_samples) >= num_samples:
                print(f"✅ Collected {len(valid_samples)} valid samples")
                break

            if i >= max_to_process:
                print(
                    f"⚠️  Reached processing limit ({max_to_process}), stopping with {len(valid_samples)} valid samples"
                )
                break

            all_samples_processed += 1

            # Basic text validation
            if not example["text"] or len(example["text"].strip()) <= 50:
                rejected_too_short += 1
                continue

            # Word count filtering
            text = example["text"].strip()
            if filter_by_word_count(text, word_count_min, word_count_max):
                valid_samples.append(text)
            else:
                rejected_word_count += 1

            # Progress reporting every 1000 samples
            if all_samples_processed % 1000 == 0:
                acceptance_rate = (
                    len(valid_samples) / all_samples_processed * 100
                    if all_samples_processed > 0
                    else 0
                )
                print(
                    f"Progress: {len(valid_samples)}/{num_samples} valid samples | "
                    f"Processed: {all_samples_processed} | Acceptance rate: {acceptance_rate:.1f}%"
                )

        # Final statistics
        acceptance_rate = (
            len(valid_samples) / all_samples_processed * 100
            if all_samples_processed > 0
            else 0
        )
        print(f"\nFiltering Statistics:")
        print(f"  Total processed: {all_samples_processed}")
        print(f"  Valid samples: {len(valid_samples)}")
        print(f"  Rejected (too short): {rejected_too_short}")
        print(f"  Rejected (word count): {rejected_word_count}")
        print(f"  Acceptance rate: {acceptance_rate:.1f}%")

        # Save to cache (save all valid samples for future use)
        if valid_samples:
            with open(cache_file, "wb") as f:
                pickle.dump(valid_samples, f)
            volume.commit()  # Make changes visible to other function calls
            print(f"✅ Cached {len(valid_samples)} valid samples")

        return (
            valid_samples[:num_samples]
            if len(valid_samples) >= num_samples
            else valid_samples
        )

    except Exception as e:
        print(f"Error: {e}")
        return []


@app.local_entrypoint()
def main(
    num_samples: int = 5000,
    seed: int = 42,
    overwrite: bool = True,
    save_local: bool = True,
    local_path: str = "src/training_data/training_data.pkl",
    word_count_min: int = 700,
    word_count_max: int = 1500,
):
    """
    Fetch training data and optionally save locally.

    Args:
        num_samples: Number of samples to fetch (default: 5000)
        seed: Random seed (default: 42)
        overwrite: Ignore cache and fetch fresh data (default: False)
        save_local: Download data to local filesystem (default: False)
        local_path: Local file path to save data (default: training_data.pkl)
        word_count_min: Minimum word count for filtering (default: 700)
        word_count_max: Maximum word count for filtering (default: 1500)

    Examples:
        modal run src/training_data/get_training_data.py
        modal run src/training_data/get_training_data.py --num_samples 1000 --word_count_min 500 --word_count_max 2000
        modal run src/training_data/get_training_data.py --save_local --overwrite
    """
    # Fetch data (will use cache unless overwrite=True)
    print(
        f"Fetching {num_samples} samples with word count {word_count_min}-{word_count_max} (overwrite={overwrite})..."
    )
    texts = fetch_and_cache_data.remote(
        num_samples=num_samples,
        seed=seed,
        overwrite=overwrite,
        word_count_min=word_count_min,
        word_count_max=word_count_max,
    )

    if not texts:
        print("No data retrieved")
        return

    print(f"✅ Retrieved {len(texts)} samples")

    # Show preview with word count info
    if texts:
        first_sample_words = len(texts[0].split())
        print(f"\nFirst sample preview ({first_sample_words} words):")
        print(f"{texts[0][:200]}...")

    # Download locally if requested
    if save_local:
        print(f"\nSaving to {local_path}...")
        try:
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        except Exception:
            pass
        with open(local_path, "wb") as f:
            pickle.dump(texts, f)
        print(f"✅ Saved {len(texts)} samples to {local_path}")
    else:
        print(f"\nData cached in Modal volume. To download locally:")
        print(f"modal volume get pile-dataset-cache training_data.pkl ./{local_path}")
