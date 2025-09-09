import modal
import random
import pickle
import os
from typing import List

app = modal.App("training-data-fetcher")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "torch", "huggingface_hub"
)

volume = modal.Volume.from_name("pile-dataset-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/cache": volume},
    timeout=60 * 10,  # 10 minutes
    memory=2048,  # 2GB RAM
)
def fetch_and_cache_data(
    num_samples: int = 5000, seed: int = 42, overwrite: bool = False
) -> List[str]:
    """
    Fetch training data from C4 dataset and cache it.

    Args:
        num_samples: Number of text samples to return (default: 5000)
        seed: Random seed for reproducibility (default: 42)
        overwrite: If True, ignore cache and fetch fresh data (default: False)

    Returns:
        List of text strings from C4 dataset
    """
    from datasets import load_dataset

    os.environ["HF_DATASETS_CACHE"] = "/cache"
    cache_file = "/cache/training_data.pkl"

    # Check cache first (unless overwriting)
    if not overwrite and os.path.exists(cache_file):
        print("Loading cached training data...")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            print(f"Found {len(cached_data)} samples in existing cache")
            if len(cached_data) >= num_samples:
                print(f"Using {num_samples} samples from {len(cached_data)} cached")
                random.seed(seed)
                return (
                    random.sample(cached_data, num_samples)
                    if len(cached_data) > num_samples
                    else cached_data
                )
        except Exception as e:
            print(f"Cache error: {e}, fetching fresh data...")
    elif overwrite:
        print(f"Overwrite=True, deleting existing cache if it exists")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Deleted old cache file")

    # Fetch fresh data
    print(f"Fetching {num_samples} samples from C4 dataset...")
    try:
        dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
        random.seed(seed)

        # Buffer more samples for better randomness
        buffer_size = num_samples * 2  # Remove arbitrary cap
        buffer = []

        for i, example in enumerate(dataset):
            if i >= buffer_size:
                break
            if example["text"] and len(example["text"].strip()) > 50:
                buffer.append(example["text"])

        # Sample from buffer
        selected_texts = (
            random.sample(buffer, num_samples) if len(buffer) >= num_samples else buffer
        )

        # Save to cache (save the buffer for future use)
        with open(cache_file, "wb") as f:
            pickle.dump(buffer, f)
        volume.commit()  # Make changes visible to other function calls
        print(f"Cached {len(buffer)} samples, returning {len(selected_texts)}")

        return selected_texts

    except Exception as e:
        print(f"Error: {e}")
        return []


@app.local_entrypoint()
def main(
    num_samples: int = 5000,
    seed: int = 42,
    overwrite: bool = False,
    save_local: bool = False,
    local_path: str = "src/training_data/training_data.pkl",
):
    """
    Fetch training data and optionally save locally.

    Args:
        num_samples: Number of samples to fetch (default: 5000)
        seed: Random seed (default: 42)
        overwrite: Ignore cache and fetch fresh data (default: False)
        save_local: Download data to local filesystem (default: False)
        local_path: Local file path to save data (default: training_data.pkl)
    """
    # Fetch data (will use cache unless overwrite=True)
    print(f"Fetching {num_samples} samples (overwrite={overwrite})...")
    texts = fetch_and_cache_data.remote(
        num_samples=num_samples, seed=seed, overwrite=overwrite
    )

    if not texts:
        print("No data retrieved")
        return

    print(f"✅ Retrieved {len(texts)} samples")

    # Show preview
    if texts:
        print(f"\nFirst sample preview:")
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
