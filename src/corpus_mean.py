"""
Parallel Corpus Mean Computation for Pythia-12B Activation Vectors

Uses Modal's .spawn_map() for fire-and-forget parallel processing.
Each document is processed independently and saved as individual .safetensors files.

Two-Stage Architecture:
1. Extract: Process 1000+ documents in parallel, save activation matrices individually
2. Aggregate: Load all saved matrices and compute corpus mean with checkpointing

Features:
- Fire-and-forget processing: Launch all jobs and disconnect
- Perfect fault tolerance: Each document saved independently
- Checkpointed aggregation: Auto-resume from failures
- Detached execution: Computer can disconnect during aggregation

Usage:
    modal run src/corpus_mean.py extract --max_docs 1000
    modal run src/corpus_mean.py aggregate
"""

import modal
import torch
from typing import List, Dict, Any, Optional
import os
import json
import time
import hashlib
import glob

# Reference the deployed extractor class
Pythia12BExtractor = modal.Cls.from_name(
    "activation-vector-project",
    "Pythia12BActivationExtractor",
)

# Reference training data volume
training_volume = modal.Volume.from_name("training-data-volume", create_if_missing=True)

# Create corpus mean computation app
app = modal.App("corpus-mean-extraction")

# Define the container image
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.1.0",
    "safetensors==0.4.1",
)


def generate_document_id(text: str) -> str:
    """Generate stable hash ID for document."""
    text_normalized = text.strip()
    doc_hash = hashlib.sha256(text_normalized.encode("utf-8")).hexdigest()
    return doc_hash[:16]


def load_training_documents(
    data_dir: str, max_docs: int, start_index: int = 0
) -> List[str]:
    """Load training documents from pickle file."""
    import pickle

    # Files are at root level of mounted volume
    # Volume is mounted at /training_data, files are directly there
    pickle_path = f"{data_dir}/training_data.pkl"
    print(f"Loading documents from {pickle_path}...")

    try:
        with open(pickle_path, "rb") as f:
            all_documents = pickle.load(f)

        print(f"Loaded {len(all_documents)} documents from pickle file")
        end_index = start_index + max_docs
        documents = (
            all_documents[start_index:end_index]
            if len(all_documents) > end_index
            else all_documents[start_index:]
        )
        print(
            f"Using documents {start_index}-{start_index + len(documents) - 1} ({len(documents)} documents) for processing"
        )

        return documents
    except Exception as e:
        print(f"Error loading pickle file {pickle_path}: {str(e)}")
        # Try the alternative filename
        alt_pickle_path = f"{data_dir}/training_data_700_1500.pkl"
        print(f"Trying alternative path: {alt_pickle_path}...")
        try:
            with open(alt_pickle_path, "rb") as f:
                all_documents = pickle.load(f)
            print(f"Loaded {len(all_documents)} documents from alternative file")
            end_index = start_index + max_docs
            documents = (
                all_documents[start_index:end_index]
                if len(all_documents) > end_index
                else all_documents[start_index:]
            )
            print(
                f"Using documents {start_index}-{start_index + len(documents) - 1} ({len(documents)} documents) for processing"
            )
            return documents
        except Exception as e2:
            print(f"Error loading alternative pickle file: {str(e2)}")
            return []


@app.function(
    image=image,
    volumes={"/training_data": training_volume},
    retries=3,
    timeout=600,  # 10 minutes per document
)
def extract_single_document(
    document_text: str,
    doc_index: int,
    word_count_min: int = 700,
    word_count_max: int = 1500,
) -> Dict[str, Any]:
    """
    Extract activation matrix for a single document and save to volume.

    Each container processes ONE document independently and saves the result.
    Perfect fault tolerance - if this fails, only this document needs reprocessing.

    Args:
        document_text: Raw text content
        doc_index: Document index for logging
        word_count_min: Minimum word count for filtering
        word_count_max: Maximum word count for filtering

    Returns:
        Processing result dictionary
    """
    from safetensors.torch import save_file

    try:
        # Basic validation
        if not document_text or not document_text.strip():
            return {
                "status": "skipped",
                "reason": "empty_document",
                "doc_index": doc_index,
            }

        text = document_text.strip()
        word_count = len(text.split())

        # Word count filtering
        if not (word_count_min <= word_count <= word_count_max):
            return {
                "status": "skipped",
                "reason": "word_count_filter",
                "word_count": word_count,
                "doc_index": doc_index,
            }

        print(f"🔧 Processing document {doc_index}: {word_count} words")

        # Create instance of deployed extractor
        extractor = Pythia12BExtractor()

        # Extract activation matrix
        result = extractor.get_activation_matrix.remote(text=text)

        if result is None:
            return {
                "status": "skipped",
                "reason": "extractor_filtered",
                "word_count": word_count,
                "doc_index": doc_index,
            }

        # Convert to tensor
        activation_matrix = torch.tensor(
            result["activation_matrix"], dtype=torch.float32
        )

        # Generate unique document ID
        doc_id = generate_document_id(text)

        # Create activations directory
        activations_dir = "/training_data/activations"
        os.makedirs(activations_dir, exist_ok=True)

        # Save activation matrix with unique filename
        save_path = f"{activations_dir}/doc_{doc_id}.safetensors"
        save_file({"activation_matrix": activation_matrix}, save_path)

        print(
            f"✅ Saved document {doc_index}: {list(activation_matrix.shape)} → {save_path}"
        )

        return {
            "status": "completed",
            "doc_index": doc_index,
            "doc_id": doc_id,
            "word_count": word_count,
            "shape": list(activation_matrix.shape),
            "save_path": save_path,
            "tokens": activation_matrix.shape[1],
        }

    except Exception as e:
        print(f"❌ Error processing document {doc_index}: {str(e)}")
        return {"status": "error", "doc_index": doc_index, "error": str(e)}


def update_rolling_mean(
    corpus_mean: Optional[torch.Tensor],
    counts: Optional[torch.Tensor],
    new_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update rolling mean with new activation matrix."""
    d_model, seq_len = new_matrix.shape
    assert d_model == 5120, f"Expected d_model=5120, got {d_model}"

    # Initialize if this is the first document
    if corpus_mean is None:
        corpus_mean = torch.zeros(5120, seq_len, dtype=torch.float32)
        counts = torch.zeros(seq_len, dtype=torch.int64)

    # Expand if new document is longer than current max
    elif seq_len > corpus_mean.shape[1]:
        old_len = corpus_mean.shape[1]

        # Create new expanded tensors
        new_corpus_mean = torch.zeros(5120, seq_len, dtype=torch.float32)
        new_counts = torch.zeros(seq_len, dtype=torch.int64)

        # Copy existing data
        new_corpus_mean[:, :old_len] = corpus_mean
        new_counts[:old_len] = counts

        corpus_mean, counts = new_corpus_mean, new_counts

    # Update rolling mean for each token position
    for pos in range(seq_len):
        N = counts[pos].item()
        corpus_mean[:, pos] = (corpus_mean[:, pos] * N + new_matrix[:, pos]) / (N + 1)
        counts[pos] += 1

    return corpus_mean, counts


# Checkpoint System for Robust Aggregation


def save_checkpoint(
    corpus_mean: torch.Tensor,
    counts: torch.Tensor,
    processed_files: set,
    checkpoint_dir: str,
    processed_docs: int,
    total_tokens: int,
) -> str:
    """Save aggregation checkpoint to volume."""
    from safetensors.torch import save_file

    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = int(time.time())
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{timestamp}.safetensors"

    # Save tensors
    save_file({"corpus_mean": corpus_mean, "counts": counts}, checkpoint_path)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "processed_docs": processed_docs,
        "total_tokens": total_tokens,
        "processed_files": list(processed_files),
        "corpus_mean_shape": list(corpus_mean.shape),
        "checkpoint_path": checkpoint_path,
    }

    metadata_path = f"{checkpoint_dir}/checkpoint_{timestamp}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Checkpoint saved: {processed_docs} docs processed → {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Load the most recent checkpoint from directory."""
    from safetensors.torch import load_file

    if not os.path.exists(checkpoint_dir):
        return None

    # Find most recent checkpoint
    metadata_files = glob.glob(f"{checkpoint_dir}/checkpoint_*_metadata.json")
    if not metadata_files:
        return None

    # Sort by timestamp (newest first)
    metadata_files.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)
    latest_metadata_path = metadata_files[0]

    try:
        # Load metadata
        with open(latest_metadata_path, "r") as f:
            metadata = json.load(f)

        # Load tensors
        tensors = load_file(metadata["checkpoint_path"])

        print(
            f"📂 Loaded checkpoint: {metadata['processed_docs']} docs already processed"
        )

        return {
            "corpus_mean": tensors["corpus_mean"],
            "counts": tensors["counts"],
            "processed_files": set(metadata["processed_files"]),
            "processed_docs": metadata["processed_docs"],
            "total_tokens": metadata["total_tokens"],
        }

    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        return None


@app.function(
    image=image,
    volumes={"/training_data": training_volume},
    timeout=10800,  # 3 hours timeout
    memory=32768,  # 32GB for batch loading
    cpu=8,  # 8 CPU cores for parallel I/O
)
def compute_corpus_mean_checkpointed(checkpoint_interval: int = 500) -> Dict[str, Any]:
    """
    Robust corpus mean computation with automatic checkpointing and resumption.

    Features:
    - Auto-resume from last checkpoint on restart/failure
    - Save checkpoint every N files processed (default 500)
    - Duplicate file prevention via processed_files tracking
    - Same mathematical correctness as original algorithm

    Args:
        checkpoint_interval: Save checkpoint every N files (default 500)

    Returns:
        Processing results and final corpus mean path
    """
    from safetensors.torch import load_file, save_file
    import concurrent.futures

    print("=" * 80)
    print("CHECKPOINTED CORPUS MEAN AGGREGATION")
    print("=" * 80)

    # Setup directories
    activations_dir = "/training_data/activations"
    checkpoint_dir = "/training_data/checkpoints"
    output_dir = "/training_data/corpus_mean_output"

    # Find all activation files
    pattern = f"{activations_dir}/doc_*.safetensors"
    all_activation_files = glob.glob(pattern)
    print(f"Found {len(all_activation_files)} activation matrices")

    if not all_activation_files:
        raise ValueError(f"No activation matrices found in {activations_dir}")

    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_dir)

    if checkpoint_data:
        # Resume from checkpoint
        corpus_mean = checkpoint_data["corpus_mean"]
        counts = checkpoint_data["counts"]
        processed_files = checkpoint_data["processed_files"]
        processed_docs = checkpoint_data["processed_docs"]
        total_tokens = checkpoint_data["total_tokens"]
        print(f"🔄 Resuming from checkpoint: {processed_docs} docs already processed")
    else:
        # Start fresh
        corpus_mean = None
        counts = None
        processed_files = set()
        processed_docs = 0
        total_tokens = 0
        print("🆕 Starting fresh aggregation")

    # Filter out already processed files
    remaining_files = [f for f in all_activation_files if f not in processed_files]
    print(f"📝 {len(remaining_files)} files remaining to process")

    if not remaining_files:
        print("✅ All files already processed! Loading final result...")
        # Return existing result
        return {"status": "already_complete", "processed_docs": processed_docs}

    # Batch processing setup
    batch_size = 16
    batches = [
        remaining_files[i : i + batch_size]
        for i in range(0, len(remaining_files), batch_size)
    ]

    start_time = time.time()
    last_checkpoint = processed_docs

    def load_single_file(file_path):
        """Load a single activation matrix file"""
        try:
            tensors = load_file(file_path)
            return file_path, tensors["activation_matrix"]
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {str(e)}")
            return file_path, None

    print(f"🔄 Processing {len(batches)} batches with automatic checkpointing...")

    # Process batches with checkpointing
    for batch_idx, batch_files in enumerate(batches):
        # Load batch in parallel
        batch_matrices = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {
                executor.submit(load_single_file, file_path): file_path
                for file_path in batch_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path, activation_matrix = future.result()
                if activation_matrix is not None:
                    batch_matrices.append((file_path, activation_matrix))

        # Process batch sequentially (preserves math correctness)
        for file_path, activation_matrix in batch_matrices:
            # Update rolling mean
            corpus_mean, counts = update_rolling_mean(
                corpus_mean, counts, activation_matrix
            )
            processed_docs += 1
            total_tokens += activation_matrix.shape[1]
            processed_files.add(file_path)

        # Progress logging
        elapsed = time.time() - start_time
        docs_per_sec = processed_docs / elapsed if elapsed > 0 else 0
        print(
            f"📊 Batch {batch_idx + 1}/{len(batches)} | Processed {processed_docs}/{len(all_activation_files)} | Rate: {docs_per_sec:.1f}/sec"
        )

        # Checkpoint if interval reached
        if processed_docs - last_checkpoint >= checkpoint_interval:
            save_checkpoint(
                corpus_mean,
                counts,
                processed_files,
                checkpoint_dir,
                processed_docs,
                total_tokens,
            )
            last_checkpoint = processed_docs

    # Final processing
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("CHECKPOINTED AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Total docs processed: {processed_docs}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Processing rate: {processed_docs / total_time:.2f} docs/sec")
    print(f"Final corpus mean shape: {list(corpus_mean.shape)}")

    # Save final result
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    corpus_mean_path = f"{output_dir}/corpus_mean_{timestamp}.safetensors"

    save_file({"corpus_mean": corpus_mean, "counts": counts}, corpus_mean_path)
    print(f"✅ Final corpus mean saved: {corpus_mean_path}")

    # Save final metadata
    metadata = {
        "timestamp": timestamp,
        "processed_docs": processed_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": total_tokens / processed_docs,
        "corpus_mean_shape": list(corpus_mean.shape),
        "processing_time_minutes": total_time / 60,
        "algorithm": "checkpointed_rolling_mean",
        "checkpoint_interval": checkpoint_interval,
    }

    metadata_path = f"{output_dir}/corpus_mean_{timestamp}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "processed_docs": processed_docs,
        "total_time_minutes": total_time / 60,
        "docs_per_second": processed_docs / total_time,
        "final_shape": list(corpus_mean.shape),
        "corpus_mean_path": corpus_mean_path,
        "metadata_path": metadata_path,
        "status": "completed",
    }


# Detached Execution and Monitoring


@app.function(
    image=image,
    volumes={"/training_data": training_volume},
    timeout=60,
)
def check_aggregation_progress() -> Dict[str, Any]:
    """Check progress of running aggregation by examining checkpoints and final output."""

    checkpoint_dir = "/training_data/checkpoints"
    output_dir = "/training_data/corpus_mean_output"
    activations_dir = "/training_data/activations"

    # Count total files to process
    total_files = len(glob.glob(f"{activations_dir}/doc_*.safetensors"))

    # Check for final completion
    final_outputs = glob.glob(f"{output_dir}/corpus_mean_*.safetensors")
    if final_outputs:
        # Sort by timestamp, get latest
        latest_output = max(
            final_outputs, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        return {
            "status": "completed",
            "final_output": latest_output,
            "total_files": total_files,
            "message": "Aggregation completed successfully!",
        }

    # Check checkpoint progress
    checkpoint_data = load_checkpoint(checkpoint_dir)
    if checkpoint_data:
        processed = checkpoint_data["processed_docs"]
        progress_pct = (processed / total_files * 100) if total_files > 0 else 0

        return {
            "status": "in_progress",
            "processed_files": processed,
            "total_files": total_files,
            "progress_percent": round(progress_pct, 1),
            "message": f"Processing: {processed}/{total_files} files ({progress_pct:.1f}%)",
        }

    # No checkpoints found
    return {
        "status": "not_started",
        "total_files": total_files,
        "message": "No checkpoints found. Aggregation may not be running.",
    }


# Load documents with volume access
@app.function(
    image=image,
    volumes={"/training_data": training_volume},
    timeout=300,
)
def load_documents_with_volume(max_docs: int, start_index: int = 0) -> List[str]:
    """Load training documents from volume-mounted pickle file."""
    return load_training_documents("/training_data", max_docs, start_index)


# Stage 1: Fire-and-forget extraction
@app.local_entrypoint()
def extract(
    max_docs: int = 1000,
    word_count_min: int = 700,
    word_count_max: int = 1500,
):
    """
    Stage 1: Extract activation matrices for all documents in parallel.

    Uses .spawn_map() for fire-and-forget processing.
    Each document processed independently and saved to volume.

    Usage:
        modal run src/corpus_mean.py extract --max_docs 1000
    """
    print("=" * 80)
    print("STAGE 1: PARALLEL ACTIVATION EXTRACTION")
    print("=" * 80)
    print(f"Processing {max_docs} documents")
    print(f"Word count range: {word_count_min}-{word_count_max}")
    print("Using .spawn_map() for fire-and-forget processing")
    print()

    # Load training documents via Modal function with volume access
    documents = load_documents_with_volume.remote(max_docs)

    if not documents:
        raise ValueError("No training documents found")

    print(f"🚀 Launching {len(documents)} extraction jobs with .spawn_map()")
    print("Each document will be processed independently and saved to volume")
    print()

    # Fire and forget - process all documents in parallel
    extract_single_document.spawn_map(
        documents,
        range(len(documents)),
        [word_count_min] * len(documents),
        [word_count_max] * len(documents),
    )

    print("✅ All extraction jobs launched successfully!")
    print("Jobs are running in background. Use 'modal logs' to monitor progress.")
    print(
        f"Expected output: ~{len(documents)} .safetensors files in /training_data/activations/"
    )
    print()
    print("Next step: Run aggregation once all jobs complete:")
    print("    modal run src/corpus_mean.py aggregate")


# Stage 2: Aggregate saved matrices with checkpointing
@app.local_entrypoint()
def aggregate(checkpoint_interval: int = 500):
    """
    Robust corpus mean aggregation with automatic checkpointing.

    Now uses detached execution by default for fault tolerance.
    Your computer can disconnect and the job will continue running.

    Args:
        checkpoint_interval: Save checkpoint every N files (default 500)

    Usage:
        modal run src/corpus_mean.py aggregate --checkpoint_interval 500
    """
    print("🚀 Starting robust corpus mean aggregation with checkpointing...")
    print("This will launch a detached job that can survive disconnections.")
    print()

    # Launch detached checkpointed aggregation
    job = compute_corpus_mean_checkpointed.spawn(checkpoint_interval)

    print(f"✅ Detached aggregation job launched: {job}")
    print()
    print("📋 Monitor with:")
    print("  modal run src/corpus_mean.py check_progress")
    print("  modal logs corpus-mean-extraction")
    print()
    print(
        f"The job will checkpoint every {checkpoint_interval} files and auto-resume on failure."
    )

    return {"job_id": str(job), "status": "launched", "type": "detached_checkpointed"}


# Stage 1.5: Process remaining documents from existing dataset
@app.local_entrypoint()
def extract_remaining(
    max_docs: int = 4000,
    start_index: int = 1000,
    word_count_min: int = 700,
    word_count_max: int = 1500,
):
    """
    Stage 1.5: Extract activation matrices for remaining documents (1000-4999).

    Processes the remaining documents from the existing 5000-document dataset.
    Uses same parallel processing architecture as the original extract() function.

    Usage:
        modal run src/corpus_mean.py extract_remaining --max_docs 4000 --start_index 1000
    """
    print("=" * 80)
    print("STAGE 1.5: EXTRACT REMAINING DOCUMENTS")
    print("=" * 80)
    print(f"Processing {max_docs} documents starting from index {start_index}")
    print(f"Word count range: {word_count_min}-{word_count_max}")
    print("Using .spawn_map() for fire-and-forget processing")
    print()

    # Load training documents via Modal function with volume access
    documents = load_documents_with_volume.remote(max_docs, start_index)

    if not documents:
        raise ValueError("No training documents found")

    print(f"🚀 Launching {len(documents)} extraction jobs with .spawn_map()")
    print("Each document will be processed independently and saved to volume")
    print()

    # Fire and forget - process all documents in parallel
    extract_single_document.spawn_map(
        documents,
        range(start_index, start_index + len(documents)),  # Adjust indices for logging
        [word_count_min] * len(documents),
        [word_count_max] * len(documents),
    )

    print("✅ All remaining extraction jobs launched successfully!")
    print("Jobs are running in background. Use 'modal logs' to monitor progress.")
    print(
        f"Expected output: ~{len(documents)} additional .safetensors files in /training_data/activations/"
    )
    print()
    print("Next step: Run aggregation once all jobs complete:")
    print("    modal run src/corpus_mean.py aggregate")


# Monitoring
@app.local_entrypoint()
def check_progress():
    """Check the progress of a running aggregation job."""
    print("🔍 Checking aggregation progress...")

    result = check_aggregation_progress.remote()

    print(f"\n📊 Status: {result['status'].upper()}")
    print(f"📝 {result['message']}")

    if result["status"] == "in_progress":
        print(f"🗂️  Files: {result['processed_files']}/{result['total_files']}")
        print(f"📈 Progress: {result['progress_percent']}%")

    elif result["status"] == "completed":
        print(f"🎉 Final output: {result['final_output']}")

    return result
