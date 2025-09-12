# Activation Vector Mean Computation - Implementation Todo

**Date**: 2025-01-18  
**Goal**: Create parallel processing system to compute mean activation vectors from training data

## Phase 1: Setup and Data Loading

- [ ] **Create new file**: `src/compute_activation_mean.py`
- [ ] **Import dependencies**: Modal, pickle, torch, existing extract_vector module
- [ ] **Setup Modal app** with proper GPU configuration (A100-80GB)
- [ ] **Create volume** for storing activation vectors: `activation-vectors-cache`
- [ ] **Load training data** from existing `training_data.pkl` (5k-10k samples)

## Phase 2: Parallel Vector Extraction

- [ ] **Create batch processing function** using Modal's `.map()`:
  ```python
  @app.function(gpu="A100", volumes={"/vectors": volume})
  def extract_single_vector(text: str, mode: str = "short") -> dict
  ```
- [ ] **Implement chunking strategy** (process 100-500 texts per batch to manage memory)
- [ ] **Use existing `Pythia12BActivationExtractor`** for vector extraction
- [ ] **Configure target layers** [25, 26, 27] and pooling mode ("short" or "long")
- [ ] **Add error handling** for failed extractions

## Phase 3: Storage and Incremental Processing

- [ ] **Design volume structure**:
  ```
  /vectors/
  ├── training_data.pkl (input)
  ├── vectors_short/batch_000.pkl, batch_001.pkl...
  ├── vectors_long/batch_000.pkl, batch_001.pkl...
  ├── mean_vector_short.pkl
  ├── mean_vector_long.pkl
  └── metadata.json
  ```
- [ ] **Implement incremental saving** to avoid OOM issues
- [ ] **Add progress tracking** and resume capability
- [ ] **Use volume.commit()** after each batch save

## Phase 4: Mean Computation

- [ ] **Create mean calculation function**:
  ```python
  @app.function(volumes={"/vectors": volume})
  def compute_mean_from_batches(mode: str) -> dict
  ```
- [ ] **Load all saved vectors** from batch files
- [ ] **Compute mean** with proper numerical stability (fp32)
- [ ] **Handle both "short" and "long" modes**
- [ ] **Normalize final mean vector** (L2 normalization)

## Phase 5: Integration and Testing

- [ ] **Create local entrypoint** with configurable parameters:
  - `num_samples`: How many training samples to process
  - `batch_size`: Vectors per batch (default: 200)
  - `mode`: "short", "long", or "both"
  - `overwrite`: Clear existing vectors and recompute
- [ ] **Add comprehensive logging** and progress indicators
- [ ] **Test with small dataset** (100 samples) first
- [ ] **Validate mean vector dimensions** match expected sizes

## Phase 6: Optimization and Production

- [ ] **Benchmark processing speed** (target: 1000+ samples/hour)
- [ ] **Implement checkpointing** for long-running jobs
- [ ] **Add memory monitoring** to prevent OOM
- [ ] **Create utility functions**:
  - Load mean vector for comparison
  - Compute cosine similarity against mean
  - Export mean vectors to different formats

## Technical Implementation Notes

### Modal Configuration
```python
@app.function(
    gpu="A100-80GB",
    memory=65536,  # 64GB RAM
    volumes={"/vectors": volume},
    timeout=60*30,  # 30 min timeout
    enable_memory_snapshot=True
)
```

### Parallel Processing Pattern
```python
# Use Modal's .map() for parallel extraction
texts = load_training_data()
vectors = extract_single_vector.map(texts, mode="short")

# Save in batches to volume
for i, batch in enumerate(chunk(vectors, batch_size=200)):
    save_vectors_batch(batch, i)
```

### Mean Computation Strategy
- Load vectors incrementally to avoid memory issues
- Use fp32 for numerical stability
- Apply L2 normalization to final mean
- Save both raw mean and normalized mean

## Success Criteria

✅ **Performance**: Process 5,000 training samples in < 2 hours  
✅ **Memory**: Handle large datasets without OOM errors  
✅ **Accuracy**: Mean vectors have expected dimensions (5120 for short, 20480 for long)  
✅ **Reproducibility**: Same mean vector for same input data and seed  
✅ **Usability**: Simple CLI interface for running mean computation  

## Future Extensions

- [ ] Support for different target layers combinations
- [ ] Multiple pooling strategies comparison
- [ ] Incremental mean updates (add new samples without full recompute)
- [ ] Integration with existing activation comparison workflows
- [ ] Export to common ML formats (safetensors, numpy, etc.)

---

**Dependencies**: Requires completed training data pipeline (`get_training_data.py`) and activation extraction (`extract_vector.py`)
