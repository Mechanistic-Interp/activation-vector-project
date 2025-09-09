# Activation Vector Project

Pythia-12B activation vector extraction system using TransformerLens on Modal.

## Project Structure

```
activation-vector-project/
├── .claude/
│   └── settings.local.json
├── memories/
│   ├── activation_extraction_documentation.md
│   ├── activation_mean_computation_todo.md
│   ├── gpu_snapshot_optimization.md
│   ├── notes.md
│   └── pythia_12b_setup_todo.md
├── outputs/
│   ├── activation_vector_short_30.safetensors
│   └── activation_vector_short_30_metadata.json
└── src/
    ├── __init__.py
    ├── .claude/
    │   └── settings.local.json
    ├── deploy.py
    ├── extract_vector.py
    ├── pooling.py
    ├── pythia_12b_modal_snapshot.py
    └── training_data/
        ├── get_training_data.py
        ├── inspect_data.ipynb
        ├── inspect_data.py
        └── training_data.pkl
```

## Directory Descriptions

### `src/`
Main source code directory containing the activation extraction system.

- **`extract_vector.py`** - Primary Modal application for Pythia-12B activation extraction. Contains the `Pythia12BActivationExtractor` class with methods for getting activation vectors and model information. Includes web endpoints and local entrypoint for testing.

- **`pooling.py`** - Token pooling utilities for activation matrices. Implements four pooling strategies: exponential decay, mean, last token, and softmax normalization. All computations performed in float32 for numerical stability.

- **`deploy.py`** - Deployment utilities and configuration helpers.

- **`pythia_12b_modal_snapshot.py`** - GPU memory snapshot management for faster Modal container cold starts.

### `src/training_data/`
Training data pipeline for C4 dataset processing.

- **`get_training_data.py`** - Modal function to fetch and cache C4 dataset samples. Supports configurable sampling, caching to Modal volumes, and local file export.

- **`inspect_data.py`** - Data analysis utilities for examining training samples.

- **`inspect_data.ipynb`** - Jupyter notebook for interactive data exploration.

- **`training_data.pkl`** - Cached training data samples in pickle format.

### `memories/`
Research documentation and development notes.

- **`activation_extraction_documentation.md`** - Comprehensive system documentation including technical details, usage examples, and API specifications.

- **`activation_mean_computation_todo.md`** - Notes on mean vector computation strategies.

- **`gpu_snapshot_optimization.md`** - GPU optimization research and memory management notes.

- **`notes.md`** - General research notes and observations.

- **`pythia_12b_setup_todo.md`** - Model setup and configuration documentation.

### `outputs/`
Generated activation vectors and associated metadata.

- **`activation_vector_short_30.safetensors`** - Saved activation vector in SafeTensors format (fp16).

- **`activation_vector_short_30_metadata.json`** - Vector metadata including extraction parameters, model configuration, and shape information.

### `.claude/`
Claude AI assistant configuration files for project-specific settings.

## Technical Specifications

**Model**: EleutherAI/pythia-12b
- Hidden dimensions: 5120 (d_model)
- Total layers: 36
- Target layers: 25, 26, 27
- Precision: FP16 for storage, FP32 for pooling computations

**Infrastructure**: Modal serverless platform
- GPU: A100-80GB
- Memory: 64GB RAM
- Features: GPU memory snapshots, volume caching

**Vector Output Modes**:
- Short: 5120-dimensional (mean across layers)
- Long: 15360-dimensional (concatenated layers)

## Usage

### Local Testing
```bash
modal run src/extract_vector.py --text "Your text here"
```

### Deployment
```bash
modal deploy src/extract_vector.py
```

### Training Data Fetch
```bash
modal run src/training_data/get_training_data.py --num_samples 1000 --save_local
```

## API Endpoints

### Extract Activation Vector
```
POST /extract-activation
{
  "text": "Input text",
  "mode": "short",
  "target_layers": [25, 26, 27],
  "pooling_strategy": "exp"
}
```

### Health Check
```
GET /health
```

## Dependencies

- Modal
- TransformerLens
- PyTorch
- Transformers
- SafeTensors
- Datasets (for C4 data)