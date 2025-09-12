# Pythia-12B Activation Vector Extraction System

## Overview
This system uses TransformerLens to extract residual stream activation vectors from layers 25, 26, and 27 of Pythia-12B model for document similarity analysis and mechanistic interpretability research.

## Files Created

### 1. `src/extract_vector.py`
Modal deployment file that provides:
- **Pythia12BActivationExtractor**: Core class with GPU memory snapshots for fast cold starts
- **get_activation_vector(text, pooling_strategy)**: Extract activation vectors
  - `short`: 5120-dim (mean across layers → exp-weight across tokens)
  - `long`: 20480-dim (concat of [last token, 2.3%, 6.7%, 15.9% exp-weighted pools])
- **get_activation_matrix(text)**: [5120, tokens] (mean across layers, reversed order)

### 2. `cosine_similarity.py`
Example client script for testing and analysis via Modal calls.

## Technical Details

### Model Architecture
- **Model**: EleutherAI/pythia-12b
- **Hidden Size (d_model)**: 5120
- **Total Layers**: 36
- **Target Layers**: 25, 26, 27 (high-level semantic representations)
- **GPU**: A100 for efficient processing

### Activation Extraction Process
1. Text is tokenized using Pythia tokenizer
2. Forward pass through model with activation caching
3. Extract `resid_post` (residual stream after attention + MLP) from layers 25-27
4. Pool activations:
   - **Short mode**: Mean across layers → exp-weight across tokens → 5120-dim vector
   - **Long mode**: Concat of [last token, 2.3%, 6.7%, 15.9% exp-weighted pools] → 20480-dim vector
5. L2-normalize vectors for cosine similarity

### TransformerLens Features Used
- `HookedTransformer.from_pretrained()` for model loading
- `run_with_cache()` for activation extraction
- Hook points system for accessing specific layer activations
- Automatic handling of residual stream components

## Usage Examples

### Deploy to Modal
```bash
modal deploy -m src.extract_vector
```

### Local Testing
```bash
# Test extraction
modal run -m src.extract_vector --text "Your text here"

# Test similarity (see cosine_similarity.py for full example)
modal run -m src.cosine_similarity --text1 "A" --text2 "B"

# Generate CSVs for samples
modal run -m src.generate_vector_csv
```

### Using the Client
```bash
# Interactive mode
modal run -m src.cosine_similarity --text1 "First" --text2 "Second"

# Compare two texts
modal run -m src.cosine_similarity --text1 "First text" --text2 "Second text"

# Batch comparison
# See generate_vector_csv.py for batch export

# Create similarity matrix
# Visualization is not included in this repo
```

## Modal Calls

Use `modal.Cls.from_name("activation-vector-project", "Pythia12BActivationExtractor")` and call `.get_activation_vector.remote(text=..., pooling_strategy="short"|"long")`. See `cosine_similarity.py` for a concrete example.

## Key Features

1. **GPU Memory Snapshots**: Fast cold starts on Modal using memory snapshots
2. **Flexible Pooling**: Short (5120-dim) or long (20480-dim) vectors
3. **Mean Centering**: Subtract mean vector from training data for better discrimination
4. **Batch Processing**: Efficiently process multiple documents
5. **Visualization**: Built-in plotting for similarity analysis
6. **Interactive Mode**: Explore similarities interactively

## Research Applications

- Document similarity analysis
- Semantic search and retrieval
- Mechanistic interpretability studies
- Representation analysis across transformer layers
- Study information flow through residual stream
- Analyze how concepts are encoded in activation space

## Dependencies

- **Modal**: Serverless GPU deployment
- **TransformerLens**: Mechanistic interpretability library
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model loading
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization

## Performance Notes

- First deployment creates GPU memory snapshot (slower)
- Subsequent cold starts are much faster due to snapshots
- A100-80GB GPU handles 12B parameters efficiently
- FP16 precision for memory optimization
- Activation caching focused on target layers only

## Future Enhancements

- Add support for more layers or custom layer selection
- Implement different pooling strategies (max, attention-weighted)
- Add streaming support for large document batches
- Include activation steering capabilities
- Support for comparing across different models
