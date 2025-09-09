# Pythia-12B Activation Vector Extraction System

## Overview
This system uses TransformerLens to extract residual stream activation vectors from layers 25, 26, and 27 of Pythia-12B model for document similarity analysis and mechanistic interpretability research.

## Files Created

### 1. `activation_extraction_modal.py`
Main Modal deployment file that provides:
- **Pythia12BActivationExtractor**: Core class with GPU memory snapshots for fast cold starts
- **get_activation_vector(text, mode)**: Extract activation vectors
  - `short` mode: Mean-pooled 5120-dimensional vector
  - `long` mode: Concatenated 15360-dimensional vector (3 layers × 5120)
- **compute_mean_vector(documents, mode)**: Compute mean vector from training documents for centering
- **compute_cosine_similarity(text1, text2, mode, subtract_mean)**: Calculate similarity between texts
- **batch_compute_similarities()**: Compare reference text against multiple documents
- Web API endpoints for remote access

### 2. `activation_client.py`
Client script for testing and analysis:
- Interactive mode for exploring document similarities
- Visualization tools (similarity plots, heatmaps)
- Batch processing capabilities
- Support for mean-centered vectors

## Technical Details

### Model Architecture
- **Model**: EleutherAI/pythia-12b
- **Hidden Size (d_model)**: 5120
- **Total Layers**: 36
- **Target Layers**: 25, 26, 27 (high-level semantic representations)
- **GPU**: A100-80GB for efficient processing

### Activation Extraction Process
1. Text is tokenized using Pythia tokenizer
2. Forward pass through model with activation caching
3. Extract `resid_post` (residual stream after attention + MLP) from layers 25-27
4. Pool activations:
   - **Short mode**: Mean across layers → 5120-dim vector
   - **Long mode**: Concatenate layers → 15360-dim vector
5. L2-normalize vectors for cosine similarity

### TransformerLens Features Used
- `HookedTransformer.from_pretrained()` for model loading
- `run_with_cache()` for activation extraction
- Hook points system for accessing specific layer activations
- Automatic handling of residual stream components

## Usage Examples

### Deploy to Modal
```bash
modal deploy activation_extraction_modal.py
```

### Local Testing
```bash
# Test extraction
modal run activation_extraction_modal.py --text "Your text here"

# Test similarity
modal run activation_extraction_modal.py --action similarity

# Test batch similarity
modal run activation_extraction_modal.py --action batch
```

### Using the Client
```bash
# Interactive mode
python activation_client.py --endpoint <MODAL_ENDPOINT_URL>

# Compare two texts
python activation_client.py --endpoint <URL> --action compare \
  --text1 "First text" --text2 "Second text"

# Batch comparison
python activation_client.py --endpoint <URL> --action batch

# Create similarity matrix
python activation_client.py --endpoint <URL> --action matrix --visualize
```

## API Endpoints

### Extract Activation Vector
```json
POST /extract-activation
{
  "text": "Your input text",
  "mode": "short",
  "return_tokens": false
}
```

### Compute Similarity
```json
POST /extract-activation
{
  "action": "similarity",
  "text1": "First text",
  "text2": "Second text",
  "mode": "short",
  "subtract_mean": false
}
```

### Compute Mean Vector
```json
POST /extract-activation
{
  "action": "mean_vector",
  "documents": ["doc1", "doc2", "doc3"],
  "mode": "short"
}
```

### Batch Similarities
```json
POST /extract-activation
{
  "action": "batch_similarity",
  "reference_text": "Reference text",
  "comparison_texts": ["text1", "text2"],
  "mode": "short",
  "subtract_mean": false,
  "mean_documents": ["doc1", "doc2"]
}
```

## Key Features

1. **GPU Memory Snapshots**: Fast cold starts on Modal using memory snapshots
2. **Flexible Pooling**: Short (5120-dim) or long (15360-dim) vectors
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