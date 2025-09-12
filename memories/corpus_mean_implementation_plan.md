# Corpus Mean Implementation Plan

## Overview
Create corpus mean computation system using Modal's direct function invocation APIs, with shared helper functions to minimize code duplication.

## Phase 1: Extract Common Logic into Helper Functions

### 1.1 Create Helper Functions in `extract_vector.py`
```python
def _extract_and_prepare_activations(self, text: str, target_layers: List[int]) -> torch.Tensor:
    """Extract blocks 25-27 → Mean pool → Reverse token order → [5120, tokens]"""

def _apply_pooling_and_normalize(self, activation_matrix: torch.Tensor, 
                                mode: str, pooling_strategy: str, 
                                pooling_params: Dict) -> torch.Tensor:
    """Apply pooling strategies and normalization"""
```

### 1.2 Add New Method: `get_activation_matrix`  
```python
@modal.method()
def get_activation_matrix(self, text: str, target_layers: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Extract intermediate [5120, tokens] activation matrix for corpus mean.
    Returns: {"activation_matrix": List[List[float]], "shape": [5120, tokens]}
    """
```

## Phase 2: Remove HTTP Endpoints (Keep Only Modal Methods)

### 2.1 Clean Up `extract_vector.py`
- Remove all `@modal.fastapi_endpoint` decorators  
- Remove `extract_activation_endpoint` and `health_check` functions
- Keep only `@modal.method()` decorated functions
- Keep `@modal.local_entrypoint()` for testing

## Phase 3: Create Corpus Mean Script with CORRECT Modal APIs

### 3.1 Create `src/corpus_mean.py` - Using Proper Modal Function References
```python
import modal
import torch
from typing import List, Dict, Any

# CORRECT Modal API - Reference the deployed class
Pythia12BExtractor = modal.Cls.from_name(
    "pythia-12b-activation-extraction",  # app name
    "Pythia12BActivationExtractor"       # class name
)

# Create volume reference for training data
training_volume = modal.Volume.from_name("activation-vector-project")

app = modal.App("corpus-mean-computation")

@app.function(
    volumes={"/training_data": training_volume},
    timeout=3600,  # 1 hour timeout
    memory=16384   # 16GB RAM
)
def compute_corpus_mean(word_count_min: int = 750, word_count_max: int = 1500, max_docs: int = 10000):
    """Compute corpus mean from training data using deployed extractor"""
    
    # Create instance of deployed extractor
    extractor = Pythia12BExtractor()
    
    # Initialize corpus mean tracking
    corpus_mean = None  # Will be [5120, max_tokens]  
    counts = None       # Will be [max_tokens]
    processed_docs = 0
    
    # Read training documents from volume
    training_files = list_training_files("/training_data")
    
    for file_path in training_files:
        if processed_docs >= max_docs:
            break
            
        with open(file_path, 'r') as f:
            text = f.read()
            
        # Filter by word count
        word_count = len(text.split())
        if not (word_count_min <= word_count <= word_count_max):
            continue
        
        # CORRECT Modal API - Call deployed function
        result = extractor.get_activation_matrix.remote(text=text)
        
        # Extract activation matrix
        activation_matrix = torch.tensor(result["activation_matrix"], dtype=torch.float32)  # [5120, tokens]
        
        # Update rolling mean
        corpus_mean, counts = update_rolling_mean(corpus_mean, counts, activation_matrix)
        processed_docs += 1
        
        if processed_docs % 100 == 0:
            print(f"Processed {processed_docs} documents...")
    
    # Save corpus mean
    save_corpus_mean(corpus_mean, counts, processed_docs)
    return {"processed_docs": processed_docs, "final_shape": list(corpus_mean.shape)}

def update_rolling_mean(corpus_mean, counts, new_matrix):
    """Update rolling mean with new activation matrix"""
    seq_len = new_matrix.shape[1]
    
    # Initialize or expand if needed
    if corpus_mean is None:
        corpus_mean = torch.zeros(5120, seq_len, dtype=torch.float32)
        counts = torch.zeros(seq_len, dtype=torch.int64)
    elif seq_len > corpus_mean.shape[1]:
        # Expand matrices for longer document
        old_len = corpus_mean.shape[1]
        new_corpus_mean = torch.zeros(5120, seq_len, dtype=torch.float32)
        new_counts = torch.zeros(seq_len, dtype=torch.int64)
        new_corpus_mean[:, :old_len] = corpus_mean
        new_counts[:old_len] = counts
        corpus_mean, counts = new_corpus_mean, new_counts
    
    # Update rolling mean for each token position
    for pos in range(seq_len):
        N = counts[pos].item()
        corpus_mean[:, pos] = (corpus_mean[:, pos] * N + new_matrix[:, pos]) / (N + 1)
        counts[pos] += 1
    
    return corpus_mean, counts

@app.local_entrypoint()
def main():
    result = compute_corpus_mean.remote()
    print(f"Corpus mean computation completed: {result}")
```

## Phase 4: Key Corrections - Proper Modal APIs

### ✅ **Correct Function Reference**
```python
# CORRECT - Use modal.Cls.from_name()
extractor_cls = modal.Cls.from_name("app-name", "ClassName")  
extractor = extractor_cls()
result = extractor.method_name.remote(args)

# WRONG - This doesn't exist
# deployed_app = modal.App.lookup(...)
```

### ✅ **Direct Method Invocation**  
```python
# Call the deployed method directly
result = extractor.get_activation_matrix.remote(text=document_text)
activation_matrix = torch.tensor(result["activation_matrix"])
```

### ✅ **Volume Access**
```python
# Reference existing volume
training_volume = modal.Volume.from_name("activation-vector-project")

# Mount in function
@app.function(volumes={"/training_data": training_volume})
```

## Implementation Steps
1. ✅ Add helper functions and new method to `extract_vector.py`
2. ✅ Remove all HTTP endpoints from `extract_vector.py` 
3. ✅ Create `corpus_mean.py` with proper Modal APIs
4. ✅ Deploy and test the corpus mean computation
5. ✅ Validate the rolling mean algorithm and output

## Algorithm Requirements
- Feed ~10,000 training texts through Pythia-12B
- Extract activations (blocks 25–27) → Mean pool → Reverse token order
- Filter documents: 750-1500 words only
- Update rolling mean: `new_mean = (old_mean * N + new_entry) / (N + 1)`
- Maintain count vector per token position
- Store final corpus mean matrix (length = max observed tokens)

## Expected Outputs
- `corpus_mean.safetensors`: [5120, max_tokens] matrix
- `corpus_mean_metadata.json`: Processing statistics
- Ready for corpus mean subtraction in document processing