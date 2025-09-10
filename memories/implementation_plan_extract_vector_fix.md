# Implementation Plan: Fix extract_vector.py to Match Algorithm Specification

## Overview
Fix the current implementation to match the exact algorithm specification for Pythia-12B activation vector extraction.

## Current Issues
- ❌ Single layer (26) instead of layers 25-27 with mean pooling
- ❌ No token order reversal (last token should be column 0)
- ❌ Wrong exponential weights - using arbitrary percentiles instead of Normal σ = -2.0, -1.5, -1.0
- ❌ Incorrect algorithm flow - pooling logic doesn't match specification

## Implementation Steps

### 1. Fix Layer Extraction (25-27 → Mean Pool)
- [ ] Change from single `target_layer = 26` back to `target_layers = [25, 26, 27]`
- [ ] Extract activations from all 3 layers simultaneously
- [ ] Mean pool across the 3 layers → `[d_model, seq_len]` shape
- [ ] Update function signatures and documentation

### 2. Implement Token Order Reversal
- [ ] After mean pooling, reverse token order: `activation_matrix = activation_matrix.flip(dims=[1])`
- [ ] Now column 0 = last token, column 1 = second-last token, etc.
- [ ] This is critical for proper exponential weighting

### 3. Fix Exponential Weight Functions in src/utils/pooling.py
- [ ] Replace current `exp_weight_977`, `exp_weight_841`, `exp_weight_933` functions
- [ ] Implement correct Normal distribution depths:
  - **ExpWeight-A**: σ = -2.0 (50% weight at ≈ 2.3% of doc length)
  - **ExpWeight-B**: σ = -1.5 (50% weight at ≈ 6.7% of doc length)  
  - **ExpWeight-C**: σ = -1.0 (50% weight at ≈ 15.9% of doc length)
- [ ] Use cumulative Normal distribution for depth calculations

### 4. Update Algorithm Flow
- [ ] **Step 1**: Extract blocks 25-27 → `[3, seq_len, d_model]`
- [ ] **Step 2**: Mean pool across 3 layers → `[seq_len, d_model]` → transpose → `[d_model, seq_len]`
- [ ] **Step 3**: Reverse token order → `[d_model, seq_len]` (column 0 = last token)
- [ ] **Step 4**: Apply pooling strategies
  - **Short**: Single exponential weight → `[d_model]`
  - **Long**: `[last_token, ExpWeight-A, ExpWeight-B, ExpWeight-C]` → `[4×d_model]`

### 5. Update Function Signatures & Documentation
- [ ] Revert parameter names: `target_layer` → `target_layers` 
- [ ] Update docstrings to match new algorithm
- [ ] Update API endpoints and CLI to reflect changes
- [ ] Ensure output dimensions: Short=5120, Long=5120×4

### 6. Skip Corpus Mean for Now
- [ ] Don't implement centering/corpus mean subtraction yet
- [ ] Focus on getting the basic extraction algorithm correct first
- [ ] Leave placeholder comments for future corpus mean integration

## Target Algorithm Flow
1. **Extract activations from blocks 25–27** → shape: `[5120, Tokens, 3]`
2. **Mean-pool across the 3 layers** → shape: `[5120, Tokens]`
3. **Reverse token order** so column 0 = last token, column 1 = second-last, etc.
4. **Apply exponential decay weights** across tokens (heaviest on last token)
5. **Compute weighted average** across tokens → shape: `[5120]`

## Outputs
- **Short vector**: Step 5 result → `[5120]`
- **Long vector**: Concatenation of four 5120-dim vectors:
  - Last token only (no weights)
  - ExpWeight-A (50% weight depth ≈ 2.3% of doc length)
  - ExpWeight-B (≈ 6.7%)  
  - ExpWeight-C (≈ 15.9%)
  - Total: `[4×5120]`

## Success Criteria
- ✅ Extracts from layers 25-27 and mean pools correctly
- ✅ Token order is reversed (last token = column 0)
- ✅ Exponential weights use correct Normal distribution depths
- ✅ Short mode outputs `[5120]` vector
- ✅ Long mode outputs `[4×5120]` concatenated vector
- ✅ Ready for corpus mean integration in next phase
