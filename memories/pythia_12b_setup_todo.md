# Pythia 12B Modal Setup - Todo List
**Date Created:** September 3, 2025  
**Project:** Deploy EleutherAI Pythia 12B on Modal for single token generation

## Setup Tasks

### ✅ Completed
- [x] Research Modal deployment patterns and requirements
- [x] Research Pythia 12B model specifications and requirements
- [x] Understand HuggingFace transformers integration with GPTNeoX
- [x] Plan architecture for single token generation

### 🔄 In Progress
- [x] Create markdown file with todo list in memories/ folder

### 📋 Pending
- [ ] Create main Modal deployment script (pythia_12b_modal.py)
  - [ ] Set up container image with dependencies
  - [ ] Configure GPU requirements (A100 80GB)
  - [ ] Implement model loading with GPTNeoXForCausalLM
  - [ ] Add caching with Modal volumes
  - [ ] Create inference function with max_new_tokens=1
  - [ ] Add web endpoint for API access
- [ ] Create client script for testing inference (client.py)
  - [ ] Simple request function
  - [ ] Input text handling
  - [ ] Display single token output
- [ ] Create requirements.txt file
- [ ] Create README with usage instructions
  - [ ] Deployment commands
  - [ ] Local testing instructions
  - [ ] API usage examples

## Technical Specifications

### Model Details
- **Model ID:** EleutherAI/pythia-12b
- **Parameters:** 11.3B
- **Architecture:** GPTNeoX
- **Memory Requirements:** ~24GB (FP16)
- **GPU Requirements:** A100 80GB recommended

### Modal Configuration
- **GPU:** A100 (80GB VRAM)
- **Memory:** 65536 MB (64GB)
- **Container Timeout:** 120 seconds idle
- **Precision:** FP16 for memory efficiency
- **Caching:** Modal volume for model weights

### Key Features
- Single token generation (max_new_tokens=1)
- Web endpoint for API access
- Model weight caching to avoid re-downloads
- Efficient memory usage with FP16 precision

## Notes
- Using Modal's serverless infrastructure for cost-effective deployment
- Model will automatically scale to zero when not in use
- First load will download ~24GB of model weights to cache