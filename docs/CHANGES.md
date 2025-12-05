# Code Changes for GPU Memory Management and CPU Offloading Support

## Overview

This document describes changes made to support GPU memory-constrained environments and CPU offloading for large language model pruning. The modifications enable pruning on systems with limited GPU memory by automatically offloading layers to CPU when needed, while maintaining compatibility with existing GPU-only workflows.

## Problem Statement

The original code assumed all model layers could fit in GPU memory simultaneously. On systems with limited GPU memory (e.g., 12GB GPUs for Llama-2-7b), this caused:
- CUDA out-of-memory (OOM) errors during model loading
- OOM errors during calibration input preparation
- OOM errors during pruning operations (sorting, masking)
- Inability to prune models that exceed GPU memory capacity

## New Command-Line Flags

### `--use_cpu` Flag

**Purpose**: Forces CPU-only execution, bypassing GPU entirely.

**Usage**:
```bash
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.6 ... --use_cpu
```

**Behavior**:
- Loads entire model on CPU using `float16` precision (~7GB RAM)
- Processes all layers on CPU (no GPU required)
- Enables complete pruning of all layers (no partial pruning)
- Significantly slower than GPU mode but ensures all layers are pruned

**When to Use**:
- Systems without GPU
- GPU memory insufficient for even partial model loading
- Need to prune all layers uniformly (not just GPU-resident layers)
- Willing to trade speed for complete pruning

**Trade-offs**:
- ✅ All layers pruned uniformly
- ✅ No GPU memory constraints
- ✅ Works on CPU-only systems
- ❌ 10-50x slower than GPU mode
- ❌ Requires ~8-10GB free RAM

## Changes Made

### 1. Model Loading with Automatic Device Mapping (`main.py`)

**File**: `main.py`, function `get_llm()`

**Changes**:
- Added `device_map="auto"` with `max_memory` configuration for GPU mode
- Calculates available GPU memory and reserves 15% (minimum 2GB) for pruning operations
- Allows CPU offloading as fallback when GPU memory is insufficient
- Added `--use_cpu` flag for explicit CPU-only mode
- CPU mode uses `float16` to reduce memory usage (~7GB vs ~14GB for float32)
- Automatic device detection: uses GPU if available unless `--use_cpu` is specified

**Code Location**: Lines 16-59

**Benefits**:
- Models can be loaded on GPUs with limited memory
- Automatic fallback to CPU offloading prevents OOM during loading
- Explicit CPU mode enables full model pruning when GPU memory is insufficient
- Backward compatible: existing commands work unchanged (GPU used by default)

### 2. Calibration Input Device Management (`lib/prune.py`)

**File**: `lib/prune.py`, function `prepare_calibration_input()`

**Changes**:
- Detects when model layers are offloaded to CPU/meta device
- Automatically places calibration inputs on CPU when offloaded layers detected
- Prevents OOM from attempting to create large calibration tensors on GPU

**Code Location**: Lines 73-139

**Benefits**:
- Prevents OOM during calibration input preparation
- Handles mixed GPU/CPU device maps correctly
- Reduces memory pressure on GPU

### 3. Meta Tensor Detection and Materialization (`lib/prune.py`)

**File**: `lib/prune.py`, function `prune_wanda()`

**Changes**:
- Detects weights on "meta" device (offloaded but not materialized)
- Attempts to materialize offloaded weights before pruning
- Skips pruning with warning if weights cannot be materialized
- Handles device mapping for layers across GPU and CPU

**Code Location**: Lines 239-370

**Benefits**:
- Prevents errors from attempting to prune meta tensors
- Provides clear warnings when layers cannot be pruned
- Allows partial pruning to complete successfully

### 4. Memory-Efficient Pruning Operations (`lib/prune.py`)

**File**: `lib/prune.py`, function `prune_wanda()`

**Changes**:
- Sample-by-sample processing to avoid moving large tensors to GPU
- Conditional CPU sorting when GPU memory usage exceeds 85%
- Explicit deletion of intermediate tensors (`W_metric`, `W_mask`, `sort_res`)
- Garbage collection after every 4 layers in CPU mode
- CPU fallback for `sort_mask.sum()` operation when GPU memory is constrained

**Code Location**: Lines 488-578, 141-195

**Benefits**:
- Reduces peak memory usage during pruning
- Prevents OOM during sorting operations
- Enables pruning on systems with limited GPU memory

### 5. Sparsity Check Improvements (`lib/prune.py`)

**File**: `lib/prune.py`, function `check_sparsity()`

**Changes**:
- Skips weights on meta device instead of raising errors
- Handles division by zero when all weights in a layer are skipped
- Provides informative messages for skipped layers

**Code Location**: Lines 32-71

**Benefits**:
- Prevents crashes when checking sparsity of partially pruned models
- Provides accurate sparsity reporting for pruned layers

### 6. Evaluation Improvements (`lib/eval.py`)

**File**: `lib/eval.py`, function `eval_ppl_wikitext()`

**Changes**:
- Detects offloaded layers and uses appropriate device for inputs
- Checks GPU memory availability before evaluation
- Graceful error handling for OOM during evaluation
- Periodic GPU cache clearing during evaluation

**Code Location**: Lines 84-150

**Benefits**:
- Evaluation works with models that have offloaded layers
- Prevents OOM during evaluation
- Allows model saving even if evaluation fails

### 7. Main Script Error Handling (`main.py`)

**File**: `main.py`, function `main()`

**Changes**:
- Graceful handling of evaluation failures
- Model saving proceeds even if evaluation is skipped
- Informative error messages with suggestions

**Code Location**: Lines 143-164

**Benefits**:
- Pruned models are saved even if evaluation fails
- Users receive clear guidance on next steps

## Technical Details

### Device Mapping Strategy

The code uses HuggingFace's `device_map="auto"` with `max_memory` constraints:
- Calculates available GPU memory
- Reserves memory for pruning operations (15% or minimum 2GB)
- Allows CPU offloading for layers that don't fit on GPU
- Automatically handles device placement during forward passes

### Memory Management

For CPU mode:
- Uses `float16` to halve memory requirements
- Explicit garbage collection every 4 layers
- Deletes intermediate tensors immediately after use
- Processes samples individually to reduce peak memory

For GPU mode:
- Monitors GPU memory usage
- Moves operations to CPU when GPU memory exceeds 85%
- Clears GPU cache after operations
- Processes samples individually to avoid large tensor transfers

### Meta Tensor Handling

When layers are offloaded, weights may remain on "meta" device (not materialized). The code:
1. Detects meta tensors before pruning
2. Attempts materialization through state_dict access or dummy forward pass
3. Skips pruning with warning if materialization fails
4. Allows partial pruning to complete successfully

## Compatibility

- **Backward Compatible**: All changes maintain compatibility with existing GPU-only workflows
- **Optional Features**: CPU offloading only activates when GPU memory is insufficient
- **No Breaking Changes**: Existing command-line arguments and workflows continue to work

## Testing

The changes have been tested with:
- Llama-2-7b-hf on 12GB GPU (partial pruning: layers 0-22, layers 23-31 offloaded to CPU)
- Llama-2-7b-hf on CPU-only mode with `--use_cpu` flag (full pruning: all 32 layers)
- Various GPU memory configurations
- Systems with and without GPU

## Usage Examples

### Default Behavior (GPU with Automatic CPU Offloading)

```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.6 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save ./out/ \
  --save_model ./workspace/pruned_models/llama_7b_wanda_0.4 \
  --cache_dir ./llm_weights \
  --use_variant
```

**Result**: Uses GPU if available, automatically offloads layers to CPU if GPU memory is insufficient. May result in partial pruning if some layers cannot be materialized.

### CPU-Only Mode (Complete Pruning)

```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.6 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save ./out/ \
  --save_model ./workspace/pruned_models/llama_7b_wanda_0.4_cpu \
  --cache_dir ./llm_weights \
  --use_variant \
  --use_cpu
```

**Result**: Entire model loaded on CPU, all 32 layers pruned uniformly. Slower but ensures complete pruning.

## Utility Scripts for Model Inference

Two utility scripts are provided to simplify loading and using both pruned and original models for inference.

### `use_pruned_model.py`

Loads and runs inference with a pruned model saved from the pruning process.

**Usage**:
```bash
python use_pruned_model.py \
  --model_path ./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4 \
  --prompt "Your prompt here" \
  --max_length 100 \
  --temperature 0.7 \
  --use_cpu  # Optional: force CPU usage
```

**Arguments**:
- `--model_path`: Path to the saved pruned model directory (required)
- `--prompt`: Input text prompt for generation (required)
- `--max_length`: Maximum length of generated text (default: 100)
- `--temperature`: Sampling temperature for generation (default: 0.7)
- `--use_cpu`: Force CPU usage even if GPU is available (optional)

**Example**:
```bash
python use_pruned_model.py \
  --model_path ./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4 \
  --prompt "The capital of France is"
```

### `use_original_model.py`

Loads and runs inference with the original (unpruned) model from HuggingFace.

**Usage**:
```bash
python use_original_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "Your prompt here" \
  --max_length 100 \
  --temperature 0.7 \
  --cache_dir ./llm_weights \
  --use_cpu  # Optional: force CPU usage
```

**Arguments**:
- `--model`: HuggingFace model name (default: "meta-llama/Llama-2-7b-hf")
- `--prompt`: Input text prompt for generation (required)
- `--max_length`: Maximum length of generated text (default: 100)
- `--temperature`: Sampling temperature for generation (default: 0.7)
- `--cache_dir`: Directory to cache the model (default: "llm_weights")
- `--use_cpu`: Force CPU usage even if GPU is available (optional)

**Example**:
```bash
python use_original_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "The capital of France is"
```

**Note**: Both scripts automatically handle device placement for models with offloaded layers, ensuring inputs are placed on the correct device (GPU or CPU) based on the model's device map.

## Limitations

- Partial pruning occurs when layers cannot be materialized (layers remain unpruned)
- CPU-only mode is significantly slower than GPU mode
- Evaluation may be skipped if GPU memory is insufficient (model still saves successfully)

## Future Improvements

Potential enhancements:
- More robust meta tensor materialization strategies
- Configurable memory reservation percentages
- Support for 8-bit quantization to further reduce memory
- Batch processing optimizations for CPU mode

## Rotary Embedding Initialization Fixes

### Problem Statement

When loading models on CPU (especially with `transformers` 4.45.2 and 4.57.3), rotary positional embeddings (`rotary_emb`) may not be properly initialized, causing `TypeError: cannot unpack non-iterable NoneType object` during pruning operations.

### Solution

**File**: `main.py`, function `get_llm()`

**Changes**:
- Added pre-initialization of rotary embeddings for CPU-loaded models
- Detects when models are loaded on CPU or have CPU/meta device layers
- Materializes `rotary_emb.inv_freq` from `model.state_dict()` to CPU device
- Explicitly calls `rotary_emb()` with dummy inputs to trigger cache initialization (`_cos_cached`, `_sin_cached`)
- Handles different `transformers` API versions (4.45.2 requires `x` and `position_ids` parameters)
- Falls back to manual creation of `LlamaRotaryEmbedding` if missing
- Provides informative warnings if initialization fails

**Code Location**: Lines 133-287

**File**: `lib/prune.py`, function `prune_wanda()`

**Changes**:
- Added robust error handling for rotary embedding issues during pruning
- Multiple fallback strategies if rotary embeddings return `None`:
  1. Materialize `inv_freq` from `state_dict`
  2. Call `rotary_emb()` directly with dummy `position_ids` (2D tensor shape)
  3. Perform tiny forward pass through `layer.self_attn` to trigger initialization
  4. Force materialization and multiple initialization attempts
- Handles `transformers` version differences:
  - 4.45.2: `rotary_emb.forward(x, position_ids)` requires both parameters
  - 4.57.3: `rotary_emb` may not exist and needs manual creation
- Ensures `position_ids` is 2D `(batch_size, seq_len)` not 1D `(seq_len,)`

**Code Location**: Lines 400-600, 610-690

**Benefits**:
- Prevents `TypeError` during pruning on CPU-loaded models
- Works with both `transformers` 4.45.2 and 4.57.3
- Graceful fallback if pre-initialization fails
- Compatible with GPU-only runs (skips initialization when not needed)

**Transformers Version Compatibility**:
- Tested with `transformers==4.45.2` (project default)
- Compatible with `transformers>=4.45.2` including 4.57.3
- Handles API differences between versions automatically

