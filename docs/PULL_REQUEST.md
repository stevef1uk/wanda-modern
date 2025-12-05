# Pull Request: GPU Memory Management and CPU Offloading Support

## Summary

This PR adds support for GPU memory-constrained environments and CPU offloading, enabling pruning of large language models on systems with limited GPU memory. The changes maintain full backward compatibility while adding automatic memory management and graceful degradation.

## Motivation

Large language models like Llama-2-7b require significant GPU memory (~14GB in float16). On systems with limited GPU memory (e.g., 12GB GPUs), the original code fails with CUDA out-of-memory errors. This PR enables:

1. Automatic GPU memory management with CPU offloading fallback
2. Pruning on systems with insufficient GPU memory
3. CPU-only mode for complete model pruning
4. Graceful handling of memory constraints

## Key Changes

### 1. Model Loading (`main.py`)
- Added `device_map="auto"` with `max_memory` configuration
- Automatic GPU memory reservation for pruning operations
- CPU offloading support when GPU memory is insufficient
- **New `--use_cpu` flag**: Forces CPU-only execution for complete model pruning

### New Feature: CPU-Only Mode

The `--use_cpu` flag enables running the entire pruning pipeline on CPU:

- **Complete Pruning**: All layers are pruned uniformly (no partial pruning)
- **No GPU Required**: Works on systems without GPU or with insufficient GPU memory
- **Memory Efficient**: Uses `float16` to reduce RAM usage (~7GB vs ~14GB)
- **Trade-off**: Significantly slower than GPU mode (10-50x) but ensures all layers are processed

**Example**:
```bash
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda \
  --sparsity_ratio 0.6 --use_cpu
```

### 2. Memory Management (`lib/prune.py`)
- Sample-by-sample processing to reduce peak memory
- Conditional CPU sorting when GPU memory is constrained
- Explicit tensor cleanup and garbage collection
- Meta tensor detection and materialization attempts

### 3. Error Handling
- Graceful handling of OOM errors
- Informative warnings for skipped layers
- Model saving even when evaluation fails

### 4. Evaluation (`lib/eval.py`)
- Support for models with offloaded layers
- Memory-aware evaluation with fallback options

### 5. Rotary Embedding Initialization (`main.py`, `lib/prune.py`)
- Fixed `TypeError: cannot unpack non-iterable NoneType object` for CPU-loaded models
- Pre-initializes rotary embeddings when loading models on CPU
- Handles `transformers` version differences (4.45.2 vs 4.57.3)
- Robust fallback strategies during pruning if initialization fails
- Ensures `position_ids` are correctly shaped (2D tensors)

## Testing

Tested configurations:
- **GPU mode (12GB GPU)**: Successfully prunes layers 0-22, offloads layers 23-31 to CPU
- **CPU-only mode**: Successfully prunes all 32 layers (slower but complete)
- **Memory constraints**: Handles OOM gracefully with informative messages

## Backward Compatibility

✅ All existing functionality preserved
✅ No breaking changes to API or command-line interface
✅ Optional features only activate when needed
✅ Existing workflows continue to work unchanged

## Performance Impact

- **GPU mode**: Minimal impact, slight overhead from device mapping
- **CPU mode**: Significantly slower but enables pruning on systems without sufficient GPU memory
- **Memory usage**: Reduced peak memory through optimizations

## Documentation

See `CHANGES.md` for detailed technical documentation of all changes.

## Example Usage

```bash
# GPU mode with automatic CPU offloading (existing behavior with memory management)
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.6 ...

# CPU-only mode (new feature)
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.6 ... --use_cpu
```

## Utility Scripts

Two utility scripts are provided for easy model inference:

**Using a pruned model**:
```bash
python use_pruned_model.py \
  --model_path ./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4 \
  --prompt "Your prompt here"
```

**Using the original model**:
```bash
python use_original_model.py \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "Your prompt here"
```

Both scripts support `--use_cpu` flag for CPU-only inference and handle device placement automatically for models with offloaded layers. See `CHANGES.md` for detailed usage instructions.

## Files Modified

- `main.py`: Model loading with device mapping and rotary embedding initialization
- `lib/prune.py`: Memory management, meta tensor handling, and rotary embedding fixes
- `lib/eval.py`: Evaluation improvements for offloaded models

## Files Added

- `CHANGES.md`: Detailed technical documentation
- `use_pruned_model.py`: Example script for using pruned models
- `use_original_model.py`: Example script for using original models
- `PRUNE_CPU_ONLY.md`: Guide for CPU-only pruning

