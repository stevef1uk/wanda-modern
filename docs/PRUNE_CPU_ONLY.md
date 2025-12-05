# CPU-Only Pruning Guide

This guide shows how to prune the model using CPU only, which ensures **all layers** are processed (not just layers 0-22).

## Why CPU-Only?

When using GPU with limited memory, some layers get offloaded to CPU/meta device and cannot be pruned. Using CPU-only mode ensures:
- ✅ All 32 layers are pruned (not just 23)
- ✅ Uniform sparsity across all layers
- ✅ No GPU memory constraints
- ⚠️ Much slower (CPU is slower than GPU)

## How to Run

### Basic Command

```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.6 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save ./out/ \
  --save_model ./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4_cpu \
  --cache_dir ./llm_weights \
  --use_variant \
  --use_cpu
```

### Key Differences

- **`--use_cpu`**: Forces CPU-only mode (all layers will be pruned)
- **Save path**: Use a different name (e.g., `_cpu` suffix) to distinguish from GPU-pruned model
- **Speed**: Expect 10-50x slower than GPU (depending on your CPU)

### Full Example

```bash
# CPU-only pruning with all layers
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.6 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save ./out/ \
  --save_model ./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4_cpu_full \
  --cache_dir ./llm_weights \
  --use_variant \
  --use_cpu
```

## What to Expect

1. **Loading**: Model loads entirely on CPU (slower than GPU)
2. **Pruning**: All 32 layers will be pruned (you'll see "pruning layer 0" through "pruning layer 31")
3. **Time**: May take several hours depending on your CPU
4. **Memory**: Uses CPU RAM (ensure you have enough - model needs ~14GB in float32)

## Memory Requirements

- **Model size**: ~7GB in float16 (CPU mode uses float16)
- **Calibration data**: ~1-2GB
- **Temporary operations**: ~2-4GB
- **Total**: ~10-13GB RAM recommended

## Using All CPU Cores for Faster Processing

CPU-only pruning can be significantly accelerated by using multiple CPU cores. PyTorch automatically parallelizes operations when configured correctly.

### Find Your CPU Core Count

```bash
# Check number of CPU cores
nproc
# or
lscpu | grep "^CPU(s):"
```

### Set Thread Environment Variables

Before running the pruning command, set these environment variables:

```bash
# Replace 16 with your actual core count (or slightly less to leave cores free)
export OMP_NUM_THREADS=16        # OpenMP threads for PyTorch operations
export MKL_NUM_THREADS=16         # Intel MKL threads (if using Intel CPU)
export NUMEXPR_NUM_THREADS=16     # NumExpr threads
export TORCH_NUM_THREADS=16       # PyTorch internal threading

# Then run your pruning command
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda \
  --sparsity_ratio 0.6 --sparsity_type unstructured --nsamples 8 \
  --save ./out/ --save_model ./workspace/pruned_models/llama_7b_wanda_0.4_cpu_full \
  --cache_dir ./llm_weights --use_variant --use_cpu
```

### Recommended Settings

- **All cores**: Set to your total core count (e.g., `OMP_NUM_THREADS=16` for 16 cores)
- **Leave cores free**: Set to 75-90% of cores to avoid system slowdown (e.g., `OMP_NUM_THREADS=12` for 16 cores)
- **Hyperthreading**: Count physical cores, not logical cores (hyperthreads)

### Example: 16-Core System

```bash
# Use 14 cores (leave 2 free for system)
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export NUMEXPR_NUM_THREADS=14
export TORCH_NUM_THREADS=14

python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda \
  --sparsity_ratio 0.6 --sparsity_type unstructured --nsamples 8 \
  --save ./out/ --save_model ./workspace/pruned_models/llama_7b_wanda_0.4_cpu_full \
  --cache_dir ./llm_weights --use_variant --use_cpu
```

### What Operations Benefit from Multi-Core

- **Matrix multiplications**: Automatically parallelized
- **Sorting operations**: `torch.sort()` uses multiple threads
- **Element-wise operations**: Large tensor operations benefit
- **BLAS operations**: If MKL/OpenBLAS is available

### Performance Impact

- **Single core**: Baseline (slowest)
- **4 cores**: ~3-4x faster
- **8 cores**: ~6-7x faster
- **16 cores**: ~10-12x faster (diminishing returns due to memory bandwidth)

### Persistent Configuration

To make these settings permanent for your session, add to your `~/.bashrc`:

```bash
# Add to ~/.bashrc
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TORCH_NUM_THREADS=16
```

Then reload: `source ~/.bashrc`

## Comparison

| Mode | Layers Pruned | Speed | GPU Memory | CPU Memory |
|------|---------------|-------|------------|------------|
| GPU (auto) | 23/32 (layers 0-22) | Fast | ~9GB | ~5GB |
| CPU only | 32/32 (all layers) | Slow (10-50x) | 0GB | ~10-13GB |
| CPU only (multi-core) | 32/32 (all layers) | Moderate (3-12x faster than single-core) | 0GB | ~10-13GB |

## After Pruning

Use the pruned model the same way:

```bash
python use_pruned_model.py \
  --model_path ./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4_cpu_full \
  --prompt "Your prompt here"
```

## Troubleshooting

### Rotary Embedding Initialization Errors

If you encounter `TypeError: cannot unpack non-iterable NoneType object` during pruning:

1. **Check transformers version**: Ensure you're using `transformers==4.45.2`:
   ```bash
   pip show transformers
   ```

2. **Reinstall correct version**:
   ```bash
   pip install transformers==4.45.2
   ```

3. **The code automatically handles this**: The code includes automatic rotary embedding initialization for CPU-loaded models. If you still see errors, check:
   - Model is loading correctly
   - Sufficient RAM available
   - Transformers version matches project requirements

### Common Issues

- **"Rotary embeddings already initialized"**: This is normal - the code checks if initialization is needed
- **"Warning: Failed to initialize X layers"**: Usually non-fatal - the code will attempt initialization during pruning
- **Memory errors**: Ensure you have ~10-13GB free RAM for Llama-2-7b

## Tips

1. **Use all cores**: Set `OMP_NUM_THREADS` to your core count for 3-12x speedup
2. **Run overnight**: CPU-only pruning takes a long time (even with multi-core)
3. **Monitor RAM**: Make sure you have enough free RAM (~10-13GB)
4. **Be patient**: It's much slower than GPU but ensures complete pruning
5. **Compare results**: You can compare GPU-partial vs CPU-full pruning results
6. **Check CPU usage**: Use `htop` or `top` to verify all cores are being used
7. **Use correct transformers version**: `transformers==4.45.2` is required for proper CPU mode operation

