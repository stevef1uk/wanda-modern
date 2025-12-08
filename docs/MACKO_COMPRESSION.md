# MACKO-SpMV Compression and Inference Guide

This guide explains how to compress your pruned model using [MACKO-SpMV](https://github.com/vlejd/macko_spmv) and run inference on an A100 GPU.

## Prerequisites

1. **Pruned Model**: You should have already pruned a model using `test_modal.py` and downloaded it locally.
   - Default location: `./pruned_models/gptt-oss-20b-_test/`
   - To download from Modal: `./scripts/download_from_modal.sh`

2. **NVIDIA A100 GPU**: MACKO-SpMV is optimized for A100 GPUs, though it may work on other GPUs.

3. **Dependencies**:
   - CUDA toolkit
   - `uv` package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Quick Start

Run the automated script:

```bash
./scripts/compress_and_inference_macko.sh
```

This script will:
1. Check for the pruned model
2. Clone the macko_spmv repository (if needed)
3. Set up the environment
4. Compress the model (~4 minutes for 20 workers)
5. Run inference benchmark

## Manual Steps

If you prefer to run steps manually:

### 1. Clone macko_spmv Repository

```bash
cd ..
git clone https://github.com/vlejd/macko_spmv.git
cd macko_spmv
```

### 2. Set Up Environment

```bash
uv venv
source .venv/bin/activate
uv sync --all-groups
```

### 3. Test Installation

```bash
cd python
pytest test_macko_spmv.py -v
cd ..
```

### 4. Compress Model

```bash
python python_scripts/util_compress_llm.py \
    /path/to/wanda-modern/pruned_models/gptt-oss-20b-_test \
    20
```

Replace the path with your actual model path. The `20` is the number of workers (adjust based on your system).

### 5. Run Inference Benchmark

```bash
python python_scripts/util_run_pruned_llm.py \
    /path/to/wanda-modern/pruned_models/gptt-oss-20b-_test \
    /path/to/wanda-modern/pruned_models/gptt-oss-20b-_test_compressed \
    --make_sparse=1 \
    | tee run_sparse.txt
```

## Understanding the Output

- **Compression**: Creates a compressed version of your pruned model, reducing storage overhead.
- **Inference Benchmark**: Compares performance between uncompressed and compressed models.

The benchmark will show:
- Memory usage
- Inference speed
- Throughput metrics

## Troubleshooting

### Model Not Found

If you get an error about the model not being found:
1. Check that you've downloaded the model from Modal:
   ```bash
   ./scripts/download_from_modal.sh
   ```
2. Verify the model path in the script matches your actual model directory.

### GPU Not Detected

- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Verify CUDA is available: `nvcc --version`
- Check that your GPU is accessible: `python -c "import torch; print(torch.cuda.is_available())"`

### Compression Fails

- Check that you have enough disk space (compressed models can still be large)
- Verify the model directory contains all required files (config.json, safetensors files, etc.)
- Check the macko_spmv repository structure matches expectations

### Inference Errors

- Ensure the compressed model was created successfully
- Check GPU memory: `nvidia-smi`
- Try reducing batch size if you encounter OOM errors

## Performance Expectations

According to the MACKO-SpMV paper:
- **Memory Reduction**: Up to 1.5x reduction at 50% sparsity
- **Speedup**: 1.2-1.5x speedup over dense representations at 50% sparsity

Actual performance will depend on:
- Model size and sparsity level
- GPU model (A100 recommended)
- Batch size and sequence length

## References

- MACKO-SpMV Repository: https://github.com/vlejd/macko_spmv
- Technical README: https://github.com/vlejd/macko_spmv/blob/master/TECHNICAL_README.md
- Paper: https://arxiv.org/abs/2511.13061

