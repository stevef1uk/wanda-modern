# Running MACKO-SpMV Compression on Modal

This guide explains how to compress your pruned model and run inference using MACKO-SpMV on Modal with an A100 GPU.

## Prerequisites

1. **Pruned Model on Modal**: You should have already run `test_modal.py` which created a pruned model stored in the Modal volume `wanda-cache` at `/cache/pruned_models/gptt-oss-20b-_test/`

2. **Modal Setup**: Ensure you have Modal installed and configured:
   ```bash
   pip install modal
   modal setup
   ```

## Quick Start

Run the compression and inference script:

```bash
modal run compress_macko_modal.py
```

This will:
1. Clone the macko_spmv repository on Modal
2. Set up the environment with all dependencies
3. Compress your pruned model (~4 minutes for 20 workers)
4. Run inference benchmarks comparing compressed vs uncompressed
5. Save all results back to the Modal volume

## What Happens

The script (`compress_macko_modal.py`) will:

1. **Load Pruned Model**: Reads the pruned model from `/cache/pruned_models/gptt-oss-20b-_test/` on the Modal volume

2. **Setup macko_spmv**: 
   - Clones the repository if needed
   - Creates a virtual environment
   - Installs all dependencies (including `fire` and `tqdm`)

3. **Compress Model**: 
   - Runs `util_compress_llm.py` with 20 workers
   - Creates compressed version at `/cache/pruned_models/gptt-oss-20b-_test_compressed`

4. **Run Inference Benchmark**:
   - Compares performance of uncompressed vs compressed models
   - Saves results to `/cache/results/run_gptt-oss-20b-_test_sparse.txt`

5. **Save Results**: All outputs are committed to the Modal volume for persistence

## Downloading Results

After the script completes, download the results locally:

```bash
# Download compressed model
modal volume get wanda-cache pruned_models/gptt-oss-20b-_test_compressed* ./pruned_models/

# Download inference results
modal volume get wanda-cache results/run_gptt-oss-20b-_test_sparse.txt ./results/
```

## GPU Configuration

The script is configured to use:
- **GPU**: A100 80GB (best performance for MACKO-SpMV)
- **Timeout**: 2 hours (compression + inference can take time)

To change the GPU type, edit `compress_macko_modal.py`:

```python
gpu=modal.gpu.A100(size="80GB")  # Change to "40GB" or use H100
```

## Troubleshooting

### Model Not Found

If you get an error that the pruned model is not found:
1. Verify you've run `test_modal.py` successfully
2. Check the model path matches: `/cache/pruned_models/gptt-oss-20b-_test/`
3. List volume contents: `modal volume ls wanda-cache pruned_models/`

### Import Errors

If you encounter import errors for `macko_spmv`:
- The script automatically adds the `src` directory to PYTHONPATH
- It also tries to install the package using `pip install -e`
- Check the logs for specific error messages

### Compression Fails

- Check GPU memory: The script uses A100 80GB which should be sufficient
- Verify the pruned model files are complete (all safetensors files present)
- Check the logs for specific error messages

### Timeout Issues

If the script times out:
- Increase timeout in the script: `timeout=7200` (2 hours)
- Or split into two separate functions (one for compression, one for inference)

## Performance Expectations

According to MACKO-SpMV research:
- **Memory Reduction**: Up to 1.5x reduction at 50% sparsity
- **Speedup**: 1.2-1.5x speedup over dense representations at 50% sparsity
- **Compression Time**: ~4 minutes for 20 workers on A100

## References

- MACKO-SpMV Repository: https://github.com/vlejd/macko_spmv
- Technical README: https://github.com/vlejd/macko_spmv/blob/master/TECHNICAL_README.md

