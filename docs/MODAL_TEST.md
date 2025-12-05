# Testing on Modal.com

This guide helps you test the pruning code on Modal.com with a GPU that has sufficient RAM to verify backward compatibility.

## Setup on Modal

### Local Setup (For Modal CLI)

You only need to install Modal CLI locally to submit jobs. The remote execution happens in Modal's containers (no virtual env needed there).

**Option A: Using a virtual environment (recommended for local CLI)**:
```bash
# Create and activate a virtual environment
python3 -m venv modal-env
source modal-env/bin/activate  # On Windows: modal-env\Scripts\activate

# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup
```

**Option B: Install globally**:
```bash
pip install modal
modal setup
```

### Option 1: Using Modal Script (Recommended)

1. **Verify HuggingFace secret exists** (if needed for LLaMA-2):
   - The script uses `hf-secret` (you should already have this set up)
   - If you need to create/update it: `modal secret create hf-secret HF_TOKEN=your_token_here`

2. **Upload your code to Modal** (or mount it):
   - The `test_modal.py` script assumes your code is available in the container
   - You can either:
     - Upload files: `modal volume put wanda-cache /path/to/your/code`
     - Or modify `test_modal.py` to clone from git

3. **Run the test script**:
```bash
modal run test_modal.py
```

### Option 2: Direct Command Execution

You can also run commands directly on Modal using their CLI:

```bash
modal run --gpu A100 python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.6 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save ./out/ \
  --save_model ./workspace/pruned_models/llama_7b_test \
  --cache_dir ./llm_weights \
  --use_variant
```

### Option 3: Modal Shell (Interactive Testing)

For interactive testing:
```bash
modal shell --gpu A100
# Then run commands interactively inside the Modal environment
```

## What to Test

Run the **original GPU mode** (without `--use_cpu` flag) to verify:
1. ✅ Model loads successfully on GPU
2. ✅ All layers are pruned (should fit on large GPU)
3. ✅ Evaluation completes successfully
4. ✅ Results match expected behavior

## Expected Behavior

With a GPU that has sufficient RAM (e.g., A100 40GB or 80GB):
- **All 32 layers** should be pruned (not just 0-22)
- **No CPU offloading** should occur
- **Evaluation** should complete without OOM
- **Performance** should be fast (GPU-accelerated)

This verifies that the changes maintain backward compatibility - when GPU memory is sufficient, the code behaves exactly like the original.

## Test Command

The test will run:
```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.6 \
  --sparsity_type unstructured \
  --nsamples 8 \
  --save ./out/ \
  --save_model ./workspace/pruned_models/llama_7b_test \
  --cache_dir ./llm_weights \
  --use_variant
```

**Note**: No `--use_cpu` flag - this tests the GPU path with automatic memory management.

