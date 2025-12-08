# Debugging MACKO Compression on Modal

This guide explains how to use the debug flags to skip earlier stages and debug issues faster.

## Quick Debug Commands

### Test Import Only (Fastest - Skip Everything Except Import Test)

This is the fastest way to test CUDA_HOME and import issues:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install --test-import-only
```

**Note**: This assumes the repository, venv, and dependencies are already set up from a previous run.

### Skip Clone (Use Existing Repository)

If the repository is already cloned:

```bash
modal run compress_macko_modal.py --skip-clone
```

### Skip Setup (Use Existing Virtual Environment)

If the virtual environment already exists:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup
```

### Skip Install (Use Existing Dependencies)

If dependencies are already installed:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install
```

### Skip Verify (Skip Import Check)

If you want to skip the import verification:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install --skip-verify
```

## Common Debug Scenarios

### Scenario 1: Debugging CUDA_HOME Issue

After the first run, you can quickly test CUDA_HOME detection:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install --test-import-only
```

This will:
1. Skip cloning (uses existing repo)
2. Skip venv setup (uses existing venv)
3. Skip dependency installation (uses existing packages)
4. Only test the macko_spmv import (which requires CUDA_HOME)

### Scenario 2: Testing After Fixing CUDA_HOME

If you've fixed CUDA_HOME detection, test it quickly:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install --test-import-only
```

### Scenario 3: Full Run After Successful Import

Once import works, run the full pipeline:

```bash
modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install
```

This skips setup but runs compression and inference.

## Available Flags

- `--skip-clone`: Skip cloning macko_spmv repository
- `--skip-setup`: Skip virtual environment setup
- `--skip-install`: Skip dependency installation
- `--skip-verify`: Skip import verification
- `--test-import-only`: Only test macko_spmv import (exit after verification)

## Typical Workflow

1. **First Run** (Full setup):
   ```bash
   modal run compress_macko_modal.py
   ```

2. **Debug Import Issue** (Fast iteration):
   ```bash
   modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install --test-import-only
   ```

3. **Once Import Works** (Run compression):
   ```bash
   modal run compress_macko_modal.py --skip-clone --skip-setup --skip-install
   ```

## Notes

- The flags are cumulative - you can combine them as needed
- If you skip a stage, the script will verify that the required resources exist
- If a required resource doesn't exist when skipped, the script will exit with an error
- The `--test-import-only` flag is useful for debugging CUDA_HOME and compilation issues

