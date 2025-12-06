"""
Modal script to test GPU pruning with sufficient RAM.
This verifies backward compatibility - when GPU has enough memory,
all layers should be pruned on GPU without offloading.
"""

import modal

# Create a Modal image with required dependencies and local code
# Use Python 3.12 to match pyproject.toml requirements (numpy>=2.3.3 needs Python >=3.11)
# Note: Modal caches images. To force rebuild with latest code, set MODAL_IGNORE_CACHE=true
# or use: modal run --ignore-cache test_modal.py
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8.0",
        "transformers==4.45.2",  # Exact version from pyproject.toml
        "accelerate>=1.10.1",
        "datasets>=4.1.1",
        "sentencepiece>=0.2.1",
        "numpy>=2.3.3",
        "protobuf>=6.32.1",
    )
    .add_local_dir(".", "/workspace")  # Mount local code directory
)

# Create a Modal app
app = modal.App("wanda-pruning-test", image=image)

# Define a volume for model cache and outputs
volume = modal.Volume.from_name("wanda-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",  # Use A100 with 40GB (or "A100-80GB" for 80GB)
    volumes={"/cache": volume},
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("hf-secret")],  # For HuggingFace token
)
def test_gpu_pruning(vol: modal.Volume):
    """Test GPU pruning with sufficient RAM."""
    import subprocess
    import sys
    import os
    
    # Set Modal environment variable so main.py can detect it
    os.environ["MODAL_ENVIRONMENT"] = "1"
    
    # Set working directory to mounted code
    os.chdir("/workspace")
    
    # Verify code is available and show what's being used
    print("=" * 60)
    print("Verifying mounted code...")
    print("=" * 60)
    
    if not os.path.exists("main.py"):
        print("ERROR: main.py not found in /workspace")
        print("Make sure you're running 'modal run test_modal.py' from the repo root directory")
        sys.exit(1)
    
    # Check if this is the modified version by looking for --use_cpu flag and exist_ok fix
    with open("main.py", "r") as f:
        main_content = f.read()
        has_use_cpu = "--use_cpu" in main_content
        has_exist_ok_fix = "os.makedirs(args.save, exist_ok=True)" in main_content or ("if args.save:" in main_content and "exist_ok=True" in main_content)
        
        if has_use_cpu:
            print("✅ Using MODIFIED code (with GPU memory management)")
        else:
            print("⚠️  WARNING: Code doesn't appear to have --use_cpu flag")
            print("   Make sure you're running from the directory with your changes")
        
        if has_exist_ok_fix:
            print("✅ Code has exist_ok=True fix for directory creation")
        else:
            print("⚠️  WARNING: Code doesn't appear to have exist_ok=True fix")
            print("   This may cause FileExistsError if directory already exists")
    
    # Show current directory and key files
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"main.py exists: {os.path.exists('main.py')}")
    print(f"lib/prune.py exists: {os.path.exists('lib/prune.py')}")
    print(f"lib/eval.py exists: {os.path.exists('lib/eval.py')}")
    print("=" * 60)
    print()
    
    # Clean up output directory if it exists (to avoid FileExistsError)
    import shutil
    out_dir = "./out"
    if os.path.exists(out_dir):
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
            print(f"Cleaned up existing directory: {out_dir}")
        else:
            os.remove(out_dir)
            print(f"Removed existing file: {out_dir}")
    
    # Run the pruning command (original GPU mode, no --use_cpu)
    # Note: Modal mounts the local directory, so code should be fresh
    # If you see stale code, run: MODAL_IGNORE_CACHE=true modal run test_modal.py
    cmd = [
        "python", "main.py",
        "--model", "meta-llama/Llama-2-7b-hf",
        "--prune_method", "wanda",
        "--sparsity_ratio", "0.6",
        "--sparsity_type", "unstructured",
        "--nsamples", "8",
        "--save", "./out/",
        "--save_model", "./workspace/pruned_models/llama_7b_test",
        "--cache_dir", "./llm_weights",
        "--use_variant",
    ]
    
    print("=" * 60)
    print("Testing GPU pruning with sufficient RAM")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the command - capture output so we can show errors if it fails
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output in real-time style (for better visibility)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Check results
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ TEST PASSED: Pruning completed successfully!")
        print("=" * 60)
        
        # Copy output files to Modal volume for persistence
        import shutil
        import pathlib
        
        # Copy output directory to volume
        out_dir = pathlib.Path("./out")
        if out_dir.exists():
            print(f"\nCopying output files from ./out/ to /cache/out/...")
            cache_out = pathlib.Path("/cache/out")
            cache_out.mkdir(parents=True, exist_ok=True)
            for f in out_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, cache_out / f.name)
                    print(f"  Copied: {f.name}")
                elif f.is_dir():
                    shutil.copytree(f, cache_out / f.name, dirs_exist_ok=True)
                    print(f"  Copied directory: {f.name}")
        
        # Copy pruned model to volume
        model_dir = pathlib.Path("./workspace/pruned_models/llama_7b_test")
        if model_dir.exists():
            print(f"\nCopying pruned model to /cache/pruned_models/llama_7b_test...")
            cache_model = pathlib.Path("/cache/pruned_models/llama_7b_test")
            cache_model.parent.mkdir(parents=True, exist_ok=True)
            if cache_model.exists():
                shutil.rmtree(cache_model)
            shutil.copytree(model_dir, cache_model)
            print(f"  Model copied successfully")
        
        # Commit volume changes - Modal volumes need explicit commit to persist
        print("\nCommitting changes to Modal volume...")
        vol.commit()
        print("✅ Files saved to Modal volume 'wanda-cache'")
        print("\nTo download files locally, run:")
        print("  modal volume get wanda-cache out/ ./out/")
        print("  modal volume get wanda-cache pruned_models/ ./pruned_models/")
    else:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED: Pruning encountered errors")
        print("=" * 60)
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        print(f"\nExit code: {result.returncode}")
        print("=" * 60)
        sys.exit(1)
    
    return result.returncode

@app.local_entrypoint()
def main():
    """Entry point for local execution."""
    test_gpu_pruning.remote(volume)

