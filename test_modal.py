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
        "transformers>=4.46.0",  # Need 4.46.0+ for gpt_oss support
        "accelerate>=1.10.1",
        "datasets>=4.1.1",
        "sentencepiece>=0.2.1",
        "numpy>=2.3.3",
        "protobuf>=6.32.1",
    )
    # Only mount necessary code files, not large directories like llm_weights, modal-env, etc.
    .add_local_file("main.py", "/workspace/main.py")
    .add_local_file("main_opt.py", "/workspace/main_opt.py")
    .add_local_file("eval_pruned_model.py", "/workspace/eval_pruned_model.py")
    .add_local_dir("lib", "/workspace/lib")
    .add_local_dir("dense_ft", "/workspace/dense_ft")
    .add_local_dir("lora_ft", "/workspace/lora_ft")
    .add_local_dir("image_classifiers", "/workspace/image_classifiers")
    .add_local_dir("scripts", "/workspace/scripts")
    .add_local_dir("tests", "/workspace/tests")
    .add_local_file("pyproject.toml", "/workspace/pyproject.toml")
    .add_local_file("README.md", "/workspace/README.md")
    .add_local_file("INSTALL.md", "/workspace/INSTALL.md")
    .add_local_file("LICENSE", "/workspace/LICENSE")
)

# Create a Modal app
app = modal.App("wanda-pruning-test1", image=image)

# Define a volume for model cache and outputs
volume = modal.Volume.from_name("wanda-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # Use H100 with 80GB (more memory for large models)
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
        "--model", "openai/gpt-oss-20b",
        "--prune_method", "wanda",
        "--sparsity_ratio", "0.6",
        "--sparsity_type", "unstructured",
        "--nsamples", "8",
        "--save", "./out/",
        "--save_model", "./workspace/pruned_models/gptt-oss-20b",
        "--cache_dir", "./llm_weights",
        "--use_variant",
    ]
    
    print("=" * 60)
    print("Testing GPU pruning with sufficient RAM")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the command with real-time output streaming
    # This allows us to see progress as the model downloads and loads
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Stream output in real-time
    stdout_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)  # Print immediately
        stdout_lines.append(line)
    
    # Wait for process to complete
    returncode = process.wait()
    stdout = ''.join(stdout_lines)
    stderr = ''  # Already captured in stdout
    
    # Check results
    if returncode == 0:
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
        model_dir = pathlib.Path("./workspace/pruned_models/gptt-oss-20b")
        if model_dir.exists():
            print(f"\nCopying pruned model to /cache/pruned_models/gptt-oss-20b...")
            cache_model = pathlib.Path("/cache/pruned_models/gptt-oss-20b-_test")
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
        if stdout:
            print("\nSTDOUT:")
            print(stdout)
        if stderr:
            print("\nSTDERR:")
            print(stderr)
        print(f"\nExit code: {returncode}")
        print("=" * 60)
        sys.exit(1)
    
    return returncode

@app.function(
    image=image,
    gpu="H100",  # Use H100 with 80GB (more memory for large models)
    volumes={"/cache": volume},
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("hf-secret")],  # For HuggingFace token
)
def eval_pruned_model_ppl(model_path: str = None, use_cpu: bool = False, skip_sparsity_check: bool = False):
    """Evaluate perplexity on an already pruned model without re-pruning."""
    import subprocess
    import sys
    import os
    
    # Set Modal environment variable
    os.environ["MODAL_ENVIRONMENT"] = "1"
    
    # Set working directory to mounted code
    os.chdir("/workspace")
    
    # Default model path if not provided
    if model_path is None:
        model_path = "/cache/pruned_models/gptt-oss-20b-_test"
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Available models in /cache/pruned_models/:")
        pruned_models_dir = "/cache/pruned_models"
        if os.path.exists(pruned_models_dir):
            for item in os.listdir(pruned_models_dir):
                item_path = os.path.join(pruned_models_dir, item)
                item_type = 'dir' if os.path.isdir(item_path) else 'file'
                print(f"  {item} ({item_type})")
        sys.exit(1)
    
    # Build command
    cmd = [
        "python", "eval_pruned_model.py",
        "--model_path", model_path,
    ]
    
    if use_cpu:
        cmd.append("--use_cpu")
    
    if skip_sparsity_check:
        cmd.append("--skip_sparsity_check")
    
    print("=" * 60)
    print("Evaluating Perplexity on Pruned Model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the command with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output in real-time
    stdout_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        stdout_lines.append(line)
    
    # Wait for process to complete
    returncode = process.wait()
    stdout = ''.join(stdout_lines)
    
    if returncode == 0:
        print("\n" + "=" * 60)
        print("✅ Perplexity evaluation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Perplexity evaluation failed")
        print("=" * 60)
        if stdout:
            print("\nSTDOUT:")
            print(stdout)
        print(f"\nExit code: {returncode}")
        sys.exit(1)
    
    return returncode

@app.local_entrypoint()
def main(skip_pruning: bool = False, model_path: str = None, use_cpu: bool = False, skip_sparsity_check: bool = False):
    """
    Entry point for local execution.
    
    Args:
        skip_pruning: If True, skip pruning and only run perplexity evaluation
        model_path: Path to pruned model (required if skip_pruning=True)
        use_cpu: Force CPU usage for evaluation
        skip_sparsity_check: Skip sparsity check during evaluation
    """
    if skip_pruning:
        eval_pruned_model_ppl.remote(model_path, use_cpu, skip_sparsity_check)
    else:
        test_gpu_pruning.remote(volume)

