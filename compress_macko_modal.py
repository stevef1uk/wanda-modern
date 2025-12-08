"""
Modal script to compress GPT-OSS-20B model using macko_spmv.
This script loads the model from a Modal volume and compresses it.
"""

import modal

# Create Modal app
app = modal.App("compress-gpt-oss-20b")

# Create or reference the volume
volume = modal.Volume.from_name("wanda-cache", create_if_missing=True)

# Define the image with all dependencies
# Use Modal's GPU image which has CUDA pre-installed
# Updated to CUDA 12.4.0 (12.1.0 was deprecated)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    })
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "fire==0.7.0",
        "tqdm==4.67.1",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "ninja",  # Required for compiling C++ extensions
    )
    .run_commands(
        # Clone and install macko_spmv from the correct repo
        "git clone https://github.com/vlejd/macko_spmv.git /tmp/macko_spmv",
        "cd /tmp/macko_spmv && pip install .",
    )
    .add_local_file("eval_compressed_ppl.py", "/workspace/eval_compressed_ppl.py")
    .add_local_file("benchmark_compressed_inference.py", "/workspace/benchmark_compressed_inference.py")
    .add_local_dir("lib", "/workspace/lib")
)

# Compression function
@app.function(
    image=image,
    volumes={"/cache": volume},
    gpu="A100-80GB",  # Fixed: Updated from deprecated syntax
    timeout=7200,  # 2 hours
    memory=200000,  # 200GB RAM
)
def compress_model(model_path: str = None):
    """Compress the GPT-OSS-20B model using the fixed compression script.
    
    Args:
        model_path: Optional path to the pruned model. Defaults to /cache/pruned_models/gptt-oss-20b-_test
    """
    import os
    import sys
    import torch
    from transformers import AutoModelForCausalLM
    from collections import OrderedDict, deque
    import tqdm
    import time
    import macko_spmv
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    # Environment
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Paths
    if model_path is None:
        model_path = "/cache/pruned_models/gptt-oss-20b-_test"
    
    # Determine output path based on model path
    if model_path.endswith("/"):
        model_path = model_path.rstrip("/")
    output_path = model_path + "_compressed.pt"
    
    logging.info(f"Starting compression of {model_path}")
    logging.info(f"Output will be saved to {output_path}")
    
    # Debug: List what's in the cache directory
    cache_dir = "/cache"
    if os.path.exists(cache_dir):
        logging.info(f"Contents of {cache_dir}:")
        try:
            for root, dirs, files in os.walk(cache_dir):
                level = root.replace(cache_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                logging.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # Limit to first 10 files per directory
                    logging.info(f"{subindent}{file}")
                if len(files) > 10:
                    logging.info(f"{subindent}... and {len(files) - 10} more files")
        except Exception as e:
            logging.warning(f"Error walking cache directory: {e}")
    else:
        logging.warning(f"Cache directory {cache_dir} does not exist!")
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try to find similar paths
        pruned_models_dir = "/cache/pruned_models"
        if os.path.exists(pruned_models_dir):
            logging.info(f"Contents of {pruned_models_dir}:")
            try:
                items = os.listdir(pruned_models_dir)
                if items:
                    for item in items:
                        item_path = os.path.join(pruned_models_dir, item)
                        item_type = 'dir' if os.path.isdir(item_path) else 'file'
                        size_info = ""
                        if os.path.isdir(item_path):
                            try:
                                # Count files in directory
                                file_count = sum(len(files) for _, _, files in os.walk(item_path))
                                size_info = f" ({file_count} files)"
                            except:
                                pass
                        logging.info(f"  {item} ({item_type}){size_info}")
                else:
                    logging.warning(f"  {pruned_models_dir} is empty!")
            except Exception as e:
                logging.warning(f"Error listing {pruned_models_dir}: {e}")
        else:
            logging.warning(f"Pruned models directory {pruned_models_dir} does not exist!")
            # Try to create it to verify volume is writable
            try:
                os.makedirs(pruned_models_dir, exist_ok=True)
                logging.info(f"Created {pruned_models_dir} - volume is writable")
            except Exception as e:
                logging.error(f"Cannot create {pruned_models_dir}: {e}")
        
        error_msg = (
            f"Model not found at {model_path}\n"
            f"Please ensure the pruned model has been saved to the Modal volume.\n"
            f"Run 'test_modal.py' first to create the pruned model, or upload it manually:\n"
            f"  modal volume put wanda-cache /path/to/model /cache/pruned_models/gptt-oss-20b-_test"
        )
        raise FileNotFoundError(error_msg)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model from {model_path} into {device} memory...")
    
    model_dense = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    state_dict = model_dense.state_dict()
    logging.info(f"Model loaded. Total tensors: {len(state_dict)}")
    
    # Pre-inspection: Log all MLP expert layer shapes
    logging.info("=" * 80)
    logging.info("Pre-compression inspection: MLP Expert Layer Shapes")
    logging.info("=" * 80)
    mlp_expert_keys = [k for k in state_dict.keys() if "mlp.experts" in k and ("gate_up_proj" in k or "down_proj" in k) and not k.endswith("_bias")]
    router_keys = [k for k in state_dict.keys() if "mlp.router" in k and "weight" in k]
    
    logging.info(f"\nFound {len(mlp_expert_keys)} expert layer keys and {len(router_keys)} router keys")
    logging.info("\nFirst 3 layers - Expert layer shapes:")
    for k in sorted(mlp_expert_keys)[:6]:  # First 6 (3 layers * 2 expert types)
        v = state_dict[k]
        output_dim = v.shape[-1] if len(v.shape) >= 2 else None
        is_router = output_dim is not None and output_dim <= 32
        router_marker = " [ROUTER - WILL SKIP]" if is_router else " [EXPERT - WILL COMPRESS]"
        logging.info(f"  {k:60s} shape: {str(v.shape):30s} output_dim: {output_dim}{router_marker}")
    
    logging.info("\nFirst 3 layers - Router layer shapes (should NOT be compressed):")
    for k in sorted(router_keys)[:3]:
        v = state_dict[k]
        output_dim = v.shape[-1] if len(v.shape) >= 2 else None
        logging.info(f"  {k:60s} shape: {str(v.shape):30s} output_dim: {output_dim}")
    logging.info("=" * 80)
    logging.info("Starting compression...")
    logging.info("=" * 80)
    
    # Compress
    compressed_state_dict = OrderedDict()
    pbar = tqdm.tqdm(total=len(state_dict))
    start_time = time.time()
    last_times = deque(maxlen=5)
    
    for idx, (k, v) in enumerate(state_dict.items()):
        t0 = time.time()
        v = v.to(device) if device == "cuda" else v
        
        # Determine if this tensor should be compressed
        # CRITICAL: Check for bias FIRST to prevent compressing bias tensors
        
        # Skip bias terms (they are vectors, not matrices)
        if "bias" in k:
            should_compress = False
        # Skip router weights (they're small routing matrices)
        elif "router" in k:
            should_compress = False
        # Compress self-attention weights
        elif "self_attn" in k and "weight" in k:
            should_compress = True
        # Compress MLP expert layers (gate_up_proj and down_proj)
        # Must exclude bias terms
        # CRITICAL: Must validate output dimension to distinguish expert layers from router layers
        # Router layers have output_dim <= 32 (number of experts), expert layers have much larger output_dim
        elif "mlp.experts" in k and ("gate_up_proj" in k or "down_proj" in k):
            if not k.endswith("_bias"):
                # Check output dimension - router layers have output_dim <= 32
                # Expert layers have shape [num_experts, in_features, out_features]:
                #   - gate_up_proj: [32, 2880, 5760] (output_dim=5760)
                #   - down_proj: [32, 2880, 2880] (output_dim=2880)
                # Router layers have output_dim <= 32 (the number of experts)
                output_dim = v.shape[-1] if len(v.shape) >= 2 else None
                
                # Additional check: if it's a 3D tensor, also check if first dim is num_experts (32)
                # and output_dim matches num_experts, it's likely a router layer
                is_router_by_shape = False
                if len(v.shape) == 3 and v.shape[0] == 32 and output_dim == 32:
                    # Shape [32, X, 32] with first and last dim both 32 suggests router layer
                    is_router_by_shape = True
                elif len(v.shape) == 2 and v.shape[0] == 32 and output_dim == 32:
                    # Shape [32, 32] suggests router layer
                    is_router_by_shape = True
                
                if output_dim is None or output_dim <= 32 or is_router_by_shape:
                    # This is a router layer - skip compression
                    logging.warning(f"Skipping {k} with shape {v.shape} (output_dim={output_dim}) - router layer detected (output_dim<=32 or shape pattern matches router)")
                    should_compress = False
                elif len(v.shape) == 3:
                    # 3D tensor with output_dim > 32 - this is an expert layer
                    logging.info(f"Compressing {k} with shape {v.shape} (output_dim={output_dim}) - 3D expert layer")
                    should_compress = True
                elif len(v.shape) == 2:
                    # 2D tensor with output_dim > 32 - could be flattened expert layer
                    logging.info(f"Compressing {k} with shape {v.shape} (output_dim={output_dim}) - 2D tensor (flattened expert)")
                    should_compress = True
                else:
                    logging.warning(f"Skipping {k} - unexpected shape {v.shape} (not 2D or 3D)")
                    should_compress = False
            else:
                should_compress = False
        # Compress other MLP weights
        elif "mlp" in k and "weight" in k:
            should_compress = True
        else:
            should_compress = False
        
        if should_compress:
            logging.debug(f"Compressing: {k} (shape: {v.shape})")
            compressed = macko_spmv.compress(v)
            for i in range(5):
                # For expert layers, append .c_{i} directly
                if "mlp.experts" in k and ("gate_up_proj" in k or "down_proj" in k):
                    new_k = f"{k}.c_{i}"
                else:
                    # For other weights, replace .weight with .c_{i}
                    new_k = k.replace("weight", f"c_{i}")
                val = compressed[i]
                if isinstance(val, torch.Tensor):
                    val = val.cpu()
                compressed_state_dict[new_k] = val
        else:
            logging.debug(f"Keeping uncompressed: {k} (shape: {v.shape})")
            compressed_state_dict[k] = v.cpu()
        
        dt = time.time() - t0
        last_times.append(dt)
        avg_time = sum(last_times) / len(last_times)
        remaining = len(state_dict) - (idx + 1)
        eta = remaining * avg_time
        pbar.set_description(f"{k} ({dt:.2f}s) ETA {eta:.1f}s")
        pbar.update(1)
        
        # Save intermediate checkpoint every 20 tensors
        if (idx + 1) % 20 == 0:
            tmp_file = output_path.replace(".pt", "_tmp.pt")
            logging.info(f"Saving intermediate checkpoint: {tmp_file}")
            torch.save(compressed_state_dict, tmp_file)
            # Ensure file is flushed to disk
            sys.stdout.flush()
            os.sync()  # Force write to disk
    
    pbar.close()
    total_time = time.time() - start_time
    
    # Save final compressed model
    logging.info(f"Saving final compressed model: {output_path}")
    torch.save(compressed_state_dict, output_path)
    
    # Ensure file is flushed to disk before committing volume
    sys.stdout.flush()
    os.sync()  # Force write to disk
    
    # Verify file exists and has size > 0
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        logging.info(f"Saved file size: {file_size / (1024**3):.2f} GB")
        if file_size == 0:
            raise RuntimeError(f"Output file {output_path} is empty!")
    else:
        raise RuntimeError(f"Output file {output_path} was not created!")
    
    # Commit volume changes - this persists files to Modal volume
    logging.info("Committing changes to Modal volume...")
    volume.commit()
    logging.info("Volume committed successfully")
    
    # Print summary
    compressed_count = sum(1 for k in compressed_state_dict.keys() if ".c_0" in k)
    skipped_router_count = sum(1 for k in compressed_state_dict.keys() if "mlp.experts" in k and ("gate_up_proj" in k or "down_proj" in k) and ".c_0" not in k and "_bias" not in k)
    logging.info("=" * 60)
    logging.info("Compression Complete!")
    logging.info(f"  Total tensors: {len(compressed_state_dict)}")
    logging.info(f"  Compressed layers: {compressed_count}")
    logging.info(f"  Skipped router layers: {skipped_router_count}")
    logging.info(f"  Time taken: {total_time:.1f}s")
    logging.info(f"  Output file: {output_path}")
    logging.info("=" * 60)
    
    return {
        "success": True,
        "output_path": output_path,
        "total_tensors": len(compressed_state_dict),
        "compressed_layers": compressed_count,
        "time_seconds": total_time,
        "model_path": model_path,
        "compressed_path": output_path
    }


# Inference benchmark function
@app.function(
    image=image,
    volumes={"/cache": volume},
    gpu="A100-80GB",
    timeout=1800,  # 30 minutes
    memory=200000,  # 200GB RAM
)
def run_inference_benchmark(model_path: str, compressed_path: str):
    """Run inference speed benchmark on compressed model."""
    import subprocess
    import sys
    import os
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    logging.info("=" * 60)
    logging.info("Running Inference Benchmark")
    logging.info("=" * 60)
    
    # Run the inference benchmark script
    cmd = [
        sys.executable,
        "/workspace/benchmark_compressed_inference.py",
        model_path,
        compressed_path,
        "--device", "cuda",
        "--num-runs", "5",
        "--max-new-tokens", "50"
    ]
    
    logging.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"Inference benchmark failed with code {result.returncode}")
        logging.error(f"STDOUT: {result.stdout}")
        logging.error(f"STDERR: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logging.info("Inference benchmark completed successfully")
    logging.info(result.stdout)
    
    # Try to extract tokens/second from output
    import re
    tok_per_sec_match = re.search(r'Tokens per second[:\s]+([\d.]+)', result.stdout)
    tokens_per_second = float(tok_per_sec_match.group(1)) if tok_per_sec_match else None
    
    return {
        "success": True,
        "output": result.stdout,
        "tokens_per_second": tokens_per_second
    }


# Perplexity evaluation function
@app.function(
    image=image,
    volumes={"/cache": volume},
    gpu="A100-80GB",
    timeout=3600,  # 1 hour
    memory=200000,  # 200GB RAM
)
def run_perplexity_eval(model_path: str, compressed_path: str):
    """Run perplexity evaluation on compressed model."""
    import subprocess
    import sys
    import os
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    logging.info("=" * 60)
    logging.info("Running Perplexity Evaluation")
    logging.info("=" * 60)
    
    # Run the perplexity evaluation script
    cmd = [
        sys.executable,
        "/workspace/eval_compressed_ppl.py",
        model_path,
        compressed_path,
        "--device", "cuda"
    ]
    
    logging.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"Perplexity evaluation failed with code {result.returncode}")
        logging.error(f"STDOUT: {result.stdout}")
        logging.error(f"STDERR: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logging.info("Perplexity evaluation completed successfully")
    logging.info(result.stdout)
    
    # Try to extract perplexity value from output
    import re
    ppl_match = re.search(r'Perplexity[:\s]+([\d.]+)', result.stdout)
    perplexity = float(ppl_match.group(1)) if ppl_match else None
    
    return {
        "success": True,
        "output": result.stdout,
        "perplexity": perplexity
    }


@app.local_entrypoint()
def main(model_path: str = None, skip_compression: bool = False, skip_inference: bool = False, skip_ppl: bool = False):
    """Run the compression, inference benchmark, and perplexity evaluation.
    
    Args:
        model_path: Optional path to the pruned model in the Modal volume.
                   Defaults to /cache/pruned_models/gptt-oss-20b-_test
                   Example: /cache/pruned_models/my-model
        skip_compression: Skip compression step (use existing compressed model) (default: False)
        skip_inference: Skip inference benchmark (default: False)
        skip_ppl: Skip perplexity evaluation (default: False)
    """
    if model_path:
        print(f"Using custom model path: {model_path}")
    else:
        print("Using default model path: /cache/pruned_models/gptt-oss-20b-_test")
        print("(To use a different path, run: modal run compress_macko_modal.py --model-path /cache/pruned_models/your-model)")
    
    # Determine paths
    if model_path is None:
        model_path = "/cache/pruned_models/gptt-oss-20b-_test"
    if model_path.endswith("/"):
        model_path = model_path.rstrip("/")
    compressed_path = model_path + "_compressed.pt"
    
    # Step 1: Compress the model (unless skipped)
    if not skip_compression:
        print("\n" + "=" * 60)
        print("Starting model compression...")
        print("=" * 60)
        result = compress_model.remote(model_path)
        print(f"\n✅ Compression completed successfully!")
        print(f"   Output: {result['output_path']}")
        print(f"   Compressed layers: {result['compressed_layers']}")
        print(f"   Time: {result['time_seconds']:.1f}s")
        compressed_path = result['compressed_path']
    else:
        print("\n⏭️  Skipping compression (--skip-compression)")
        print(f"   Using existing compressed model: {compressed_path}")
        result = {
            "model_path": model_path,
            "compressed_path": compressed_path
        }
    
    # Step 2: Run inference benchmark
    if not skip_inference:
        print("\n" + "=" * 60)
        print("Running Inference Benchmark...")
        print("=" * 60)
        inference_result = run_inference_benchmark.remote(
            result['model_path'],
            result['compressed_path']
        )
        if inference_result.get('success'):
            if inference_result.get('tokens_per_second'):
                print(f"✅ Inference benchmark completed successfully")
                print(f"   Tokens per second: {inference_result['tokens_per_second']:.2f}")
            else:
                print("✅ Inference benchmark completed (check logs for details)")
        else:
            print(f"⚠️  Inference benchmark had issues: {inference_result.get('error', 'Unknown error')}")
    else:
        print("\n⏭️  Skipping inference benchmark (--skip-inference)")
    
    # Step 3: Run perplexity evaluation
    if not skip_ppl:
        print("\n" + "=" * 60)
        print("Running Perplexity Evaluation...")
        print("=" * 60)
        ppl_result = run_perplexity_eval.remote(
            result['model_path'],
            result['compressed_path']
        )
        if ppl_result.get('success'):
            if ppl_result.get('perplexity'):
                print(f"✅ Perplexity evaluation completed successfully")
                print(f"   WikiText2 Perplexity: {ppl_result['perplexity']:.4f}")
            else:
                print("✅ Perplexity evaluation completed (check logs for value)")
        else:
            print(f"⚠️  Perplexity evaluation had issues: {ppl_result.get('error', 'Unknown error')}")
    else:
        print("\n⏭️  Skipping perplexity evaluation (--skip-ppl)")
    
    print("\n" + "=" * 60)
    print("✅ All tasks completed!")
    print("=" * 60)