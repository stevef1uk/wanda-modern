import fire
import torch
from transformers import AutoModelForCausalLM
from collections import OrderedDict, deque
import tqdm
import os
import time
import macko_spmv
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# ----------------------------
# Environment
# ----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------
# Load model state dict
# ----------------------------
def get_state_dict(model_path, device="cpu"):
    logging.info(f"Loading model from {model_path} into {device} memory...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    state_dict = model_dense.state_dict()
    logging.info(f"Model loaded. Total tensors: {len(state_dict)}")
    return state_dict

# ----------------------------
# Compress routine with ETA
# ----------------------------
def compress(model_path, use_gpu=False, save_interval=20, eta_window=5):
    model_path = model_path.rstrip("/")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    state_dict = get_state_dict(model_path, device=device)
    compressed_state_dict = OrderedDict()
    pbar = tqdm.tqdm(total=len(state_dict))
    start_time = time.time()
    last_times = deque(maxlen=eta_window)  # For moving average of last few tensors
    
    for idx, (k, v) in enumerate(state_dict.items()):
        t0 = time.time()
        v = v.to(device) if device == "cuda" else v
        
        # Determine if this tensor should be compressed
        # CRITICAL FIX: Check for bias FIRST before checking layer type
        # This prevents compressing bias tensors which are small and break dimensions
        
        # Skip bias terms (they are vectors, not matrices)
        if "bias" in k:
            should_compress = False
        # Skip router weights (they're small routing matrices, not large linear layers)
        elif "router" in k:
            should_compress = False
        # Compress self-attention weights (they have .weight suffix)
        elif "self_attn" in k and "weight" in k:
            should_compress = True
        # Compress MLP expert layers (gpt-oss-20b: gate_up_proj and down_proj)
        # These are actual weight tensors without .weight suffix
        # Must exclude bias terms (gate_up_proj_bias, down_proj_bias)
        # CRITICAL: Must validate output dimension to distinguish expert layers from router layers
        # Router layers have output_dim <= 32 (number of experts), expert layers have much larger output_dim
        elif "mlp.experts" in k and ("gate_up_proj" in k or "down_proj" in k):
            # Double-check this is not a bias (already checked above, but be explicit)
            if not k.endswith("_bias"):
                # Check output dimension - router layers have output_dim <= 32
                # Expert layers have shape [num_experts, in_features, out_features]:
                #   - gate_up_proj: [32, 2880, 5760] (output_dim=5760)
                #   - down_proj: [32, 2880, 2880] (output_dim=2880)
                # Router layers have output_dim <= 32 (the number of experts)
                output_dim = v.shape[-1] if len(v.shape) >= 2 else None
                
                if output_dim is None or output_dim <= 32:
                    # This is a router layer - skip compression
                    logging.warning(f"Skipping {k} with shape {v.shape} (output_dim={output_dim}) - router layer detected")
                    should_compress = False
                elif len(v.shape) == 3:
                    # 3D tensor with output_dim > 32 - this is an expert layer
                    logging.info(f"✓ Compressing {k} with shape {v.shape} (output_dim={output_dim}) - 3D expert layer")
                    should_compress = True
                elif len(v.shape) == 2:
                    # 2D tensor with output_dim > 32 - could be flattened expert layer
                    logging.info(f"✓ Compressing {k} with shape {v.shape} (output_dim={output_dim}) - 2D tensor (flattened expert)")
                    should_compress = True
                else:
                    logging.warning(f"Skipping {k} - unexpected shape {v.shape} (not 2D or 3D)")
                    should_compress = False
            else:
                should_compress = False
        # Compress other MLP weights (for LLaMA-style models with .weight suffix)
        elif "mlp" in k and "weight" in k:
            should_compress = True
        else:
            should_compress = False

        if should_compress:
            logging.debug(f"Compressing: {k} (shape: {v.shape})")
            compressed = macko_spmv.compress(v)
            for i in range(5):
                # For expert layers, append .c_{i} directly as they don't have .weight
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
        
        # Save intermediate checkpoint
        if (idx + 1) % save_interval == 0:
            tmp_file = model_path + "_compressed_tmp.pt"
            logging.info(f"Saving intermediate checkpoint: {tmp_file}")
            torch.save(compressed_state_dict, tmp_file)
    
    pbar.close()
    total_time = time.time() - start_time
    
    # Save final compressed model
    final_file = model_path + "_compressed.pt"
    logging.info(f"Saving final compressed model: {final_file} (total time {total_time:.1f}s)")
    torch.save(compressed_state_dict, final_file)
    
    # Print summary
    compressed_count = sum(1 for k in compressed_state_dict.keys() if ".c_0" in k)
    logging.info(f"Compression complete!")
    logging.info(f"  Total tensors: {len(compressed_state_dict)}")
    logging.info(f"  Compressed layers: {compressed_count}")
    logging.info(f"  Time taken: {total_time:.1f}s")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    fire.Fire(compress)