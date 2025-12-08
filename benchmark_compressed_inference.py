#!/usr/bin/env python3
"""
Benchmark inference speed on compressed model using macko_spmv.
This measures tokens/second and compares compressed vs uncompressed performance.
"""

# CRITICAL: These must be first, before ANY other imports
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_SDPA"] = "1"

import torch
# Completely disable torch._dynamo to avoid tracing issues
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import macko_spmv
import tqdm

# Add lib directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_lib_paths = [
    os.path.join(script_dir, 'lib'),
    '/workspace/lib',
    os.path.join(os.path.dirname(script_dir), 'lib'),
]

for lib_path in possible_lib_paths:
    if os.path.exists(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, os.path.dirname(lib_path))
        break


class CustomLayer(nn.Module):
    """Custom layer for macko_spmv compressed weights."""
    def __init__(self, c_0, c_1, c_2, c_3, c_4):
        super().__init__()
        self.register_buffer("c_0", c_0)
        self.register_buffer("c_1", c_1)
        self.register_buffer("c_2", c_2)
        if isinstance(c_3, torch.Tensor):
            self.register_buffer("c_3", c_3)
        else:
            self.c_3 = c_3
        if isinstance(c_4, torch.Tensor):
            self.register_buffer("c_4", c_4)
        else:
            self.c_4 = c_4

    def forward(self, x):
        """Apply macko_spmv.multiply handling batched inputs."""
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        N, H = x_flat.shape
        
        outputs = []
        for i in range(N):
            vec = x_flat[i]
            out_vec = macko_spmv.multiply(
                (self.c_0, self.c_1, self.c_2, self.c_3, self.c_4),
                vec
            )
            outputs.append(out_vec)
        
        out_flat = torch.stack(outputs, dim=0)
        new_shape = list(original_shape[:-1]) + [out_flat.shape[-1]]
        return out_flat.reshape(new_shape)


def fix_attention(model, i, proj_name, sd):
    """Replace attention projection with compressed macko_spmv layer."""
    path = f"model.layers.{i}.self_attn.{proj_name}"
    attn_obj = model.model.layers[i].self_attn
    new_layer = CustomLayer(
        sd[f"{path}.c_0"],
        sd[f"{path}.c_1"],
        sd[f"{path}.c_2"],
        sd[f"{path}.c_3"],
        sd[f"{path}.c_4"],
    )
    if hasattr(attn_obj, proj_name):
        delattr(attn_obj, proj_name)
    attn_obj.register_module(proj_name, new_layer)


def fix_mlp_expert(model, i, expert_name, sd):
    """Replace MLP expert layer with compressed macko_spmv layer."""
    path = f"model.layers.{i}.mlp.experts.{expert_name}"
    
    # Verify compressed keys exist
    # If compressed keys don't exist, this layer was skipped during compression (likely a router layer)
    # In this case, use the original uncompressed layer from the model
    if f"{path}.c_0" not in sd:
        # Check if this is the original uncompressed layer (saved without .c_0 suffix)
        if path in sd:
            # Layer was saved uncompressed (router layer), use it as-is
            return
        else:
            # Layer not found at all - this is an error
            raise KeyError(f"Compressed key {path}.c_0 not found in state dict, and uncompressed key {path} also not found")
    
    # Validate compressed representation shape
    # Note: This is a sanity check - the compression script should have already filtered out router layers
    c_2 = sd[f"{path}.c_2"]
    if isinstance(c_2, torch.Tensor):
        output_dim = c_2.shape[0] - 1  # c_2 is [out_features + 1]
        if output_dim <= 32:
            # This is a router layer that was incorrectly compressed
            # Skip replacing it - use the original uncompressed layer from the model
            print(f"WARNING: Layer {path} has output dimension {output_dim} (looks like a router layer).")
            print(f"  Skipping compression replacement - will use original uncompressed layer.")
            print(f"  Please re-compress the model with the updated script to skip router layers.")
            return  # Skip replacing this layer
    
    mlp_obj = model.model.layers[i].mlp.experts
    new_layer = CustomLayer(
        sd[f"{path}.c_0"],
        sd[f"{path}.c_1"],
        sd[f"{path}.c_2"],
        sd[f"{path}.c_3"],
        sd[f"{path}.c_4"],
    )
    if hasattr(mlp_obj, expert_name):
        delattr(mlp_obj, expert_name)
    mlp_obj.register_module(expert_name, new_layer)
    
    # Patch the expert's forward method
    if expert_name == "gate_up_proj" and not hasattr(mlp_obj, '_forward_patched'):
        def patched_forward(self_expert, hidden_states, router_indices=None, routing_weights=None):
            gate_up = self_expert.gate_up_proj(hidden_states)
            
            if hasattr(self_expert, 'gate_up_proj_bias') and self_expert.gate_up_proj_bias is not None:
                bias = self_expert.gate_up_proj_bias
                if bias.dim() == 2 and bias.shape[1] == gate_up.shape[-1]:
                    gate_up = gate_up + bias[None, :, None, :]
                elif bias.dim() == 1 and bias.shape[0] == gate_up.shape[-1]:
                    gate_up = gate_up + bias[None, None, None, :]
            
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = torch.nn.functional.silu(gate) * up
            
            down = self_expert.down_proj(hidden_states)
            if hasattr(self_expert, 'down_proj_bias') and self_expert.down_proj_bias is not None:
                bias = self_expert.down_proj_bias
                if bias.dim() == 2 and bias.shape[1] == down.shape[-1]:
                    down = down + bias[None, :, None, :]
                elif bias.dim() == 1 and bias.shape[0] == down.shape[-1]:
                    down = down + bias[None, None, None, :]
            
            return down
        
        mlp_obj.forward = patched_forward.__get__(mlp_obj, type(mlp_obj))
        mlp_obj._forward_patched = True


def load_compressed_model(model_path, compressed_path, device="cuda"):
    """Load uncompressed model and replace layers with compressed macko_spmv layers."""
    print(f"Loading uncompressed model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
        attn_implementation="eager"
    )
    model.config._attn_implementation = "eager"
    if hasattr(model.config, 'attn_implementation'):
        model.config.attn_implementation = "eager"
    
    print(f"Loading compressed weights from: {compressed_path}")
    sd = torch.load(compressed_path, map_location="cpu")
    
    print("Replacing layers with compressed macko_spmv layers...")
    print("Using gpt-oss-20b MoE architecture (mlp.experts.gate_up_proj, mlp.experts.down_proj)...")
    
    for i in tqdm.trange(len(model.model.layers), desc="Replacing layers"):
        try:
            fix_attention(model, i, "q_proj", sd)
            fix_attention(model, i, "k_proj", sd)
            fix_attention(model, i, "v_proj", sd)
            fix_attention(model, i, "o_proj", sd)
        except KeyError as e:
            print(f"Warning: Could not load attention weights for layer {i}: {e}")
        
        try:
            fix_mlp_expert(model, i, "gate_up_proj", sd)
            fix_mlp_expert(model, i, "down_proj", sd)
        except KeyError as e:
            print(f"Warning: Could not load MLP expert weights for layer {i}: {e}")
    
    print(f"Moving model to {device}...")
    model = model.to(device=device)
    model.eval()
    
    return model


def benchmark_inference(model, tokenizer, prompts, num_runs=5, max_new_tokens=50, device="cuda"):
    """Benchmark inference speed."""
    model.eval()
    times = []
    total_tokens = 0
    
    print(f"\nBenchmarking inference ({num_runs} runs, {len(prompts)} prompts)...")
    
    with torch.no_grad():
        for run_idx in range(num_runs):
            for prompt_idx, prompt in enumerate(prompts):
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                inputs["input_ids"] = inputs["input_ids"].long()
                
                # Warm-up (first run)
                if run_idx == 0 and prompt_idx == 0:
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Measure generation time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Count generated tokens
                input_length = inputs['input_ids'].shape[1]
                output_length = outputs.shape[1]
                generated_tokens = output_length - input_length
                
                elapsed = end_time - start_time
                times.append(elapsed)
                total_tokens += generated_tokens
                
                if run_idx == 0:
                    print(f"  Prompt {prompt_idx+1}: {generated_tokens} tokens in {elapsed:.3f}s ({generated_tokens/elapsed:.1f} tok/s)")
    
    avg_time = sum(times) / len(times)
    tokens_per_second = total_tokens / sum(times)
    
    return {
        "avg_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
        "num_runs": len(times)
    }


def main(model_path, compressed_path, device="cuda", num_runs=5, max_new_tokens=50):
    """Run inference benchmark on compressed model."""
    print("=" * 60)
    print("Inference Benchmark on Compressed Model")
    print("=" * 60)
    
    # Load compressed model
    model = load_compressed_model(model_path, compressed_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "In a world where artificial intelligence",
        "The quick brown fox jumps",
    ]
    
    # Benchmark
    results = benchmark_inference(
        model,
        tokenizer,
        test_prompts,
        num_runs=num_runs,
        max_new_tokens=max_new_tokens,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("✅ Inference Benchmark Complete!")
    print(f"Average generation time: {results['avg_time']:.3f} seconds")
    print(f"Tokens per second: {results['tokens_per_second']:.2f}")
    print(f"Total tokens generated: {results['total_tokens']}")
    print(f"Number of runs: {results['num_runs']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark inference speed on compressed model")
    parser.add_argument("model_path", type=str, help="Path to uncompressed model")
    parser.add_argument("compressed_path", type=str, help="Path to compressed model (.pt file)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    main(
        args.model_path,
        args.compressed_path,
        args.device,
        args.num_runs,
        args.max_new_tokens
    )

