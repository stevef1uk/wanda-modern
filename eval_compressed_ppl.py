#!/usr/bin/env python3
"""
Evaluate perplexity on a compressed model using macko_spmv.
This loads the compressed model and runs WikiText2 perplexity evaluation.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import macko_spmv
import tqdm

# Add lib directory to path to import eval functions
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

from lib.eval import eval_ppl_wikitext
from lib.data import get_loaders


class CustomLayer(nn.Module):
    """Custom layer for macko_spmv compressed weights."""
    def __init__(self, c_0, c_1, c_2, c_3, c_4):
        super().__init__()
        self.register_buffer("c_0", c_0)
        self.register_buffer("c_1", c_1)
        self.register_buffer("c_2", c_2)
        # c_3 and c_4 might be integers or small tensors
        if isinstance(c_3, torch.Tensor):
            self.register_buffer("c_3", c_3)
        else:
            self.c_3 = c_3
        if isinstance(c_4, torch.Tensor):
            self.register_buffer("c_4", c_4)
        else:
            self.c_4 = c_4
        self._debug_once = True

    def forward(self, x):
        """
        Apply macko_spmv.multiply handling batched inputs.
        """
        original_shape = x.shape
        
        # Flatten to [N, H]
        x_flat = x.reshape(-1, x.shape[-1])
        N, H = x_flat.shape
        
        # Process each vector individually - macko_spmv.multiply expects 1D input
        outputs = []
        for i in range(N):
            vec = x_flat[i]  # [H]
            out_vec = macko_spmv.multiply(
                (self.c_0, self.c_1, self.c_2, self.c_3, self.c_4),
                vec
            )
            outputs.append(out_vec)
        
        out_flat = torch.stack(outputs, dim=0)  # [N, H_out]
        
        if self._debug_once:
            print(f"CustomLayer: input {original_shape} -> output_flat {out_flat.shape}")
            print(f"  c_0 shape: {self.c_0.shape}, c_1 shape: {self.c_1.shape}, c_2 shape: {self.c_2.shape}")
            print(f"  c_3 type: {type(self.c_3)}, c_4 type: {type(self.c_4)}")
            if hasattr(self.c_3, 'shape'):
                print(f"  c_3 shape: {self.c_3.shape}, c_4 shape: {self.c_4.shape}")
            # Check if this looks like a router layer (output size = num_experts)
            if out_flat.shape[-1] == 32:
                print(f"  WARNING: Output size is 32 (num_experts) - this might be a router layer!")
            self._debug_once = False
        
        # Reshape back
        new_shape = list(original_shape[:-1]) + [out_flat.shape[-1]]
        out = out_flat.reshape(new_shape)
        
        return out


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
    
    # Patch the expert's forward method only if both gate_up_proj and down_proj are valid (not router layers)
    if expert_name == "gate_up_proj" and not hasattr(mlp_obj, '_forward_patched'):
        # Check if down_proj is also a valid expert layer (not a router layer)
        down_proj_path = f"model.layers.{i}.mlp.experts.down_proj"
        should_patch = True
        if f"{down_proj_path}.c_2" in sd:
            down_c_2 = sd[f"{down_proj_path}.c_2"]
            if isinstance(down_c_2, torch.Tensor):
                down_output_dim = down_c_2.shape[0] - 1
                if down_output_dim <= 32:
                    should_patch = False  # down_proj is a router layer, don't patch
        
        if not should_patch:
            return  # Skip patching if down_proj is a router layer
        def patched_forward(self_expert, hidden_states, router_indices=None, routing_weights=None):
            # hidden_states shape: [num_experts, batch, seq_len, hidden_dim] for MoE
            # We need to handle the expert dimension correctly
            
            # Use CustomLayer directly
            gate_up = self_expert.gate_up_proj(hidden_states)
            
            # Validate output shape - gate_up should be [num_experts, batch, seq_len, 2*intermediate_dim]
            # If output_dim <= 32, this is likely a router layer that was incorrectly compressed
            # In this case, we can't proceed with the normal flow, so raise an error with helpful message
            if gate_up.shape[-1] <= 32:
                raise RuntimeError(
                    f"gate_up_proj output dimension is {gate_up.shape[-1]} (expected ~5760). "
                    f"This is a router layer that was incorrectly compressed. "
                    f"Please re-compress the model with the updated script that skips router layers."
                )
            
            # Add bias if it exists and has correct shape
            if hasattr(self_expert, 'gate_up_proj_bias') and self_expert.gate_up_proj_bias is not None:
                bias = self_expert.gate_up_proj_bias
                if bias.dim() == 2 and bias.shape[1] == gate_up.shape[-1]:
                    gate_up = gate_up + bias[None, :, None, :]
                elif bias.dim() == 1 and bias.shape[0] == gate_up.shape[-1]:
                    gate_up = gate_up + bias[None, None, None, :]
            
            # Split and activate
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = torch.nn.functional.silu(gate) * up
            
            # Down projection
            down = self_expert.down_proj(hidden_states)
            
            # Validate output shape - down should be [num_experts, batch, seq_len, hidden_dim]
            # If output_dim <= 32, this is likely a router layer that was incorrectly compressed
            # Just return it as-is (it's already computed)
            if down.shape[-1] <= 32:
                # This is a router layer - return as-is
                return down
            
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
    import sys
    sys.stdout.flush()  # Ensure output is flushed
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        local_files_only=True,
        attn_implementation="eager"
    )
    print("Model loaded successfully")
    import sys
    sys.stdout.flush()
    
    # Force eager attention in config
    model.config._attn_implementation = "eager"
    if hasattr(model.config, 'attn_implementation'):
        model.config.attn_implementation = "eager"
    
    print(f"Loading compressed weights from: {compressed_path}")
    import sys
    sys.stdout.flush()
    
    sd = torch.load(compressed_path, map_location="cpu")
    print(f"Loaded compressed state dict with {len(sd)} keys")
    sys.stdout.flush()
    
    print("Replacing layers with compressed macko_spmv layers...")
    print("Using gpt-oss-20b MoE architecture (mlp.experts.gate_up_proj, mlp.experts.down_proj)...")
    sys.stdout.flush()
    
    for i in tqdm.trange(len(model.model.layers), desc="Replacing layers"):
        # Attention projections
        try:
            fix_attention(model, i, "q_proj", sd)
            fix_attention(model, i, "k_proj", sd)
            fix_attention(model, i, "v_proj", sd)
            fix_attention(model, i, "o_proj", sd)
        except KeyError as e:
            print(f"Warning: Could not load attention weights for layer {i}: {e}")
        
        # MLP expert layers
        try:
            fix_mlp_expert(model, i, "gate_up_proj", sd)
            fix_mlp_expert(model, i, "down_proj", sd)
        except KeyError as e:
            print(f"Warning: Could not load MLP expert weights for layer {i}: {e}")
    
    print(f"Moving model to {device}...")
    import sys
    sys.stdout.flush()
    
    model = model.to(device=device)
    model.eval()
    print("Model moved to device and set to eval mode")
    sys.stdout.flush()
    
    # Set seqlen for evaluation
    if not hasattr(model, 'seqlen'):
        model.seqlen = model.config.max_position_embeddings
    
    return model


def eval_compressed_ppl(model_path, compressed_path, device="cuda"):
    """Evaluate perplexity on compressed model."""
    print("=" * 60)
    print("Evaluating Perplexity on Compressed Model")
    print("=" * 60)
    
    # Load compressed model
    model = load_compressed_model(model_path, compressed_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    
    # Get test loader for WikiText2
    dataset = "wikitext2"
    eval_seqlen = min(model.seqlen, 2048)
    if model.seqlen > 2048:
        print(f"Note: Model supports {model.seqlen} tokens, but using {eval_seqlen} for evaluation")
    
    print(f"Loading {dataset} test dataset...")
    import sys
    sys.stdout.flush()
    
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=eval_seqlen, tokenizer=tokenizer
    )
    print(f"Dataset loaded, {len(testloader)} samples")
    sys.stdout.flush()
    
    # Evaluate perplexity
    print("Running perplexity evaluation...")
    sys.stdout.flush()
    
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, bs=1, device=device)
    
    print("=" * 60)
    print(f"✅ Perplexity Evaluation Complete!")
    print(f"WikiText2 Perplexity: {ppl:.4f}")
    print("=" * 60)
    
    return ppl


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate perplexity on compressed model")
    parser.add_argument("model_path", type=str, help="Path to uncompressed model")
    parser.add_argument("compressed_path", type=str, help="Path to compressed model (.pt file)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    eval_compressed_ppl(args.model_path, args.compressed_path, args.device)