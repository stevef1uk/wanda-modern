#!/usr/bin/env python3
"""Inspect tensor shapes in the model to understand the structure."""
import torch
from transformers import AutoModelForCausalLM
import sys

def inspect_shapes(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )
    
    print("\n" + "=" * 80)
    print("Inspecting MLP Expert Layer Shapes")
    print("=" * 80)
    
    state_dict = model.state_dict()
    
    # Find all MLP expert related keys
    mlp_keys = [k for k in state_dict.keys() if "mlp" in k.lower()]
    
    print(f"\nFound {len(mlp_keys)} MLP-related keys")
    print("\nFirst layer MLP keys:")
    for k in sorted(mlp_keys):
        if "layers.0" in k:
            v = state_dict[k]
            print(f"  {k:60s} shape: {str(v.shape):30s} dtype: {v.dtype}")
    
    print("\n" + "=" * 80)
    print("Detailed Expert Layer Analysis")
    print("=" * 80)
    
    # Check first few layers
    for layer_idx in range(min(3, len(model.model.layers))):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
            print(f"\nLayer {layer_idx}:")
            
            # Check gate_up_proj
            if hasattr(experts, 'gate_up_proj'):
                gate_up = experts.gate_up_proj
                if hasattr(gate_up, 'weight'):
                    print(f"  gate_up_proj.weight: {gate_up.weight.shape}")
                else:
                    print(f"  gate_up_proj (direct): {gate_up.shape if hasattr(gate_up, 'shape') else 'N/A'}")
                    # Try to get from state dict
                    key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
                    if key in state_dict:
                        print(f"  gate_up_proj (state_dict): {state_dict[key].shape}")
            
            # Check down_proj
            if hasattr(experts, 'down_proj'):
                down = experts.down_proj
                if hasattr(down, 'weight'):
                    print(f"  down_proj.weight: {down.weight.shape}")
                else:
                    print(f"  down_proj (direct): {down.shape if hasattr(down, 'shape') else 'N/A'}")
                    key = f"model.layers.{layer_idx}.mlp.experts.down_proj"
                    if key in state_dict:
                        print(f"  down_proj (state_dict): {state_dict[key].shape}")
            
            # Check router
            if hasattr(layer.mlp, 'router'):
                router = layer.mlp.router
                if hasattr(router, 'weight'):
                    print(f"  router.weight: {router.weight.shape}")
                else:
                    key = f"model.layers.{layer_idx}.mlp.router.weight"
                    if key in state_dict:
                        print(f"  router.weight (state_dict): {state_dict[key].shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model_shapes.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inspect_shapes(model_path)

