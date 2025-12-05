#!/usr/bin/env python3
"""
Quick script to measure GPU memory usage of a single model.
"""

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved,
            "max_allocated": max_allocated
        }
    return None


def get_model_gpu_memory(model):
    """Calculate actual GPU memory used by model parameters."""
    if not torch.cuda.is_available():
        return {"gpu_memory_gb": 0.0, "cpu_layers": 0, "gpu_layers": 0, "gpu_params": 0, "cpu_params": 0}
    
    gpu_memory_bytes = 0
    cpu_layers = 0
    gpu_layers = 0
    gpu_params = 0
    cpu_params = 0
    
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            gpu_memory_bytes += param.nelement() * param.element_size()
            gpu_params += param.nelement()
            gpu_layers += 1
        else:
            cpu_params += param.nelement()
            cpu_layers += 1
    
    for name, buffer in model.named_buffers():
        if buffer.device.type == 'cuda':
            gpu_memory_bytes += buffer.nelement() * buffer.element_size()
    
    gpu_memory_gb = gpu_memory_bytes / (1024**3)
    return {
        "gpu_memory_gb": gpu_memory_gb,
        "cpu_layers": cpu_layers,
        "gpu_layers": gpu_layers,
        "gpu_params": gpu_params,
        "cpu_params": cpu_params,
        "total_params": gpu_params + cpu_params
    }


def main():
    parser = argparse.ArgumentParser(description="Check GPU memory usage of a model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (local) or HuggingFace model name"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="llm_weights",
        help="Cache directory for HuggingFace models"
    )
    parser.add_argument(
        "--is_pruned",
        action="store_true",
        help="Model is a local pruned model"
    )
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Disable CPU offloading (load entire model on GPU if it fits)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("GPU Memory Check")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    # Clear memory first
    clear_gpu_memory()
    time.sleep(0.5)
    
    mem_before = get_memory_usage()
    print(f"\nBefore loading:")
    print(f"  Allocated: {mem_before['allocated']:.2f} GB")
    print(f"  Reserved: {mem_before['reserved']:.2f} GB")
    print(f"  Free: {mem_before['free']:.2f} GB / {mem_before['total']:.2f} GB total")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    
    try:
        import accelerate
        has_accelerate = True
    except ImportError:
        has_accelerate = False
        print("Warning: accelerate not installed, using device_map='auto' may fail")
    
    load_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    if args.no_offload:
        # Disable CPU offloading - load entire model on GPU
        if has_accelerate:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved_gb = max(2.0, gpu_memory_gb * 0.15)
            max_memory_gb = gpu_memory_gb - reserved_gb
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {
                0: f"{max_memory_gb:.1f}GiB",
                "cpu": "0GiB"  # Disable CPU offloading
            }
        else:
            # Without accelerate, just move to GPU manually
            load_kwargs["device_map"] = None
    elif has_accelerate:
        # Allow CPU offloading
        load_kwargs["device_map"] = "auto"
        if not args.is_pruned:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved_gb = max(2.0, gpu_memory_gb * 0.15)
            max_memory_gb = gpu_memory_gb - reserved_gb
            load_kwargs["max_memory"] = {
                0: f"{max_memory_gb:.1f}GiB",
                "cpu": "50GiB"
            }
    
    if args.is_pruned:
        model_path = os.path.abspath(os.path.expanduser(args.model))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **load_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, **load_kwargs)
    
    if not has_accelerate and torch.cuda.is_available() and load_kwargs.get("device_map") is None:
        model = model.to("cuda:0")
    
    model.eval()
    
    # Wait for memory to settle
    time.sleep(0.5)
    
    mem_after = get_memory_usage()
    memory_delta = mem_after['allocated'] - mem_before['allocated']
    model_mem = get_model_gpu_memory(model)
    
    print(f"\nAfter loading:")
    print(f"  Allocated: {mem_after['allocated']:.2f} GB (delta: +{memory_delta:.2f} GB)")
    print(f"  Reserved: {mem_after['reserved']:.2f} GB")
    print(f"  Free: {mem_after['free']:.2f} GB / {mem_after['total']:.2f} GB total")
    print(f"  Peak allocated: {mem_after['max_allocated']:.2f} GB")
    
    print(f"\nModel GPU Memory:")
    print(f"  GPU parameters: {model_mem['gpu_memory_gb']:.2f} GB")
    print(f"  GPU param count: {model_mem['gpu_params']:,}")
    print(f"  Layers on GPU: {model_mem['gpu_layers']}")
    if model_mem['cpu_layers'] > 0:
        print(f"  Layers on CPU: {model_mem['cpu_layers']} (offloaded)")
        print(f"  CPU param count: {model_mem['cpu_params']:,}")
    print(f"  Total parameters: {model_mem['total_params']:,}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Model GPU memory: {model_mem['gpu_memory_gb']:.2f} GB")
    print(f"GPU parameters: {model_mem['gpu_params']:,}")
    if model_mem['cpu_layers'] > 0:
        print(f"CPU parameters: {model_mem['cpu_params']:,} ({model_mem['cpu_params']/model_mem['total_params']*100:.1f}% offloaded)")
    print(f"Total GPU allocated: {mem_after['allocated']:.2f} GB")
    print(f"Memory delta: +{memory_delta:.2f} GB")
    
    # Cleanup
    del model
    del tokenizer
    clear_gpu_memory()


if __name__ == "__main__":
    main()

