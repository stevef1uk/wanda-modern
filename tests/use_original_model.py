#!/usr/bin/env python3
"""
Example script to load and use the original (unpruned) model for inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_original_model(model_name, cache_dir="llm_weights", use_cpu=False):
    """
    Load the original unpruned model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
        cache_dir: Directory to cache the model
        use_cpu: If True, force CPU usage even if GPU is available
    
    Returns:
        model, tokenizer
    """
    print(f"Loading original model: {model_name}")
    
    # Determine device
    if use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        device_map = "cpu"
        torch_dtype = torch.float32
    else:
        device = torch.device("cuda:0")
        # Use device_map="auto" to handle memory constraints
        device_map = "auto"
        torch_dtype = torch.float16
        
        # Check GPU memory and set max_memory if needed
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved_gb = max(2.0, gpu_memory_gb * 0.15)
            max_memory_gb = gpu_memory_gb - reserved_gb
            
            max_memory = {
                0: f"{max_memory_gb:.1f}GiB",
                "cpu": "50GiB"
            }
            print(f"GPU detected: {gpu_memory_gb:.2f} GB total, reserving {reserved_gb:.1f} GB")
            print(f"Model will use up to {max_memory_gb:.1f} GB GPU memory, with CPU offloading as fallback")
        else:
            max_memory = None
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    
    # Load model
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        **load_kwargs
    )
    
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, device=None):
    """
    Generate text using the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        device: Device to use (if None, will be determined automatically)
    
    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Determine device for inputs
    if device is None:
        # For models with device_map, find the device of the embedding layer
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # Check embedding layer device
            if "model.embed_tokens" in model.hf_device_map:
                embed_dev = model.hf_device_map["model.embed_tokens"]
                if isinstance(embed_dev, str):
                    input_device = torch.device(embed_dev)
                elif isinstance(embed_dev, torch.device):
                    input_device = embed_dev
                elif isinstance(embed_dev, int):
                    input_device = torch.device(f"cuda:{embed_dev}")
                else:
                    input_device = torch.device("cpu")
            else:
                # Fallback: get device from first parameter
                input_device = next(model.parameters()).device
        else:
            # Get device from model parameters
            input_device = next(model.parameters()).device
    else:
        input_device = device
    
    # Move inputs to the correct device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Use the original (unpruned) model for inference")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="llm_weights",
        help="Directory to cache the model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt for text generation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_original_model(
        args.model, 
        cache_dir=args.cache_dir,
        use_cpu=args.use_cpu
    )
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    generated = generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(f"\nGenerated text:\n{generated}\n")

if __name__ == "__main__":
    main()

