#!/usr/bin/env python3
"""
Example script to load and use a pruned model for inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_pruned_model(model_path, use_cpu=False):
    """
    Load a pruned model from disk.
    
    Args:
        model_path: Path to the saved pruned model directory
        use_cpu: If True, force CPU usage even if GPU is available
    
    Returns:
        model, tokenizer
    """
    print(f"Loading pruned model from: {model_path}")
    
    # Determine device
    if use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        device_map = "cpu"
        torch_dtype = torch.float32
    else:
        device = torch.device("cuda:0")
        # Use device_map="auto" to handle offloaded layers
        device_map = "auto"
        torch_dtype = torch.float16
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, device=None):
    """
    Generate text using the pruned model.
    
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
    
    parser = argparse.ArgumentParser(description="Use a pruned model for inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4",
        help="Path to the saved pruned model"
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
    model, tokenizer = load_pruned_model(args.model_path, use_cpu=args.use_cpu)
    
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

