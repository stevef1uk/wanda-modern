#!/usr/bin/env python3
"""
Evaluate a saved pruned model on WikiText2 perplexity.

This script allows you to evaluate a pruned model separately from the pruning process,
which is useful when:
- Pruning was done with --skip_eval flag
- Evaluation failed due to OOM during pruning
- You want to evaluate on a different machine with more memory
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.eval import eval_ppl
from lib.prune import check_sparsity


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
        torch_dtype = torch.float16  # Use float16 for CPU to save memory
        print("Using CPU mode (float16)")
    else:
        device = torch.device("cuda:0")
        device_map = "auto"  # Use device_map="auto" to handle offloaded layers
        torch_dtype = torch.float16
        print("Using GPU mode (float16)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model.eval()
    
    # Set seqlen for evaluation (required by eval_ppl)
    if not hasattr(model, 'seqlen'):
        model.seqlen = model.config.max_position_embeddings
    
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved pruned model on WikiText2 perplexity"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the saved pruned model directory'
    )
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    parser.add_argument(
        '--skip_sparsity_check',
        action='store_true',
        help='Skip sparsity check (faster)'
    )
    
    args = parser.parse_args()
    
    # Load the pruned model
    model, tokenizer, device = load_pruned_model(args.model_path, args.use_cpu)
    
    # Check sparsity
    if not args.skip_sparsity_check:
        print("\n" + "="*60)
        print("Checking sparsity...")
        print("="*60)
        try:
            sparsity_ratio = check_sparsity(model)
            print(f"Sparsity: {sparsity_ratio:.4f} ({sparsity_ratio*100:.2f}% pruned)")
        except Exception as e:
            print(f"Warning: Could not calculate sparsity: {e}")
        print("="*60 + "\n")
    
    # Create a minimal args object for eval_ppl
    # eval_ppl only needs args.model for dataset selection, but we can use a dummy
    class EvalArgs:
        def __init__(self):
            self.model = "meta-llama/Llama-2-7b-hf"  # Dummy, only used for dataset selection
    
    eval_args = EvalArgs()
    
    # Run evaluation
    print("\n" + "="*60)
    print("Starting WikiText2 perplexity evaluation...")
    print("="*60)
    
    try:
        ppl_test = eval_ppl(eval_args, model, tokenizer, device)
        print("\n" + "="*60)
        print(f"✅ Evaluation complete!")
        print(f"WikiText2 Perplexity: {ppl_test:.4f}")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ Evaluation failed: {e}")
        print("="*60)
        raise


if __name__ == '__main__':
    main()

