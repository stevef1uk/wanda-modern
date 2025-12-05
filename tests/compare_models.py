#!/usr/bin/env python3
"""
Compare original and pruned models on performance and accuracy metrics.
Measures:
- Inference speed (tokens/second)
- Perplexity on WikiText-2
- Memory usage
- Model size
"""

import argparse
import time
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lib.eval import eval_ppl
from lib.prune import check_sparsity


def get_model_size(model):
    """Calculate model size in GB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb / 1024  # Convert to GB


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Get peak memory allocated during this process
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved,
            "max_allocated": max_allocated
        }
    return None


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()


def measure_inference_speed(model, tokenizer, prompts, num_runs=5, max_new_tokens=50, device=None):
    """
    Measure inference speed in tokens/second.
    
    Args:
        model: The model to test
        tokenizer: The tokenizer
        prompts: List of prompt strings
        num_runs: Number of runs to average over
        max_new_tokens: Number of tokens to generate
        device: Device to use (if None, auto-detect)
    
    Returns:
        Dictionary with timing metrics
    """
    # Determine device
    if device is None:
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
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
                input_device = next(model.parameters()).device
        else:
            input_device = next(model.parameters()).device
    else:
        input_device = device
    
    model.eval()
    times = []
    total_tokens = 0
    
    print(f"Measuring inference speed ({num_runs} runs)...")
    
    with torch.no_grad():
        for run_idx in range(num_runs):
            for prompt in prompts:
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(input_device) for k, v in inputs.items()}
                
                # Warm-up (first run)
                if run_idx == 0:
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
                    do_sample=False,  # Use greedy decoding for consistent timing
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
    
    avg_time = sum(times) / len(times)
    tokens_per_second = total_tokens / sum(times)
    
    return {
        "avg_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
        "num_runs": len(times)
    }


def load_model(model_path_or_name, cache_dir="llm_weights", use_cpu=False, is_pruned=False, load_in_8bit=False):
    """Load either original or pruned model with proper caching and offloading."""
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading {'pruned' if is_pruned else 'original'} model...")
    print(f"{'='*60}")

    # Resolve local path if pruned
    if is_pruned:
        model_path_or_name = os.path.abspath(os.path.expanduser(model_path_or_name))
        if not os.path.exists(model_path_or_name):
            raise FileNotFoundError(f"Pruned model path does not exist: {model_path_or_name}")
        print(f"Loading from local path: {model_path_or_name}")

    # Decide device
    if use_cpu or not torch.cuda.is_available():
        device = "cpu"
        device_map = None
    else:
        device = "cuda"
        device_map = "auto"  # enables CPU offloading

    # Tokenizer
    tokenizer_kwargs = {}
    if is_pruned:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name,
        cache_dir=None if is_pruned else cache_dir,
        use_fast=False,
        **tokenizer_kwargs
    )

    # Model load kwargs
    model_kwargs = {
        "torch_dtype": torch.float16 if not use_cpu else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    if is_pruned:
        model_kwargs["local_files_only"] = True
    else:
        model_kwargs["cache_dir"] = cache_dir

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        **model_kwargs
    )

    model.eval()

    # Ensure seqlen attribute
    if not hasattr(model, "seqlen"):
        model.seqlen = model.config.max_position_embeddings

    print(f"Using device: {device}")
    return model, tokenizer

 
    
def main():
    parser = argparse.ArgumentParser(description="Compare original and pruned models")
    parser.add_argument(
        "--original_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name for original model"
    )
    parser.add_argument(
        "--pruned_model",
        type=str,
        default="/home/stevef/dev/orig/wanda-modern/pruned_models/llama_7b_test",
        help="Path to pruned model directory"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="llm_weights",
        help="Directory to cache original model"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    parser.add_argument(
        "--skip_perplexity",
        action="store_true",
        help="Skip perplexity evaluation (faster, for quick tests)"
    )
    parser.add_argument(
        "--num_inference_runs",
        type=int,
        default=5,
        help="Number of inference runs for speed measurement"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Model Comparison: Original vs Pruned")
    print("="*60)
    
    # Test prompts for inference speed
    test_prompts = [
        "The capital of France is",
        "In the field of artificial intelligence,",
        "The weather today is",
        "Machine learning is",
        "The history of computing"
    ]
    
    results = {}
    
    # Test original model
    print("\n" + "="*60)
    print("TESTING ORIGINAL MODEL")
    print("="*60)
    
    try:
        original_model, original_tokenizer = load_model(
            args.original_model,
            cache_dir=args.cache_dir,
            use_cpu=args.use_cpu,
            is_pruned=False
        )
        
        # Model size
        original_size = get_model_size(original_model)
        print(f"\nModel size: {original_size:.2f} GB")
        
        # Memory usage - clear and measure before loading
        if torch.cuda.is_available() and not args.use_cpu:
            clear_gpu_memory()
            import time
            time.sleep(0.5)  # Let memory settle
            mem_before = get_memory_usage()
            print(f"\nGPU Memory before loading:")
            print(f"  Allocated: {mem_before['allocated']:.2f} GB")
            print(f"  Reserved: {mem_before['reserved']:.2f} GB")
            print(f"  Free: {mem_before['free']:.2f} GB / {mem_before['total']:.2f} GB total")
        
        # Inference speed
        print("\nMeasuring inference speed...")
        inference_results = measure_inference_speed(
            original_model,
            original_tokenizer,
            test_prompts,
            num_runs=args.num_inference_runs,
            device=None
        )
        print(f"Average generation time: {inference_results['avg_time']:.3f} seconds")
        print(f"Tokens per second: {inference_results['tokens_per_second']:.2f}")
        
        results['original'] = {
            'model_size_gb': original_size,
            'inference_speed': inference_results,
            'perplexity': None
        }
        
        # Perplexity evaluation
        if not args.skip_perplexity:
            print("\nEvaluating perplexity on WikiText-2...")
            try:
                # Create a dummy args object for eval_ppl
                class DummyArgs:
                    pass
                dummy_args = DummyArgs()
                
                device = torch.device("cpu" if args.use_cpu else "cuda:0")
                ppl = eval_ppl(dummy_args, original_model, original_tokenizer, device)
                print(f"Perplexity: {ppl:.4f}")
                results['original']['perplexity'] = ppl
            except Exception as e:
                print(f"Perplexity evaluation failed: {e}")
                results['original']['perplexity'] = None
        
        # Memory usage after - measure after a brief delay to let things settle
        if torch.cuda.is_available() and not args.use_cpu:
            import time
            time.sleep(0.5)  # Let memory allocation settle
            mem_after = get_memory_usage()
            memory_delta = mem_after['allocated'] - mem_before['allocated']
            print(f"\nGPU Memory after loading:")
            print(f"  Allocated: {mem_after['allocated']:.2f} GB (delta: +{memory_delta:.2f} GB)")
            print(f"  Reserved: {mem_after['reserved']:.2f} GB")
            print(f"  Free: {mem_after['free']:.2f} GB / {mem_after['total']:.2f} GB total")
            print(f"  Peak allocated: {mem_after['max_allocated']:.2f} GB")
            results['original']['gpu_memory_gb'] = mem_after['allocated']
            results['original']['gpu_memory_delta_gb'] = memory_delta
            results['original']['gpu_memory_peak_gb'] = mem_after['max_allocated']
        
        # Clean up - aggressively clear memory
        del original_model
        del original_tokenizer
        clear_gpu_memory()
        import time
        time.sleep(1.0)  # Give time for memory to be freed
        
    except Exception as e:
        print(f"Error testing original model: {e}")
        import traceback
        traceback.print_exc()
        results['original'] = None
    
    # Test pruned model
    print("\n" + "="*60)
    print("TESTING PRUNED MODEL")
    print("="*60)
   

    # --- Extra cleanup before loading pruned model ---
    if torch.cuda.is_available() and not args.use_cpu:
        import gc, time
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        time.sleep(3.0)  # Increase wait time to 3 seconds for memory to settle
 
    try:
        pruned_model, pruned_tokenizer = load_model(
            args.pruned_model,
            cache_dir=args.cache_dir,
            use_cpu=args.use_cpu,
            is_pruned=True
        )
        
        # Model size
        pruned_size = get_model_size(pruned_model)
        print(f"\nModel size: {pruned_size:.2f} GB")
        
        # Sparsity
        try:
            sparsity = check_sparsity(pruned_model)
            print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}% pruned)")
            results['pruned'] = {'sparsity': sparsity}
        except:
            print("Could not calculate sparsity")
        
        # Memory usage - clear and measure before loading
        if torch.cuda.is_available() and not args.use_cpu:
            clear_gpu_memory()
            import time
            time.sleep(0.5)  # Let memory settle
            mem_before = get_memory_usage()
            print(f"\nGPU Memory before loading:")
            print(f"  Allocated: {mem_before['allocated']:.2f} GB")
            print(f"  Reserved: {mem_before['reserved']:.2f} GB")
            print(f"  Free: {mem_before['free']:.2f} GB / {mem_before['total']:.2f} GB total")
        
        # Inference speed
        print("\nMeasuring inference speed...")
        inference_results = measure_inference_speed(
            pruned_model,
            pruned_tokenizer,
            test_prompts,
            num_runs=args.num_inference_runs,
            device=None
        )
        print(f"Average generation time: {inference_results['avg_time']:.3f} seconds")
        print(f"Tokens per second: {inference_results['tokens_per_second']:.2f}")
        
        results['pruned'] = results.get('pruned', {})
        results['pruned'].update({
            'model_size_gb': pruned_size,
            'inference_speed': inference_results,
            'perplexity': None
        })
        
        # Perplexity evaluation
        if not args.skip_perplexity:
            print("\nEvaluating perplexity on WikiText-2...")
            try:
                class DummyArgs:
                    pass
                dummy_args = DummyArgs()
                
                device = torch.device("cpu" if args.use_cpu else "cuda:0")
                ppl = eval_ppl(dummy_args, pruned_model, pruned_tokenizer, device)
                print(f"Perplexity: {ppl:.4f}")
                results['pruned']['perplexity'] = ppl
            except Exception as e:
                print(f"Perplexity evaluation failed: {e}")
                results['pruned']['perplexity'] = None
        
        # Memory usage after - measure after a brief delay to let things settle
        if torch.cuda.is_available() and not args.use_cpu:
            import time
            time.sleep(0.5)  # Let memory allocation settle
            mem_after = get_memory_usage()
            memory_delta = mem_after['allocated'] - mem_before['allocated']
            print(f"\nGPU Memory after loading:")
            print(f"  Allocated: {mem_after['allocated']:.2f} GB (delta: +{memory_delta:.2f} GB)")
            print(f"  Reserved: {mem_after['reserved']:.2f} GB")
            print(f"  Free: {mem_after['free']:.2f} GB / {mem_after['total']:.2f} GB total")
            print(f"  Peak allocated: {mem_after['max_allocated']:.2f} GB")
            results['pruned']['gpu_memory_gb'] = mem_after['allocated']
            results['pruned']['gpu_memory_delta_gb'] = memory_delta
            results['pruned']['gpu_memory_peak_gb'] = mem_after['max_allocated']
        
        # Clean up - aggressively clear memory
        del pruned_model
        del pruned_tokenizer
        clear_gpu_memory()
        import time
        time.sleep(1.0)  # Give time for memory to be freed
        
    except Exception as e:
        print(f"Error testing pruned model: {e}")
        import traceback
        traceback.print_exc()
        results['pruned'] = None
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if results.get('original') and results.get('pruned'):
        orig = results['original']
        prun = results['pruned']
        
        print(f"\n{'Metric':<30} {'Original':<20} {'Pruned':<20} {'Improvement':<20}")
        print("-" * 90)
        
        # Model size
        if 'model_size_gb' in orig and 'model_size_gb' in prun:
            size_reduction = (1 - prun['model_size_gb'] / orig['model_size_gb']) * 100
            print(f"{'Model Size (GB)':<30} {orig['model_size_gb']:<20.2f} {prun['model_size_gb']:<20.2f} {size_reduction:>19.1f}%")
        
        # Inference speed
        if 'inference_speed' in orig and 'inference_speed' in prun:
            orig_speed = orig['inference_speed']['tokens_per_second']
            prun_speed = prun['inference_speed']['tokens_per_second']
            speedup = prun_speed / orig_speed if orig_speed > 0 else 0
            print(f"{'Inference Speed (tok/s)':<30} {orig_speed:<20.2f} {prun_speed:<20.2f} {speedup:>19.2f}x")
            
            orig_time = orig['inference_speed']['avg_time']
            prun_time = prun['inference_speed']['avg_time']
            time_reduction = (1 - prun_time / orig_time) * 100 if orig_time > 0 else 0
            print(f"{'Avg Generation Time (s)':<30} {orig_time:<20.3f} {prun_time:<20.3f} {time_reduction:>19.1f}%")
        
        # Perplexity
        if orig.get('perplexity') and prun.get('perplexity'):
            orig_ppl = orig['perplexity']
            prun_ppl = prun['perplexity']
            ppl_increase = ((prun_ppl - orig_ppl) / orig_ppl) * 100 if orig_ppl > 0 else 0
            print(f"{'Perplexity (WikiText-2)':<30} {orig_ppl:<20.4f} {prun_ppl:<20.4f} {ppl_increase:>19.1f}%")
        
        # GPU Memory - show both allocated and delta
        if 'gpu_memory_gb' in orig and 'gpu_memory_gb' in prun:
            mem_reduction = (1 - prun['gpu_memory_gb'] / orig['gpu_memory_gb']) * 100
            print(f"{'GPU Memory Allocated (GB)':<30} {orig['gpu_memory_gb']:<20.2f} {prun['gpu_memory_gb']:<20.2f} {mem_reduction:>19.1f}%")
            if 'gpu_memory_delta_gb' in orig and 'gpu_memory_delta_gb' in prun:
                delta_reduction = (1 - prun['gpu_memory_delta_gb'] / orig['gpu_memory_delta_gb']) * 100 if orig['gpu_memory_delta_gb'] > 0 else 0
                print(f"{'GPU Memory Delta (GB)':<30} {orig['gpu_memory_delta_gb']:<20.2f} {prun['gpu_memory_delta_gb']:<20.2f} {delta_reduction:>19.1f}%")
            if 'gpu_memory_peak_gb' in orig and 'gpu_memory_peak_gb' in prun:
                peak_reduction = (1 - prun['gpu_memory_peak_gb'] / orig['gpu_memory_peak_gb']) * 100
                print(f"{'GPU Memory Peak (GB)':<30} {orig['gpu_memory_peak_gb']:<20.2f} {prun['gpu_memory_peak_gb']:<20.2f} {peak_reduction:>19.1f}%")
        
        # Sparsity
        if 'sparsity' in prun:
            print(f"{'Sparsity':<30} {'0.0000':<20} {prun['sparsity']:<20.4f} {'N/A':>19}")
        
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        
        if 'model_size_gb' in orig and 'model_size_gb' in prun:
            print(f"✓ Model size reduced by {(1 - prun['model_size_gb'] / orig['model_size_gb']) * 100:.1f}%")
        
        if 'gpu_memory_gb' in orig and 'gpu_memory_gb' in prun:
            mem_reduction = (1 - prun['gpu_memory_gb'] / orig['gpu_memory_gb']) * 100
            if mem_reduction > 0:
                print(f"✓ GPU memory reduced by {mem_reduction:.1f}%")
            elif mem_reduction < 0:
                print(f"⚠ GPU memory increased by {abs(mem_reduction):.1f}% (may be due to measurement timing)")
        
        if 'inference_speed' in orig and 'inference_speed' in prun:
            speedup = prun['inference_speed']['tokens_per_second'] / orig['inference_speed']['tokens_per_second']
            print(f"✓ Inference speed: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        if orig.get('perplexity') and prun.get('perplexity'):
            print(f"✓ Perplexity: {orig['perplexity']:.4f} → {prun['perplexity']:.4f} "
                  f"({((prun['perplexity'] - orig['perplexity']) / orig['perplexity']) * 100:+.1f}%)")
    
    else:
        print("Could not complete comparison - one or both models failed to load/test")
    
    print()


if __name__ == "__main__":
    main()

