import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count() if torch.cuda.is_available() else 0)

def get_llm(model_name, cache_dir="llm_weights", use_gpu=True):
    """
    Load model with GPU support and memory management to avoid OOM.
    Uses device_map="auto" with max_memory limits to allow CPU offloading if needed.
    For Modal/cloud environments, prevents CPU offloading to avoid rotary_emb initialization issues.
    """
    # Use float16 for efficiency (works on both GPU and CPU)
    torch_dtype = torch.float16
    
    # For Modal/cloud environments, prevent offloading to avoid rotary_emb initialization issues
    is_modal = os.getenv("MODAL_ENVIRONMENT") or os.getenv("MODAL")
    
    if use_gpu and torch.cuda.is_available():
        # Calculate GPU memory limits - reserve some for operations
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Load config explicitly with trust_remote_code for custom architectures
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Reserve some GPU memory for operations (pruning needs temporary memory for sorting, etc.)
        reserved_gb = max(2.0, gpu_memory_gb * 0.15)  # Reserve 15% or at least 2GB for pruning operations
        max_memory_gb = gpu_memory_gb - reserved_gb
        
        if is_modal:
            # On Modal, prevent CPU offloading to avoid rotary_emb initialization issues
            # H100 (80GB) has enough memory for large models without offloading
            print(f"Modal environment detected: Preventing CPU offloading to avoid rotary_emb issues")
            max_memory = {
                0: f"{max_memory_gb:.1f}GiB",
                "cpu": "0GiB"  # Disable CPU offloading on Modal
            }
            print(f"GPU detected: {gpu_memory_gb:.2f} GB total, reserving {reserved_gb:.1f} GB")
            print(f"Model will use up to {max_memory_gb:.1f} GB GPU memory (CPU offloading disabled)")
        else:
            # Set max_memory to allow CPU offloading if GPU runs out (for local runs)
            max_memory = {
                0: f"{max_memory_gb:.1f}GiB",
                "cpu": "50GiB"  # Allow CPU offloading with 50GB limit
            }
            print(f"GPU detected: {gpu_memory_gb:.2f} GB total, reserving {reserved_gb:.1f} GB")
            print(f"Model will use up to {max_memory_gb:.1f} GB GPU memory, with CPU offloading as fallback")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            config=config,
            torch_dtype=torch_dtype, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True
        )
    else:
        print("Using CPU (GPU not available or disabled)")
        # Use float16 for CPU to reduce memory usage (~7GB vs ~14GB for float32)
        torch_dtype = torch.float16
        print("Using float16 on CPU (~7GB RAM required)")
        print("Loading model (this may take 5-10 minutes on CPU, please be patient)...")
        print("Note: Loading progress will be shown below. If it seems stuck, it's just slow on CPU.")
        
        try:
            import time
            start_time = time.time()
            
            # Load config explicitly with trust_remote_code for custom architectures
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch_dtype, 
                cache_dir=cache_dir, 
                low_cpu_mem_usage=True, 
                device_map="cpu",
                trust_remote_code=True
            )
            load_time = time.time() - start_time
            print(f"Model loaded successfully! (took {load_time/60:.1f} minutes)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print("\n" + "="*60)
                print("ERROR: Not enough RAM to load the model!")
                print("="*60)
                print("The model needs ~7-8GB free RAM in float16 mode.")
                print(f"Current available RAM: Check with 'free -h'")
                print("\nOptions:")
                print("1. Free up RAM (close other applications)")
                print("2. Wait for other processes to finish")
                print("3. Consider using GPU mode (remove --use_cpu flag)")
                print("   This will prune layers 0-22 on GPU, which is faster")
                print("="*60)
            raise
        except Exception as e:
            print(f"\nError loading model: {e}")
            if "Killed" in str(e) or "killed" in str(e).lower():
                print("\n" + "="*60)
                print("Process was killed by system (likely OOM killer)")
                print("="*60)
                print("Your system ran out of memory during loading.")
                print("Check available RAM with: free -h")
                print("\nSolutions:")
                print("1. Free up RAM - close other applications")
                print("2. Wait and try again when more RAM is available")
                print("3. Use GPU mode instead (remove --use_cpu flag)")
                print("="*60)
            raise

    model.seqlen = model.config.max_position_embeddings 
    
    # Initialize rotary embeddings for CPU-loaded models to avoid NoneType errors
    # This is necessary for transformers 4.57+ when loading on CPU with device_map="cpu"
    # Only run if: (1) using CPU mode, OR (2) model has CPU/meta layers in device_map
    # Skip on GPU-only runs (like Modal) where all layers are on GPU
    needs_rotary_init = False
    if not use_gpu:
        needs_rotary_init = True
    elif hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Check if any layers are on CPU or meta device
        has_cpu_layers = any(
            d in ["cpu", "meta"] or 
            (isinstance(d, str) and "cpu" in d.lower()) or
            (isinstance(d, torch.device) and d.type == "cpu")
            for d in model.hf_device_map.values()
        )
        if has_cpu_layers:
            needs_rotary_init = True
    
    if needs_rotary_init:
        print("Initializing rotary embeddings for CPU-loaded model...")
        try:
            # In transformers 4.55.0+, rotary_emb is stored at model.model.rotary_emb (shared across all layers)
            # Not in each layer's self_attn like in older versions
            if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                state_dict = model.state_dict()
                cpu_device = torch.device("cpu")
                rotary_emb = model.model.rotary_emb
                
                # Materialize rotary_emb.inv_freq from state_dict if it's on meta device
                rotary_key = "model.rotary_emb.inv_freq"
                if rotary_key in state_dict:
                    inv_freq = state_dict[rotary_key]
                    if hasattr(inv_freq, 'device'):
                        if inv_freq.device.type == 'meta' or (hasattr(rotary_emb, 'inv_freq') and 
                            hasattr(rotary_emb.inv_freq, 'device') and rotary_emb.inv_freq.device.type == 'meta'):
                            rotary_emb.inv_freq = inv_freq.to(cpu_device)
                        elif hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq.device != inv_freq.device:
                            rotary_emb.inv_freq = inv_freq.to(rotary_emb.inv_freq.device)
                
                # Initialize rotary_emb by calling it with dummy inputs
                # This ensures the cache is set up properly
                try:
                    with torch.no_grad():
                        seq_len = 128
                        dummy_pos = torch.arange(0, seq_len, dtype=torch.long, device=cpu_device).unsqueeze(0)
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                        dummy_x = torch.zeros((1, seq_len, head_dim), dtype=torch.float16, device=cpu_device)
                        
                        # Try multiple calling methods for compatibility
                        try:
                            result = rotary_emb(dummy_x, dummy_pos)
                        except TypeError:
                            try:
                                result = rotary_emb(dummy_x, position_ids=dummy_pos)
                            except TypeError:
                                result = rotary_emb(x=dummy_x, position_ids=dummy_pos)
                        
                        # Verify it returns a tuple
                        if result is not None and isinstance(result, (tuple, list)) and len(result) == 2:
                            print(f"Successfully initialized rotary embeddings (shared across all layers).")
                        else:
                            print(f"Warning: rotary_emb returned unexpected result: {type(result)}")
                except Exception as e_init:
                    print(f"Warning: Could not initialize rotary_emb by calling it: {e_init}")
                    print("Will attempt to initialize during pruning...")
            else:
                print("Warning: model.model.rotary_emb not found - rotary embeddings may not be initialized.")
                print("Will attempt to initialize during pruning...")
        except Exception as e:
            print(f"Warning: Could not pre-initialize rotary embeddings: {e}")
            print("Will attempt to initialize during pruning...")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--skip_eval", action="store_true", help="Skip perplexity evaluation (useful for CPU mode or when memory is constrained)")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    use_gpu = not args.use_cpu and torch.cuda.is_available()
    model = get_llm(args.model, args.cache_dir, use_gpu=use_gpu)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Determine device - use GPU if available and not forced to CPU
    # Note: If model is offloaded, device will be CPU for calibration inputs
    if use_gpu and torch.cuda.is_available():
        # For models with device_map, check if any layers are actually on GPU
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # Check if model has any GPU layers
            # device_map can return: torch.device, str ("cuda:0", "cpu"), or int (GPU index)
            has_gpu_layers = any(
                (isinstance(d, torch.device) and d.type == "cuda") or
                (isinstance(d, str) and "cuda" in d.lower()) or
                (isinstance(d, int))  # Integer means GPU index
                for d in model.hf_device_map.values()
            )
            if has_gpu_layers:
                # Get device from first GPU layer
                for key, dev in model.hf_device_map.items():
                    if isinstance(dev, torch.device) and dev.type == "cuda":
                        device = dev
                        break
                    elif isinstance(dev, str) and "cuda" in dev.lower():
                        device = torch.device(dev)
                        break
                    elif isinstance(dev, int):
                        # Integer means GPU index
                        device = torch.device(f"cuda:{dev}")
                        break
                else:
                    device = torch.device("cuda:0")
            else:
                # All layers offloaded to CPU
                device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    
    # Save model BEFORE evaluation to ensure it's saved even if evaluation fails/kills process
    # This is especially important for CPU mode where evaluation can run out of RAM
    if args.save_model:
        print(f"\n{'='*60}")
        print("Saving pruned model before evaluation...")
        print(f"{'='*60}\n")
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"✅ Model saved to {args.save_model}")
        print("   (Saved before evaluation to ensure it's preserved even if evaluation fails)\n")
    
    # Skip evaluation if requested or if using CPU mode (evaluation is very memory-intensive on CPU)
    ppl_test = None
    if args.skip_eval:
        print("Skipping evaluation (--skip_eval flag set)")
        if not args.save_model:
            print("⚠️  WARNING: Model not saved (use --save_model to save the pruned model)")
        ppl_test = None
    elif args.use_cpu:
        print(f"\n{'='*60}")
        print("⚠️  WARNING: Evaluation on CPU is very memory-intensive and may cause OOM.")
        print("   Pruning completed successfully.")
        if args.save_model:
            print("   Model has been saved.")
        else:
            print("   ⚠️  Model NOT saved (use --save_model to save it).")
        print("   To evaluate the model:")
        print("   1. Load the saved model separately with more RAM available")
        print("   2. Or use GPU for evaluation: python eval_pruned_model.py --model_path <path>")
        print("   3. Or add --skip_eval flag to skip evaluation entirely")
        print(f"{'='*60}\n")
        ppl_test = None
    else:
        # Clear GPU cache aggressively before evaluation to free up memory
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection again after sync
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory before eval: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
            free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            print(f"Free GPU memory: {free_memory:.2f} GB")
        
        # Try evaluation, but skip if OOM occurs
        try:
            ppl_test = eval_ppl(args, model, tokenizer, device)
            print(f"wikitext perplexity {ppl_test}")
        except Exception as e:
            if "OutOfMemoryError" in str(type(e).__name__) or "out of memory" in str(e).lower():
                print(f"\n{'='*60}")
                print("Evaluation skipped due to GPU memory constraints.")
                print("Model has been pruned successfully and has been saved.")
                print("You can evaluate the model separately with:")
                print("  - More GPU memory available")
                print("  - Using CPU-only evaluation (--use_cpu flag)")
                print("  - Or loading the saved model in a separate session")
                print(f"{'='*60}\n")
                ppl_test = None
            else:
                raise

    if args.save:
        # Normalize the path to handle trailing slashes
        save_path = os.path.normpath(args.save)
        # Handle case where path exists as a file instead of directory
        if os.path.exists(save_path):
            if os.path.isfile(save_path):
                # Remove the file if it exists as a file
                os.remove(save_path)
            elif not os.path.isdir(save_path):
                raise ValueError(f"Save path '{save_path}' exists but is not a directory or file")
        # Create directory if it doesn't exist (exist_ok=True handles existing directories)
        os.makedirs(save_path, exist_ok=True)
        # Update args.save to use normalized path for consistency
        args.save = save_path
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            if ppl_test is not None:
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
            else:
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\tN/A (evaluation skipped)", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    # Model already saved before evaluation (see above)
    # Only save again if save_model path changed or if we want to update it
    if args.save_model:
        # Check if model was already saved (to avoid duplicate saves)
        # The model was saved before evaluation, so we're done
        pass

if __name__ == '__main__':
    main()
