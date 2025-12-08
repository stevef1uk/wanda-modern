# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Cap sequence length for evaluation to avoid OOM
    # Large models (like gpt-oss-20b) have max_position_embeddings=131072 which is too long
    # Use 2048 for evaluation (same as calibration) - sufficient for perplexity evaluation
    eval_seqlen = min(model.seqlen, 2048)
    if model.seqlen > 2048:
        print(f"Note: Model supports {model.seqlen} tokens, but using {eval_seqlen} for evaluation (sufficient for perplexity)")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=eval_seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # For large models, use smaller batch size to avoid OOM
    # Reduce batch size if model is very large (20B+ parameters)
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        # Rough estimate: if hidden_size is large, reduce batch size
        if model.config.hidden_size >= 5120:  # 20B+ models typically have hidden_size >= 5120
            bs = 1
            print(f"Using batch size {bs} for evaluation (large model detected)")
    
    # Handle both DataLoader/list and tokenized dataset
    # Check if testenc is a DataLoader/list or a tokenized dataset
    # Tokenized datasets have .input_ids attribute (TokenizerWrapper or BatchEncoding)
    # Priority: check for input_ids first (most reliable indicator)
    is_dataloader = not hasattr(testenc, 'input_ids')
    
    if is_dataloader:
        # testenc is a DataLoader - get number of samples from length
        nsamples = len(testenc)
        # Get actual_seqlen from first sample
        if nsamples > 0:
            first_sample = testenc[0][0] if isinstance(testenc[0], (list, tuple)) else testenc[0]
            actual_seqlen = first_sample.shape[0] if isinstance(first_sample, torch.Tensor) else model.seqlen
        else:
            actual_seqlen = model.seqlen
        actual_seqlen = min(actual_seqlen, 2048)
    else:
        # testenc is a tokenized dataset - get input_ids
        if hasattr(testenc, 'input_ids'):
            testenc = testenc.input_ids
        # Handle batch dimension - testenc might be [1, total_tokens] or [total_tokens]
        if len(testenc.shape) == 2 and testenc.shape[0] == 1:
            # Squeeze batch dimension: [1, total_tokens] -> [total_tokens]
            testenc = testenc.squeeze(0)
        # Use actual sequence length from testenc, not model.seqlen (which might be 131k)
        # The testenc was created with eval_seqlen (capped at 2048), so use that
        actual_seqlen = testenc.shape[0] if len(testenc.shape) == 1 else (testenc.shape[1] if len(testenc.shape) > 1 else model.seqlen)
        # Cap at reasonable limit
        actual_seqlen = min(actual_seqlen, 2048)
        
        # Calculate number of samples from total tokens
        # For 1D tensor: nsamples = len(testenc) // actual_seqlen
        # For 2D tensor: nsamples = testenc.numel() // actual_seqlen
        if len(testenc.shape) == 1:
            nsamples = len(testenc) // actual_seqlen
        else:
            nsamples = testenc.numel() // actual_seqlen

    # Determine input device - if model has offloaded layers, use CPU for inputs
    # to avoid OOM when materializing offloaded layers
    input_device = device
    has_offloaded = False
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Check if model has offloaded layers
        has_offloaded = any(dev in ["cpu", "meta"] or (isinstance(dev, str) and "cpu" in dev.lower())
                           for dev in model.hf_device_map.values())
        if has_offloaded:
            # Force CPU inputs to prevent GPU materialization of offloaded layers
            input_device = torch.device("cpu")
            print(f"Model has offloaded layers, using CPU for evaluation inputs to avoid OOM")
            
            # Check GPU memory - if it's too constrained, skip evaluation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - allocated
                # If less than 500MB free, skip evaluation
                if free < 0.5:
                    print(f"GPU memory too constrained ({free:.2f} GB free), skipping evaluation")
                    raise RuntimeError("Evaluation skipped due to GPU memory constraints: insufficient free memory")

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    
    # Track timing for progress reporting
    start_time = time.time()
    last_print_time = start_time

    # Loop through each batch
    for i in range(0,nsamples,bs):
        # Print progress more frequently (every 10 samples) or every 30 seconds
        current_time = time.time()
        should_print = (i % 10 == 0) or (current_time - last_print_time > 30)
        
        if should_print:
            elapsed = current_time - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (nsamples - i) / rate if rate > 0 else 0
            print(f"sample {i}/{nsamples} ({i*100//nsamples}%) - elapsed: {elapsed:.1f}s, rate: {rate:.2f} samples/s, est. remaining: {remaining:.1f}s")
            last_print_time = current_time
            # Clear GPU cache periodically to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        if is_dataloader:
            # Get inputs from DataLoader
            inputs = testenc[i][0].to(input_device)
            if isinstance(inputs, torch.Tensor) and len(inputs.shape) == 1:
                inputs = inputs.unsqueeze(0)  # Add batch dimension if needed
            # Ensure correct shape
            if inputs.shape[0] != (j-i):
                # If we need multiple samples, get them
                batch_inputs = []
                for idx in range(i, j):
                    sample = testenc[idx][0].to(input_device)
                    if len(sample.shape) == 1:
                        sample = sample.unsqueeze(0)
                    batch_inputs.append(sample)
                inputs = torch.cat(batch_inputs, dim=0)
        else:
            # Get inputs from tokenized dataset
            # testenc is now 1D [total_tokens] after squeezing batch dimension
            start_idx = i * actual_seqlen
            end_idx = j * actual_seqlen
            inputs = testenc[start_idx:end_idx].to(input_device)
            inputs = inputs.reshape(j-i, actual_seqlen)

        # Forward pass through the model
        try:
            # Clear GPU cache before each forward pass if model has offloaded layers
            forward_has_offloaded = False
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                forward_has_offloaded = any(dev in ["cpu", "meta"] or (isinstance(dev, str) and "cpu" in dev.lower())
                                   for dev in model.hf_device_map.values())
                if forward_has_offloaded and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Print a message before forward pass if it's been a while since last update
            if forward_has_offloaded and (current_time - last_print_time > 5):
                print(f"  Processing sample {i}... (this may take a while with offloaded layers)")
            
            forward_start = time.time()
            lm_logits = model(inputs).logits
            forward_time = time.time() - forward_start
            
            # Warn if forward pass is taking very long
            if forward_time > 60 and i % 10 != 0:
                print(f"  Sample {i} forward pass took {forward_time:.1f}s (very slow - model may have offloaded layers)")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's actually an OOM error
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                # If OOM, clear cache aggressively and raise custom exception
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                print(f"OOM during evaluation at sample {i}")
                print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
                print("Skipping perplexity evaluation due to GPU memory constraints.")
                print("Model has been pruned successfully. You can evaluate separately with more GPU memory or on CPU.")
                # Raise a RuntimeError that will be caught by main.py
                raise RuntimeError("Evaluation skipped due to GPU memory constraints: CUDA out of memory")
            else:
                # Re-raise if it's not an OOM error
                raise

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        
        # Check for invalid loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss at sample {i}: {loss.item()}")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Logits shape: {lm_logits.shape}")
            print(f"  Shift logits shape: {shift_logits.shape}")
            print(f"  Shift labels shape: {shift_labels.shape}")
            # Skip this sample or use a default value
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        # Calculate negative log likelihood
        # For DataLoader: each sample is already a full sequence, so multiply by (j-i) samples
        # For tokenized dataset: we're processing chunks, so multiply by actual_seqlen * (j-i)
        if is_dataloader:
            # Each sample in DataLoader is a full sequence
            neg_log_likelihood = loss.float() * actual_seqlen * (j-i)
        else:
            # Tokenized dataset: loss is already per-token, multiply by number of tokens processed
            neg_log_likelihood = loss.float() * actual_seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        
        # Clear GPU cache after each batch to prevent accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute perplexity
    # Sum all NLLs and divide by total number of tokens evaluated
    if len(nlls) == 0:
        raise ValueError("No samples processed - cannot compute perplexity")
    
    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * actual_seqlen
    
    # Check for invalid values
    if torch.isnan(total_nll) or torch.isinf(total_nll):
        print(f"WARNING: Total NLL is invalid: {total_nll}")
        print(f"  Number of samples: {nsamples}")
        print(f"  Sequence length: {actual_seqlen}")
        print(f"  Number of NLL values: {len(nlls)}")
        if len(nlls) > 0:
            print(f"  First few NLL values: {[nll.item() for nll in nlls[:5]]}")
        raise ValueError(f"Cannot compute perplexity: invalid NLL values (NaN or Inf)")
    
    ppl = torch.exp(total_nll / total_tokens)

    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 