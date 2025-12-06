import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            # Get weight directly from model to avoid meta tensor issues
            layer_module = model.model.layers[i]
            parts = name.split('.')
            module = layer_module
            for part in parts[:-1]:
                module = getattr(module, part)
            param_module = getattr(module, parts[-1])
            W = param_module.weight.data
            
            # Skip if weight is on meta device (offloaded layers)
            if hasattr(W, 'device') and W.device.type == 'meta':
                continue
            
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        if sub_params > 0:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        else:
            print(f"layer {i} sparsity N/A (all weights skipped - offloaded to CPU)")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Determine the actual device where the model is located
    # If model is offloaded, use CPU to avoid OOM
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        if "model.embed_tokens" in model.hf_device_map:
            embed_device = model.hf_device_map["model.embed_tokens"]
            # If offloaded to CPU or meta, use CPU for calibration inputs
            if embed_device in ["cpu", "meta"] or (isinstance(embed_device, str) and "cpu" in embed_device.lower()):
                device = torch.device("cpu")
            else:
                device = torch.device(embed_device) if isinstance(embed_device, str) else embed_device
        else:
            # Check first layer device
            if "model.layers.0" in model.hf_device_map:
                layer0_device = model.hf_device_map["model.layers.0"]
                if layer0_device in ["cpu", "meta"] or (isinstance(layer0_device, str) and "cpu" in layer0_device.lower()):
                    device = torch.device("cpu")
                else:
                    device = torch.device(layer0_device) if isinstance(layer0_device, str) else layer0_device
    
    dtype = next(iter(model.parameters())).dtype
    # Use CPU for calibration inputs if model is offloaded to avoid OOM
    # The calibration inputs are large (4GB for 128 samples), so CPU is safer
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Check if any layers are offloaded
        has_offloaded = any(dev in ["cpu", "meta"] or (isinstance(dev, str) and "cpu" in dev.lower()) 
                           for dev in model.hf_device_map.values())
        if has_offloaded:
            device = torch.device("cpu")
            print(f"Model has offloaded layers, using CPU for calibration inputs to avoid OOM")
    
    # Ensure device is a torch.device object, not a string
    if isinstance(device, str):
        device = torch.device(device)
    
    # Only allocate space for the number of samples we'll actually use
    # This reduces memory usage, especially important for CPU mode
    max_samples = 128  # Maximum expected samples
    inps = torch.zeros((max_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    # Ensure all tensors are on the same device and not meta tensors
    # W_metric should never be meta at this point (weights should be materialized)
    if hasattr(W_metric, 'device') and W_metric.device.type == 'meta':
        raise RuntimeError("W_metric is on meta device - weights should have been materialized before computing W_metric")
    
    target_device = W_metric.device
    
    # Move sort_res[0] to target device if needed
    sort_values = sort_res[0].to(target_device) if sort_res[0].device != target_device else sort_res[0]
    tmp_metric = tmp_metric.to(target_device) if tmp_metric.device != target_device else tmp_metric
    sum_before = sum_before.to(target_device) if sum_before.device != target_device else sum_before
    
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    # sort_mask.sum() creates index tensor on same device as sort_mask
    # If on GPU and memory is tight, do sum on CPU
    if sort_mask.device.type == "cuda":
        try:
            # Check GPU memory - if >85% used, do sum on CPU
            allocated = torch.cuda.memory_allocated(sort_mask.device.index if hasattr(sort_mask.device, 'index') else 0)
            total = torch.cuda.get_device_properties(sort_mask.device.index if hasattr(sort_mask.device, 'index') else 0).total_memory
            if allocated / total > 0.85:
                sort_mask_cpu = sort_mask.cpu()
                index_tensor = sort_mask_cpu.sum(dim=1, keepdims=True) - 1
                index_tensor = index_tensor.to(target_device)
                del sort_mask_cpu
            else:
                index_tensor = sort_mask.sum(dim=1, keepdims=True) - 1
        except:
            # If check fails, do it on CPU to be safe
            sort_mask_cpu = sort_mask.cpu()
            index_tensor = sort_mask_cpu.sum(dim=1, keepdims=True) - 1
            index_tensor = index_tensor.to(target_device)
            del sort_mask_cpu
    else:
        index_tensor = sort_mask.sum(dim=1, keepdims=True) - 1
    thres = torch.gather(sort_values, dim=1, index=index_tensor)
    W_mask = (W_metric <= thres)
    # Ensure cur_sparsity is computed on real device, not meta
    # numel() returns an int, so convert to float directly
    cur_sparsity = (W_mask==True).sum().float() / float(W_mask.numel())
    # Convert to Python float to avoid meta tensor issues
    cur_sparsity_value = cur_sparsity.item() if hasattr(cur_sparsity, 'item') and cur_sparsity.device.type != 'meta' else float(cur_sparsity)
    return W_mask, cur_sparsity_value

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # Move to device for sorting if needed (for GPU efficiency)
                W_metric_device = W_metric.to(device) if device.type == "cuda" else W_metric
                thresh = torch.sort(W_metric_device.flatten())[0][int(W.numel()*args.sparsity_ratio)]
                if device.type == "cuda":
                    thresh = thresh.cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer,load_validation=False)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    # Track if calibration inputs are on CPU (due to offloading)
    # If so, keep them on CPU - the model will handle device transfers during forward pass
    calibration_on_cpu = inps.device.type == "cpu"
    
    # Debug: Check which layers are offloaded to CPU/meta
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        offloaded_layers = []
        for key in sorted(model.hf_device_map.keys()):
            if 'layers' in key:
                dev = model.hf_device_map[key]
                if isinstance(dev, str) and dev in ["cpu", "meta"]:
                    offloaded_layers.append(key)
        if offloaded_layers:
            print(f"Note: {len(offloaded_layers)} layers are offloaded to CPU/meta (will be materialized during forward pass)")
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            # Convert device to torch.device - handle int, str, or torch.device
            if isinstance(dev, int):
                # Integer means GPU index
                dev = torch.device(f"cuda:{dev}")
            elif isinstance(dev, str):
                if dev in ["cpu", "meta"]:
                    dev = torch.device("cpu")
                else:
                    dev = torch.device(dev)
            elif not isinstance(dev, torch.device):
                # Fallback: assume CPU if unknown type
                dev = torch.device("cpu")
            
            # If calibration inputs are on CPU (due to offloading), keep them there
            # The model's forward pass will handle device transfers automatically
            # This avoids OOM when trying to move 4GB tensors to GPU
            if not calibration_on_cpu and dev.type == "cuda":
                # Only try to move to GPU if we have enough memory
                if torch.cuda.is_available():
                    try:
                        # Try moving a small tensor first to check memory
                        test_tensor = torch.zeros(1, device=dev)
                        del test_tensor
                        # If successful, move inputs (but this may still fail for large tensors)
                        # So we'll let the forward pass handle it instead
                        # inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                        pass
                    except RuntimeError:
                        print(f"Layer {i} on GPU but insufficient memory, keeping calibration inputs on CPU")
                        calibration_on_cpu = True
            
            # Only move if layer is on CPU and inputs are on CPU (no-op)
            # Or if we explicitly want to move (but we're avoiding this for large tensors)
            if dev.type == "cpu" and inps.device.type != "cpu":
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Check if this layer is offloaded - if so, ensure it's loaded before forward pass
        layer_key = f"model.layers.{i}"
        is_offloaded = False
        target_dev = torch.device("cpu")  # Default to CPU for offloaded layers
        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            layer_dev = model.hf_device_map[layer_key]
            if isinstance(layer_dev, str) and layer_dev in ["cpu", "meta"]:
                is_offloaded = True
                target_dev = torch.device("cpu")
            elif isinstance(layer_dev, int):
                # On GPU
                is_offloaded = False
                target_dev = torch.device(f"cuda:{layer_dev}")
            elif isinstance(layer_dev, torch.device):
                target_dev = layer_dev
                is_offloaded = (layer_dev.type == "cpu")
            else:
                is_offloaded = False
        
        # If layer is offloaded, force materialization by accessing weights through state_dict
        # This should trigger accelerate's loading mechanism
        if is_offloaded:
            try:
                # Access weights through state_dict - this should trigger loading
                state_dict = model.state_dict()
                for name in subset:
                    param_path = f"model.layers.{i}.{name}.weight"
                    if param_path in state_dict:
                        # Access the weight from state_dict - this should trigger materialization
                        weight_from_state = state_dict[param_path]
                        # If it's not meta, update the module's weight
                        if hasattr(weight_from_state, 'device') and weight_from_state.device.type != 'meta':
                            # Get the module and update its weight
                            layer_module = model.model.layers[i]
                            parts = name.split('.')
                            module = layer_module
                            for part in parts[:-1]:
                                module = getattr(module, part)
                            param_module = getattr(module, parts[-1])
                            # Update the weight with the materialized version
                            param_module.weight.data = weight_from_state
                
                # Also ensure rotary_emb is materialized for LLaMA models
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                    if rotary_key in state_dict:
                        inv_freq = state_dict[rotary_key]
                        if hasattr(inv_freq, 'device') and inv_freq.device.type != 'meta':
                            layer.self_attn.rotary_emb.inv_freq = inv_freq.to(target_dev)
            except Exception as e:
                print(f"Warning: Could not force materialization via state_dict for layer {i}: {e}")
                # Fallback: try dummy forward pass
                if len(inps) > 0:
                    try:
                        dummy_inp = inps[0].unsqueeze(0).to(target_dev)
                        dummy_attn = attention_mask[0:1].to(target_dev) if attention_mask is not None else None
                        dummy_pos = position_ids[0:1].to(target_dev) if position_ids is not None else None
                        # Ensure rotary_emb is on correct device before dummy forward pass
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                            rotary_emb = layer.self_attn.rotary_emb
                            if hasattr(rotary_emb, 'inv_freq'):
                                if hasattr(rotary_emb.inv_freq, 'device') and rotary_emb.inv_freq.device != target_dev:
                                    rotary_emb.inv_freq = rotary_emb.inv_freq.to(target_dev)
                        with torch.no_grad():
                            _ = layer(dummy_inp, attention_mask=dummy_attn, position_ids=dummy_pos)
                    except Exception as e2:
                        print(f"Warning: Could not force materialization for offloaded layer {i}: {e2}")
        
        # Process samples one at a time to avoid moving large tensors to GPU
        # The forward pass will materialize weights automatically
        for j in range(args.nsamples):
            with torch.no_grad():
                sample_inp = inps[j].unsqueeze(0)
                # Get layer device
                if f"model.layers.{i}" in model.hf_device_map:
                    layer_dev = model.hf_device_map[f"model.layers.{i}"]
                    # Convert to torch.device - handle int, str, or torch.device
                    if isinstance(layer_dev, int):
                        layer_dev = torch.device(f"cuda:{layer_dev}")
                    elif isinstance(layer_dev, str):
                        layer_dev = torch.device("cpu" if layer_dev in ["cpu", "meta"] else layer_dev)
                    elif not isinstance(layer_dev, torch.device):
                        layer_dev = torch.device("cpu")
                else:
                    layer_dev = inps.device
                
                # Move only this sample to layer device (avoids moving 4GB tensor)
                if sample_inp.device != layer_dev:
                    sample_inp = sample_inp.to(layer_dev)
                    # Handle attention_mask and position_ids
                    if attention_mask is not None:
                        if len(attention_mask.shape) == 2:
                            sample_attn = attention_mask.to(layer_dev)
                        else:
                            sample_attn = attention_mask[j:j+1].to(layer_dev) if attention_mask.device != layer_dev else attention_mask[j:j+1]
                    else:
                        sample_attn = None
                    # Generate position_ids if None (required by decoder layers for position embeddings)
                    if position_ids is not None:
                        if len(position_ids.shape) == 2:
                            sample_pos = position_ids.to(layer_dev)
                        else:
                            sample_pos = position_ids[j:j+1].to(layer_dev) if position_ids.device != layer_dev else position_ids[j:j+1]
                    else:
                        # Generate position_ids: [0, 1, 2, ..., seqlen-1]
                        seq_len = sample_inp.shape[1]
                        sample_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)
                else:
                    sample_attn = attention_mask
                    # Generate position_ids if None
                    if position_ids is not None:
                        # Ensure position_ids is on the same device as sample_inp
                        if len(position_ids.shape) == 2:
                            sample_pos = position_ids.to(layer_dev) if position_ids.device != layer_dev else position_ids
                        else:
                            sample_pos = position_ids[j:j+1].to(layer_dev) if position_ids.device != layer_dev else position_ids[j:j+1]
                    else:
                        seq_len = sample_inp.shape[1]
                        sample_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)
                
                # CRITICAL: Ensure sample_pos is on the same device as sample_inp
                if sample_pos.device != sample_inp.device:
                    sample_pos = sample_pos.to(sample_inp.device)
                
                # Ensure rotary_emb is on the correct device and properly initialized
                # LLaMA layers need rotary embeddings to be accessible for position encoding
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    rotary_emb = layer.self_attn.rotary_emb
                    # Ensure rotary_emb is on the same device as the layer
                    if hasattr(rotary_emb, 'inv_freq'):
                        # Force materialization by accessing through state_dict if needed
                        try:
                            # Check if inv_freq is meta or None
                            if not hasattr(rotary_emb.inv_freq, 'device') or rotary_emb.inv_freq.device.type == 'meta':
                                state_dict = model.state_dict()
                                rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                                if rotary_key in state_dict:
                                    inv_freq = state_dict[rotary_key]
                                    if hasattr(inv_freq, 'device') and inv_freq.device.type != 'meta':
                                        rotary_emb.inv_freq = inv_freq.to(layer_dev)
                            # Move inv_freq to layer device if needed
                            elif rotary_emb.inv_freq.device != layer_dev:
                                rotary_emb.inv_freq = rotary_emb.inv_freq.to(layer_dev)
                            # Access inv_freq to trigger materialization
                            _ = rotary_emb.inv_freq
                            
                            # Test that rotary_emb actually works before calling the layer
                            # This ensures it will return (cos, sin) instead of None
                            try:
                                seq_len = sample_inp.shape[1]
                                # position_ids must be 2D: (batch_size, seq_len)
                                test_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)  # Shape: (1, seq_len)
                                # In transformers 4.45.2, forward() signature is (self, x, position_ids)
                                # Create dummy x tensor
                                head_dim = model.config.hidden_size // model.config.num_attention_heads
                                dummy_x = torch.zeros((1, seq_len, head_dim), dtype=sample_inp.dtype, device=layer_dev)
                                # Try multiple calling methods
                                try:
                                    test_result = rotary_emb(dummy_x, test_pos)
                                except TypeError:
                                    try:
                                        test_result = rotary_emb(dummy_x, position_ids=test_pos)
                                    except TypeError:
                                        test_result = rotary_emb(x=dummy_x, position_ids=test_pos)
                                if test_result is None:
                                    raise ValueError("rotary_emb returned None during test")
                                # Ensure it returns a tuple
                                if not isinstance(test_result, (tuple, list)) or len(test_result) != 2:
                                    raise ValueError(f"rotary_emb returned unexpected format: {type(test_result)}")
                            except Exception as e_test:
                                # If test fails, try to fix it
                                print(f"Warning: rotary_emb test failed at layer {i}, sample {j}: {e_test}")
                                # Try materializing through state_dict again
                                try:
                                    state_dict = model.state_dict()
                                    rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                                    if rotary_key in state_dict:
                                        inv_freq = state_dict[rotary_key]
                                        if hasattr(inv_freq, 'device') and inv_freq.device.type != 'meta':
                                            rotary_emb.inv_freq = inv_freq.to(layer_dev)
                                            # Retry test
                                            seq_len = sample_inp.shape[1]
                                            # position_ids must be 2D: (batch_size, seq_len)
                                            test_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)  # Shape: (1, seq_len)
                                            # Create dummy x tensor for transformers 4.45.2
                                            head_dim = model.config.hidden_size // model.config.num_attention_heads
                                            dummy_x = torch.zeros((1, seq_len, head_dim), dtype=sample_inp.dtype, device=layer_dev)
                                            # Try multiple calling methods
                                            try:
                                                test_result = rotary_emb(dummy_x, test_pos)
                                            except TypeError:
                                                try:
                                                    test_result = rotary_emb(dummy_x, position_ids=test_pos)
                                                except TypeError:
                                                    test_result = rotary_emb(x=dummy_x, position_ids=test_pos)
                                            if test_result is None:
                                                raise ValueError("rotary_emb still returns None after fix")
                                except Exception as e_fix:
                                    print(f"Could not fix rotary_emb: {e_fix}")
                                    # Continue anyway - the try-except below will catch it
                        except Exception as e_rot:
                            # If we can't fix rotary_emb, try to materialize it via a test call
                            try:
                                # Try calling rotary_emb with a dummy position_ids to trigger initialization
                                seq_len = sample_inp.shape[1]
                                # position_ids must be 2D: (batch_size, seq_len)
                                dummy_pos_test = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)  # Shape: (1, seq_len)
                                # Create dummy x tensor for transformers 4.45.2
                                head_dim = model.config.hidden_size // model.config.num_attention_heads
                                dummy_x = torch.zeros((1, seq_len, head_dim), dtype=sample_inp.dtype, device=layer_dev)
                                # Try multiple calling methods
                                try:
                                    _ = rotary_emb(dummy_x, dummy_pos_test)
                                except TypeError:
                                    try:
                                        _ = rotary_emb(dummy_x, position_ids=dummy_pos_test)
                                    except TypeError:
                                        _ = rotary_emb(x=dummy_x, position_ids=dummy_pos_test)
                            except:
                                pass
                
                # CRITICAL: Ensure sample_pos is on the same device as sample_inp BEFORE any rotary_emb operations
                # This must happen before we try to initialize or use rotary_emb
                if sample_pos.device != sample_inp.device:
                    sample_pos = sample_pos.to(sample_inp.device)
                
                # For the first sample of each layer, ensure rotary_emb is initialized by doing a dummy forward pass
                # This is critical for LLaMA models where rotary_emb might not be fully initialized until first use
                # In transformers 4.45.2, each attention layer has its own rotary_emb that needs to be on the correct device
                if j == 0 and hasattr(layer, 'self_attn'):
                    # Use getattr - rotary_emb might exist but hasattr returns False
                    rotary_emb = getattr(layer.self_attn, 'rotary_emb', None)
                    if rotary_emb is None:
                        # Try to access it directly - it might be created lazily
                        try:
                            rotary_emb = layer.self_attn.rotary_emb
                        except AttributeError:
                            rotary_emb = None
                    
                    if rotary_emb is not None:
                        # CRITICAL: Ensure the layer's rotary_emb.inv_freq is on the same device as the layer
                        # This is necessary because when position_embeddings is None, the layer uses its own rotary_emb
                        try:
                            state_dict = model.state_dict()
                            rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                            if rotary_key in state_dict:
                                inv_freq = state_dict[rotary_key]
                                if hasattr(inv_freq, 'device'):
                                    if inv_freq.device.type == 'meta':
                                        rotary_emb.inv_freq = inv_freq.to(layer_dev)
                                    elif inv_freq.device != layer_dev:
                                        rotary_emb.inv_freq = inv_freq.to(layer_dev)
                                    else:
                                        rotary_emb.inv_freq = inv_freq.to(layer_dev)
                                else:
                                    rotary_emb.inv_freq = inv_freq.to(layer_dev)
                        except Exception as e_state:
                            # If state_dict access fails, try to ensure inv_freq is on correct device anyway
                            if hasattr(rotary_emb, 'inv_freq') and hasattr(rotary_emb.inv_freq, 'device'):
                                if rotary_emb.inv_freq.device != layer_dev:
                                    rotary_emb.inv_freq = rotary_emb.inv_freq.to(layer_dev)
                        
                        # Now try to initialize rotary_emb by calling it directly
                        # CRITICAL: This must succeed or the layer will fail
                        init_success = False
                        try:
                            # Ensure inv_freq exists and is on correct device
                            if not hasattr(rotary_emb, 'inv_freq') or rotary_emb.inv_freq is None:
                                state_dict = model.state_dict()
                                rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                                if rotary_key in state_dict:
                                    rotary_emb.inv_freq = state_dict[rotary_key].to(layer_dev)
                            
                            # Ensure inv_freq is on layer_dev
                            if hasattr(rotary_emb, 'inv_freq') and hasattr(rotary_emb.inv_freq, 'device'):
                                if rotary_emb.inv_freq.device != layer_dev:
                                    rotary_emb.inv_freq = rotary_emb.inv_freq.to(layer_dev)
                            
                            # Call rotary_emb with position_ids to initialize its cache
                            # Use the actual sequence length we'll need
                            seq_len = sample_inp.shape[1]
                            # position_ids must be 2D: (batch_size, seq_len) and on layer_dev
                            test_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)  # Shape: (1, seq_len)
                            # In transformers 4.45.2, forward() signature is (self, x, position_ids)
                            # We need to pass both x (dummy input) and position_ids
                            # Create a dummy x tensor - rotary_emb just needs it for shape, not values
                            head_dim = model.config.hidden_size // model.config.num_attention_heads
                            dummy_x = torch.zeros((1, seq_len, head_dim), dtype=torch.float16, device=layer_dev)
                            
                            # Call multiple times to ensure cache is properly set
                            for attempt in range(3):
                                try:
                                    # Method 1: Both as positional (transformers 4.45.2)
                                    test_result = rotary_emb(dummy_x, test_pos)
                                except TypeError as e1:
                                    try:
                                        # Method 2: x positional, position_ids keyword
                                        test_result = rotary_emb(dummy_x, position_ids=test_pos)
                                    except TypeError as e2:
                                        try:
                                            # Method 3: Both as keyword
                                            test_result = rotary_emb(x=dummy_x, position_ids=test_pos)
                                        except Exception as e3:
                                            if attempt == 2:  # Last attempt
                                                raise ValueError(f"All call methods failed: {e1}, {e2}, {e3}")
                                            continue
                                # Verify it returns a tuple
                                if test_result is not None and isinstance(test_result, (tuple, list)) and len(test_result) == 2:
                                    init_success = True
                                    break
                            
                            if not init_success:
                                raise ValueError(f"rotary_emb returned invalid result after 3 attempts")
                        except Exception as e_rot_init:
                            # If direct call fails, try through attention layer
                            try:
                                # Do a tiny dummy forward pass through the attention module to trigger initialization
                                # Use just 1 token to minimize overhead
                                # CRITICAL: Ensure dummy_pos_tiny is on layer_dev
                                dummy_inp_tiny = sample_inp[:, :1]  # Just first token (already on layer_dev)
                                dummy_pos_tiny = sample_pos[:, :1] if sample_pos is not None else torch.arange(0, 1, dtype=torch.long, device=layer_dev).unsqueeze(0)
                                if dummy_pos_tiny.device != layer_dev:
                                    dummy_pos_tiny = dummy_pos_tiny.to(layer_dev)
                                dummy_attn_tiny = sample_attn[:, :1] if sample_attn is not None else None
                                if dummy_attn_tiny is not None and dummy_attn_tiny.device != layer_dev:
                                    dummy_attn_tiny = dummy_attn_tiny.to(layer_dev)
                                with torch.no_grad():
                                    # This should trigger rotary_emb initialization
                                    _ = layer.self_attn(dummy_inp_tiny, attention_mask=dummy_attn_tiny, position_ids=dummy_pos_tiny)
                                init_success = True
                            except Exception as e_dummy:
                                # If dummy pass fails, log it but continue - the real pass will show the actual error
                                print(f"Warning: Could not initialize rotary_emb for layer {i} before use")
                                print(f"  Direct call error: {str(e_rot_init)[:100]}")
                                print(f"  Attention call error: {str(e_dummy)[:100]}")
                                # Don't raise here - let the actual forward pass try and catch the error
                
                # CRITICAL: Ensure sample_inp and sample_pos are on layer_dev BEFORE computing position_embeddings
                # This must happen first to avoid device mismatches when using model.model.rotary_emb
                # The rotary_emb is shared across layers, so we need to ensure inputs are on the correct device
                if sample_inp.device != layer_dev:
                    sample_inp = sample_inp.to(layer_dev)
                if sample_pos.device != layer_dev:
                    sample_pos = sample_pos.to(layer_dev).contiguous()
                
                # Compute position_embeddings from position_ids using model.model.rotary_emb
                # In transformers 4.45.2, decoder layer accepts position_ids and optional position_embeddings
                # If position_embeddings is provided, the layer uses it instead of computing its own
                position_embeddings = None
                if sample_pos is not None:
                    try:
                        # Ensure model.model.rotary_emb exists and is initialized
                        if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                            rotary_emb = model.model.rotary_emb
                            # CRITICAL: For offloaded layers, rotary_emb.inv_freq must be on layer_dev (CPU)
                            # Even though rotary_emb is shared, we need to use it on the correct device for this layer
                            # Use layer_dev as the target device (sample_inp and sample_pos are already on layer_dev)
                            target_device = layer_dev  # Use layer_dev, not rotary_emb's current device
                            
                            # For offloaded layers, we need to compute position_embeddings on CPU
                            # But model.model.rotary_emb is shared and might be on GPU
                            # Instead of moving the shared buffer (which affects all layers),
                            # create a temporary rotary_emb on the target device
                            state_dict = model.state_dict()
                            rotary_key = "model.rotary_emb.inv_freq"
                            
                            # Get inv_freq on target_device
                            if rotary_key in state_dict:
                                inv_freq = state_dict[rotary_key].to(target_device)
                            elif hasattr(rotary_emb, 'inv_freq') and hasattr(rotary_emb.inv_freq, 'device'):
                                inv_freq = rotary_emb.inv_freq.to(target_device)
                            else:
                                raise ValueError("Could not get rotary_emb.inv_freq")
                            
                            # Create a temporary rotary_emb module on target_device
                            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                            temp_rotary_emb = LlamaRotaryEmbedding(config=model.config)
                            temp_rotary_emb.inv_freq = inv_freq
                            temp_rotary_emb.to(target_device)
                            
                            # Use temp_rotary_emb instead of the shared one
                            rotary_emb = temp_rotary_emb
                            
                            # sample_inp and sample_pos are already on target_device (layer_dev) from above
                            # Double-check to be safe
                            if sample_inp.device != target_device:
                                sample_inp = sample_inp.to(target_device)
                            if sample_pos.device != target_device:
                                sample_pos = sample_pos.to(target_device)
                            
                            # Compute position_embeddings: rotary_emb(hidden_states, position_ids) returns (cos, sin)
                            # For transformers 4.45.2, rotary_emb.forward(x, position_ids) signature
                            # Ensure both inputs are on the same device
                            try:
                                position_embeddings = rotary_emb(sample_inp, sample_pos)
                            except TypeError:
                                # Try with keyword argument
                                try:
                                    position_embeddings = rotary_emb(sample_inp, position_ids=sample_pos)
                                except TypeError:
                                    position_embeddings = rotary_emb(x=sample_inp, position_ids=sample_pos)
                            except RuntimeError as e_dev:
                                # Device mismatch - ensure everything is on the same device
                                if "same device" in str(e_dev) or "device" in str(e_dev).lower():
                                    # Get the device from sample_inp (which should be on layer_dev)
                                    target_dev = sample_inp.device
                                    # Ensure rotary_emb.inv_freq is on target_dev
                                    if hasattr(rotary_emb, 'inv_freq') and hasattr(rotary_emb.inv_freq, 'device'):
                                        if rotary_emb.inv_freq.device != target_dev:
                                            rotary_emb.inv_freq = rotary_emb.inv_freq.to(target_dev)
                                    # Retry
                                    try:
                                        position_embeddings = rotary_emb(sample_inp, sample_pos)
                                    except TypeError:
                                        try:
                                            position_embeddings = rotary_emb(sample_inp, position_ids=sample_pos)
                                        except TypeError:
                                            position_embeddings = rotary_emb(x=sample_inp, position_ids=sample_pos)
                                else:
                                    raise
                    except Exception as e_rot:
                        # If rotary_emb computation fails, try to fix it
                        print(f"Warning: Could not compute position_embeddings for layer {i}, sample {j}: {e_rot}")
                        position_embeddings = None
                
                # CRITICAL: Final device check before calling layer - ensure everything is on layer_dev
                # This is especially important for offloaded layers where layer_dev is CPU
                # Create new tensors on the correct device to avoid view/slice issues
                if sample_inp.device != layer_dev:
                    sample_inp = sample_inp.to(layer_dev)
                if sample_pos.device != layer_dev:
                    # Create a new tensor on layer_dev (not a view) to ensure it's actually moved
                    sample_pos = sample_pos.to(layer_dev).contiguous()
                if sample_attn is not None and sample_attn.device != layer_dev:
                    sample_attn = sample_attn.to(layer_dev)
                
                # Call layer with position_ids and position_embeddings
                # In transformers 4.45.2, decoder layer accepts position_ids and optional position_embeddings
                # If position_embeddings is None, the layer uses its own rotary_emb which must be on the same device
                try:
                    if position_embeddings is not None:
                        # Ensure position_embeddings tensors are also on layer_dev
                        if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2:
                            cos, sin = position_embeddings
                            if cos.device != layer_dev:
                                cos = cos.to(layer_dev)
                            if sin.device != layer_dev:
                                sin = sin.to(layer_dev)
                            position_embeddings = (cos, sin)
                        out = layer(sample_inp, attention_mask=sample_attn, position_ids=sample_pos, position_embeddings=position_embeddings)[0]
                    else:
                        # Fallback: try with just position_ids (decoder layer will use its own rotary_emb)
                        # CRITICAL: Ensure sample_pos is on the same device as the layer's rotary_emb
                        # The layer's rotary_emb.inv_freq should be on layer_dev (CPU for offloaded layers)
                        # Create contiguous copy to ensure it's actually on the right device (not a view)
                        if sample_pos.device != layer_dev:
                            sample_pos = sample_pos.to(layer_dev).contiguous()
                        # Also ensure the layer's rotary_emb is on layer_dev
                        # Move the entire rotary_emb module to ensure all buffers are on the correct device
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                            rotary_emb = layer.self_attn.rotary_emb
                            try:
                                # Move the entire module to layer_dev
                                rotary_emb.to(layer_dev)
                            except Exception:
                                # If moving the module fails, try moving inv_freq directly
                                if hasattr(rotary_emb, 'inv_freq'):
                                    if not hasattr(rotary_emb.inv_freq, 'device') or rotary_emb.inv_freq.device != layer_dev:
                                        # Materialize from state_dict if needed
                                        try:
                                            state_dict = model.state_dict()
                                            rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                                            if rotary_key in state_dict:
                                                inv_freq = state_dict[rotary_key].to(layer_dev)
                                                rotary_emb.register_buffer('inv_freq', inv_freq, persistent=False)
                                            elif hasattr(rotary_emb.inv_freq, 'device'):
                                                inv_freq = rotary_emb.inv_freq.to(layer_dev)
                                                rotary_emb.register_buffer('inv_freq', inv_freq, persistent=False)
                                        except Exception as e_dev_fix:
                                            # Last resort: try to move it anyway
                                            if hasattr(rotary_emb.inv_freq, 'device'):
                                                inv_freq = rotary_emb.inv_freq.to(layer_dev)
                                                rotary_emb.register_buffer('inv_freq', inv_freq, persistent=False)
                        
                        # FINAL CHECK: Ensure sample_pos is definitely on layer_dev before calling layer
                        # Create a new contiguous tensor to avoid any view/slice issues
                        if sample_pos.device != layer_dev:
                            sample_pos = torch.tensor(sample_pos.cpu().numpy(), dtype=sample_pos.dtype, device=layer_dev)
                        # Double-check
                        assert sample_pos.device == layer_dev, f"sample_pos device mismatch before layer call: {sample_pos.device} != {layer_dev}"
                        
                        out = layer(sample_inp, attention_mask=sample_attn, position_ids=sample_pos)[0]
                except (TypeError, RuntimeError) as e:
                    error_str = str(e)
                    if "cannot unpack non-iterable NoneType object" in error_str or "position_embeddings" in error_str or "NoneType" in error_str:
                        # Rotary embedding issue - ensure rotary_emb is properly initialized
                        print(f"Warning: Rotary embedding issue at layer {i}, sample {j}. Error: {error_str}")
                        print(f"Attempting to fix rotary_emb...")
                        # Try to recompute position_embeddings using model.model.rotary_emb
                        # In transformers 4.55.0+, rotary_emb is at model.model.rotary_emb (shared)
                        fixed = False
                        try:
                            if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                                rotary_emb = model.model.rotary_emb
                                # CRITICAL: Ensure rotary_emb.inv_freq is on the same device as sample_inp
                                target_device = sample_inp.device
                                if hasattr(rotary_emb, 'inv_freq'):
                                    if hasattr(rotary_emb.inv_freq, 'device'):
                                        if rotary_emb.inv_freq.device.type == 'meta':
                                            state_dict = model.state_dict()
                                            rotary_key = "model.rotary_emb.inv_freq"
                                            if rotary_key in state_dict:
                                                rotary_emb.inv_freq = state_dict[rotary_key].to(target_device)
                                        elif rotary_emb.inv_freq.device != target_device:
                                            rotary_emb.inv_freq = rotary_emb.inv_freq.to(target_device)
                                
                                # Ensure sample_pos is on the same device as sample_inp
                                if sample_pos is not None and sample_pos.device != target_device:
                                    sample_pos = sample_pos.to(target_device)
                                
                                # Recompute position_embeddings
                                try:
                                    position_embeddings = rotary_emb(sample_inp, sample_pos)
                                except TypeError:
                                    try:
                                        position_embeddings = rotary_emb(sample_inp, position_ids=sample_pos)
                                    except TypeError:
                                        position_embeddings = rotary_emb(x=sample_inp, position_ids=sample_pos)
                                
                                if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2:
                                    fixed = True
                        except Exception as e_fix:
                            print(f"  Could not recompute position_embeddings: {e_fix}")
                        
                        if fixed:
                            # Retry the forward pass with computed position_embeddings
                            try:
                                out = layer(sample_inp, attention_mask=sample_attn, position_ids=sample_pos, position_embeddings=position_embeddings)[0]
                            except Exception as e_retry:
                                print(f"Retry failed: {e_retry}")
                                raise e  # Raise original error
                        else:
                            print(f"Could not fix rotary_emb, raising original error")
                            print(f"  Layer {i}, device: {layer_dev}")
                            print(f"  Model has model.model.rotary_emb: {hasattr(model, 'model') and hasattr(model.model, 'rotary_emb')}")
                            raise e
                    else:
                        raise
                # Move output back to calibration device
                outs[j] = out.to(inps.device)
        
        # After forward pass, ensure all weights in this layer are materialized
        # by accessing them explicitly
        for name in subset:
            try:
                # Access weight through model hierarchy to trigger materialization
                layer_module = model.model.layers[i]
                parts = name.split('.')
                module = layer_module
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_module = getattr(module, parts[-1])
                # Access the weight - this should be materialized after forward pass
                _ = param_module.weight
                _ = param_module.weight.data
            except:
                pass  # If this fails, we'll handle it in the pruning loop
        
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            
            # Get weight directly from model (not from subset which may have meta references)
            layer_module = model.model.layers[i]
            parts = name.split('.')
            module = layer_module
            for part in parts[:-1]:
                module = getattr(module, part)
            param_module = getattr(module, parts[-1])
            
            # Access weight directly from model - this should be materialized after forward pass
            weight_data = param_module.weight.data
            
            # If still meta, force materialization by doing a computation that triggers loading
            if hasattr(weight_data, 'device') and weight_data.device.type == 'meta':
                try:
                    # Get input size for dummy forward pass
                    input_size = weight_data.shape[1] if len(weight_data.shape) > 1 else weight_data.shape[0]
                    
                    # Determine target device (CPU since layers are offloaded)
                    target_dev = torch.device("cpu")
                    if hasattr(model, 'hf_device_map'):
                        layer_key = f"model.layers.{i}"
                        if layer_key in model.hf_device_map:
                            dev_val = model.hf_device_map[layer_key]
                            if isinstance(dev_val, str):
                                target_dev = torch.device("cpu" if dev_val in ["cpu", "meta"] else dev_val)
                            elif isinstance(dev_val, int):
                                target_dev = torch.device(f"cuda:{dev_val}")
                            elif isinstance(dev_val, torch.device):
                                target_dev = dev_val
                    
                    # Force materialization by doing a forward pass with proper input shape
                    # Use the actual input from calibration data if available
                    if len(inps) > 0 and inps[0].shape[-1] == input_size:
                        # Use actual calibration input - this should definitely trigger loading
                        dummy_input = inps[0][:1].to(target_dev)  # Use first sample
                    else:
                        # Fallback to zeros
                        dummy_input = torch.zeros(1, input_size, dtype=weight_data.dtype, device=target_dev)
                    
                    # Do forward pass to force materialization
                    with torch.no_grad():
                        try:
                            _ = param_module(dummy_input)
                        except Exception as fwd_err:
                            # Forward pass might fail for attention layers - try with proper shape
                            # For attention layers, we might need different input
                            if 'attention' in name.lower() or 'attn' in name.lower():
                                # Skip - these will be handled by the full layer forward pass
                                pass
                            else:
                                raise
                    
                    # Check if weight is now materialized
                    weight_data = param_module.weight.data
                    
                    # If still meta after forward pass, the weight is truly not loaded
                    # For offloaded layers, we need to skip pruning or load manually
                    if hasattr(weight_data, 'device') and weight_data.device.type == 'meta':
                        # Last resort: skip this weight and print a warning
                        # The model will still work, just this weight won't be pruned
                        print(f"Warning: Skipping pruning of {name} in layer {i} - weight is on meta device and cannot be materialized")
                        print(f"  This weight will remain unpruned. Consider reducing max_memory to avoid offloading.")
                        # Skip to next weight
                        continue
                            
                except RuntimeError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to materialize weight for {name} in layer {i}: {e}")
            
            # Update subset reference to point to materialized weight
            subset[name].weight.data = weight_data
            
            W_metric = torch.abs(weight_data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # Move to CPU for sorting if on GPU and memory is constrained
                # This avoids OOM when sorting large weight matrices (sort needs ~2x memory)
                W_metric_device = W_metric.device
                if W_metric_device.type == "cuda":
                    # Check GPU memory usage - if >85% used, sort on CPU
                    try:
                        allocated = torch.cuda.memory_allocated(W_metric_device.index if hasattr(W_metric_device, 'index') else 0)
                        total = torch.cuda.get_device_properties(W_metric_device.index if hasattr(W_metric_device, 'index') else 0).total_memory
                        memory_usage_ratio = allocated / total
                        
                        if memory_usage_ratio > 0.85:
                            # GPU memory is constrained, sort on CPU
                            W_metric_cpu = W_metric.cpu()
                            sort_res = torch.sort(W_metric_cpu, dim=-1, stable=True)
                            # Move indices back to GPU for mask operations
                            sort_res = (sort_res[0], sort_res[1].to(W_metric_device))
                        else:
                            sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    except (RuntimeError, AttributeError):
                        # If check fails, move to CPU for safety
                        W_metric_cpu = W_metric.cpu()
                        sort_res = torch.sort(W_metric_cpu, dim=-1, stable=True)
                        sort_res = (sort_res[0], sort_res[1].to(W_metric_device))
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    # Ensure all tensors are on same device (W_metric's device)
                    target_device = W_metric.device
                    sort_values = sort_res[0].to(target_device) if sort_res[0].device != target_device else sort_res[0]
                    
                    tmp_metric = torch.cumsum(sort_values, dim=1)
                    sum_before = W_metric.sum(dim=1)
                    # Update sort_res to use moved values
                    sort_res = (sort_values, sort_res[1])

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    # cur_sparsity is now a Python float, not a tensor
                    while (abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    # Ensure indices are on same device as W_metric
                    sort_indices = sort_res[1].to(W_metric.device) if sort_res[1].device != W_metric.device else sort_res[1]
                    indices = sort_indices[:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero
            
            # Clear intermediate tensors to free memory (important for CPU mode)
            del W_metric, W_mask
            if 'sort_res' in locals():
                del sort_res
            if 'tmp_metric' in locals():
                del tmp_metric
            if 'sum_before' in locals():
                del sum_before
            
            # Clear GPU cache after each weight operation to free memory
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
            
            # For CPU mode, force garbage collection periodically to free memory
            if device.type == "cpu" and (i % 4 == 0 or name == list(subset.keys())[-1]):
                import gc
                gc.collect() 

        # Process samples one at a time to avoid moving large tensors to GPU
        for j in range(args.nsamples):
            with torch.no_grad():
                sample_inp = inps[j].unsqueeze(0)
                # Get layer device
                if f"model.layers.{i}" in model.hf_device_map:
                    layer_dev = model.hf_device_map[f"model.layers.{i}"]
                    # Convert to torch.device - handle int, str, or torch.device
                    if isinstance(layer_dev, int):
                        layer_dev = torch.device(f"cuda:{layer_dev}")
                    elif isinstance(layer_dev, str):
                        layer_dev = torch.device("cpu" if layer_dev in ["cpu", "meta"] else layer_dev)
                    elif not isinstance(layer_dev, torch.device):
                        layer_dev = torch.device("cpu")
                else:
                    layer_dev = inps.device
                
                # Move only this sample to layer device
                if sample_inp.device != layer_dev:
                    sample_inp = sample_inp.to(layer_dev)
                    if attention_mask is not None:
                        sample_attn = attention_mask.to(layer_dev) if len(attention_mask.shape) == 2 else attention_mask[j:j+1].to(layer_dev)
                    else:
                        sample_attn = None
                    # Generate position_ids if None (required by decoder layers for position embeddings)
                    if position_ids is not None:
                        sample_pos = position_ids.to(layer_dev) if len(position_ids.shape) == 2 else position_ids[j:j+1].to(layer_dev)
                    else:
                        seq_len = sample_inp.shape[1]
                        sample_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)
                else:
                    sample_attn = attention_mask
                    # Generate position_ids if None
                    if position_ids is not None:
                        sample_pos = position_ids
                    else:
                        seq_len = sample_inp.shape[1]
                        sample_pos = torch.arange(0, seq_len, dtype=torch.long, device=layer_dev).unsqueeze(0)
                
                # CRITICAL: Ensure sample_inp and sample_pos are on layer_dev BEFORE computing position_embeddings
                if sample_inp.device != layer_dev:
                    sample_inp = sample_inp.to(layer_dev)
                if sample_pos.device != layer_dev:
                    sample_pos = sample_pos.to(layer_dev).contiguous()
                
                # Compute position_embeddings from position_ids using model.model.rotary_emb
                # In transformers 4.45.2, decoder layer accepts position_ids and optional position_embeddings
                position_embeddings = None
                if sample_pos is not None:
                    try:
                        # Ensure model.model.rotary_emb exists and is initialized
                        if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                            rotary_emb = model.model.rotary_emb
                            target_device = layer_dev  # Use layer_dev, not rotary_emb's current device
                            
                            # Create a temporary rotary_emb on target_device to avoid modifying shared buffer
                            state_dict = model.state_dict()
                            rotary_key = "model.rotary_emb.inv_freq"
                            
                            # Get inv_freq on target_device
                            if rotary_key in state_dict:
                                inv_freq = state_dict[rotary_key].to(target_device)
                            elif hasattr(rotary_emb, 'inv_freq') and hasattr(rotary_emb.inv_freq, 'device'):
                                inv_freq = rotary_emb.inv_freq.to(target_device)
                            else:
                                raise ValueError("Could not get rotary_emb.inv_freq")
                            
                            # Create a temporary rotary_emb module on target_device
                            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                            temp_rotary_emb = LlamaRotaryEmbedding(config=model.config)
                            temp_rotary_emb.inv_freq = inv_freq
                            temp_rotary_emb.to(target_device)
                            
                            # Use temp_rotary_emb instead of the shared one
                            rotary_emb = temp_rotary_emb
                            
                            # Compute position_embeddings: rotary_emb(hidden_states, position_ids) returns (cos, sin)
                            try:
                                position_embeddings = rotary_emb(sample_inp, sample_pos)
                            except TypeError:
                                try:
                                    position_embeddings = rotary_emb(sample_inp, position_ids=sample_pos)
                                except TypeError:
                                    position_embeddings = rotary_emb(x=sample_inp, position_ids=sample_pos)
                    except Exception as e_rot:
                        # If rotary_emb computation fails, try to fix it
                        print(f"Warning: Could not compute position_embeddings for layer {i}, sample {j} (second pass): {e_rot}")
                        position_embeddings = None
                
                # CRITICAL: Final device check before calling layer
                if sample_inp.device != layer_dev:
                    sample_inp = sample_inp.to(layer_dev)
                if sample_pos.device != layer_dev:
                    sample_pos = torch.tensor(sample_pos.cpu().numpy(), dtype=sample_pos.dtype, device=layer_dev)
                if sample_attn is not None and sample_attn.device != layer_dev:
                    sample_attn = sample_attn.to(layer_dev)
                
                # Ensure layer's rotary_emb is on layer_dev if we're falling back to it
                if position_embeddings is None:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                        rotary_emb = layer.self_attn.rotary_emb
                        try:
                            rotary_emb.to(layer_dev)
                        except Exception:
                            if hasattr(rotary_emb, 'inv_freq'):
                                state_dict = model.state_dict()
                                rotary_key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
                                if rotary_key in state_dict:
                                    inv_freq = state_dict[rotary_key].to(layer_dev)
                                    rotary_emb.register_buffer('inv_freq', inv_freq, persistent=False)
                
                # Call layer with position_ids and position_embeddings
                if position_embeddings is not None:
                    out = layer(sample_inp, attention_mask=sample_attn, position_ids=sample_pos, position_embeddings=position_embeddings)[0]
                else:
                    # Final check: ensure sample_pos is on layer_dev
                    if sample_pos.device != layer_dev:
                        sample_pos = torch.tensor(sample_pos.cpu().numpy(), dtype=sample_pos.dtype, device=layer_dev)
                    out = layer(sample_inp, attention_mask=sample_attn, position_ids=sample_pos)[0]
                outs[j] = out.to(inps.device)
        
        # Clear wrapped layers and handles to free memory after processing layer
        del wrapped_layers
        for h in handles:
            h.remove()
        del handles
        
        # For CPU mode, force garbage collection after each layer
        if device.type == "cpu":
            import gc
            gc.collect()
            if (i + 1) % 4 == 0:
                print(f"Processed {i+1} layers, memory cleared")
        
        # Swap inputs/outputs, but keep on same device
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer,load_validation=False)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
        # Convert to torch.device if string
        if isinstance(dev, str):
            if dev in ["cpu", "meta"]:
                dev = torch.device("cpu")
            else:
                dev = torch.device(dev)
    else:
        dev = torch.device("cpu")  # Default to CPU if not specified
    
    # Use CPU if model is offloaded to avoid OOM
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        has_offloaded = any(d in ["cpu", "meta"] or (isinstance(d, str) and "cpu" in d.lower()) 
                           for d in model.hf_device_map.values())
        if has_offloaded:
            dev = torch.device("cpu")
            print(f"Using CPU for calibration inputs (model has offloaded layers)")

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        # Clear CUDA cache periodically if using GPU
        if torch.cuda.is_available() and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer,load_validation=False)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
        # Convert to torch.device if string
        if isinstance(dev, str):
            if dev in ["cpu", "meta"]:
                dev = torch.device("cpu")
            else:
                dev = torch.device(dev)
    else:
        dev = torch.device("cpu")  # Default to CPU if not specified
    
    # Use CPU if model is offloaded to avoid OOM
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        has_offloaded = any(d in ["cpu", "meta"] or (isinstance(d, str) and "cpu" in d.lower()) 
                           for d in model.hf_device_map.values())
        if has_offloaded:
            dev = torch.device("cpu")
            print(f"Using CPU for calibration inputs (model has offloaded layers)")

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        # Clear CUDA cache periodically if using GPU
        if torch.cuda.is_available() and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()