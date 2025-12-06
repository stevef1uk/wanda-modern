# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import time
from datasets import load_dataset
import requests

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer, load_validation=True):
    # Cap calibration sequence length to reasonable value (2048 tokens)
    # Even if model supports longer sequences, calibration doesn't need them
    # This avoids issues with datasets that don't have very long samples
    max_calib_seqlen = 2048
    calib_seqlen = min(seqlen, max_calib_seqlen)
    if seqlen > max_calib_seqlen:
        print(f"Note: Model supports {seqlen} tokens, but using {calib_seqlen} for calibration (sufficient for pruning)")
    
    # Load train dataset (always needed for calibration)
    print(f"Loading C4 dataset for calibration (this may take a few minutes)...")
    # Add retry logic for network timeouts
    import time
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, Exception) as e:
            if attempt < max_retries - 1:
                print(f"Dataset download timed out (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to load C4 dataset after {max_retries} attempts.")
                print("This is likely a network timeout issue. Please try again later.")
                raise
    print(f"C4 dataset loaded: {len(traindata)} examples available")

    # Generate samples from training set
    print(f"Generating {nsamples} calibration samples (length: {calib_seqlen} tokens)...")
    random.seed(seed)
    trainloader = []
    for sample_idx in range(nsamples):
        attempts = 0
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > calib_seqlen:
                break
            attempts += 1
            if attempts > 1000:  # Safety limit to avoid infinite loop
                print(f"Warning: Having trouble finding samples > {calib_seqlen} tokens after {attempts} attempts")
                # Use whatever length we have, pad if needed
                if trainenc.input_ids.shape[1] > calib_seqlen // 2:
                    break
        i = random.randint(0, max(0, trainenc.input_ids.shape[1] - calib_seqlen - 1))
        j = min(i + calib_seqlen, trainenc.input_ids.shape[1])
        inp = trainenc.input_ids[:, i:j]
        # Pad to calib_seqlen if needed
        if inp.shape[1] < calib_seqlen:
            padding = torch.zeros((1, calib_seqlen - inp.shape[1]), dtype=inp.dtype)
            inp = torch.cat([inp, padding], dim=1)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        if (sample_idx + 1) % max(1, nsamples // 4) == 0 or sample_idx == nsamples - 1:
            print(f"  Generated {sample_idx + 1}/{nsamples} calibration samples")

    # Only load validation dataset if requested (not needed for pruning/calibration)
    if load_validation:
        valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        # Prepare validation dataset
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return trainloader, valenc
    else:
        return trainloader, None

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, load_validation=True):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, load_validation=load_validation)