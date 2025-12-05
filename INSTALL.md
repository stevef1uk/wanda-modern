# Installation  

## Option 1: Using pyproject.toml (Recommended)

The project uses `pyproject.toml` for dependency management. Install dependencies using:

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

This will install:
- `transformers==4.45.2` (required for rotary embedding compatibility)
- `torch>=2.8.0`
- `accelerate>=1.10.1`
- `datasets>=4.1.1`
- And other required dependencies

## Option 2: Manual Installation

Step 1: Create a new conda environment:
```
conda create -n prune_llm python=3.12
conda activate prune_llm
```

Step 2: Install relevant packages
```
conda install pytorch>=2.8.0 torchvision torchaudio -c pytorch -c conda-forge
pip install transformers==4.45.2 datasets>=4.1.1 accelerate>=1.10.1 wandb sentencepiece
```

**Note**: The project requires `transformers==4.45.2` for proper rotary embedding initialization on CPU. Using other versions may cause `TypeError: cannot unpack non-iterable NoneType object` errors when pruning CPU-loaded models.

There are known [issues](https://github.com/huggingface/transformers/issues/22222) with older versions of the transformers library on loading the LLaMA tokenizer correctly. The current version (4.45.2) resolves these issues.