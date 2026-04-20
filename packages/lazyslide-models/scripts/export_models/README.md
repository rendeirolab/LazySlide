# LazySlide export models

This repo keeps track of reproducible codes to build deployment version of models used in LazySlide.

Because some models relied on a specific version of a package, or it cannot be loaded easily from huggingface, 
we will try to build a static version to easily load it for inference.

The export scripts are named as `export_*.py`. They are all self-contained, please run with `uv run --script`.

# Upload weights to HuggingFace

```bash

uv run hf upload RendeiroLab/LazySlide-models checkpoints/xxx.pt xxx/xxx.pt

# For GPL license models
uv run hf upload RendeiroLab/LazySlide-models-gpl checkpoints/xxx.pt xxx/xxx.pt

```