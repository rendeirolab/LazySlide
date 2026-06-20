# Installation and environments

## How do I verify my installation?

```bash
python -c "import lazyslide as zs; print(zs.__version__)"
```

Then run a minimal, model-free smoke test:

```python
import lazyslide as zs

wsi = zs.datasets.sample(with_data=False)
zs.pp.find_tissues(wsi, level=-1)
print(len(wsi.shapes["tissues"]))
```

## Which slide reader should I install?

TiffSlide and OpenSlide cover many bright-field WSI formats. fastslide provides a high-performance native reader; Bio-Formats supports broad microscopy coverage; cuCIM supports GPU-accelerated I/O; pyisyntax handles Philips iSyntax; and pylibCZIrw handles Zeiss CZI. Reader installation details are on the [Installation](../installation) page.

Test the selected reader against representative files from your scanner. A filename extension alone does not guarantee that every vendor variant is supported.

## How do I access a gated Hugging Face model?

Request access on the model's Hugging Face page, create a read token, and authenticate once:

```bash
hf auth login
```

Do not place tokens in notebooks, command history, or committed configuration. Confirm access by instantiating the model before launching a long slide job.

## How do I run models on an offline compute node?

Download the weights on a networked machine or login node:

```bash
hf download OWNER/MODEL
```

Copy or mount the same Hugging Face cache on the compute node, then set offline mode before importing model libraries:

```bash
export HF_HUB_OFFLINE=1
```

Set `HF_HOME` consistently if the cache is not under its default location. See the [Model Zoo](../avail_models) for the complete offline workflow.

## How do I install the development documentation environment?

From a repository checkout:

```bash
uv sync --group docs --group tutorials
uv run task doc-build
```

Contributor setup is documented in [Development setup](../contributing/setup).
