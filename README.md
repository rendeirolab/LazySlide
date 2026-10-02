# LazySlide

<p align="center">
    <picture align="center">
    <img src="https://raw.githubusercontent.com/rendeirolab/lazyslide/main/assets/logo.png" alt="LazySlide logo" width="150px">
    </picture>
</p>
<p align="center">
  <i>Accessible and interoperable whole slide image analysis</i>
</p>


[![Documentation Status](https://readthedocs.org/projects/lazyslide/badge/?version=stable&style=flat-square)](https://lazyslide.readthedocs.io/en/stable)
[![pypi version](https://img.shields.io/pypi/v/lazyslide?color=0098FF&logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/lazyslide)
[![conda version](https://img.shields.io/conda/vn/conda-forge/lazyslide?style=flat-square&logo=anaconda&logoColor=white&color=%2344A833)](https://anaconda.org/conda-forge/lazyslide)
[![PyPI - License](https://img.shields.io/pypi/l/lazyslide?color=FFD43B&style=flat-square)](https://github.com/rendeirolab/lazyslide/blob/main/LICENSE)
[![scverse ecosystem](https://img.shields.io/badge/scverse_ecosystem-gray.svg?style=flat-square&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgOTEgOTEiIHZlcnNpb249IjEuMSIKICAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4bWw6c3BhY2U9InByZXNlcnZlIgogICAgeG1sbnM6c2VyaWY9Imh0dHA6Ly93d3cuc2VyaWYuY29tLyIgc3R5bGU9ImZpbGwtcnVsZTpldmVub2RkO2NsaXAtcnVsZTpldmVub2RkO3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtbWl0ZXJsaW1pdDoyOyI+CiAgICA8ZyBpZD0iRWJlbmVfMyI+CiAgICAgICAgPGc+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNSw4OS42Yy0yMi4zLC0zLjQgLTMwLjYsLTE5LjggLTMwLjYsLTE5LjhjMTAuOCwxNi45IDQzLDkuMSA1Mi45LDIuNWMxMi40LC04LjMgOCwtMTUuMyA2LjgsLTE4LjFjNS40LDcuMiA1LjMsMjMuNSAtMS4xLDI5LjRjLTUuNiw1LjEgLTE1LjMsNy45IC0yOCw2WiIgc3R5bGU9ImZpbGw6I2ZmZjtmaWxsLXJ1bGU6bm9uemVybztzdHJva2U6IzAwMDtzdHJva2Utd2lkdGg6MXB4OyIvPgogICAgICAgICAgICA8cGF0aCBkPSJNODMuOSw0My41YzIuOSwtNy4xIDAuOCwtMTIuNSAwLjUsLTEzLjNjLTAuNywtMS4zIC0xLjUsLTIuMyAtMi40LC0zLjFjLTE2LjEsLTEyLjYgLTU1LjksMSAtNzAuOSwxNi44Yy0xMC45LDExLjUgLTEwLjEsMjAgLTYuNywyNS44YzMuMSw0LjggNy45LDcuNiAxMy40LDljLTExLjUsLTEyLjQgOS44LC0zMS4xIDI5LC0zOGMyMSwtNy41IDMyLjUsLTMgMzcuMSwyLjhaIiBzdHlsZT0iZmlsbDojMzQzNDM0O2ZpbGwtcnVsZTpub256ZXJvO3N0cm9rZTojMDAwO3N0cm9rZS13aWR0aDoxcHg7Ii8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik03OS42LDUwLjRjOSwtMTAuNSA1LC0xOS43IDQuOCwtMjAuNGMtMCwwIDQuNCw3LjEgMi4yLDIyLjZjLTEuMiw4LjUgLTUuNCwxNiAtMTAuMSwxMS44Yy0yLjEsLTEuOCAtMywtNi45IDMuMSwtMTRaIiBzdHlsZT0iZmlsbDojZmZmO2ZpbGwtcnVsZTpub256ZXJvO3N0cm9rZTojMDAwO3N0cm9rZS13aWR0aDoxcHg7Ii8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik02NCw1NC4yYy0zLjMsLTQuOCAtOC4xLC03LjQgLTEyLjMsLTEwLjhjLTIuMiwtMS43IC0xNi40LC0xMS4yIC0xOS4yLC0xNS4xYy02LjQsLTYuNCAtOS41LC0xNi45IC0zLjQsLTIzLjFjLTQuNCwtMC44IC04LjIsMC4yIC0xMC42LDEuNWMtMS4xLDAuNiAtMi4xLDEuMiAtMi44LDJjLTYuNyw2LjIgLTUuOCwxNyAtMS42LDI0LjNjNC41LDcuOCAxMy4yLDE1LjQgMjQuMywyMi44YzUuMSwzLjQgMTUuNiw4LjQgMTkuMywxNmMxMS43LC04LjEgNy42LC0xNC45IDYuMywtMTcuNloiIHN0eWxlPSJmaWxsOiNiNGI0YjQ7ZmlsbC1ydWxlOm5vbnplcm87c3Ryb2tlOiMwMDA7c3Ryb2tlLXdpZHRoOjFweDsiLz4KICAgICAgICAgICAgPHBhdGggZD0iTTM4LjcsOS44YzcuOSw2LjMgMTIuNCw5LjggMjAsOC41YzUuNywtMSA0LjksLTcuOSAtNCwtMTMuNmMtNC40LC0yLjggLTkuNCwtNC4yIC0xNS43LC00LjJjLTcuNSwtMCAtMTYuMywzLjkgLTIwLjYsNi40YzQsLTIuMyAxMS45LC0zLjggMjAuMywyLjlaIiBzdHlsZT0iZmlsbDojZmZmO2ZpbGwtcnVsZTpub256ZXJvO3N0cm9rZTojMDAwO3N0cm9rZS13aWR0aDoxcHg7Ii8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=)](https://scverse.org/packages/#ecosystem)
[![Nature Methods](https://img.shields.io/badge/DOI-10.1038%2Fs41592--026--03044--7-blue?style=flat-square)](https://doi.org/10.1038/s41592-026-03044-7)

[Installation](https://lazyslide.readthedocs.io/en/stable/installation.html) | 
[Tutorials](https://lazyslide.readthedocs.io/en/stable/tutorials/index.html) |
[Preprint](https://doi.org/10.1101/2025.05.28.656548) | 
[Nature Methods](https://doi.org/10.1038/s41592-026-03044-7)

LazySlide is a Python framework for whole slide image (WSI) analysis in digital and computational pathology. From a raw slide to tissue masks, tiles, foundation-model features, cell segmentations and zero-shot predictions in a few lines of code. Everything is stored as [SpatialData](https://spatialdata.scverse.org), so results go straight into [scverse](https://scverse.org) tools such as scanpy, anndata and squidpy.

## Key features

- **Preprocessing**: tissue detection, tiling at any resolution, artifact QC
- **Pathology foundation models**: tile features from 30+ models (UNI, Virchow, Prov-GigaPath, H-optimus, …) or any timm model
- **Segmentation**: cells (InstanSeg, Cellpose, …), tissue and artifacts
- **Vision-language models**: zero-shot classification and segmentation, slide captioning, text search (CONCH, PLIP, TITAN, …)
- **Spatial and multimodal analysis**: spatial domains, tile graphs, linking morphology to gene expression
- **Any slide format**: SVS, NDPI, MRXS, DICOM, CZI, iSyntax and more via [wsidata](https://github.com/rendeirolab/wsidata)
- **Deep learning ready**: PyTorch datasets for training your own models

![LazySlide overview: tissue segmentation, tiling, foundation-model feature extraction, cell segmentation, spatial domains, zero-shot classification and captioning, genomic data integration](https://raw.githubusercontent.com/rendeirolab/lazyslide/main/assets/Figure.png)

## Installation

LazySlide supports Python 3.11–3.14 on Linux, macOS and Windows.

```bash
pip install lazyslide   # or: uv add lazyslide
```

For extra slide readers (CZI, iSyntax, BioFormats) and gated models, see the [installation guide](https://lazyslide.readthedocs.io/en/stable/installation.html) and [model zoo](https://lazyslide.readthedocs.io/en/stable/avail_models.html).

## Quick start

Detect tissue, tile it and extract features from a sample slide in a few lines of code:

```python
import lazyslide as zs

wsi = zs.datasets.sample()

# Pipeline
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
zs.tl.feature_extraction(wsi, model="resnet50")

# Access the features
features = wsi["resnet50_tiles"]

# Color tiles by feature dimensions 1 and 99
zs.pl.tiles(wsi, feature_key="resnet50", color=["1", "99"])
```

To open your own slide:

```python
wsi = zs.open_wsi("path/to/slide.svs")
```

## Documentation

New to digital pathology? Start with the [getting started guide](https://lazyslide.readthedocs.io/en/stable/getting-started/index.html). The [documentation](https://lazyslide.readthedocs.io) also has [tutorials](https://lazyslide.readthedocs.io/en/stable/tutorials/index.html), [how-to guides](https://lazyslide.readthedocs.io/en/stable/how-to/index.html), the [API reference](https://lazyslide.readthedocs.io/en/stable/api/index.html) and the [model zoo](https://lazyslide.readthedocs.io/en/stable/avail_models.html).

## Citation

If you use LazySlide in your research, please cite:

> Zheng Y, Abila E, Chrenková E, Buljan I, Winkler J, Rendeiro AF. LazySlide: accessible and interoperable whole-slide image analysis. *Nature Methods* 23, 728–731 (2026). <https://doi.org/10.1038/s41592-026-03044-7>

<details><summary>BibTeX</summary>

```bibtex
@article{zheng2026lazyslide,
  title   = {LazySlide: accessible and interoperable whole-slide image analysis},
  author  = {Zheng, Yimin and Abila, Ernesto and Chrenkov{\'a}, Eva and Buljan, Iva and Winkler, Juliane and Rendeiro, Andr{\'e} F.},
  journal = {Nature Methods},
  volume  = {23},
  number  = {4},
  pages   = {728--731},
  year    = {2026},
  doi     = {10.1038/s41592-026-03044-7}
}
```

</details>

## Contributing

Contributions to documentation, tests and features are welcome, and so are suggestions. Open an [issue](https://github.com/rendeirolab/lazyslide/issues) or a pull request, and see the [contributing guide](https://lazyslide.readthedocs.io/en/latest/contributing/index.html).

## Licence

LazySlide is released under the [MIT License](https://github.com/rendeirolab/lazyslide/blob/main/LICENSE).
