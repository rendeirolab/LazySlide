"""Prime Hugging Face cache entries needed by CI tests."""

from huggingface_hub import hf_hub_download

import lazyslide as zs
from lazyslide.tools._features import load_models

TIMM_TEST_MODEL_REPOS = (
    "timm/resnet50.a1_in1k",
    "timm/test_resnet.r160_in1k",
    "timm/test_vit.r160_in1k",
)

HF_MODEL_FILES = (
    (
        "RendeiroLab/LazySlide-models-gpl",
        "PathProfiler/PathProfiler_tissue_seg_exported.pt2",
    ),
)


def main():
    zs.datasets.sample()
    zs.datasets.gtex_artery()

    for repo_id in TIMM_TEST_MODEL_REPOS:
        hf_hub_download(repo_id, "model.safetensors")

    for repo_id, filename in HF_MODEL_FILES:
        hf_hub_download(repo_id, filename)

    load_models("resnet50")


if __name__ == "__main__":
    main()
