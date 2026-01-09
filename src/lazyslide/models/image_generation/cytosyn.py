from importlib import import_module
from importlib.util import find_spec

import torch

from .._model_registry import register
from .._utils import hf_access
from ..base import ImageGenerationModel, ModelTask


@register(
    key="cytosyn",
    task=ModelTask.image_generation,
    is_gated=True,
    license="CC BY-NC-ND 4.0",
    description="A REPA-E Histopathology Image Generation Model",
    commercial=False,
    github_url="https://github.com/prov-gigatime/GigaTIME",
    paper_url="https://www.owkin.com/blogs-case-studies/"
    "cytosyn-a-state-of-the-art-diffusion-model-for-histopathology-image-generation",
    param_size="766M",
)
class CytoSyn(ImageGenerationModel):
    def __init__(self, model_path=None, token=None):
        diffusers = find_spec("diffusers")
        if diffusers is None:
            raise ModuleNotFoundError(
                "Please install diffusers to use CytoSyn: `pip install diffusers`"
            )

        DiffusionPipeline = import_module("diffusers.pipelines").DiffusionPipeline
        with hf_access("Owkin-Bioptimus/CytoSyn"):
            self.model = DiffusionPipeline.from_pretrained(
                "Owkin-Bioptimus/CytoSyn",
                custom_pipeline="Owkin-Bioptimus/CytoSyn",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

    def generate(self, *args, **kwargs):
        opts = dict(
            num_images_per_prompt=1,
            num_inference_steps=250,
            guidance_scale=1.0,  # No guidance for unconditional
        )
        opts.update(kwargs)
        return self.model(**opts)["images"]

    def generate_conditionally(self, h0_mini_embeds, **kwargs):
        opts = dict(
            h0_mini_embeds=h0_mini_embeds,
            num_images_per_prompt=1,
            num_inference_steps=250,
            guidance_scale=2.5,
            guidance_low=0.0,
            guidance_high=0.75,
        )
        opts.update(kwargs)
        return self.model(**opts)["images"]
