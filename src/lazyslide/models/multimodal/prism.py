import warnings

import torch

from .._utils import hf_access
from ..base import ModelBase


class Prism(ModelBase):
    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel

        # Suppress warnings from transformers
        with warnings.catch_warnings(), hf_access(model_path):
            warnings.simplefilter("ignore")

            self.model = AutoModel.from_pretrained(
                "paige-ai/Prism",
                trust_remote_code=True,
                token=token,
            )

    @torch.inference_mode()
    def encode_slide(self, embeddings, coords=None) -> dict:
        # Make sure the embeddings has a batch dimension
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
        return self.model.slide_representations(embeddings)

    @torch.inference_mode()
    def score(
        self,
        slide_embedding,
        prompts: list[list[str]],
    ):
        if len(prompts):
            pass

        device = self.model.device

        # Flatten all prompts and track indices for class reconstruction
        flat_prompts = []
        group_lengths = []
        for group in prompts:
            flat_prompts.extend(group)
            group_lengths.append(len(group))

        token_ids = self.model.tokenize(flat_prompts)[:, :-1].to(device)

        dummy_image_latents = torch.empty(
            (len(flat_prompts), 1, self.model.text_decoder.context_dim), device=device
        )
        decoder_out = self.model.text_decoder(token_ids, dummy_image_latents)

        text_proj = self.model.text_to_latents(decoder_out["text_embedding"])
        image_proj = self.model.img_to_latents(slide_embedding)

        sim = torch.einsum("i d, j d -> i j", image_proj, text_proj)  # (image, prompt)
        sim = sim * self.model.temperature.exp()
        zero_shot_probs = torch.softmax(
            sim.to(torch.float), dim=-1
        )  # (Bi, total_prompts)

        # Sum probabilities per group (class)
        class_probs = []
        start = 0
        for length in group_lengths:
            end = start + length
            class_probs.append(zero_shot_probs[:, start:end].sum(dim=-1, keepdim=True))
            start = end

        probs = torch.cat(class_probs, dim=-1)
        return probs.detach().cpu().numpy()

    @torch.inference_mode()
    def caption(
        self,
        img_latents,
        prompt: list[str],
        max_length: int = 100,
    ):
        genned_ids = self.model.generate(
            self.model.tokenize(prompt).to(self.model.device),
            key_value_states=img_latents,
            do_sample=False,
            num_beams=5,
            num_beam_groups=1,
            max_length=max_length,
        )
        genned_caption = self.model.untokenize(genned_ids)

        return genned_caption
