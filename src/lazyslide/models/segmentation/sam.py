import torch

from ..base import ModelTask, SegmentationModel


class SAM(SegmentationModel, key="sam"):
    task = ModelTask.segmentation
    commercial = True
    license = "Apache 2.0"
    description = "SAM model for image segmentation"
    github_url = "https://github.com/facebookresearch/segment-anything"
    paper_url = "https://arxiv.org/abs/2304.02643"

    SAM_VARIENTS = [
        "facebook/sam-vit-base",
        "facebook/sam-vit-large",
        "facebook/sam-vit-huge",
    ]

    SAM_HQ_VARIENTS = [
        "syscv-community/sam-hq-vit-base",
        "syscv-community/sam-hq-vit-large",
        "syscv-community/sam-hq-vit-huge",
    ]

    def __init__(self, variant="facebook/sam-vit-base", model_path=None, token=None):
        self.variant = variant
        if variant in self.SAM_VARIENTS:
            from transformers import SamModel, SamProcessor
            # from ultralytics import SAM

            self.model = SamModel.from_pretrained(variant, use_auth_token=token)
            self.processor = SamProcessor.from_pretrained(variant, use_auth_token=token)
            self._is_hq = False

        elif variant in self.SAM_HQ_VARIENTS:
            from transformers import SamHQModel, SamHQProcessor

            self.model = SamHQModel.from_pretrained(variant, use_auth_token=token)
            self.processor = SamHQProcessor.from_pretrained(
                variant, use_auth_token=token
            )
            self._is_hq = True
        else:
            raise ValueError(
                f"Unsupported SAM variant: {variant}. "
                f"Choose from {self.SAM_VARIENTS + self.SAM_HQ_VARIENTS}."
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def get_image_embedding(self, image) -> torch.Tensor:
        """
        Get the image embedding from the SAM model.

        Returns:
            torch.Tensor: Image embedding tensor of shape (1, C, H, W).

        """
        img_inputs = self.processor(image, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(img_inputs["pixel_values"])
        if self._is_hq:
            embeddings = embeddings[0]
        return embeddings.detach().cpu()

    @torch.inference_mode()
    def segment(
        self,
        image,
        image_embedding=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        segmentation_maps=None,
        multimask_output=False,
    ) -> torch.Tensor:
        """
        Segment the input image using the SAM model.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Segmentation mask tensor of shape (H, W).
        """
        inputs = self.processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )
        if image_embedding is not None:
            del inputs["pixel_values"]
            inputs["image_embeddings"] = image_embedding

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                inputs[k] = v.to(dtype=torch.float32)

        inputs = inputs.to(self.model.device)
        outputs = self.model(**inputs, multimask_output=multimask_output)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            mask_threshold=0,
        )
        return {"probability_map": masks[0]}

    def supported_output(self):
        """
        Returns the supported output types for the SAM model.
        """
        return ("probability_map",)
