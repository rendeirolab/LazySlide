import numpy as np

from lazyslide.models.base import SegmentationModel


class Cellpose(SegmentationModel):
    def __init__(
        self,
        diam_mean=None,
        model_path=None,
        **eval_kwargs,
    ):
        try:
            from cellpose import models
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install cellpose>=4.0.0")

        self.cellpose_model = models.CellposeModel(
            pretrained_model=model_path if model_path is not None else "cpsam",
            diam_mean=diam_mean,
            gpu=True,
        )
        self.eval_kwargs = eval_kwargs

    def to(self, device):
        import torch

        self.cellpose_model.device = torch.device(device)

    def get_transform(self):
        return None

    def segment(self, image):
        if image.ndim == 4:
            # If the image is a batch, we need to make it into a list of images
            image = [img.detach().cpu().numpy() for img in image]

        masks, _, _ = self.cellpose_model.eval(
            image, batch_size=len(image), **self.eval_kwargs
        )
        if isinstance(masks, list):
            # If the masks are a list, we need to convert them to a numpy array
            masks = np.array(masks)
        elif masks.ndim == 2:
            # If the masks are a single image, we need to add a batch dimension
            masks = masks[np.newaxis, ...]
        return {"instance_map": masks}

    def supported_output(self):
        return ("instance_map",)
