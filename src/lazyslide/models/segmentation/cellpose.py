import numpy as np

from lazyslide.models.base import ModelTask, SegmentationModel


class Cellpose(SegmentationModel, key="cellpose"):
    task = ModelTask.segmentation
    license = "BSD-3-Clause"
    description = "Cell segmentation model"
    commercial = True
    github_url = "https://github.com/MouseLand/cellpose"
    hf_url = "https://huggingface.co/mouseland/cellpose-sam"
    paper_url = "https://doi.org/10.1038/s41592-020-01018-x"
    bib_key = "Stringer2021-cx"

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
