import numpy as np

from lazyslide.models.base import ModelTask, SegmentationModel


class Cellpose(SegmentationModel, key="cellpose"):
    """
    Only supports cellpose>=4.0.0

    If you want to fine-tune the cellpose model, please take a look at the following resources:

    - https://github.com/MouseLand/cellpose/blob/main/notebooks/train_Cellpose-SAM.ipynb
    - https://cellpose.readthedocs.io/en/latest/train.html

    To run a fine-tuned model, pass the `model_path` argument pointing to the fine-tuned weights.

    .. code-block:: python

       >>> zs.seg.cells(wsi, model="cellpose", model_path="fine-tuned-checkpoint.pth")

    """

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

        self.model = models.CellposeModel(
            pretrained_model=model_path if model_path is not None else "cpsam",
            diam_mean=diam_mean,
            gpu=True,
        )
        self.eval_kwargs = eval_kwargs

    def to(self, device):
        import torch

        self.model.device = torch.device(device)
        return self

    def get_transform(self):
        return None

    def segment(self, image):
        if image.ndim == 4:
            # If the image is a batch, we need to make it into a list of images
            image = [img.detach().cpu().numpy() for img in image]

        masks, _, _ = self.model.eval(image, batch_size=len(image), **self.eval_kwargs)
        if isinstance(masks, list):
            # If the masks are a list, we need to convert them to a numpy array
            masks = np.array(masks)
        elif masks.ndim == 2:
            # If the masks are a single image, we need to add a batch dimension
            masks = masks[np.newaxis, ...]
        return {"instance_map": masks}

    def supported_output(self):
        return ("instance_map",)
