import cv2
import numpy as np
import torch
from wsidata import WSIData
from wsidata.io import add_tissues
from shapely.affinity import scale

from lazyslide._const import Key
from lazyslide.cv import BinaryMask
from lazyslide.models.segmentation import SMPBase


class GrandQCTissueSegmentation(SMPBase):
    def __init__(self):
        weights = self.load_weights(
            "https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth"
        )

        super().__init__(
            arch="unetplusplus",
            encoder_name="timm-efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
            activation=None,
        )
        self.model.load_state_dict(
            torch.load(weights, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def segment(self, image):
        with torch.inference_mode():
            return self.model.predict(image)


def tissue(
    wsi: WSIData,
    level: int = None,
    chunk: bool = None,
    overlap: int = 0.2,
    device: str = "cpu",
    key_added: str = Key.tissue,
):
    """
    Return a dataset for tissue segmentation.

    Parameters
    ----------
    wsi: WSI
        The whole slide image.

    Returns
    -------
    TissueSegmentation
        A dataset for tissue segmentation.

    """

    props = wsi.properties
    if level is None:
        level_mpp = np.array(props.level_downsample) * props.mpp
        # Get the nearest level that towards mpp=10
        level = np.argmin(np.abs(level_mpp - 10))
    shape = props.level_shape[level]
    # If the image size is more than 5120*5120, we will need to run
    # inference on tile basis
    if chunk is None:
        chunk = shape[0] > 5120 or shape[1] > 5120

    model = GrandQCTissueSegmentation()
    transform = model.get_transform()

    model.to(device)

    if chunk:
        raise NotImplementedError("Chunked segmentation is not implemented yet.")
        tile_size = 512
        stride_size = int(tile_size * (1 - overlap))
        # Create grid
        x = np.arange(0, shape[1], stride_size)
        y = np.arange(0, shape[0], stride_size)
        xx, yy = np.meshgrid(x, y)
    # If the image size is less than 5120*5120, we can run inference
    # on the whole image
    else:
        # Ensure the image size is multiple of 32
        # Calculate the nearest multiples of 32
        height, width = shape
        new_height = (height + 31) // 32 * 32
        new_width = (width + 31) // 32 * 32
        img = wsi.reader.get_region(0, 0, width, height, level=level)
        downsample = props.level_downsample[level]

        # We cannot read the image directly from the reader.
        # The padding from image reader will introduce padding at only two sides
        # We need to pad the image on all four sides
        # without shifting the image equilibrium
        # Otherwise, this will introduce artifacts in the segmentation

        # # Compute padding amounts
        top_pad = (new_height - height) // 2
        bottom_pad = new_height - height - top_pad
        left_pad = (new_width - width) // 2
        right_pad = new_width - width - left_pad

        # Apply padding
        img = np.pad(
            img,
            pad_width=((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0,  # Pad with black pixels
        )

        # Simulate JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        result, img = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(img, 1)

        img = torch.tensor(img).permute(2, 0, 1)

        img_t = transform(img).unsqueeze(0)
        img_t = img_t.to(device)
        pred = model.segment(img_t)

        pred = pred.squeeze().detach().cpu().numpy()
        mask = np.argmax(pred, axis=0).astype(np.uint8)
        # Flip the mask
        mask = 1 - mask
        polygons = BinaryMask(mask).to_polygons(
            min_area=1e-3,
            min_hole_area=1e-5,
            detect_holes=True,
        )
        polygons = [
            scale(p, xfact=downsample, yfact=downsample, origin=(0, 0))
            for p in polygons
        ]
        add_tissues(wsi, key_added, polygons)
