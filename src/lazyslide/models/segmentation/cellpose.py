from lazyslide.models.base import SegmentationModel


class Cellpose(SegmentationModel):
    def __init__(self, model_type="nuclei"):
        from cellpose import models

        self.cellpose_model = models.Cellpose(model_type=model_type, gpu=False)

    def to(self, device):
        self.cellpose_model.device = device

    def get_transform(self):
        return None

    def segment(self, image):
        masks, flows, styles = self.cellpose_model.eval(
            image, diameter=30, channels=[0, 0]
        )
