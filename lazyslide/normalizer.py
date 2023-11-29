import torch
from torchvision.transforms.v2 import ToImage, ToDtype, Lambda, Compose


class Normalizer(torch.nn.Module):

    T = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Lambda(lambda x: x*255)
    ])

    def __init__(self, method="macenko"):
        super().__init__()

        import torchstain.torch.normalizers as norm

        if method == "macenko":
            normalizer = norm.TorchMacenkoNormalizer()
        elif method == "reinhard":
            normalizer = norm.TorchReinhardNormalizer()
        elif method == "reinhard_modified":
            normalizer = norm.TorchReinhardNormalizer(method="modified")
        self.normalizer = normalizer

    def forward(self, img):
        t_img = self.T(img)
        norm, _, _ = self.normalizer.normalize(I=t_img)
        return norm
