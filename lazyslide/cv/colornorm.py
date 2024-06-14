import lazy_loader

torch = lazy_loader.load("torch")


class ColorNormalizer(torch.nn.Module):
    from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Lambda

    T = Compose(
        [ToImage(), ToDtype(torch.float32, scale=True), Lambda(lambda x: x * 255)]
    )

    def __init__(self, method="macenko"):
        super().__init__()

        import torchstain.torch.normalizers as norm

        self.method = method
        if method == "macenko":
            normalizer = norm.TorchMacenkoNormalizer()
        elif method == "reinhard":
            normalizer = norm.TorchReinhardNormalizer()
            normalizer.target_means = torch.tensor([72.909996, 20.8268, -4.9465137])
            normalizer.target_stds = torch.tensor([18.560713, 14.889295, 5.6756697])
        elif method == "reinhard_modified":
            normalizer = norm.TorchReinhardNormalizer(method="modified")
            normalizer.target_means = torch.tensor([72.909996, 20.8268, -4.9465137])
            normalizer.target_stds = torch.tensor([18.560713, 14.889295, 5.6756697])
        else:
            raise NotImplementedError(f"Requested method '{method}' not implemented")
        self.normalizer = normalizer

    def __repr__(self):
        return f"ColorNormalizer(method='{self.method}')"

    def forward(self, img):
        t_img = self.T(img)
        if self.method == "macenko":
            norm, _, _ = self.normalizer.normalize(I=t_img)
        else:
            norm = self.normalizer.normalize(I=t_img)
        return norm
