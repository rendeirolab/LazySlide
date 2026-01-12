import torch
import torch.nn as nn

from .._model_registry import register
from .._utils import hf_access
from ..base import ModelTask, StyleTransferModel


@register(
    key="gigatime",
    task=ModelTask.style_transfer,
    is_gated=True,
    license="PROV-GIGATIME LICENSE",
    license_url="https://github.com/prov-gigatime/GigaTIME/blob/main/LICENSE",
    description="Multimodal AI generates virtual population for tumor microenvironment modeling",
    commercial=False,
    github_url="https://github.com/prov-gigatime/GigaTIME",
    paper_url="https://doi.org/10.1016/j.cell.2025.11.016",
    bib_key="Valanarasu2025-md",
    param_size="9M",
    flops="52.88G",
)
class GigaTIME(StyleTransferModel):
    def __init__(self, model_path: str = None, token: str = None):
        from huggingface_hub import hf_hub_download

        with hf_access("prov-gigatime/GigaTIME"):
            weights_file = hf_hub_download(
                repo_id="prov-gigatime/GigaTIME",
                filename="model.pth",
            )

        self.model = GigaTIMEModel(num_classes=23)
        self.model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image):
        return self.model(image)

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def get_channel_names(self):
        # Return channel names for the 50 protein markers
        markers = [
            "DAPI",
            "TRITC",  # background channel not used in analysis
            "Cy5",  # background channel not used in analysis
            "PD-1",
            "CD14",
            "CD4",
            "T-bet",
            "CD34",
            "CD68",
            "CD16",
            "CD11c",
            "CD138",
            "CD20",
            "CD3",
            "CD8",
            "PD-L1",
            "CK",
            "Ki67",
            "Tryptase",
            "Actin-D",
            "Caspase3-D",
            "PHH3-B",
            "Transgelin",
        ]
        return markers

    def output_shape(self):
        return 50, 256, 256

    def check_input_tile(self, mpp, size_x=None, size_y=None) -> bool:
        return True


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class GigaTIMEModel(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_2 = VGGBlock(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv2_2 = VGGBlock(
            nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2]
        )

        self.conv0_3 = VGGBlock(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_3 = VGGBlock(
            nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1]
        )

        self.conv0_4 = VGGBlock(
            nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0]
        )

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        output = self.final(x0_4)
        return output
