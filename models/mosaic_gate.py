import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two 3x3 convolutions without normalization.

    For this gate model we intentionally avoid BatchNorm / InstanceNorm so
    that the prediction does not depend on batch statistics and the model
    remains stable with small batch sizes.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.act = nn.LeakyReLU(
            negative_slope=0.10,
            inplace=True,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class MosaicGateNet(nn.Module):
    """
    Predict a mosaic-region probability map from only inference-time inputs.

    Input:
        Stage1 output      3ch
        Original Mosaic   3ch
        ----------------------
        total             6ch

    Output:
        logits            1ch

    Important:
        train_S is used only as the training target.
        train_S is NOT passed into this network as an input.

    Architecture:
        small U-Net style encoder/decoder.

    Upsampling uses bilinear interpolation rather than transposed convolution
    to avoid introducing checkerboard-like artifacts into the gate map.
    """

    def __init__(
            self,
            input_channels=6,
            base_channels=32,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock(
            input_channels,
            c1,
        )

        self.enc2 = ConvBlock(
            c1,
            c2,
        )

        self.enc3 = ConvBlock(
            c2,
            c3,
        )

        self.bottleneck = ConvBlock(
            c3,
            c4,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.dec3 = ConvBlock(
            c4 + c3,
            c3,
        )

        self.dec2 = ConvBlock(
            c3 + c2,
            c2,
        )

        self.dec1 = ConvBlock(
            c2 + c1,
            c1,
        )

        self.out_conv = nn.Conv2d(
            c1,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

        # Start with approximately 0.5 probability everywhere.
        nn.init.zeros_(
            self.out_conv.weight
        )

        if self.out_conv.bias is not None:
            nn.init.zeros_(
                self.out_conv.bias
            )

    @staticmethod
    def _upsample_like(x, reference):
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
            self,
            stage1,
            mosaic,
    ):
        if stage1.shape != mosaic.shape:
            raise RuntimeError(
                "Stage1 / Mosaic tensor size mismatch:\n"
                f"Stage1 = {tuple(stage1.shape)}\n"
                f"Mosaic = {tuple(mosaic.shape)}"
            )

        x = torch.cat(
            [
                stage1,
                mosaic,
            ],
            dim=1,
        )

        if x.shape[1] != self.input_channels:
            raise RuntimeError(
                "Unexpected MosaicGateNet input channels:\n"
                f"Expected = {self.input_channels}\n"
                f"Actual   = {x.shape[1]}"
            )

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self._upsample_like(
            b,
            e3,
        )

        d3 = self.dec3(
            torch.cat(
                [
                    d3,
                    e3,
                ],
                dim=1,
            )
        )

        d2 = self._upsample_like(
            d3,
            e2,
        )

        d2 = self.dec2(
            torch.cat(
                [
                    d2,
                    e2,
                ],
                dim=1,
            )
        )

        d1 = self._upsample_like(
            d2,
            e1,
        )

        d1 = self.dec1(
            torch.cat(
                [
                    d1,
                    e1,
                ],
                dim=1,
            )
        )

        logits = self.out_conv(
            d1
        )

        return logits
