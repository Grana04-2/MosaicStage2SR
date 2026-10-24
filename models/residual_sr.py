import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    BatchNorm / InstanceNorm は使用しない。

    SRでは正確な輝度・色を維持したいため、
    normalizationを入れない方が扱いやすい。
    """

    def __init__(
            self,
            channels=64,
            block_res_scale=0.10
    ):
        super().__init__()

        self.block_res_scale = (
            block_res_scale
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        residual = self.conv1(
            x
        )

        residual = self.act(
            residual
        )

        residual = self.conv2(
            residual
        )

        return (
                x
                + self.block_res_scale
                * residual
        )


class ResidualSRNet(nn.Module):
    """
    Stage 1 output:
        768 x 432

    Stage 2 output:
        1280 x 720

    Scale:
        5 / 3

    処理:
        LR
         ↓
        Bicubic 5/3
         ↓
        CNNでresidual推定
         ↓
        Bicubic + residual
    """

    def __init__(
            self,
            input_channels=3,
            channels=64,
            num_blocks=8,
            block_res_scale=0.10,
            output_res_scale=0.20,
            scale_num=5,
            scale_den=3
    ):
        super().__init__()

        self.input_channels = input_channels

        self.scale_num = scale_num
        self.scale_den = scale_den

        self.output_res_scale = (
            output_res_scale
        )

        # ================================================
        # Head
        # ================================================

        self.head = nn.Conv2d(
            input_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ================================================
        # Residual trunk
        # ================================================

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    channels=channels,
                    block_res_scale=block_res_scale
                )
                for _ in range(
                    num_blocks
                )
            ]
        )

        self.body_tail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ================================================
        # RGB residual
        # ================================================

        self.tail = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                    module,
                    nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

        # ================================================
        # 最後のresidual出力だけゼロ初期化
        #
        # 学習開始時:
        # output == bicubic
        #
        # これにより初期状態で画像を壊さない。
        # ================================================

        nn.init.zeros_(
            self.tail.weight
        )

        if self.tail.bias is not None:

            nn.init.zeros_(
                self.tail.bias
            )

    def upscale_bicubic(
            self,
            lr
    ):

        height = lr.shape[-2]
        width = lr.shape[-1]

        if (
                height % self.scale_den != 0
                or
                width % self.scale_den != 0
        ):

            raise ValueError(
                "Input height/width must be "
                f"divisible by {self.scale_den}. "
                f"Got {width}x{height}"
            )

        hr_height = (
            height
            * self.scale_num
            // self.scale_den
        )

        hr_width = (
            width
            * self.scale_num
            // self.scale_den
        )

        base = F.interpolate(
            lr,
            size=(
                hr_height,
                hr_width
            ),
            mode="bicubic",
            align_corners=False
        )

        return base

    def forward(
            self,
            lr,
            mosaic=None,
            stage1_mask=None
    ):
        # ========================================================
        # Stage1 base
        # ========================================================

        base = self.upscale_bicubic(
            lr
        )

        # ========================================================
        # Input composition
        # ========================================================

        if self.input_channels == 3:

            detail_input = (
                base
            )

        elif self.input_channels == 6:

            if mosaic is None:
                raise RuntimeError(
                    "input_channels=6 requires mosaic input."
                )

            mosaic_base = (
                self.upscale_bicubic(
                    mosaic
                )
            )

            # Stage1が除去した成分
            mosaic_residual = (
                    mosaic_base
                    -
                    base
            )

            # 3ch train_A
            # +
            # 3ch (train_M - train_A)
            detail_input = torch.cat(
                [
                    base,
                    mosaic_residual,
                ],
                dim=1
            )

        elif self.input_channels == 7:

            if mosaic is None:
                raise RuntimeError(
                    "input_channels=7 requires mosaic input."
                )

            if stage1_mask is None:
                raise RuntimeError(
                    "input_channels=7 requires "
                    "stage1_mask input."
                )

            mosaic_base = (
                self.upscale_bicubic(
                    mosaic
                )
            )

            mask_base = F.interpolate(
                stage1_mask,
                size=base.shape[-2:],
                mode="nearest"
            )

            detail_input = torch.cat(
                [
                    base,
                    mosaic_base,
                    mask_base,
                ],
                dim=1
            )

        else:

            raise RuntimeError(
                "Unsupported input_channels: "
                f"{self.input_channels}"
            )

        # ========================================================
        # Safety check
        # ========================================================

        if (
                detail_input.shape[1]
                != self.input_channels
        ):
            raise RuntimeError(
                "Unexpected model input channels:\n"
                f"Expected = {self.input_channels}\n"
                f"Actual   = {detail_input.shape[1]}"
            )

        # ========================================================
        # Residual SR trunk
        # ========================================================

        features = self.head(
            detail_input
        )

        trunk = self.body(
            features
        )

        trunk = self.body_tail(
            trunk
        )

        features = (
                features
                +
                trunk
        )

        residual = self.tail(
            features
        )

        sr = (
                base
                +
                self.output_res_scale
                * residual
        )

        return sr

