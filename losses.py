import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def build_region_score_weight_map(
        reference: torch.Tensor,
        target: torch.Tensor,
        region_boost: float = 1.0,
        region_threshold: float = 0.01,
        region_range: float = 0.10,
        region_smooth: int = 5,
):
    """
    reference, target: [N, C, H, W], range [0, 1]
    戻り値:
        score_map  : [N, 1, H, W], 0..1
        weight_map : [N, 1, H, W], 1..(1+region_boost)
    """

    # RGB平均の絶対差を region score の元にする
    diff_map = torch.abs(reference - target).mean(dim=1, keepdim=True)

    # threshold 以下は0, threshold+range 以上は1へ
    denom = max(float(region_range), 1e-8)
    score_map = (diff_map - float(region_threshold)) / denom
    score_map = torch.clamp(score_map, 0.0, 1.0)

    # 平滑化
    k = int(region_smooth)
    if k > 1:
        if k % 2 == 0:
            k += 1
        pad = k // 2
        score_map = F.avg_pool2d(
            F.pad(score_map, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=k,
            stride=1,
        )

    weight_map = 1.0 + float(region_boost) * score_map
    return score_map, weight_map


# ============================================================
# Hard Region Weight
#
# Regionの中でも高Gradientな部分を追加強調する。
#
# base_weight:
#   従来のRegion Weight
#
# region_score:
#   build_region_weight() が返す 0..1 score
#
# target:
#   HR target
# ============================================================
@torch.no_grad()
def build_hard_region_weight(
        target,
        base_weight,
        region_score,
        hard_region_boost=0.5,
        gradient_threshold=0.02,
        gradient_range=0.04,
        smooth_kernel=3
):
    target = (
        target
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    base_weight = (
        base_weight
        .detach()
        .float()
    )

    region_score = (
        region_score
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    # ========================================================
    # RGB -> luminance
    # ========================================================

    if target.shape[1] == 3:

        luma = (
                0.299
                * target[:, 0:1]
                +
                0.587
                * target[:, 1:2]
                +
                0.114
                * target[:, 2:3]
        )

    else:

        luma = target.mean(
            dim=1,
            keepdim=True
        )

    # ========================================================
    # Gradient map
    #
    # evaluate_region_characteristics.py と近い定義
    # ========================================================

    gradient_x = (
            luma[
            :,
            :,
            :,
            1:
            ]
            -
            luma[
            :,
            :,
            :,
            :-1
            ]
    ).abs()

    gradient_y = (
            luma[
            :,
            :,
            1:,
            :
            ]
            -
            luma[
            :,
            :,
            :-1,
            :
            ]
    ).abs()

    # 元サイズへ戻す
    gradient_x = F.pad(
        gradient_x,
        (
            0,
            1,
            0,
            0
        ),
        mode="replicate"
    )

    gradient_y = F.pad(
        gradient_y,
        (
            0,
            0,
            0,
            1
        ),
        mode="replicate"
    )

    gradient_map = (
                           gradient_x
                           +
                           gradient_y
                   ) * 0.5

    # ========================================================
    # Optional smoothing
    # ========================================================

    if (
            smooth_kernel is not None
            and
            smooth_kernel > 1
    ):

        if smooth_kernel % 2 == 0:
            raise ValueError(
                "smooth_kernel must be odd."
            )

        gradient_map = F.avg_pool2d(
            gradient_map,
            kernel_size=smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2
        )

    # ========================================================
    # Gradient -> 0..1 Hardness
    #
    # threshold以下      : 0
    # threshold+range以上: 1
    # 中間              : linear
    # ========================================================

    gradient_score = (
            (
                    gradient_map
                    -
                    gradient_threshold
            )
            /
            max(
                gradient_range,
                1e-8
            )
    ).clamp(
        0.0,
        1.0
    )

    # ========================================================
    # Hard Region
    #
    # Region外では0になる。
    # ========================================================

    hard_score = (
            region_score
            *
            gradient_score
    )

    # ========================================================
    # Detail weight
    #
    # additiveにすることで暴走を抑える
    # ========================================================

    detail_weight = (
            base_weight
            +
            float(
                hard_region_boost
            )
            *
            hard_score
    )

    return (
        detail_weight,
        hard_score,
        gradient_score
    )


# ============================================================
# Charbonnier
# ============================================================

def charbonnier_map(
        prediction,
        target,
        epsilon=1e-3
):
    difference = (
            prediction
            - target
    )

    return torch.sqrt(
        difference * difference
        + epsilon * epsilon
    )


def charbonnier_distance(
        prediction,
        target,
        epsilon=1e-3
):
    return charbonnier_map(
        prediction,
        target,
        epsilon=epsilon
    ).mean()


class CharbonnierLoss(nn.Module):

    def __init__(
            self,
            epsilon=1e-3
    ):
        super().__init__()

        self.epsilon = epsilon

    def forward(
            self,
            prediction,
            target
    ):
        return charbonnier_distance(
            prediction,
            target,
            epsilon=self.epsilon
        )


# ============================================================
# Weighted mean
# ============================================================

def weighted_mean(
        value,
        weight
):
    """
    value:
        [B, C, H, W]

    weight:
        [B, 1, H, W]
        または value と同shape
    """

    if weight.shape[1] == 1:
        weight = weight.expand_as(
            value
        )

    numerator = (
            value
            * weight
    ).sum()

    denominator = (
            weight.sum()
            + 1e-8
    )

    return (
            numerator
            / denominator
    )


# ============================================================
# Region map
# ============================================================

def build_region_weight(
        bicubic,
        target,
        threshold=0.01,
        change_range=0.10,
        region_boost=2.0,
        smooth_kernel=5
):
    """
    Bicubic(Stage1 output) と clean HR の差から
    「復元が必要な領域」をsoft score化する。

    score:
        0.0 = ほぼ既に一致
        1.0 = 差が大きい

    weight:
        region_boost=1.0 の場合
        1.0 ～ 2.0

    Region map自体には勾配を流さない。
    """

    with torch.no_grad():

        difference = torch.abs(
            bicubic.detach()
            - target.detach()
        ).mean(
            dim=1,
            keepdim=True
        )

        score = (
                        difference
                        - threshold
                ) / max(
            change_range,
            1e-8
        )

        score = torch.clamp(
            score,
            0.0,
            1.0
        )

        # ================================================
        # 点単位ではなく領域として滑らかにする
        # ================================================

        if smooth_kernel > 1:

            if (
                    smooth_kernel
                    % 2
                    == 0
            ):
                raise ValueError(
                    "smooth_kernel must be odd."
                )

            score = F.avg_pool2d(
                score,
                kernel_size=smooth_kernel,
                stride=1,
                padding=smooth_kernel // 2
            )

        weight = (
                1.0
                + region_boost
                * score
        )

    return (
        weight,
        score
    )


# ============================================================
# Region-weighted Pixel loss
# ============================================================

def weighted_charbonnier_loss(
        prediction,
        target,
        weight,
        epsilon=1e-3
):
    loss_map = charbonnier_map(
        prediction,
        target,
        epsilon=epsilon
    )

    return weighted_mean(
        loss_map,
        weight
    )


# ============================================================
# Gradient
# ============================================================

def gradient_loss(
        prediction,
        target
):
    pred_dx = (
            prediction[:, :, :, 1:]
            - prediction[:, :, :, :-1]
    )

    target_dx = (
            target[:, :, :, 1:]
            - target[:, :, :, :-1]
    )

    pred_dy = (
            prediction[:, :, 1:, :]
            - prediction[:, :, :-1, :]
    )

    target_dy = (
            target[:, :, 1:, :]
            - target[:, :, :-1, :]
    )

    loss_x = charbonnier_distance(
        pred_dx,
        target_dx
    )

    loss_y = charbonnier_distance(
        pred_dy,
        target_dy
    )

    return (
            loss_x
            + loss_y
    ) * 0.5


def weighted_gradient_loss(
        prediction,
        target,
        weight
):
    pred_dx = (
            prediction[:, :, :, 1:]
            - prediction[:, :, :, :-1]
    )

    target_dx = (
            target[:, :, :, 1:]
            - target[:, :, :, :-1]
    )

    pred_dy = (
            prediction[:, :, 1:, :]
            - prediction[:, :, :-1, :]
    )

    target_dy = (
            target[:, :, 1:, :]
            - target[:, :, :-1, :]
    )

    # 隣接する2pixelのweight平均
    weight_x = (
                       weight[:, :, :, 1:]
                       + weight[:, :, :, :-1]
               ) * 0.5

    weight_y = (
                       weight[:, :, 1:, :]
                       + weight[:, :, :-1, :]
               ) * 0.5

    loss_x_map = charbonnier_map(
        pred_dx,
        target_dx
    )

    loss_y_map = charbonnier_map(
        pred_dy,
        target_dy
    )

    loss_x = weighted_mean(
        loss_x_map,
        weight_x
    )

    loss_y = weighted_mean(
        loss_y_map,
        weight_y
    )

    return (
            loss_x
            + loss_y
    ) * 0.5


# ============================================================
# RGB Laplacian
# ============================================================

def rgb_laplacian(
        x
):
    channels = (
        x.shape[1]
    )

    kernel = torch.tensor(
        [
            [
                0.0,
                -1.0,
                0.0
            ],
            [
                -1.0,
                4.0,
                -1.0
            ],
            [
                0.0,
                -1.0,
                0.0
            ]
        ],
        dtype=x.dtype,
        device=x.device
    )

    kernel = kernel.view(
        1,
        1,
        3,
        3
    )

    kernel = kernel.repeat(
        channels,
        1,
        1,
        1
    )

    padded = F.pad(
        x,
        (
            1,
            1,
            1,
            1
        ),
        mode="reflect"
    )

    return F.conv2d(
        padded,
        kernel,
        groups=channels
    )


def laplacian_loss(
        prediction,
        target
):
    pred_lap = rgb_laplacian(
        prediction
    )

    target_lap = rgb_laplacian(
        target
    )

    return charbonnier_distance(
        pred_lap,
        target_lap
    )


def weighted_laplacian_loss(
        prediction,
        target,
        weight
):
    pred_lap = rgb_laplacian(
        prediction
    )

    target_lap = rgb_laplacian(
        target
    )

    loss_map = charbonnier_map(
        pred_lap,
        target_lap
    )

    return weighted_mean(
        loss_map,
        weight
    )
