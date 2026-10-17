import argparse
import csv
import math
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.stage2_dataset import (
    load_manifest,
    split_records,
    split_records_by_sequence,
    get_sequence_id_from_stem,
    Stage2SRDataset,
)

from models.residual_sr import (
    ResidualSRNet,
)

from losses import (
    build_region_weight,
    build_hard_region_weight,
)


# ============================================================
# PSNR
# ============================================================

def psnr_from_stats(
        squared_error_sum,
        value_count
):
    if value_count <= 0:
        return float("nan")

    mse = (
            squared_error_sum
            / value_count
    )

    if mse <= 1e-12:
        return 99.0

    return (
            10.0
            * math.log10(
        1.0 / mse
    )
    )


# ============================================================
# Masked squared error
# ============================================================

def masked_error_stats(
        prediction,
        target,
        mask
):
    prediction = (
        prediction
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    target = (
        target
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    mask = mask.to(
        device=prediction.device,
        dtype=prediction.dtype
    )

    if (
            mask.shape[1] == 1
            and
            prediction.shape[1] == 3
    ):
        mask = mask.expand(
            -1,
            3,
            -1,
            -1
        )

    squared_error = (
            prediction
            - target
    ).pow(2)

    squared_error_sum = (
            squared_error
            * mask
    ).sum().item()

    value_count = (
        mask.sum().item()
    )

    return (
        squared_error_sum,
        value_count
    )


# ============================================================
# RGB -> luminance
# ============================================================

def rgb_to_luma(
        image
):
    # image:
    # [B, 3, H, W]
    #
    # 0..1 RGB -> luminance

    return (
            0.299
            * image[:, 0:1]
            +
            0.587
            * image[:, 1:2]
            +
            0.114
            * image[:, 2:3]
    )


# ============================================================
# Tensor -> grayscale numpy
# ============================================================

def tensor_to_luma_numpy(
        image
):

    luma = rgb_to_luma(
        image
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    array = (
        luma[
            0,
            0
        ]
        .cpu()
        .numpy()
    )

    return (
        array
        * 255.0
    ).astype(
        np.uint8
    )


# ============================================================
# Optical flow
#
# current -> neighbor
# ============================================================

def calculate_optical_flow(
        current_image,
        neighbor_image
):
    current_gray = (
        tensor_to_luma_numpy(
            current_image
        )
    )

    neighbor_gray = (
        tensor_to_luma_numpy(
            neighbor_image
        )
    )

    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        neighbor_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow


# ============================================================
# Warp neighbor image to current coordinates
# ============================================================

def warp_neighbor_to_current(
        neighbor_image,
        flow
):
    neighbor_np = (
        neighbor_image[
            0
        ]
        .detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0
        )
        .permute(
            1,
            2,
            0
        )
        .numpy()
        .astype(
            np.float32
        )
    )

    height, width = (
        flow.shape[
        :2
        ]
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(
            width,
            dtype=np.float32
        ),
        np.arange(
            height,
            dtype=np.float32
        )
    )

    map_x = (
            grid_x
            + flow[:, :, 0]
    )

    map_y = (
            grid_y
            + flow[:, :, 1]
    )

    warped = cv2.remap(
        neighbor_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    warped_tensor = (
        torch
        .from_numpy(
            warped
        )
        .permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
        .to(
            device=neighbor_image.device,
            dtype=torch.float32
        )
    )

    return warped_tensor


# ============================================================
# Consistency mask
# ============================================================

def build_temporal_valid_mask(
        current_bicubic,
        warped_neighbor_bicubic,
        region_mask,
        consistency_threshold=0.05
):
    current_luma = rgb_to_luma(
        current_bicubic
        .detach()
        .float()
    )

    neighbor_luma = rgb_to_luma(
        warped_neighbor_bicubic
        .detach()
        .float()
    )

    difference = (
            current_luma
            - neighbor_luma
    ).abs()

    valid_mask = (
            region_mask.bool()
            &
            (
                    difference
                    <= consistency_threshold
            )
    )

    return valid_mask


# ============================================================
# Neighbor detail availability
# ============================================================
def make_empty_detail():
    return {
        "usable_fraction":
            float("nan"),

        "current_target_gradient":
            float("nan"),

        "neighbor_target_gradient":
            float("nan"),

        "target_detail_ratio":
            float("nan"),

        "target_detail_delta":
            float("nan"),

        "current_input_gradient":
            float("nan"),

        "neighbor_input_gradient":
            float("nan"),

        "input_detail_ratio":
            float("nan"),

        "input_detail_delta":
            float("nan"),
    }


def calculate_neighbor_detail(
        current_bicubic,
        current_target,
        neighbor_bicubic,
        neighbor_target,
        region_mask,
        consistency_threshold=0.05,
        min_valid_pixels=50
):

    # ========================================================
    # Optical flow
    #
    # Flow itself is estimated only from input-side images.
    #
    # current -> neighbor
    # ========================================================

    flow = calculate_optical_flow(
        current_bicubic,
        neighbor_bicubic
    )

    # ========================================================
    # Warp neighbor INPUT image
    #
    # This is used ONLY to check temporal/alignment
    # consistency.
    # ========================================================

    warped_neighbor_bicubic = (
        warp_neighbor_to_current(
            neighbor_bicubic,
            flow
        )
    )

    # ========================================================
    # Temporal valid mask
    #
    # Region
    #   AND
    # photometrically consistent after alignment
    # ========================================================

    valid_mask = (
        build_temporal_valid_mask(
            current_bicubic,
            warped_neighbor_bicubic,
            region_mask,
            consistency_threshold
        )
    )

    region_pixels = int(
        region_mask
        .sum()
        .item()
    )

    valid_pixels = int(
        valid_mask
        .sum()
        .item()
    )

    usable_fraction = (
        valid_pixels
        /
        max(
            region_pixels,
            1
        )
    )

    if (
            valid_pixels
            <
            min_valid_pixels
    ):

        return {
            "usable_fraction":
                usable_fraction,

            "current_target_gradient":
                float("nan"),

            "neighbor_target_gradient":
                float("nan"),

            "target_detail_ratio":
                float("nan"),

            "target_detail_delta":
                float("nan"),

            "current_input_gradient":
                float("nan"),

            "neighbor_input_gradient":
                float("nan"),

            "input_detail_ratio":
                float("nan"),

            "input_detail_delta":
                float("nan"),
        }

    # ========================================================
    # TARGET gradient maps
    #
    # Important:
    #
    #   gradient(image)
    #       ->
    #   warp(gradient)
    #
    # NOT:
    #
    #   warp(image)
    #       ->
    #   gradient
    # ========================================================

    current_target_gradient_map = (
        calculate_detail_gradient_map(
            current_target
        )
    )

    neighbor_target_gradient_map = (
        calculate_detail_gradient_map(
            neighbor_target
        )
    )

    warped_neighbor_target_gradient_map = (
        warp_scalar_map_to_current(
            neighbor_target_gradient_map,
            flow
        )
    )

    # --------------------------------------------------------
    # Compare on exactly the same valid mask
    # --------------------------------------------------------

    current_target_gradient = (
        masked_map_mean(
            current_target_gradient_map,
            valid_mask
        )
    )

    neighbor_target_gradient = (
        masked_map_mean(
            warped_neighbor_target_gradient_map,
            valid_mask
        )
    )

    target_detail_ratio = (
        neighbor_target_gradient
        /
        max(
            current_target_gradient,
            1e-8
        )
    )

    target_detail_delta = (
        neighbor_target_gradient
        -
        current_target_gradient
    )

    # ========================================================
    # INPUT gradient maps
    #
    # Same procedure.
    # ========================================================

    current_input_gradient_map = (
        calculate_detail_gradient_map(
            current_bicubic
        )
    )

    neighbor_input_gradient_map = (
        calculate_detail_gradient_map(
            neighbor_bicubic
        )
    )

    warped_neighbor_input_gradient_map = (
        warp_scalar_map_to_current(
            neighbor_input_gradient_map,
            flow
        )
    )

    current_input_gradient = (
        masked_map_mean(
            current_input_gradient_map,
            valid_mask
        )
    )

    neighbor_input_gradient = (
        masked_map_mean(
            warped_neighbor_input_gradient_map,
            valid_mask
        )
    )

    input_detail_ratio = (
        neighbor_input_gradient
        /
        max(
            current_input_gradient,
            1e-8
        )
    )

    input_detail_delta = (
        neighbor_input_gradient
        -
        current_input_gradient
    )

    return {
        "usable_fraction":
            usable_fraction,

        "current_target_gradient":
            current_target_gradient,

        "neighbor_target_gradient":
            neighbor_target_gradient,

        "target_detail_ratio":
            target_detail_ratio,

        "target_detail_delta":
            target_detail_delta,

        "current_input_gradient":
            current_input_gradient,

        "neighbor_input_gradient":
            neighbor_input_gradient,

        "input_detail_ratio":
            input_detail_ratio,

        "input_detail_delta":
            input_detail_delta,
    }

# ============================================================
# Detail gradient map
#
# image:
#   [B, 3, H, W]
#
# return:
#   [B, 1, H, W]
#
# Important:
#   Gradient is calculated BEFORE temporal warping.
# ============================================================

def calculate_detail_gradient_map(
        image
):

    image = (
        image
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    luma = rgb_to_luma(
        image
    )

    gradient_x = torch.zeros_like(
        luma
    )

    gradient_y = torch.zeros_like(
        luma
    )

    # --------------------------------------------------------
    # Horizontal forward difference
    # --------------------------------------------------------

    gradient_x[
        :,
        :,
        :,
        :-1
    ] = (
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

    # --------------------------------------------------------
    # Vertical forward difference
    # --------------------------------------------------------

    gradient_y[
        :,
        :,
        :-1,
        :
    ] = (
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

    # Mean absolute spatial gradient.
    #
    # Ratio比較が主目的なので、
    # current / neighborで同じ定義を使うことが重要。
    gradient_map = (
        gradient_x
        +
        gradient_y
    ) * 0.5

    return gradient_map

# ============================================================
# Warp scalar map to current coordinates
#
# scalar_map:
#   [1, 1, H, W]
#
# flow:
#   current -> neighbor
#
# return:
#   [1, 1, H, W]
# ============================================================

def warp_scalar_map_to_current(
        scalar_map,
        flow
):

    map_np = (
        scalar_map[
            0,
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    height, width = (
        flow.shape[
            :2
        ]
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(
            width,
            dtype=np.float32
        ),
        np.arange(
            height,
            dtype=np.float32
        )
    )

    map_x = (
        grid_x
        +
        flow[
            :,
            :,
            0
        ]
    )

    map_y = (
        grid_y
        +
        flow[
            :,
            :,
            1
        ]
    )

    warped = cv2.remap(
        map_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    warped_tensor = (
        torch
        .from_numpy(
            warped
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            device=scalar_map.device,
            dtype=scalar_map.dtype
        )
    )

    return warped_tensor

# ============================================================
# Masked scalar-map mean
# ============================================================

def masked_map_mean(
        value_map,
        mask
):

    value_map = (
        value_map
        .detach()
        .float()
    )

    mask = (
        mask
        .detach()
        .bool()
    )

    if mask.dim() == 3:

        mask = mask.unsqueeze(
            1
        )

    if not mask.any():

        return float(
            "nan"
        )

    return (
        value_map[
            mask
        ]
        .mean()
        .item()
    )

# ============================================================
# Masked gradient mean
# ============================================================

def calculate_masked_gradient_mean(
        image,
        mask
):
    image = (
        image
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    mask = (
        mask
        .detach()
        .bool()
    )

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if not mask.any():
        return float("nan")

    luma = rgb_to_luma(
        image
    )

    dx = (
            luma[:, :, :, 1:]
            -
            luma[:, :, :, :-1]
    ).abs()

    dy = (
            luma[:, :, 1:, :]
            -
            luma[:, :, :-1, :]
    ).abs()

    mask_x = (
            mask[:, :, :, 1:]
            &
            mask[:, :, :, :-1]
    )

    mask_y = (
            mask[:, :, 1:, :]
            &
            mask[:, :, :-1, :]
    )

    grad_sum = 0.0
    grad_count = 0

    if mask_x.any():
        grad_sum += (
            dx[
                mask_x
            ]
            .sum()
            .item()
        )

        grad_count += int(
            mask_x
            .sum()
            .item()
        )

    if mask_y.any():
        grad_sum += (
            dy[
                mask_y
            ]
            .sum()
            .item()
        )

        grad_count += int(
            mask_y
            .sum()
            .item()
        )

    if grad_count <= 0:
        return float("nan")

    return (
            grad_sum
            / grad_count
    )


# ============================================================
# Neighbor stem
# ============================================================

def get_neighbor_stems(
        stem
):
    try:

        sequence_id, frame_text = (
            stem.rsplit(
                "_",
                1
            )
        )

        if not frame_text.isdigit():
            return None, None

        frame_number = int(
            frame_text
        )

        width = len(
            frame_text
        )

        prev_stem = None

        if frame_number > 0:
            prev_stem = (
                f"{sequence_id}_"
                f"{frame_number - 1:0{width}d}"
            )

        next_stem = (
            f"{sequence_id}_"
            f"{frame_number + 1:0{width}d}"
        )

        return (
            prev_stem,
            next_stem
        )

    except ValueError:

        return (
            None,
            None
        )


# ============================================================
# Region masked temporal difference
# ============================================================

def masked_luma_difference(
        image_a,
        image_b,
        region_mask
):
    image_a = (
        image_a
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    image_b = (
        image_b
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    luma_a = rgb_to_luma(
        image_a
    )

    luma_b = rgb_to_luma(
        image_b
    )

    mask = (
        region_mask
        .detach()
        .bool()
    )

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if not mask.any():
        return float("nan")

    difference = (
            luma_a
            - luma_b
    ).abs()

    return (
        difference[
            mask
        ]
        .mean()
        .item()
    )


# ============================================================
# Partial correlation
# ============================================================

def calculate_partial_correlation(
        x,
        y,
        controls
):
    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    controls = np.asarray(
        controls,
        dtype=np.float64
    )

    if controls.ndim == 1:
        controls = (
            controls[
            :,
            None
            ]
        )

    valid = (
            np.isfinite(
                x
            )
            &
            np.isfinite(
                y
            )
            &
            np.all(
                np.isfinite(
                    controls
                ),
                axis=1
            )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    controls = controls[
        valid
    ]

    if len(
            x
    ) < 3:
        return float("nan")

    # intercept
    design = np.column_stack(
        [
            np.ones(
                len(
                    x
                )
            ),
            controls
        ]
    )

    beta_x = np.linalg.lstsq(
        design,
        x,
        rcond=None
    )[0]

    beta_y = np.linalg.lstsq(
        design,
        y,
        rcond=None
    )[0]

    residual_x = (
            x
            - design
            @ beta_x
    )

    residual_y = (
            y
            - design
            @ beta_y
    )

    if (
            np.std(
                residual_x
            )
            <= 1e-12
            or
            np.std(
                residual_y
            )
            <= 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            residual_x,
            residual_y
        )[0, 1]
    )

# ============================================================
# Hard Region / Detail Weight statistics
# ============================================================

@torch.no_grad()
def calculate_hard_weight_statistics(
        hard_score,
        region_weight_map,
        detail_weight_map,
        region_mask
):
    hard_score = (
        hard_score
        .detach()
        .float()
    )

    region_weight_map = (
        region_weight_map
        .detach()
        .float()
    )

    detail_weight_map = (
        detail_weight_map
        .detach()
        .float()
    )

    region_mask = (
        region_mask
        .detach()
        .bool()
    )

    if region_mask.dim() == 3:
        region_mask = region_mask.unsqueeze(
            1
        )

    # ========================================================
    # Whole image
    # ========================================================

    map_pixel_count = int(
        hard_score.numel()
    )

    hard_score_mean = float(
        hard_score.mean().item()
    )

    hard_positive_fraction = float(
        (
            hard_score > 0.0
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    hard_gt_0p5_fraction = float(
        (
            hard_score > 0.5
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    region_weight_mean = float(
        region_weight_map.mean().item()
    )

    detail_weight_mean = float(
        detail_weight_map.mean().item()
    )

    # ========================================================
    # Evaluation Region only
    # ========================================================

    region_pixel_count = int(
        region_mask.sum().item()
    )

    if region_pixel_count > 0:

        hard_region_values = (
            hard_score[
                region_mask
            ]
        )

        region_weight_values = (
            region_weight_map[
                region_mask
            ]
        )

        detail_weight_values = (
            detail_weight_map[
                region_mask
            ]
        )

        hard_region_mean = float(
            hard_region_values
            .mean()
            .item()
        )

        hard_region_positive_fraction = float(
            (
                hard_region_values > 0.0
            )
            .float()
            .mean()
            .item()
            * 100.0
        )

        hard_region_gt_0p5_fraction = float(
            (
                hard_region_values > 0.5
            )
            .float()
            .mean()
            .item()
            * 100.0
        )

        region_region_weight_mean = float(
            region_weight_values
            .mean()
            .item()
        )

        region_detail_weight_mean = float(
            detail_weight_values
            .mean()
            .item()
        )

        region_extra_weight_mean = (
            region_detail_weight_mean
            -
            region_region_weight_mean
        )

    else:

        hard_region_mean = float(
            "nan"
        )

        hard_region_positive_fraction = float(
            "nan"
        )

        hard_region_gt_0p5_fraction = float(
            "nan"
        )

        region_region_weight_mean = float(
            "nan"
        )

        region_detail_weight_mean = float(
            "nan"
        )

        region_extra_weight_mean = float(
            "nan"
        )

    return {
        "map_pixel_count":
            map_pixel_count,

        "hard_score_mean":
            hard_score_mean,

        "hard_positive_fraction_percent":
            hard_positive_fraction,

        "hard_gt_0p5_fraction_percent":
            hard_gt_0p5_fraction,

        "hard_region_mean":
            hard_region_mean,

        "hard_region_positive_fraction_percent":
            hard_region_positive_fraction,

        "hard_region_gt_0p5_fraction_percent":
            hard_region_gt_0p5_fraction,

        "region_weight_mean":
            region_weight_mean,

        "detail_weight_mean":
            detail_weight_mean,

        "region_region_weight_mean":
            region_region_weight_mean,

        "region_detail_weight_mean":
            region_detail_weight_mean,

        "region_extra_weight_mean":
            region_extra_weight_mean,
    }
# ============================================================
# Region characteristics
# ============================================================

def calculate_region_characteristics(
        target,
        region_mask,
        dark_threshold=0.15
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

    region_mask = (
        region_mask
        .detach()
        .bool()
    )

    if region_mask.dim() == 3:
        region_mask = (
            region_mask
            .unsqueeze(1)
        )

    luma = rgb_to_luma(
        target
    )

    # ========================================================
    # Region pixel count
    # ========================================================

    region_pixel_count = int(
        region_mask
        .sum()
        .item()
    )

    total_pixel_count = (
        region_mask.numel()
    )

    region_coverage = (
            100.0
            * region_pixel_count
            / max(
        total_pixel_count,
        1
    )
    )

    if region_pixel_count <= 0:
        return {
            "region_pixel_count":
                0,

            "region_coverage_percent":
                0.0,

            "luma_mean":
                float("nan"),

            "luma_std":
                float("nan"),

            "dark_fraction":
                float("nan"),

            "gradient_mean":
                float("nan"),

            "laplacian_var":
                float("nan"),
        }

    # ========================================================
    # Luminance statistics
    # ========================================================

    region_luma = (
        luma[
            region_mask
        ]
    )

    luma_mean = (
        region_luma
        .mean()
        .item()
    )

    luma_std = (
        region_luma
        .std(
            unbiased=False
        )
        .item()
    )

    dark_fraction = (
        (
                region_luma
                < dark_threshold
        )
        .float()
        .mean()
        .item()
    )

    # ========================================================
    # Gradient
    #
    # 両方の画素がRegion内である境界だけ測定
    # ========================================================

    dx = (
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

    dy = (
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

    mask_x = (
            region_mask[
            :,
            :,
            :,
            1:
            ]
            &
            region_mask[
            :,
            :,
            :,
            :-1
            ]
    )

    mask_y = (
            region_mask[
            :,
            :,
            1:,
            :
            ]
            &
            region_mask[
            :,
            :,
            :-1,
            :
            ]
    )

    grad_sum = 0.0
    grad_count = 0

    if mask_x.any():
        grad_sum += (
            dx[
                mask_x
            ]
            .sum()
            .item()
        )

        grad_count += int(
            mask_x
            .sum()
            .item()
        )

    if mask_y.any():
        grad_sum += (
            dy[
                mask_y
            ]
            .sum()
            .item()
        )

        grad_count += int(
            mask_y
            .sum()
            .item()
        )

    if grad_count > 0:

        gradient_mean = (
                grad_sum
                / grad_count
        )

    else:

        gradient_mean = (
            float("nan")
        )

    # ========================================================
    # Laplacian variance
    #
    # 値が大きい:
    #   sharp / high-frequency
    #
    # 値が小さい:
    #   smooth / blur
    # ========================================================

    laplacian_kernel = torch.tensor(
        [
            [
                [
                    [0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        ],
        dtype=luma.dtype,
        device=luma.device
    )

    luma_padded = F.pad(
        luma,
        (
            1,
            1,
            1,
            1
        ),
        mode="replicate"
    )

    laplacian = F.conv2d(
        luma_padded,
        laplacian_kernel
    )

    region_laplacian = (
        laplacian[
            region_mask
        ]
    )

    if (
            region_laplacian
                    .numel()
            > 1
    ):

        laplacian_var = (
            region_laplacian
            .var(
                unbiased=False
            )
            .item()
        )

    else:

        laplacian_var = (
            float("nan")
        )

    return {
        "region_pixel_count":
            region_pixel_count,

        "region_coverage_percent":
            region_coverage,

        "luma_mean":
            luma_mean,

        "luma_std":
            luma_std,

        "dark_fraction":
            dark_fraction,

        "gradient_mean":
            gradient_mean,

        "laplacian_var":
            laplacian_var,
    }


# ============================================================
# Pearson correlation
# ============================================================

def calculate_pearson(
        x,
        y
):
    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    valid = (
            np.isfinite(
                x
            )
            &
            np.isfinite(
                y
            )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if len(
            x
    ) < 2:
        return float("nan")

    if (
            np.std(
                x
            )
            <= 1e-12
            or
            np.std(
                y
            )
            <= 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


# ============================================================
# Quartile analysis
# ============================================================

def print_rank_quartiles(
        rows,
        metric_key,
        label
):
    valid_rows = [
        row
        for row in rows
        if (
                math.isfinite(
                    row[
                        metric_key
                    ]
                )
                and
                math.isfinite(
                    row[
                        "region_gain"
                    ]
                )
        )
    ]

    if len(
            valid_rows
    ) < 4:
        return

    valid_rows = sorted(
        valid_rows,
        key=lambda row:
        row[
            metric_key
        ]
    )

    groups = np.array_split(
        np.array(
            valid_rows,
            dtype=object
        ),
        4
    )

    print()
    print(
        f"{label} quartiles"
    )

    print(
        "----------------------------------------"
    )

    for group_index, group in enumerate(
            groups,
            start=1
    ):
        group_rows = list(
            group
        )

        metrics = np.array(
            [
                row[
                    metric_key
                ]
                for row in group_rows
            ],
            dtype=np.float64
        )

        gains = np.array(
            [
                row[
                    "region_gain"
                ]
                for row in group_rows
            ],
            dtype=np.float64
        )

        improved_percent = (
                100.0
                * np.mean(
            gains > 0.0
        )
        )

        print(
            f"Q{group_index} | "
            f"{metric_key} "
            f"{metrics.min():.6f} - "
            f"{metrics.max():.6f} | "
            f"N {len(group_rows):3d} | "
            f"Gain mean "
            f"{gains.mean():+.4f} dB | "
            f"median "
            f"{np.median(gains):+.4f} dB | "
            f"improved "
            f"{improved_percent:.1f}%"
        )


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./datasets/stage2_1280x720"
    )

    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--dark_threshold",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--min_region_pixels",
        type=int,
        default=100
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    return parser.parse_args()

# ============================================================
# Sequence Hard Region statistics
# ============================================================

def print_sequence_hard_statistics(
        rows
):
    sequence_groups = {}

    for row in rows:

        sequence_id = row[
            "sequence_id"
        ]

        if sequence_id not in sequence_groups:
            sequence_groups[
                sequence_id
            ] = []

        sequence_groups[
            sequence_id
        ].append(
            row
        )

    print()
    print(
        "========================================"
    )
    print(
        "Hard Region statistics by sequence"
    )
    print(
        "========================================"
    )

    for sequence_id in sorted(
            sequence_groups.keys()
    ):

        sequence_rows = (
            sequence_groups[
                sequence_id
            ]
        )

        total_pixels = sum(
            row[
                "map_pixel_count"
            ]
            for row in sequence_rows
        )

        region_pixels = sum(
            row[
                "region_pixel_count"
            ]
            for row in sequence_rows
        )

        total_pixels = max(
            total_pixels,
            1
        )

        region_pixels_safe = max(
            region_pixels,
            1
        )

        # ====================================================
        # Full-frame pixel-weighted statistics
        # ====================================================

        hard_mean = (
            sum(
                row[
                    "hard_score_mean"
                ]
                *
                row[
                    "map_pixel_count"
                ]
                for row in sequence_rows
            )
            /
            total_pixels
        )

        hard_positive = (
            sum(
                row[
                    "hard_positive_fraction_percent"
                ]
                / 100.0
                *
                row[
                    "map_pixel_count"
                ]
                for row in sequence_rows
            )
            /
            total_pixels
            *
            100.0
        )

        hard_gt_0p5 = (
            sum(
                row[
                    "hard_gt_0p5_fraction_percent"
                ]
                / 100.0
                *
                row[
                    "map_pixel_count"
                ]
                for row in sequence_rows
            )
            /
            total_pixels
            *
            100.0
        )

        region_weight_mean = (
            sum(
                row[
                    "region_weight_mean"
                ]
                *
                row[
                    "map_pixel_count"
                ]
                for row in sequence_rows
            )
            /
            total_pixels
        )

        detail_weight_mean = (
            sum(
                row[
                    "detail_weight_mean"
                ]
                *
                row[
                    "map_pixel_count"
                ]
                for row in sequence_rows
            )
            /
            total_pixels
        )

        # ====================================================
        # Region-only pixel-weighted statistics
        # ====================================================

        hard_region_mean = (
            sum(
                row[
                    "hard_region_mean"
                ]
                *
                row[
                    "region_pixel_count"
                ]
                for row in sequence_rows
                if math.isfinite(
                    row[
                        "hard_region_mean"
                    ]
                )
            )
            /
            region_pixels_safe
        )

        hard_region_positive = (
            sum(
                row[
                    "hard_region_positive_fraction_percent"
                ]
                / 100.0
                *
                row[
                    "region_pixel_count"
                ]
                for row in sequence_rows
                if math.isfinite(
                    row[
                        "hard_region_positive_fraction_percent"
                    ]
                )
            )
            /
            region_pixels_safe
            *
            100.0
        )

        hard_region_gt_0p5 = (
            sum(
                row[
                    "hard_region_gt_0p5_fraction_percent"
                ]
                / 100.0
                *
                row[
                    "region_pixel_count"
                ]
                for row in sequence_rows
                if math.isfinite(
                    row[
                        "hard_region_gt_0p5_fraction_percent"
                    ]
                )
            )
            /
            region_pixels_safe
            *
            100.0
        )

        region_base_weight = (
            sum(
                row[
                    "region_region_weight_mean"
                ]
                *
                row[
                    "region_pixel_count"
                ]
                for row in sequence_rows
                if math.isfinite(
                    row[
                        "region_region_weight_mean"
                    ]
                )
            )
            /
            region_pixels_safe
        )

        region_detail_weight = (
            sum(
                row[
                    "region_detail_weight_mean"
                ]
                *
                row[
                    "region_pixel_count"
                ]
                for row in sequence_rows
                if math.isfinite(
                    row[
                        "region_detail_weight_mean"
                    ]
                )
            )
            /
            region_pixels_safe
        )

        region_extra_weight = (
            region_detail_weight
            -
            region_base_weight
        )

        # ====================================================
        # Model result
        # ====================================================

        gains = np.array(
            [
                row[
                    "region_gain"
                ]
                for row in sequence_rows
                if math.isfinite(
                    row[
                        "region_gain"
                    ]
                )
            ],
            dtype=np.float64
        )

        gain_mean = (
            float(
                gains.mean()
            )
            if gains.size > 0
            else float("nan")
        )

        print()
        print(
            f"Sequence {sequence_id}"
        )

        print(
            f"  Frames            : "
            f"{len(sequence_rows)}"
        )

        print(
            f"  Region pixels     : "
            f"{region_pixels}"
        )

        print(
            f"  Region Gain mean  : "
            f"{gain_mean:+.4f} dB"
        )

        print(
            f"  Hard mean         : "
            f"{hard_mean:.6f}"
        )

        print(
            f"  Hard > 0          : "
            f"{hard_positive:.3f} %"
        )

        print(
            f"  Hard > 0.5        : "
            f"{hard_gt_0p5:.3f} %"
        )

        print(
            f"  Region Hard mean  : "
            f"{hard_region_mean:.6f}"
        )

        print(
            f"  Region Hard > 0   : "
            f"{hard_region_positive:.2f} %"
        )

        print(
            f"  Region Hard > 0.5 : "
            f"{hard_region_gt_0p5:.2f} %"
        )

        print(
            f"  Weight mean       : "
            f"{region_weight_mean:.6f}"
        )

        print(
            f"  DetailW mean      : "
            f"{detail_weight_mean:.6f}"
        )

        print(
            f"  Region Weight     : "
            f"{region_base_weight:.6f}"
        )

        print(
            f"  Region DetailW    : "
            f"{region_detail_weight:.6f}"
        )

        print(
            f"  Region extra W    : "
            f"{region_extra_weight:+.6f}"
        )
# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    checkpoint_args = (
        checkpoint.get(
            "args",
            {}
        )
    )

    checkpoint_epoch = (
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    # ========================================================
    # Training parameters
    # ========================================================

    channels = (
        checkpoint_args.get(
            "channels",
            64
        )
    )

    blocks = (
        checkpoint_args.get(
            "blocks",
            8
        )
    )

    lr_patch = (
        checkpoint_args.get(
            "lr_patch",
            192
        )
    )

    val_ratio = (
        checkpoint_args.get(
            "val_ratio",
            0.10
        )
    )

    seed = (
        checkpoint_args.get(
            "seed",
            1234
        )
    )

    split_mode = (
        checkpoint_args.get(
            "split_mode",
            "frame"
        )
    )

    region_boost = (
        checkpoint_args.get(
            "region_boost",
            1.0
        )
    )

    region_threshold = (
        checkpoint_args.get(
            "region_threshold",
            0.01
        )
    )

    region_range = (
        checkpoint_args.get(
            "region_range",
            0.10
        )
    )

    region_smooth = (
        checkpoint_args.get(
            "region_smooth",
            5
        )
    )

    hard_region_boost = (
        checkpoint_args.get(
            "hard_region_boost",
            0.0
        )
    )

    hard_gradient_threshold = (
        checkpoint_args.get(
            "hard_gradient_threshold",
            0.02
        )
    )

    hard_gradient_range = (
        checkpoint_args.get(
            "hard_gradient_range",
            0.04
        )
    )

    hard_gradient_smooth = (
        checkpoint_args.get(
            "hard_gradient_smooth",
            3
        )
    )

    print()
    print(
        "Region parameters"
    )

    print()
    print(
        "Hard Region parameters"
    )

    print(
        f"  boost           : "
        f"{hard_region_boost}"
    )

    print(
        f"  grad threshold  : "
        f"{hard_gradient_threshold}"
    )

    print(
        f"  grad range      : "
        f"{hard_gradient_range}"
    )

    print(
        f"  grad smooth     : "
        f"{hard_gradient_smooth}"
    )

    print(
        f"  train threshold : "
        f"{region_threshold}"
    )

    print(
        f"  range           : "
        f"{region_range}"
    )

    print(
        f"  smooth          : "
        f"{region_smooth}"
    )

    print(
        f"  boost           : "
        f"{region_boost}"
    )

    print(
        f"  eval threshold  : "
        f"{args.eval_threshold}"
    )

    print(
        f"  dark threshold  : "
        f"{args.dark_threshold}"
    )

    print(
        f"  min pixels      : "
        f"{args.min_region_pixels}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    records = load_manifest(
        args.dataset_root
    )

    if split_mode == "sequence":

        (
            train_records,
            val_records
        ) = split_records_by_sequence(
            records,
            val_ratio=val_ratio,
            seed=seed
        )

    else:

        (
            train_records,
            val_records
        ) = split_records(
            records,
            val_ratio=val_ratio,
            seed=seed
        )

    print()
    print(
        f"Total pairs : "
        f"{len(records)}"
    )

    print(
        f"Train pairs : "
        f"{len(train_records)}"
    )

    print(
        f"Val pairs   : "
        f"{len(val_records)}"
    )

    val_dataset = Stage2SRDataset(
        dataset_root=args.dataset_root,
        records=val_records,
        train=False,
        lr_patch_size=lr_patch,
        horizontal_flip=False
    )

    # ========================================================
    # Full dataset for temporal neighbors
    # ========================================================

    all_dataset = Stage2SRDataset(
        dataset_root=args.dataset_root,
        records=records,
        train=False,
        lr_patch_size=lr_patch,
        horizontal_flip=False
    )

    stem_to_index = {
        record[
            "stem"
        ]:
            index

        for index, record
        in enumerate(
            records
        )
    }

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
                device.type
                == "cuda"
        )
    )

    # ========================================================
    # Model
    # ========================================================

    model = ResidualSRNet(
        channels=channels,
        num_blocks=blocks,
        block_res_scale=0.10,
        output_res_scale=0.20,
        scale_num=5,
        scale_den=3
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ],
        strict=True
    )

    model.eval()

    # ========================================================
    # Evaluate
    # ========================================================

    rows = []

    for batch_index, batch in enumerate(
            val_loader,
            start=1
    ):

        stem = (
            batch[
                "stem"
            ][0]
        )

        lr = (
            batch[
                "lr"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        hr = (
            batch[
                "hr"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        sr = (
            model(
                lr
            )
            .clamp(
                0.0,
                1.0
            )
        )

        bicubic = (
            model.upscale_bicubic(
                lr
            )
            .clamp(
                0.0,
                1.0
            )
        )

        # ====================================================
        # Region Weight
        # ====================================================

        (
            region_weight_map,
            score_map
        ) = build_region_weight(
            bicubic=bicubic,
            target=hr,
            threshold=region_threshold,
            change_range=region_range,
            region_boost=region_boost,
            smooth_kernel=region_smooth
        )

        region_mask = (
                score_map
                >= args.eval_threshold
        )

        # ====================================================
        # Hard Region / Detail Weight
        #
        # Control checkpointでもHard score自体は計算する。
        # hard_region_boost=0なら
        # detail_weight == region_weight。
        # ====================================================

        (
            detail_weight_map,
            hard_score,
            _
        ) = build_hard_region_weight(
            target=hr,
            base_weight=region_weight_map,
            region_score=score_map,
            hard_region_boost=hard_region_boost,
            gradient_threshold=hard_gradient_threshold,
            gradient_range=hard_gradient_range,
            smooth_kernel=hard_gradient_smooth
        )

        hard_statistics = (
            calculate_hard_weight_statistics(
                hard_score=hard_score,
                region_weight_map=region_weight_map,
                detail_weight_map=detail_weight_map,
                region_mask=region_mask
            )
        )

        # ====================================================
        # Temporal motion
        # ====================================================

        (
            prev_stem,
            next_stem
        ) = get_neighbor_stems(
            stem
        )

        motion_input_values = []
        motion_target_values = []

        motion_input_prev = float(
            "nan"
        )

        motion_input_next = float(
            "nan"
        )

        motion_target_prev = float(
            "nan"
        )

        motion_target_next = float(
            "nan"
        )

        neighbor_count = 0

        prev_detail = make_empty_detail()
        next_detail = make_empty_detail()

        # ----------------------------------------------------
        # Previous frame
        # ----------------------------------------------------

        if (
                prev_stem is not None
                and
                prev_stem in stem_to_index
        ):

            prev_item = all_dataset[
                stem_to_index[
                    prev_stem
                ]
            ]

            prev_lr = (
                prev_item[
                    "lr"
                ]
                .unsqueeze(0)
                .to(
                    device
                )
            )

            prev_hr = (
                prev_item[
                    "hr"
                ]
                .unsqueeze(0)
                .to(
                    device
                )
            )

            prev_bicubic = (
                model.upscale_bicubic(
                    prev_lr
                )
                .clamp(
                    0.0,
                    1.0
                )
            )

            prev_detail = (
                calculate_neighbor_detail(
                    current_bicubic=bicubic,
                    current_target=hr,
                    neighbor_bicubic=prev_bicubic,
                    neighbor_target=prev_hr,
                    region_mask=region_mask,
                    consistency_threshold=0.05,
                    min_valid_pixels=50
                )
            )

            motion_input_prev = (
                masked_luma_difference(
                    bicubic,
                    prev_bicubic,
                    region_mask
                )
            )

            motion_target_prev = (
                masked_luma_difference(
                    hr,
                    prev_hr,
                    region_mask
                )
            )

            if math.isfinite(
                    motion_input_prev
            ):
                motion_input_values.append(
                    motion_input_prev
                )

            if math.isfinite(
                    motion_target_prev
            ):
                motion_target_values.append(
                    motion_target_prev
                )

            neighbor_count += 1

        # ----------------------------------------------------
        # Next frame
        # ----------------------------------------------------

        if (
                next_stem is not None
                and
                next_stem in stem_to_index
        ):

            next_item = all_dataset[
                stem_to_index[
                    next_stem
                ]
            ]

            next_lr = (
                next_item[
                    "lr"
                ]
                .unsqueeze(0)
                .to(
                    device
                )
            )

            next_hr = (
                next_item[
                    "hr"
                ]
                .unsqueeze(0)
                .to(
                    device
                )
            )

            next_bicubic = (
                model.upscale_bicubic(
                    next_lr
                )
                .clamp(
                    0.0,
                    1.0
                )
            )

            next_detail = (
                calculate_neighbor_detail(
                    current_bicubic=bicubic,
                    current_target=hr,
                    neighbor_bicubic=next_bicubic,
                    neighbor_target=next_hr,
                    region_mask=region_mask,
                    consistency_threshold=0.05,
                    min_valid_pixels=50
                )
            )

            motion_input_next = (
                masked_luma_difference(
                    bicubic,
                    next_bicubic,
                    region_mask
                )
            )

            motion_target_next = (
                masked_luma_difference(
                    hr,
                    next_hr,
                    region_mask
                )
            )

            if math.isfinite(
                    motion_input_next
            ):
                motion_input_values.append(
                    motion_input_next
                )

            if math.isfinite(
                    motion_target_next
            ):
                motion_target_values.append(
                    motion_target_next
                )

            neighbor_count += 1
        # ====================================================
        # Best neighbor detail
        # ====================================================

        detail_candidates = []

        if math.isfinite(
                prev_detail[
                    "target_detail_ratio"
                ]
        ):
            detail_candidates.append(
                prev_detail
            )

        if math.isfinite(
                next_detail[
                    "target_detail_ratio"
                ]
        ):
            detail_candidates.append(
                next_detail
            )

        if detail_candidates:

            best_detail = max(
                detail_candidates,
                key=lambda item:
                item[
                    "target_detail_ratio"
                ]
            )

            best_target_detail_ratio = (
                best_detail[
                    "target_detail_ratio"
                ]
            )

            best_target_detail_delta = (
                best_detail[
                    "target_detail_delta"
                ]
            )

            best_input_detail_ratio = (
                best_detail[
                    "input_detail_ratio"
                ]
            )

            best_usable_fraction = (
                best_detail[
                    "usable_fraction"
                ]
            )

        else:

            best_target_detail_ratio = float(
                "nan"
            )

            best_target_detail_delta = float(
                "nan"
            )

            best_input_detail_ratio = float(
                "nan"
            )

            best_usable_fraction = float(
                "nan"
            )

        # ----------------------------------------------------
        # Mean temporal motion
        # ----------------------------------------------------

        if len(
                motion_input_values
        ) > 0:

            motion_input_score = float(
                np.mean(
                    motion_input_values
                )
            )

        else:

            motion_input_score = float(
                "nan"
            )

        if len(
                motion_target_values
        ) > 0:

            motion_target_score = float(
                np.mean(
                    motion_target_values
                )
            )

        else:

            motion_target_score = float(
                "nan"
            )

        # ====================================================
        # Region PSNR
        # ====================================================

        (
            sr_error,
            sr_count
        ) = masked_error_stats(
            prediction=sr,
            target=hr,
            mask=region_mask
        )

        (
            bicubic_error,
            bicubic_count
        ) = masked_error_stats(
            prediction=bicubic,
            target=hr,
            mask=region_mask
        )

        sr_region_psnr = (
            psnr_from_stats(
                sr_error,
                sr_count
            )
        )

        bicubic_region_psnr = (
            psnr_from_stats(
                bicubic_error,
                bicubic_count
            )
        )

        region_gain = (
                sr_region_psnr
                - bicubic_region_psnr
        )

        # ====================================================
        # Target Region characteristics
        # ====================================================

        characteristics = (
            calculate_region_characteristics(
                target=hr,
                region_mask=region_mask,
                dark_threshold=args.dark_threshold
            )
        )

        sequence_id = (
            get_sequence_id_from_stem(
                stem
            )
        )

        row = {
            "stem":
                stem,

            "sequence_id":
                sequence_id,

            "bicubic_region_psnr":
                bicubic_region_psnr,

            "sr_region_psnr":
                sr_region_psnr,

            "region_gain":
                region_gain,

            "prev_stem":
                prev_stem,

            "next_stem":
                next_stem,

            "neighbor_count":
                neighbor_count,

            "motion_input_prev":
                motion_input_prev,

            "motion_input_next":
                motion_input_next,

            "motion_input_score":
                motion_input_score,

            "motion_target_prev":
                motion_target_prev,

            "motion_target_next":
                motion_target_next,

            "motion_target_score":
                motion_target_score,

            "prev_target_detail_ratio":
                prev_detail["target_detail_ratio"],

            "next_target_detail_ratio":
                next_detail["target_detail_ratio"],

            "best_target_detail_ratio":
                best_target_detail_ratio,

            "best_target_detail_delta":
                best_target_detail_delta,

            "best_input_detail_ratio":
                best_input_detail_ratio,

            "best_usable_fraction":
                best_usable_fraction,

            "prev_input_detail_ratio":
                prev_detail["input_detail_ratio"],

            "next_input_detail_ratio":
                next_detail["input_detail_ratio"],

            "prev_usable_fraction":
                prev_detail["usable_fraction"],

            "next_usable_fraction":
                next_detail["usable_fraction"],
            **hard_statistics,
            **characteristics,
        }

        rows.append(
            row
        )

        if (
                batch_index % 50 == 0
                or
                batch_index
                == len(
            val_loader
        )
        ):
            print(
                f"Evaluate "
                f"{batch_index:04d}/"
                f"{len(val_loader):04d}"
            )

    # ========================================================
    # CSV
    # ========================================================

    threshold_text = (
        f"{args.eval_threshold:.3f}"
        .replace(
            ".",
            "p"
        )
    )

    csv_path = (
            checkpoint_path.parent
            / (
                f"{checkpoint_path.stem}"
                f"_region_characteristics_"
                f"{threshold_text}.csv"
            )
    )

    fieldnames = [
        "stem",
        "sequence_id",

        "region_pixel_count",
        "region_coverage_percent",

        "bicubic_region_psnr",
        "sr_region_psnr",
        "region_gain",

        "luma_mean",
        "luma_std",
        "dark_fraction",
        "gradient_mean",
        "laplacian_var",

        "map_pixel_count",

        "hard_score_mean",
        "hard_positive_fraction_percent",
        "hard_gt_0p5_fraction_percent",

        "hard_region_mean",
        "hard_region_positive_fraction_percent",
        "hard_region_gt_0p5_fraction_percent",

        "region_weight_mean",
        "detail_weight_mean",

        "region_region_weight_mean",
        "region_detail_weight_mean",
        "region_extra_weight_mean",
        "prev_stem",
        "next_stem",
        "neighbor_count",

        "motion_input_prev",
        "motion_input_next",
        "motion_input_score",

        "motion_target_prev",
        "motion_target_next",
        "motion_target_score",
        "prev_target_detail_ratio",
        "next_target_detail_ratio",

        "best_target_detail_ratio",
        "best_target_detail_delta",

        "best_input_detail_ratio",
        "best_usable_fraction",

        "prev_input_detail_ratio",
        "next_input_detail_ratio",
        "prev_usable_fraction",
        "next_usable_fraction",

    ]

    with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )

    # ========================================================
    # Valid rows
    #
    # 極端に小さいRegionは相関を不安定にするので、
    # min_region_pixels以上のみ統計に使用
    # ========================================================

    valid_rows = [
        row
        for row in rows
        if (
                row[
                    "region_pixel_count"
                ]
                >= args.min_region_pixels
                and
                math.isfinite(
                    row[
                        "region_gain"
                    ]
                )
        )
    ]

    print_sequence_hard_statistics(
        valid_rows
    )

    print()
    print(
        "========================================"
    )

    print(
        "Region characteristic analysis"
    )

    print(
        "========================================"
    )

    print(
        f"All images      : "
        f"{len(rows)}"
    )

    print(
        f"Valid images    : "
        f"{len(valid_rows)}"
    )

    print(
        f"Min region pixel: "
        f"{args.min_region_pixels}"
    )

    # ========================================================
    # Correlations
    # ========================================================

    gains = [
        row[
            "region_gain"
        ]
        for row in valid_rows
    ]

    metric_keys = [
        (
            "region_coverage_percent",
            "Coverage"
        ),
        (
            "luma_mean",
            "Luma mean"
        ),
        (
            "luma_std",
            "Luma std"
        ),
        (
            "dark_fraction",
            "Dark fraction"
        ),
        (
            "gradient_mean",
            "Gradient mean"
        ),
        (
            "laplacian_var",
            "Laplacian variance"
        ),
    ]

    print()
    print(
        "Pearson correlation with Region Gain"
    )

    print(
        "----------------------------------------"
    )

    for metric_key, label in metric_keys:
        values = [
            row[
                metric_key
            ]
            for row in valid_rows
        ]

        correlation = (
            calculate_pearson(
                values,
                gains
            )
        )

        print(
            f"{label:20s}: "
            f"{correlation:+.4f}"
        )

    # ========================================================
    # Temporal correlation
    # ========================================================

    temporal_rows = [
        row
        for row in valid_rows
        if (
                row[
                    "neighbor_count"
                ]
                > 0
                and
                math.isfinite(
                    row[
                        "motion_input_score"
                    ]
                )
                and
                math.isfinite(
                    row[
                        "motion_target_score"
                    ]
                )
        )
    ]

    temporal_gains = np.array(
        [
            row[
                "region_gain"
            ]
            for row in temporal_rows
        ],
        dtype=np.float64
    )

    motion_input = np.array(
        [
            row[
                "motion_input_score"
            ]
            for row in temporal_rows
        ],
        dtype=np.float64
    )

    motion_target = np.array(
        [
            row[
                "motion_target_score"
            ]
            for row in temporal_rows
        ],
        dtype=np.float64
    )

    gradient_values = np.array(
        [
            row[
                "gradient_mean"
            ]
            for row in temporal_rows
        ],
        dtype=np.float64
    )

    luma_std_values = np.array(
        [
            row[
                "luma_std"
            ]
            for row in temporal_rows
        ],
        dtype=np.float64
    )

    coverage_values = np.array(
        [
            row[
                "region_coverage_percent"
            ]
            for row in temporal_rows
        ],
        dtype=np.float64
    )

    print()
    print(
        "========================================"
    )

    print(
        "Temporal motion analysis"
    )

    print(
        "========================================"
    )

    print(
        f"Temporal valid images: "
        f"{len(temporal_rows)}"
    )

    input_corr = (
        calculate_pearson(
            motion_input,
            temporal_gains
        )
    )

    target_corr = (
        calculate_pearson(
            motion_target,
            temporal_gains
        )
    )

    print()
    print(
        "Pearson correlation with Region Gain"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Input motion  : "
        f"{input_corr:+.4f}"
    )

    print(
        f"Target motion : "
        f"{target_corr:+.4f}"
    )

    controls = np.column_stack(
        [
            gradient_values,
            luma_std_values,
            coverage_values,
        ]
    )

    input_partial_corr = (
        calculate_partial_correlation(
            motion_input,
            temporal_gains,
            controls
        )
    )

    target_partial_corr = (
        calculate_partial_correlation(
            motion_target,
            temporal_gains,
            controls
        )
    )

    print()
    print(
        "Partial correlation"
    )

    print(
        "Controlled for:"
    )

    print(
        "  gradient_mean"
    )

    print(
        "  luma_std"
    )

    print(
        "  region_coverage"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Input motion  : "
        f"{input_partial_corr:+.4f}"
    )

    print(
        f"Target motion : "
        f"{target_partial_corr:+.4f}"
    )

    # ========================================================
    # Temporal quartiles
    # ========================================================

    print_rank_quartiles(
        temporal_rows,
        metric_key="motion_input_score",
        label="Input motion"
    )

    print_rank_quartiles(
        temporal_rows,
        metric_key="motion_target_score",
        label="Target motion"
    )

    # ========================================================
    # Quartiles
    # ========================================================

    print_rank_quartiles(
        valid_rows,
        metric_key="gradient_mean",
        label="Gradient mean"
    )

    print_rank_quartiles(
        valid_rows,
        metric_key="laplacian_var",
        label="Laplacian variance"
    )

    print_rank_quartiles(
        valid_rows,
        metric_key="luma_mean",
        label="Luma mean"
    )

    print_rank_quartiles(
        valid_rows,
        metric_key="dark_fraction",
        label="Dark fraction"
    )

    # ========================================================
    # Neighbor detail availability
    # ========================================================

    detail_rows = [
        row
        for row in valid_rows
        if (
            math.isfinite(
                row[
                    "best_target_detail_ratio"
                ]
            )
            and
            math.isfinite(
                row[
                    "best_input_detail_ratio"
                ]
            )
            and
            math.isfinite(
                row[
                    "best_usable_fraction"
                ]
            )
        )
    ]

    print()
    print(
        "========================================"
    )
    print(
        "Neighbor detail availability"
    )
    print(
        "========================================"
    )

    print(
        f"Valid detail images: "
        f"{len(detail_rows)}"
    )

    if detail_rows:

        target_ratios = np.array(
            [
                row[
                    "best_target_detail_ratio"
                ]
                for row in detail_rows
            ],
            dtype=np.float64
        )

        input_ratios = np.array(
            [
                row[
                    "best_input_detail_ratio"
                ]
                for row in detail_rows
            ],
            dtype=np.float64
        )

        usable = np.array(
            [
                row[
                    "best_usable_fraction"
                ]
                for row in detail_rows
            ],
            dtype=np.float64
        )

        # ====================================================
        # Useful temporal:
        #
        # prev OR next のどちらか一方が
        # 3条件すべてを満たせばUseful
        # ====================================================

        prev_useful = np.array(
            [
                (
                        math.isfinite(
                            row[
                                "prev_target_detail_ratio"
                            ]
                        )
                        and
                        math.isfinite(
                            row[
                                "prev_input_detail_ratio"
                            ]
                        )
                        and
                        math.isfinite(
                            row[
                                "prev_usable_fraction"
                            ]
                        )
                        and
                        row[
                            "prev_target_detail_ratio"
                        ] >= 1.15
                        and
                        row[
                            "prev_input_detail_ratio"
                        ] >= 1.05
                        and
                        row[
                            "prev_usable_fraction"
                        ] >= 0.50
                )
                for row in detail_rows
            ],
            dtype=bool
        )

        next_useful = np.array(
            [
                (
                        math.isfinite(
                            row[
                                "next_target_detail_ratio"
                            ]
                        )
                        and
                        math.isfinite(
                            row[
                                "next_input_detail_ratio"
                            ]
                        )
                        and
                        math.isfinite(
                            row[
                                "next_usable_fraction"
                            ]
                        )
                        and
                        row[
                            "next_target_detail_ratio"
                        ] >= 1.15
                        and
                        row[
                            "next_input_detail_ratio"
                        ] >= 1.05
                        and
                        row[
                            "next_usable_fraction"
                        ] >= 0.50
                )
                for row in detail_rows
            ],
            dtype=bool
        )

        useful_mask = (
                prev_useful
                |
                next_useful
        )

        print(
            f"Target ratio mean   : "
            f"{target_ratios.mean():.4f}"
        )

        print(
            f"Target ratio median : "
            f"{np.median(target_ratios):.4f}"
        )

        print(
            f"Input ratio mean    : "
            f"{input_ratios.mean():.4f}"
        )

        print(
            f"Input ratio median  : "
            f"{np.median(input_ratios):.4f}"
        )

        print(
            f"Usable mean         : "
            f"{usable.mean():.4f}"
        )

        print(
            f"Useful temporal     : "
            f"{useful_mask.sum()} / "
            f"{len(detail_rows)} "
            f"({100.0 * useful_mask.mean():.2f} %)"
        )

    # ========================================================
    # Worst / Best
    # ========================================================

    sorted_rows = sorted(
        valid_rows,
        key=lambda row:
        row[
            "region_gain"
        ]
    )

    print()
    print(
        "========================================"
    )

    print(
        "Worst 10"
    )

    print(
        "========================================"
    )

    for row in sorted_rows[:10]:
        print(
            f"{row['stem']} | "
            f"Gain "
            f"{row['region_gain']:+.4f} dB | "
            f"Pixels "
            f"{row['region_pixel_count']} | "
            f"Luma "
            f"{row['luma_mean']:.3f} | "
            f"Dark "
            f"{row['dark_fraction']:.3f} | "
            f"Grad "
            f"{row['gradient_mean']:.5f} | "
            f"LapVar "
            f"{row['laplacian_var']:.6f}"
        )

    print()
    print(
        "========================================"
    )

    print(
        "Best 10"
    )

    print(
        "========================================"
    )

    for row in sorted_rows[-10:][::-1]:
        print(
            f"{row['stem']} | "
            f"Gain "
            f"{row['region_gain']:+.4f} dB | "
            f"Pixels "
            f"{row['region_pixel_count']} | "
            f"Luma "
            f"{row['luma_mean']:.3f} | "
            f"Dark "
            f"{row['dark_fraction']:.3f} | "
            f"Grad "
            f"{row['gradient_mean']:.5f} | "
            f"LapVar "
            f"{row['laplacian_var']:.6f}"
        )

    print()
    print(
        f"CSV saved: "
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()
