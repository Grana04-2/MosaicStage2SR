import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# Same detector parameters as final Stage1
# ============================================================

BLOCK_WINDOW = 17

BLOCK_FLAT_THRESHOLD = 0.015
BLOCK_EDGE_THRESHOLD = 0.05

BLOCK_FLAT_RATIO_MIN = 0.55
BLOCK_EDGE_RATIO_MIN = 0.025

BLOCK_DENSITY_KERNEL = 15
BLOCK_DENSITY_MIN = 0.08

BLOCK_EXPAND_KERNEL = 9
BLOCK_CLOSE_KERNEL = 7


# ============================================================
# Load Mosaic image
#
# PIL RGB 0..255
#     ->
# Tensor [-1, 1]
# ============================================================

def load_mosaic_tensor(
        image_path,
        device
):
    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = (
        torch.from_numpy(
            array
        )
        .permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
        .to(
            device
        )
    )

    tensor = (
        tensor
        * 2.0
        - 1.0
    )

    return tensor


# ============================================================
# Same detector as final Stage1
# ============================================================

@torch.no_grad()
def detect_mosaic_block_structure(
        input_image
):
    # [-1, 1] -> [0, 1]
    x = (
        (
            input_image
            .detach()
            .float()
            + 1.0
        )
        / 2.0
    ).clamp(
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Horizontal gradient
    # --------------------------------------------------------

    gx = torch.abs(
        x[:, :, :, 1:]
        -
        x[:, :, :, :-1]
    )

    gx = gx.mean(
        dim=1,
        keepdim=True
    )

    gx = F.pad(
        gx,
        (
            0,
            1,
            0,
            0
        ),
        mode="replicate"
    )

    # --------------------------------------------------------
    # Vertical gradient
    # --------------------------------------------------------

    gy = torch.abs(
        x[:, :, 1:, :]
        -
        x[:, :, :-1, :]
    )

    gy = gy.mean(
        dim=1,
        keepdim=True
    )

    gy = F.pad(
        gy,
        (
            0,
            0,
            0,
            1
        ),
        mode="replicate"
    )

    # --------------------------------------------------------
    # Flat pixels
    # --------------------------------------------------------

    flat_x = (
        gx
        <
        BLOCK_FLAT_THRESHOLD
    ).float()

    flat_y = (
        gy
        <
        BLOCK_FLAT_THRESHOLD
    ).float()

    flat = (
        flat_x
        +
        flat_y
    ) * 0.5

    # --------------------------------------------------------
    # Strong block boundaries
    # --------------------------------------------------------

    edge_x = (
        gx
        >
        BLOCK_EDGE_THRESHOLD
    ).float()

    edge_y = (
        gy
        >
        BLOCK_EDGE_THRESHOLD
    ).float()

    # --------------------------------------------------------
    # Local statistics
    # --------------------------------------------------------

    pad = (
        BLOCK_WINDOW
        // 2
    )

    flat_ratio = F.avg_pool2d(
        flat,
        kernel_size=BLOCK_WINDOW,
        stride=1,
        padding=pad
    )

    edge_x_ratio = F.avg_pool2d(
        edge_x,
        kernel_size=BLOCK_WINDOW,
        stride=1,
        padding=pad
    )

    edge_y_ratio = F.avg_pool2d(
        edge_y,
        kernel_size=BLOCK_WINDOW,
        stride=1,
        padding=pad
    )

    block_score = (
        flat_ratio
        *
        torch.sqrt(
            edge_x_ratio
            *
            edge_y_ratio
            +
            1e-8
        )
    )

    # --------------------------------------------------------
    # Raw mask
    # --------------------------------------------------------

    raw_mask = (
        (
            flat_ratio
            >
            BLOCK_FLAT_RATIO_MIN
        )
        &
        (
            edge_x_ratio
            >
            BLOCK_EDGE_RATIO_MIN
        )
        &
        (
            edge_y_ratio
            >
            BLOCK_EDGE_RATIO_MIN
        )
    ).float()

    # --------------------------------------------------------
    # Density cleanup
    # --------------------------------------------------------

    density = F.avg_pool2d(
        raw_mask,
        kernel_size=BLOCK_DENSITY_KERNEL,
        stride=1,
        padding=(
            BLOCK_DENSITY_KERNEL
            // 2
        )
    )

    clean_mask = (
        density
        >
        BLOCK_DENSITY_MIN
    ).float()

    # --------------------------------------------------------
    # Expand
    # --------------------------------------------------------

    final_mask = F.max_pool2d(
        clean_mask,
        kernel_size=BLOCK_EXPAND_KERNEL,
        stride=1,
        padding=(
            BLOCK_EXPAND_KERNEL
            // 2
        )
    )

    # --------------------------------------------------------
    # Closing
    # --------------------------------------------------------

    final_mask = (
        1.0
        -
        F.max_pool2d(
            1.0
            -
            final_mask,
            kernel_size=BLOCK_CLOSE_KERNEL,
            stride=1,
            padding=(
                BLOCK_CLOSE_KERNEL
                // 2
            )
        )
    )

    return (
        raw_mask,
        clean_mask,
        final_mask,
        block_score
    )


# ============================================================
# Save binary mask as grayscale PNG
# ============================================================

def save_mask(
        mask,
        output_path
):
    mask = (
        mask[
            0,
            0
        ]
        .detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0
        )
    )

    array = (
        mask
        * 255.0
        + 0.5
    ).byte().numpy()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        array,
        mode="L"
    ).save(
        output_path
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=(
            "./datasets/"
            "stage2_lp21_sigma6_mask67_1280x720"
        )
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="train_M"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_stage1_mask"
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset_root
    )

    input_dir = (
        dataset_root
        / args.input_dir
    )

    output_dir = (
        dataset_root
        / args.output_dir
    )

    if not input_dir.exists():
        raise FileNotFoundError(
            input_dir
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    image_paths = sorted(
        [
            path
            for path
            in input_dir.rglob("*")
            if path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            }
        ]
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found: {input_dir}"
        )

    print(
        f"Device       : {device}"
    )

    print(
        f"Input images : {len(image_paths)}"
    )

    print(
        f"Output       : {output_dir}"
    )

    coverage_sum = 0.0

    for index, image_path in enumerate(
            image_paths,
            start=1
    ):
        input_image = load_mosaic_tensor(
            image_path,
            device
        )

        (
            _,
            _,
            mosaic_mask,
            _
        ) = detect_mosaic_block_structure(
            input_image
        )

        # ====================================================
        # EXACT Stage1 mask67
        # ====================================================

        blend_mask = F.max_pool2d(
            mosaic_mask,
            kernel_size=67,
            stride=1,
            padding=33
        )

        relative_path = (
            image_path.relative_to(
                input_dir
            )
        )

        # Mask itself is always saved losslessly as PNG.
        relative_path = (
            relative_path.with_suffix(
                ".png"
            )
        )

        output_path = (
            output_dir
            / relative_path
        )

        save_mask(
            blend_mask,
            output_path
        )

        coverage_sum += (
            blend_mask.mean().item()
        )

        if (
                index % 100 == 0
                or
                index == len(image_paths)
        ):
            print(
                f"{index:04d}/"
                f"{len(image_paths):04d}"
            )

    print()
    print(
        f"Saved masks  : "
        f"{len(image_paths)}"
    )

    print(
        f"Mean coverage: "
        f"{coverage_sum / len(image_paths):.6f}"
    )


if __name__ == "__main__":
    main()