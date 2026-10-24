import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from models.residual_sr import ResidualSRNet
from losses import (
    weighted_charbonnier_loss,
    weighted_gradient_loss,
    weighted_laplacian_loss,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Manifest
# ============================================================

def load_record(manifest_path, stem):
    stem = Path(stem).stem

    with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline=""
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError("manifest.csv has no header.")

        required = {
            "stem",
            "stage1_file",
            "clean_file",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(
                "manifest.csv is missing fields: "
                f"{sorted(missing)}"
            )

        matches = [
            row
            for row in reader
            if Path(row["stem"]).stem == stem
        ]

    if len(matches) != 1:
        raise RuntimeError(
            "Stem must match exactly one manifest record.\n"
            f"Stem    : {stem}\n"
            f"Matches : {len(matches)}"
        )

    return matches[0]


# ============================================================
# Image helpers
# ============================================================

def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_mask(path):
    with Image.open(path) as image:
        return image.convert("L").copy()


def rgb_to_tensor(image):
    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    array = np.transpose(
        array,
        (2, 0, 1)
    )

    return torch.from_numpy(
        np.ascontiguousarray(array)
    )


def mask_to_tensor(image):
    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return torch.from_numpy(
        np.ascontiguousarray(
            array[None, ...]
        )
    )


def tensor_to_pil(tensor):
    tensor = (
        tensor
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )

    if tensor.dim() == 4:
        tensor = tensor[0]

    array = (
        tensor
        .permute(1, 2, 0)
        .numpy()
    )

    array = (
            array * 255.0 + 0.5
    ).astype(np.uint8)

    return Image.fromarray(array)


def mask_tensor_to_pil(tensor):
    tensor = (
        tensor
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )

    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]

    array = (
            tensor.numpy() * 255.0 + 0.5
    ).astype(np.uint8)

    return Image.fromarray(
        array,
        mode="L"
    ).convert("RGB")


def add_title(image, title, title_height=28):
    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + title_height
        ),
        "white"
    )

    canvas.paste(
        image,
        (0, title_height)
    )

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 6),
        title,
        fill="black"
    )

    return canvas


# ============================================================
# Fixed crop selection
# ============================================================

def window_sum(integral, x, y, size):
    x2 = x + size
    y2 = y + size

    return (
            integral[y2, x2]
            - integral[y, x2]
            - integral[y2, x]
            + integral[y, x]
    )


def find_max_mask_crop(
        mask_image,
        patch_size,
        align=3
):
    mask = (
            np.asarray(mask_image)
            >= 128
    ).astype(np.int32)

    height, width = mask.shape

    if width < patch_size or height < patch_size:
        raise RuntimeError(
            "Mask image is smaller than patch size.\n"
            f"Image : {width}x{height}\n"
            f"Patch : {patch_size}"
        )

    integral = np.pad(
        mask,
        ((1, 0), (1, 0)),
        mode="constant"
    )

    integral = (
        integral
        .cumsum(axis=0)
        .cumsum(axis=1)
    )

    best_x = 0
    best_y = 0
    best_sum = -1

    for y in range(
            0,
            height - patch_size + 1,
            align
    ):
        for x in range(
                0,
                width - patch_size + 1,
                align
        ):
            current_sum = window_sum(
                integral,
                x,
                y,
                patch_size
            )

            if current_sum > best_sum:
                best_sum = current_sum
                best_x = x
                best_y = y

    coverage = (
            float(best_sum)
            / float(patch_size * patch_size)
    )

    return (
        best_x,
        best_y,
        coverage
    )


def validate_crop(
        x,
        y,
        patch_size,
        width,
        height,
        align
):
    if x < 0 or y < 0:
        raise ValueError(
            "crop_x / crop_y must be >= 0."
        )

    if x % align != 0 or y % align != 0:
        raise ValueError(
            "crop_x and crop_y must be divisible by "
            f"{align}. Got x={x}, y={y}"
        )

    if (
            x + patch_size > width
            or
            y + patch_size > height
    ):
        raise ValueError(
            "Crop is outside the LR image.\n"
            f"Crop  : x={x}, y={y}, size={patch_size}\n"
            f"Image : {width}x{height}"
        )


def build_tile_positions(
        width,
        height,
        patch_size,
        stride,
        align=3
):
    if patch_size % align != 0:
        raise ValueError(
            f"patch_size must be divisible by {align}"
        )

    if stride % align != 0:
        raise ValueError(
            f"tile_stride must be divisible by {align}"
        )

    if (
            width < patch_size
            or
            height < patch_size
    ):
        raise ValueError(
            "Image is smaller than patch."
        )

    max_x = (
            width
            - patch_size
    )

    max_y = (
            height
            - patch_size
    )

    def make_axis_positions(
            max_position
    ):
        positions = list(
            range(
                0,
                max_position + 1,
                stride
            )
        )

        if (
                not positions
                or
                positions[-1]
                != max_position
        ):
            positions.append(
                max_position
            )

        for position in positions:

            if position % align != 0:
                raise RuntimeError(
                    "Tile position is not aligned:\n"
                    f"{position}"
                )

        return positions

    x_positions = make_axis_positions(
        max_x
    )

    y_positions = make_axis_positions(
        max_y
    )

    return [
        (
            x,
            y
        )
        for y in y_positions
        for x in x_positions
    ]


@torch.no_grad()
def run_tiled_full_inference(
        model,
        lr_full,
        tile_positions,
        patch_size,
        scale_num=5,
        scale_den=3
):
    batch_size = (
        lr_full.shape[0]
    )

    lr_height = (
        lr_full.shape[-2]
    )

    lr_width = (
        lr_full.shape[-1]
    )

    hr_height = (
        lr_height
        * scale_num
        // scale_den
    )

    hr_width = (
        lr_width
        * scale_num
        // scale_den
    )

    hr_patch_size = (
        patch_size
        * scale_num
        // scale_den
    )

    output_sum = torch.zeros(
        (
            batch_size,
            3,
            hr_height,
            hr_width
        ),
        device=lr_full.device,
        dtype=lr_full.dtype
    )

    weight_sum = torch.zeros(
        (
            batch_size,
            1,
            hr_height,
            hr_width
        ),
        device=lr_full.device,
        dtype=lr_full.dtype
    )

    # ================================================
    # Center-weighted blending window
    #
    # Tile中央を強く、
    # Tile端を弱くする。
    # 端を完全な0にはしない。
    # ================================================

    window_1d = torch.hann_window(
        hr_patch_size,
        periodic=False,
        device=lr_full.device,
        dtype=lr_full.dtype
    )

    window_2d = (
        window_1d[:, None]
        * window_1d[None, :]
    )

    window_2d = window_2d.clamp_min(
        1e-3
    )

    tile_weight = window_2d[
        None,
        None,
        :,
        :
    ]

    for (
            x_lr,
            y_lr
    ) in tile_positions:

        lr_tile = lr_full[
            :,
            :,
            y_lr:y_lr + patch_size,
            x_lr:x_lr + patch_size
        ]

        sr_tile = (
            model(
                lr_tile
            )
            .clamp(
                0.0,
                1.0
            )
        )

        x_hr = (
            x_lr
            * scale_num
            // scale_den
        )

        y_hr = (
            y_lr
            * scale_num
            // scale_den
        )

        output_sum[
            :,
            :,
            y_hr:y_hr + hr_patch_size,
            x_hr:x_hr + hr_patch_size
        ] += (
            sr_tile
            * tile_weight
        )

        weight_sum[
            :,
            :,
            y_hr:y_hr + hr_patch_size,
            x_hr:x_hr + hr_patch_size
        ] += tile_weight

    return (
        output_sum
        /
        weight_sum.clamp_min(
            1e-6
        )
    )


def save_crop_preview(
        train_h_image,
        mask_image,
        x,
        y,
        patch_size,
        save_path
):
    input_panel = train_h_image.copy()
    input_draw = ImageDraw.Draw(input_panel)
    input_draw.rectangle(
        (
            x,
            y,
            x + patch_size - 1,
            y + patch_size - 1
        ),
        outline="red",
        width=3
    )

    mask_panel = mask_image.convert("RGB")
    mask_draw = ImageDraw.Draw(mask_panel)
    mask_draw.rectangle(
        (
            x,
            y,
            x + patch_size - 1,
            y + patch_size - 1
        ),
        outline="red",
        width=3
    )

    panels = [
        add_title(
            input_panel,
            "train_H / selected fixed crop"
        ),
        add_title(
            mask_panel,
            "train_S / selected fixed crop"
        ),
    ]

    canvas = Image.new(
        "RGB",
        (
            sum(panel.width for panel in panels),
            max(panel.height for panel in panels)
        ),
        "white"
    )

    offset_x = 0

    for panel in panels:
        canvas.paste(
            panel,
            (offset_x, 0)
        )
        offset_x += panel.width

    canvas.save(save_path)


# ============================================================
# Metrics
# ============================================================

def calculate_psnr(prediction, target):
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    mse = F.mse_loss(
        prediction,
        target
    ).item()

    if mse <= 1e-12:
        return 99.0

    return (
            10.0
            * math.log10(1.0 / mse)
    )


def calculate_masked_psnr(
        prediction,
        target,
        mask
):
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    mask_3ch = mask.expand_as(target)

    count = mask_3ch.sum().item()

    if count <= 0.0:
        return float("nan")

    squared_error = (
            (prediction - target).pow(2)
            * mask_3ch
    ).sum().item()

    mse = squared_error / count

    if mse <= 1e-12:
        return 99.0

    return (
            10.0
            * math.log10(1.0 / mse)
    )


# ============================================================
# Visualization
# ============================================================

def save_iteration_panel(
        model,
        lr,
        sr,
        hr,
        mask_hr,
        save_path,
        iteration,
        overall_psnr,
        inside_psnr
):
    with torch.no_grad():
        bicubic = (
            model.upscale_bicubic(lr)
            .clamp(0.0, 1.0)
        )

    panels = [
        add_title(
            tensor_to_pil(bicubic),
            "Bicubic / train_H"
        ),
        add_title(
            tensor_to_pil(
                sr.clamp(0.0, 1.0)
            ),
            (
                f"SR iter={iteration} "
                f"PSNR={overall_psnr:.2f}"
            )
        ),
        add_title(
            tensor_to_pil(hr),
            "Target train_B"
        ),
        add_title(
            mask_tensor_to_pil(mask_hr),
            (
                f"train_S HR "
                f"insidePSNR={inside_psnr:.2f}"
            )
        ),
    ]

    canvas = Image.new(
        "RGB",
        (
            sum(panel.width for panel in panels),
            max(panel.height for panel in panels)
        ),
        "white"
    )

    offset_x = 0

    for panel in panels:
        canvas.paste(
            panel,
            (offset_x, 0)
        )
        offset_x += panel.width

    canvas.save(save_path)


# ============================================================
# CSV logging
# ============================================================

def append_log(log_path, row):
    exists = log_path.exists()

    with open(
            log_path,
            "a",
            encoding="utf-8",
            newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys())
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Arguments
# ============================================================

def parse_args():
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
        "--stem",
        type=str,
        default="00102_00028"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="fixed_overfit_00102_00028"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=500
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=192
    )

    parser.add_argument(
        "--tile_stride",
        type=int,
        default=96
    )

    parser.add_argument(
        "--crop_x",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--crop_y",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=64
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=8
    )

    parser.add_argument(
        "--block_res_scale",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--output_res_scale",
        type=float,
        default=1.00
    )

    parser.add_argument(
        "--pixel_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--gradient_weight",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--laplacian_weight",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--outside_weight",
        type=float,
        default=0.02,
        help=(
            "Loss weight outside train_S. "
            "0.02 means train_S pixels are weighted 50x "
            "relative to outside pixels."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )

    parser.add_argument(
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")

    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1")

    if not 0.0 <= args.outside_weight <= 1.0:
        raise ValueError(
            "--outside_weight must be in [0, 1]."
        )

    scale_num = 5
    scale_den = 3

    if args.patch_size % scale_den != 0:
        raise ValueError(
            "--patch_size must be divisible by 3."
        )

    dataset_root = Path(args.dataset_root)
    manifest_path = dataset_root / "manifest.csv"

    record = load_record(
        manifest_path=manifest_path,
        stem=args.stem
    )

    relative_lr = Path(
        record["stage1_file"]
    )

    relative_hr = Path(
        record["clean_file"]
    )

    train_h_path = (
            dataset_root
            / "train_H"
            / relative_lr
    )

    train_s_path = (
            dataset_root
            / "train_S"
            / relative_lr
    )

    train_b_path = (
            dataset_root
            / "train_B"
            / relative_hr
    )

    for label, path in [
        ("train_H", train_h_path),
        ("train_S", train_s_path),
        ("train_B", train_b_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found:\n{path.resolve()}"
            )

    train_h_image = load_rgb(train_h_path)
    train_s_image = load_mask(train_s_path)
    train_b_image = load_rgb(train_b_path)

    lr_width, lr_height = train_h_image.size
    hr_width, hr_height = train_b_image.size

    expected_hr_width = (
            lr_width * scale_num // scale_den
    )

    expected_hr_height = (
            lr_height * scale_num // scale_den
    )

    if (
            hr_width != expected_hr_width
            or
            hr_height != expected_hr_height
    ):
        raise RuntimeError(
            "LR / HR scale mismatch.\n"
            f"LR       : {lr_width}x{lr_height}\n"
            f"HR       : {hr_width}x{hr_height}\n"
            f"Expected : "
            f"{expected_hr_width}x{expected_hr_height}"
        )

    if train_s_image.size != train_h_image.size:
        raise RuntimeError(
            "train_H / train_S size mismatch."
        )

    # ========================================================
    # Full-image tiled training
    # ========================================================

    if (
            args.tile_stride
            % scale_den
            != 0
    ):
        raise ValueError(
            "--tile_stride must be divisible by 3."
        )

    tile_positions = (
        build_tile_positions(
            width=lr_width,
            height=lr_height,
            patch_size=args.patch_size,
            stride=args.tile_stride,
            align=scale_den
        )
    )

    hr_patch_size = (
            args.patch_size
            * scale_num
            // scale_den
    )

    full_mask_array = (
            np.asarray(
                train_s_image
            )
            >= 128
    )

    full_mask_coverage = float(
        full_mask_array.mean()
    )


    # ========================================================
    # Output folders
    # ========================================================

    checkpoint_dir = (
            Path("checkpoints")
            / args.name
    )

    sample_dir = (
            checkpoint_dir
            / "samples"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_h_image.save(
        checkpoint_dir
        / "full_train_H.png"
    )

    train_s_image.save(
        checkpoint_dir
        / "full_train_S.png"
    )

    train_b_image.save(
        checkpoint_dir
        / "full_target_train_B.png"
    )

    config_path = (
            checkpoint_dir
            / "config.txt"
    )

    with open(
            config_path,
            "w",
            encoding="utf-8"
    ) as file:
        file.write(
            f"stem={Path(args.stem).stem}\n"
        )
        file.write(
            "mode=full_image_tiled\n"
        )
        file.write(
            f"patch_size={args.patch_size}\n"
        )
        file.write(
            f"tile_stride={args.tile_stride}\n"
        )
        file.write(
            f"tile_count={len(tile_positions)}\n"
        )
        file.write(
            f"full_mask_coverage="
            f"{full_mask_coverage:.8f}\n"
        )
        file.write(
            f"output_res_scale={args.output_res_scale}\n"
        )
        file.write(
            f"lr={args.lr}\n"
        )
        file.write(
            f"iterations={args.iterations}\n"
        )
        file.write(
            f"outside_weight={args.outside_weight}\n"
        )

    print()
    print("========================================")
    print("Full-image tiled overfit feasibility test")
    print("========================================")
    print(f"Stem             : {Path(args.stem).stem}")
    print(f"train_H          : {train_h_path.resolve()}")
    print(f"train_S          : {train_s_path.resolve()}")
    print(f"train_B          : {train_b_path.resolve()}")
    print(f"LR size          : {lr_width}x{lr_height}")
    print(f"HR size          : {hr_width}x{hr_height}")
    print(
        f"LR tile          : "
        f"{args.patch_size}x{args.patch_size}"
    )

    print(
        f"Tile stride      : "
        f"{args.tile_stride}"
    )

    print(
        f"Tile count       : "
        f"{len(tile_positions)}"
    )

    print(
        f"Full mask cover  : "
        f"{full_mask_coverage:.6f}"
    )
    print(f"Output res scale : {args.output_res_scale}")
    print(f"Outside weight   : {args.outside_weight}")
    print("========================================")

    if args.dry_run:
        print("Dry run completed. No training performed.")
        return

    # ========================================================
    # Device / tensors
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
            args.amp
            and
            device.type == "cuda"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    lr_full = (
        rgb_to_tensor(
            train_h_image
        )
        .unsqueeze(0)
        .to(
            device
        )
    )

    hr_full = (
        rgb_to_tensor(
            train_b_image
        )
        .unsqueeze(0)
        .to(
            device
        )
    )

    mask_lr_full = (
        mask_to_tensor(
            train_s_image
        )
        .unsqueeze(0)
        .to(
            device
        )
    )

    mask_hr_full = F.interpolate(
        mask_lr_full,
        size=(
            hr_height,
            hr_width
        ),
        mode="nearest"
    ).clamp(
        0.0,
        1.0
    )

    outside_mask_hr_full = (
            1.0
            - mask_hr_full
    )

    # ========================================================
    # Model / optimizer
    # ========================================================

    model = ResidualSRNet(
        input_channels=3,
        channels=args.channels,
        num_blocks=args.blocks,
        block_res_scale=args.block_res_scale,
        output_res_scale=args.output_res_scale,
        scale_num=scale_num,
        scale_den=scale_den
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(f"Device           : {device}")
    print(f"AMP              : {use_amp}")
    print(f"Parameters       : {parameter_count:,}")
    print(f"Iterations       : {args.iterations}")
    print(f"Learning rate    : {args.lr}")
    print(
        "Loss weights     : "
        f"pixel={args.pixel_weight}, "
        f"grad={args.gradient_weight}, "
        f"lap={args.laplacian_weight}"
    )

    log_path = (
            checkpoint_dir
            / "training_log.csv"
    )

    best_inside_psnr = -float("inf")
    start_time = time.time()

    # ========================================================
    # Iteration 0
    # ========================================================

    model.eval()

    with torch.no_grad():

        sr_initial_clamped = (
            run_tiled_full_inference(
                model=model,
                lr_full=lr_full,
                tile_positions=tile_positions,
                patch_size=args.patch_size,
                scale_num=scale_num,
                scale_den=scale_den
            )
        )

        initial_overall_psnr = (
            calculate_psnr(
                sr_initial_clamped,
                hr_full
            )
        )

        initial_inside_psnr = (
            calculate_masked_psnr(
                sr_initial_clamped,
                hr_full,
                mask_hr_full
            )
        )

    save_iteration_panel(
        model=model,
        lr=lr_full,
        sr=sr_initial_clamped,
        hr=hr_full,
        mask_hr=mask_hr_full,
        save_path=(
                sample_dir
                / "iter_000000.png"
        ),
        iteration=0,
        overall_psnr=initial_overall_psnr,
        inside_psnr=initial_inside_psnr
    )

    print()
    print(
        f"Iter 000000 | "
        f"Overall PSNR {initial_overall_psnr:.4f} | "
        f"Inside PSNR {initial_inside_psnr:.4f}"
    )

    # ========================================================
    # Fixed-patch overfit
    # ========================================================

    model.train()

    for iteration in range(
            1,
            args.iterations + 1
    ):

        tile_index = (
                (
                        iteration - 1
                )
                % len(
            tile_positions
        )
        )

        (
            crop_x,
            crop_y
        ) = tile_positions[
            tile_index
        ]

        crop_x_hr = (
                crop_x
                * scale_num
                // scale_den
        )

        crop_y_hr = (
                crop_y
                * scale_num
                // scale_den
        )

        lr = lr_full[
             :,
             :,
             crop_y:
             crop_y + args.patch_size,
             crop_x:
             crop_x + args.patch_size
             ]

        hr = hr_full[
             :,
             :,
             crop_y_hr:
             crop_y_hr + hr_patch_size,
             crop_x_hr:
             crop_x_hr + hr_patch_size
             ]

        weight_map = torch.ones(
            (
                1,
                1,
                hr_patch_size,
                hr_patch_size
            ),
            device=device,
            dtype=hr.dtype
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
                enabled=use_amp
        ):
            sr = model(lr)

            loss_pixel = weighted_charbonnier_loss(
                sr,
                hr,
                weight_map
            )

            loss_gradient = torch.zeros_like(
                loss_pixel
            )

            loss_laplacian = torch.zeros_like(
                loss_pixel
            )

            loss_total = (
                    args.pixel_weight
                    * loss_pixel
            )

        scaler.scale(
            loss_total
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        should_log = (
                iteration == 1
                or
                iteration % args.log_every == 0
                or
                iteration == args.iterations
        )

        should_save = (
                iteration % args.save_every == 0
                or
                iteration == args.iterations
        )

        if should_log or should_save:
            model.eval()

            with torch.no_grad():
                sr_eval = (
                    run_tiled_full_inference(
                        model=model,
                        lr_full=lr_full,
                        tile_positions=tile_positions,
                        patch_size=args.patch_size,
                        scale_num=scale_num,
                        scale_den=scale_den
                    )
                )

                overall_psnr = (
                    calculate_psnr(
                        sr_eval,
                        hr_full
                    )
                )

                inside_psnr = (
                    calculate_masked_psnr(
                        sr_eval,
                        hr_full,
                        mask_hr_full
                    )
                )

                outside_psnr = (
                    calculate_masked_psnr(
                        sr_eval,
                        hr_full,
                        outside_mask_hr_full
                    )
                )

                bicubic = (
                    model.upscale_bicubic(
                        lr_full
                    )
                    .clamp(
                        0.0,
                        1.0
                    )
                )

                bicubic_inside_psnr = (
                    calculate_masked_psnr(
                        bicubic,
                        hr_full,
                        mask_hr_full
                    )
                )

                residual_abs_mean = (
                    (
                            sr_eval
                            - bicubic
                    )
                    .abs()
                    .mean()
                    .item()
                )

            elapsed = time.time() - start_time

            if should_log:
                print(
                    f"Iter {iteration:06d} | "
                    f"Loss {loss_total.item():.6f} | "
                    f"Pix {loss_pixel.item():.6f} | "
                    f"Grad {loss_gradient.item():.6f} | "
                    f"Lap {loss_laplacian.item():.6f} | "
                    f"PSNR {overall_psnr:.4f} | "
                    f"Inside {inside_psnr:.4f} | "
                    f"Outside {outside_psnr:.4f} | "
                    f"Residual {residual_abs_mean:.6f}"
                )

                append_log(
                    log_path,
                    {
                        "iteration":
                            iteration,
                        "loss_total":
                            loss_total.item(),
                        "loss_pixel":
                            loss_pixel.item(),
                        "loss_gradient":
                            loss_gradient.item(),
                        "loss_laplacian":
                            loss_laplacian.item(),
                        "overall_psnr":
                            overall_psnr,
                        "inside_psnr":
                            inside_psnr,
                        "inside_gain_vs_bicubic":
                            (
                                    inside_psnr
                                    - bicubic_inside_psnr
                            ),
                        "outside_psnr":
                            outside_psnr,
                        "residual_abs_mean":
                            residual_abs_mean,
                        "seconds":
                            elapsed,
                    }
                )

            if should_save:
                save_iteration_panel(
                    model=model,
                    lr=lr_full,
                    sr=sr_eval,
                    hr=hr_full,
                    mask_hr=mask_hr_full,
                    save_path=(
                            sample_dir
                            / f"iter_{iteration:06d}.png"
                    ),
                    iteration=iteration,
                    overall_psnr=overall_psnr,
                    inside_psnr=inside_psnr
                )

                checkpoint = {
                    "iteration":
                        iteration,
                    "model":
                        model.state_dict(),
                    "optimizer":
                        optimizer.state_dict(),
                    "scaler":
                        scaler.state_dict(),
                    "stem":
                        Path(args.stem).stem,
                                        "mode":
                        "full_image_tiled",
                    "patch_size":
                        args.patch_size,
                    "tile_stride":
                        args.tile_stride,
                    "tile_count":
                        len(
                            tile_positions
                        ),
                    "mask_coverage":
                        full_mask_coverage,
                    "inside_psnr":
                        inside_psnr,
                    "args":
                        vars(args),
                }

                torch.save(
                    checkpoint,
                    checkpoint_dir
                    / f"iter_{iteration:06d}.pth"
                )

                if inside_psnr > best_inside_psnr:
                    best_inside_psnr = inside_psnr

                    torch.save(
                        checkpoint,
                        checkpoint_dir
                        / "best_inside_psnr.pth"
                    )

            model.train()

    # ========================================================
    # Completed
    # ========================================================

    print()
    print("========================================")
    print("Full-image tiled overfit completed")
    print("========================================")
    print(f"Best inside PSNR : {best_inside_psnr:.4f} dB")
    print(f"Output dir       : {checkpoint_dir.resolve()}")
    print(f"Samples          : {sample_dir.resolve()}")
    print(f"Log CSV          : {log_path.resolve()}")
    print("========================================")


if __name__ == "__main__":
    main()
