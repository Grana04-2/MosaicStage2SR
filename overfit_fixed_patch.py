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
    # Fixed LR crop selection
    # ========================================================

    auto_crop = (
        args.crop_x < 0
        or
        args.crop_y < 0
    )

    if auto_crop:
        (
            crop_x,
            crop_y,
            crop_coverage
        ) = find_max_mask_crop(
            mask_image=train_s_image,
            patch_size=args.patch_size,
            align=scale_den
        )

    else:
        crop_x = args.crop_x
        crop_y = args.crop_y

        validate_crop(
            x=crop_x,
            y=crop_y,
            patch_size=args.patch_size,
            width=lr_width,
            height=lr_height,
            align=scale_den
        )

        mask_array = (
            np.asarray(train_s_image)
            >= 128
        )

        crop_coverage = float(
            mask_array[
                crop_y:crop_y + args.patch_size,
                crop_x:crop_x + args.patch_size
            ].mean()
        )

    validate_crop(
        x=crop_x,
        y=crop_y,
        patch_size=args.patch_size,
        width=lr_width,
        height=lr_height,
        align=scale_den
    )

    hr_patch_size = (
        args.patch_size
        * scale_num
        // scale_den
    )

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

    lr_patch_image = train_h_image.crop(
        (
            crop_x,
            crop_y,
            crop_x + args.patch_size,
            crop_y + args.patch_size
        )
    )

    mask_patch_image = train_s_image.crop(
        (
            crop_x,
            crop_y,
            crop_x + args.patch_size,
            crop_y + args.patch_size
        )
    )

    hr_patch_image = train_b_image.crop(
        (
            crop_x_hr,
            crop_y_hr,
            crop_x_hr + hr_patch_size,
            crop_y_hr + hr_patch_size
        )
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

    preview_path = (
        checkpoint_dir
        / "crop_preview.png"
    )

    save_crop_preview(
        train_h_image=train_h_image,
        mask_image=train_s_image,
        x=crop_x,
        y=crop_y,
        patch_size=args.patch_size,
        save_path=preview_path
    )

    lr_patch_image.save(
        checkpoint_dir
        / "fixed_lr_patch.png"
    )

    mask_patch_image.save(
        checkpoint_dir
        / "fixed_mask_patch.png"
    )

    hr_patch_image.save(
        checkpoint_dir
        / "fixed_target_patch.png"
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
            f"crop_x={crop_x}\n"
        )
        file.write(
            f"crop_y={crop_y}\n"
        )
        file.write(
            f"patch_size={args.patch_size}\n"
        )
        file.write(
            f"hr_patch_size={hr_patch_size}\n"
        )
        file.write(
            f"crop_coverage={crop_coverage:.8f}\n"
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
    print("Fixed-patch overfit feasibility test")
    print("========================================")
    print(f"Stem             : {Path(args.stem).stem}")
    print(f"train_H          : {train_h_path.resolve()}")
    print(f"train_S          : {train_s_path.resolve()}")
    print(f"train_B          : {train_b_path.resolve()}")
    print(f"LR size          : {lr_width}x{lr_height}")
    print(f"HR size          : {hr_width}x{hr_height}")
    print(f"Fixed LR crop    : x={crop_x}, y={crop_y}")
    print(f"LR patch         : {args.patch_size}x{args.patch_size}")
    print(f"HR patch         : {hr_patch_size}x{hr_patch_size}")
    print(f"Mask coverage    : {crop_coverage:.6f}")
    print(f"Crop preview     : {preview_path.resolve()}")
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

    lr = (
        rgb_to_tensor(lr_patch_image)
        .unsqueeze(0)
        .to(device)
    )

    hr = (
        rgb_to_tensor(hr_patch_image)
        .unsqueeze(0)
        .to(device)
    )

    mask_lr = (
        mask_to_tensor(mask_patch_image)
        .unsqueeze(0)
        .to(device)
    )

    mask_hr = F.interpolate(
        mask_lr,
        size=(
            hr_patch_size,
            hr_patch_size
        ),
        mode="nearest"
    ).clamp(
        0.0,
        1.0
    )

    outside_mask_hr = (
        1.0
        - mask_hr
    )

    # inside train_S = 1.0
    # outside train_S = outside_weight
    weight_map = (
        mask_hr
        +
        float(args.outside_weight)
        * outside_mask_hr
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
        sr_initial = model(lr)
        sr_initial_clamped = sr_initial.clamp(
            0.0,
            1.0
        )

        initial_overall_psnr = calculate_psnr(
            sr_initial_clamped,
            hr
        )

        initial_inside_psnr = calculate_masked_psnr(
            sr_initial_clamped,
            hr,
            mask_hr
        )

    save_iteration_panel(
        model=model,
        lr=lr,
        sr=sr_initial_clamped,
        hr=hr,
        mask_hr=mask_hr,
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

            loss_gradient = weighted_gradient_loss(
                sr,
                hr,
                weight_map
            )

            loss_laplacian = weighted_laplacian_loss(
                sr,
                hr,
                weight_map
            )

            loss_total = (
                args.pixel_weight
                * loss_pixel
                +
                args.gradient_weight
                * loss_gradient
                +
                args.laplacian_weight
                * loss_laplacian
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
                    model(lr)
                    .clamp(0.0, 1.0)
                )

                overall_psnr = calculate_psnr(
                    sr_eval,
                    hr
                )

                inside_psnr = calculate_masked_psnr(
                    sr_eval,
                    hr,
                    mask_hr
                )

                outside_psnr = calculate_masked_psnr(
                    sr_eval,
                    hr,
                    outside_mask_hr
                )

                bicubic = (
                    model.upscale_bicubic(lr)
                    .clamp(0.0, 1.0)
                )

                bicubic_inside_psnr = (
                    calculate_masked_psnr(
                        bicubic,
                        hr,
                        mask_hr
                    )
                )

                residual_abs_mean = (
                    (sr_eval - bicubic)
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
                    lr=lr,
                    sr=sr_eval,
                    hr=hr,
                    mask_hr=mask_hr,
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
                    "crop_x":
                        crop_x,
                    "crop_y":
                        crop_y,
                    "patch_size":
                        args.patch_size,
                    "mask_coverage":
                        crop_coverage,
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
    print("Fixed-patch overfit completed")
    print("========================================")
    print(f"Best inside PSNR : {best_inside_psnr:.4f} dB")
    print(f"Output dir       : {checkpoint_dir.resolve()}")
    print(f"Samples          : {sample_dir.resolve()}")
    print(f"Log CSV          : {log_path.resolve()}")
    print("========================================")


if __name__ == "__main__":
    main()
