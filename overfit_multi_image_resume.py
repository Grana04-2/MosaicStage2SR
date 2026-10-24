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
from losses import weighted_charbonnier_loss


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
# Manifest / image helpers
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
            raise RuntimeError(
                "manifest.csv has no header."
            )

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
        .clamp(
            0.0,
            1.0
        )
    )

    if tensor.dim() == 4:
        tensor = tensor[0]

    array = (
        tensor
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    array = (
        array
        * 255.0
        + 0.5
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array
    )


def mask_tensor_to_pil(tensor):
    tensor = (
        tensor
        .detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0
        )
    )

    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]

    array = (
        tensor.numpy()
        * 255.0
        + 0.5
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="L"
    ).convert(
        "RGB"
    )


def add_title(
        image,
        title,
        title_height=28
):
    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + title_height
        ),
        "white"
    )

    canvas.paste(
        image,
        (
            0,
            title_height
        )
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            8,
            6
        ),
        title,
        fill="black"
    )

    return canvas


# ============================================================
# Tile layout
# ============================================================

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
        stage1_full,
        mosaic_full,
        tile_positions,
        patch_size,
        scale_num=5,
        scale_den=3
):
    batch_size = (
        stage1_full.shape[0]
    )

    lr_height = (
        stage1_full.shape[-2]
    )

    lr_width = (
        stage1_full.shape[-1]
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
        device=stage1_full.device,
        dtype=stage1_full.dtype
    )

    weight_sum = torch.zeros(
        (
            batch_size,
            1,
            hr_height,
            hr_width
        ),
        device=stage1_full.device,
        dtype=stage1_full.dtype
    )

    window_1d = torch.hann_window(
        hr_patch_size,
        periodic=False,
        device=stage1_full.device,
        dtype=stage1_full.dtype
    )

    window_2d = (
        window_1d[:, None]
        * window_1d[None, :]
    ).clamp_min(
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

        stage1_tile = stage1_full[
            :,
            :,
            y_lr:
            y_lr + patch_size,
            x_lr:
            x_lr + patch_size
        ]

        mosaic_tile = mosaic_full[
            :,
            :,
            y_lr:
            y_lr + patch_size,
            x_lr:
            x_lr + patch_size
        ]

        sr_tile = (
            model(
                stage1_tile,
                mosaic_tile
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
            y_hr:
            y_hr + hr_patch_size,
            x_hr:
            x_hr + hr_patch_size
        ] += (
            sr_tile
            * tile_weight
        )

        weight_sum[
            :,
            :,
            y_hr:
            y_hr + hr_patch_size,
            x_hr:
            x_hr + hr_patch_size
        ] += tile_weight

    return (
        output_sum
        /
        weight_sum.clamp_min(
            1e-6
        )
    )


# ============================================================
# Metrics
# ============================================================

def calculate_psnr(
        prediction,
        target
):
    prediction = prediction.clamp(
        0.0,
        1.0
    )

    target = target.clamp(
        0.0,
        1.0
    )

    mse = F.mse_loss(
        prediction,
        target
    ).item()

    if mse <= 1e-12:
        return 99.0

    return (
        10.0
        * math.log10(
            1.0 / mse
        )
    )


def calculate_masked_psnr(
        prediction,
        target,
        mask
):
    prediction = prediction.clamp(
        0.0,
        1.0
    )

    target = target.clamp(
        0.0,
        1.0
    )

    mask_3ch = mask.expand_as(
        target
    )

    count = (
        mask_3ch
        .sum()
        .item()
    )

    if count <= 0.0:
        return float("nan")

    squared_error = (
        (
            prediction
            - target
        ).pow(
            2
        )
        * mask_3ch
    ).sum().item()

    mse = (
        squared_error
        / count
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
# Visualization / log
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
            model.upscale_bicubic(
                lr
            )
            .clamp(
                0.0,
                1.0
            )
        )

    panels = [
        add_title(
            tensor_to_pil(
                bicubic
            ),
            "Bicubic / train_A"
        ),

        add_title(
            tensor_to_pil(
                sr
            ),
            (
                f"SR iter={iteration} "
                f"PSNR={overall_psnr:.2f}"
            )
        ),

        add_title(
            tensor_to_pil(
                hr
            ),
            "Target train_B"
        ),

        add_title(
            mask_tensor_to_pil(
                mask_hr
            ),
            (
                "train_S HR "
                f"insidePSNR={inside_psnr:.2f}"
            )
        ),
    ]

    canvas = Image.new(
        "RGB",
        (
            sum(
                panel.width
                for panel in panels
            ),
            max(
                panel.height
                for panel in panels
            )
        ),
        "white"
    )

    offset_x = 0

    for panel in panels:

        canvas.paste(
            panel,
            (
                offset_x,
                0
            )
        )

        offset_x += (
            panel.width
        )

    canvas.save(
        save_path
    )


def append_log(
        log_path,
        row
):
    exists = (
        log_path.exists()
    )

    with open(
            log_path,
            "a",
            encoding="utf-8",
            newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            )
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# Multi-image evaluation
# ============================================================

@torch.no_grad()
def evaluate_all_samples(
        model,
        samples,
        tile_positions,
        patch_size,
        scale_num,
        scale_den,
        iteration,
        sample_dir,
        save_images
):
    per_sample = []

    for sample in samples:

        sr = run_tiled_full_inference(
            model=model,
            stage1_full=sample[
                "stage1_full"
            ],
            mosaic_full=sample[
                "mosaic_full"
            ],
            tile_positions=tile_positions,
            patch_size=patch_size,
            scale_num=scale_num,
            scale_den=scale_den
        )

        overall_psnr = (
            calculate_psnr(
                sr,
                sample[
                    "hr_full"
                ]
            )
        )

        inside_psnr = (
            calculate_masked_psnr(
                sr,
                sample[
                    "hr_full"
                ],
                sample[
                    "mask_hr_full"
                ]
            )
        )

        outside_psnr = (
            calculate_masked_psnr(
                sr,
                sample[
                    "hr_full"
                ],
                sample[
                    "outside_mask_hr_full"
                ]
            )
        )

        bicubic = (
            model.upscale_bicubic(
                sample[
                    "stage1_full"
                ]
            )
            .clamp(
                0.0,
                1.0
            )
        )

        bicubic_inside_psnr = (
            calculate_masked_psnr(
                bicubic,
                sample[
                    "hr_full"
                ],
                sample[
                    "mask_hr_full"
                ]
            )
        )

        result = {
            "stem":
                sample[
                    "stem"
                ],

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
        }

        per_sample.append(
            result
        )

        if save_images:

            iteration_dir = (
                sample_dir
                / f"iter_{iteration:06d}"
            )

            iteration_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            save_iteration_panel(
                model=model,
                lr=sample[
                    "stage1_full"
                ],
                sr=sr,
                hr=sample[
                    "hr_full"
                ],
                mask_hr=sample[
                    "mask_hr_full"
                ],
                save_path=(
                    iteration_dir
                    / (
                        sample[
                            "stem"
                        ]
                        + ".png"
                    )
                ),
                iteration=iteration,
                overall_psnr=overall_psnr,
                inside_psnr=inside_psnr
            )

    mean_overall = float(
        np.mean(
            [
                row[
                    "overall_psnr"
                ]
                for row
                in per_sample
            ]
        )
    )

    mean_inside = float(
        np.mean(
            [
                row[
                    "inside_psnr"
                ]
                for row
                in per_sample
            ]
        )
    )

    min_inside = float(
        np.min(
            [
                row[
                    "inside_psnr"
                ]
                for row
                in per_sample
            ]
        )
    )

    return (
        per_sample,
        mean_overall,
        mean_inside,
        min_inside
    )


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
        "--stems",
        nargs=20,
        required=True,
        help=(
            "Exactly 20 stems."
        )
    )

    parser.add_argument(
        "--name",
        type=str,
        default="multi_overfit_5stems"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=75000
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=500
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
        "--channels",
        type=int,
        default=128
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=16
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
        "--seed",
        type=int,
        default=1234
    )

    parser.add_argument(
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Checkpoint path to resume from. "
            "--iterations is the final target iteration."
        )
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    if args.iterations < 1:
        raise ValueError(
            "--iterations must be >= 1"
        )

    if args.eval_every < 1:
        raise ValueError(
            "--eval_every must be >= 1"
        )

    if args.log_every < 1:
        raise ValueError(
            "--log_every must be >= 1"
        )

    scale_num = 5
    scale_den = 3

    if (
            args.patch_size
            % scale_den
            != 0
    ):
        raise ValueError(
            "--patch_size must be divisible by 3."
        )

    if (
            args.tile_stride
            % scale_den
            != 0
    ):
        raise ValueError(
            "--tile_stride must be divisible by 3."
        )

    stems = [
        Path(stem).stem
        for stem
        in args.stems
    ]

    if len(stems) != 20:
        raise RuntimeError(
            "Exactly 20 stems are required."
        )

    if len(set(stems)) != 20:
        raise RuntimeError(
            "All 20 stems must be unique."
        )

    dataset_root = Path(
        args.dataset_root
    )

    manifest_path = (
        dataset_root
        / "manifest.csv"
    )

    raw_samples = []

    reference_lr_size = None
    reference_hr_size = None

    for stem in stems:

        record = load_record(
            manifest_path=manifest_path,
            stem=stem
        )

        relative_lr = Path(
            record[
                "stage1_file"
            ]
        )

        relative_hr = Path(
            record[
                "clean_file"
            ]
        )

        train_a_path = (
                dataset_root
                / "train_A"
                / relative_lr
        )

        train_m_path = (
                dataset_root
                / "train_M"
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
            (
                    "train_A",
                    train_a_path
            ),
            (
                    "train_M",
                    train_m_path
            ),
            (
                    "train_S",
                    train_s_path
            ),
            (
                    "train_B",
                    train_b_path
            ),
        ]:

            if not path.exists():

                raise FileNotFoundError(
                    f"{label} not found:\n"
                    f"{path.resolve()}"
                )

        train_a_image = load_rgb(
            train_a_path
        )

        train_m_image = load_rgb(
            train_m_path
        )

        train_s_image = load_mask(
            train_s_path
        )

        train_b_image = load_rgb(
            train_b_path
        )

        lr_width, lr_height = (
            train_a_image.size
        )

        hr_width, hr_height = (
            train_b_image.size
        )

        expected_hr_width = (
            lr_width
            * scale_num
            // scale_den
        )

        expected_hr_height = (
            lr_height
            * scale_num
            // scale_den
        )

        if (
                hr_width
                != expected_hr_width
                or
                hr_height
                != expected_hr_height
        ):

            raise RuntimeError(
                "LR / HR scale mismatch.\n"
                f"Stem     : {stem}\n"
                f"LR       : "
                f"{lr_width}x{lr_height}\n"
                f"HR       : "
                f"{hr_width}x{hr_height}\n"
                f"Expected : "
                f"{expected_hr_width}"
                f"x"
                f"{expected_hr_height}"
            )

        if (
                train_m_image.size
                != train_a_image.size
        ):
            raise RuntimeError(
                "train_A / train_M size mismatch.\n"
                f"Stem: {stem}"
            )

        if (
                train_s_image.size
                != train_a_image.size
        ):
            raise RuntimeError(
                "train_A / train_S size mismatch.\n"
                f"Stem: {stem}"
            )

        if reference_lr_size is None:
            reference_lr_size = (
                lr_width,
                lr_height
            )

            reference_hr_size = (
                hr_width,
                hr_height
            )

        if (
                (
                    lr_width,
                    lr_height
                )
                != reference_lr_size
                or
                (
                    hr_width,
                    hr_height
                )
                != reference_hr_size
        ):
            raise RuntimeError(
                "All 20 images must have the same size."
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

        raw_samples.append(
            {
                "stem":
                    stem,

                                "train_a_image":
                    train_a_image,

                "train_m_image":
                    train_m_image,

                "train_s_image":
                    train_s_image,

                "train_b_image":
                    train_b_image,

                "mask_coverage":
                    full_mask_coverage,
            }
        )

    lr_width, lr_height = (
        reference_lr_size
    )

    hr_width, hr_height = (
        reference_hr_size
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

    # Interleave images for every tile:
    # tile0-image0, tile0-image1, ...,
    # tile1-image0, tile1-image1, ...
    training_jobs = [
        (
            sample_index,
            x,
            y
        )
        for (
            x,
            y
        ) in tile_positions
        for sample_index
        in range(
            len(
                raw_samples
            )
        )
    ]

    checkpoint_dir = (
        Path(
            "checkpoints"
        )
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
            "mode=multi_image_overfit_6ch_stage1_mosaic\n"
        )

        file.write(
            f"stems={','.join(stems)}\n"
        )

        file.write(
            f"patch_size={args.patch_size}\n"
        )

        file.write(
            f"tile_stride={args.tile_stride}\n"
        )

        file.write(
            f"tile_count_per_image="
            f"{len(tile_positions)}\n"
        )

        file.write(
            f"jobs_per_cycle="
            f"{len(training_jobs)}\n"
        )

        file.write(
            f"channels={args.channels}\n"
        )

        file.write(
            f"blocks={args.blocks}\n"
        )

        file.write(
            f"output_res_scale="
            f"{args.output_res_scale}\n"
        )

        file.write(
            f"lr={args.lr}\n"
        )

        file.write(
            f"iterations={args.iterations}\n"
        )

        for sample in raw_samples:

            file.write(
                f"coverage_{sample['stem']}="
                f"{sample['mask_coverage']:.8f}\n"
            )

    print()
    print(
        "========================================"
    )
    print(
        "Multi-image overfit feasibility test"
    )
    print(
        "========================================"
    )

    for index, sample in enumerate(
            raw_samples,
            start=1
    ):

        print(
            f"[{index}] "
            f"{sample['stem']} | "
            f"coverage "
            f"{sample['mask_coverage']:.6f}"
        )

    print(
        f"LR size          : "
        f"{lr_width}x{lr_height}"
    )

    print(
        f"HR size          : "
        f"{hr_width}x{hr_height}"
    )

    print(
        f"LR tile          : "
        f"{args.patch_size}"
        f"x"
        f"{args.patch_size}"
    )

    print(
        f"Tile stride      : "
        f"{args.tile_stride}"
    )

    print(
        f"Tiles/image      : "
        f"{len(tile_positions)}"
    )

    print(
        f"Jobs/cycle       : "
        f"{len(training_jobs)}"
    )

    print(
        f"Iterations       : "
        f"{args.iterations}"
    )

    print(
        "========================================"
    )

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
        torch.backends.cudnn.benchmark = (
            True
        )

    samples = []

    for raw_sample in raw_samples:
        stage1_full = (
            rgb_to_tensor(
                raw_sample[
                    "train_a_image"
                ]
            )
            .unsqueeze(0)
            .to(
                device
            )
        )

        mosaic_full = (
            rgb_to_tensor(
                raw_sample[
                    "train_m_image"
                ]
            )
            .unsqueeze(0)
            .to(
                device
            )
        )

        hr_full = (
            rgb_to_tensor(
                raw_sample[
                    "train_b_image"
                ]
            )
            .unsqueeze(0)
            .to(
                device
            )
        )

        mask_lr_full = (
            mask_to_tensor(
                raw_sample[
                    "train_s_image"
                ]
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

        samples.append(
            {
                "stem":
                    raw_sample[
                        "stem"
                    ],

                                "stage1_full":
                    stage1_full,

                "mosaic_full":
                    mosaic_full,

                "hr_full":
                    hr_full,

                "mask_hr_full":
                    mask_hr_full,

                "outside_mask_hr_full":
                    outside_mask_hr_full,

                "mask_coverage":
                    raw_sample[
                        "mask_coverage"
                    ],
            }
        )

    model = ResidualSRNet(
        input_channels=6,
        channels=args.channels,
        num_blocks=args.blocks,
        block_res_scale=args.block_res_scale,
        output_res_scale=args.output_res_scale,
        scale_num=scale_num,
        scale_den=scale_den
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.99
        )
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print()
    print(
        f"Device           : "
        f"{device}"
    )

    print(
        f"AMP              : "
        f"{use_amp}"
    )

    print(
        f"Parameters       : "
        f"{parameter_count:,}"
    )

    print(
        f"Learning rate    : "
        f"{args.lr}"
    )

    print(
        "Loss             : "
        "full-area Pixel-only"
    )

    summary_log_path = (
        checkpoint_dir
        / "training_log.csv"
    )

    per_stem_log_path = (
        checkpoint_dir
        / "per_stem_log.csv"
    )

    best_mean_inside_psnr = (
        -float(
            "inf"
        )
    )

    start_iteration = 1

    # ========================================================
    # Resume
    # ========================================================

    if args.resume:

        resume_path = Path(
            args.resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found:\n"
                f"{resume_path.resolve()}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location=device
        )

        checkpoint_stems = checkpoint.get(
            "stems",
            None
        )

        if (
                checkpoint_stems is not None
                and
                list(checkpoint_stems)
                != stems
        ):
            raise RuntimeError(
                "Resume checkpoint stems do not match "
                "the current --stems order.\n"
                f"Checkpoint: {checkpoint_stems}\n"
                f"Current   : {stems}"
            )

        model.load_state_dict(
            checkpoint[
                "model"
            ],
            strict=True
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

        if (
                "scaler"
                in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler"
                ]
            )

        for param_group in optimizer.param_groups:
            param_group[
                "lr"
            ] = args.lr

        completed_iteration = int(
            checkpoint[
                "iteration"
            ]
        )

        start_iteration = (
            completed_iteration
            + 1
        )

        best_path = (
            checkpoint_dir
            / "best_mean_inside_psnr.pth"
        )

        if best_path.exists():

            best_checkpoint = torch.load(
                best_path,
                map_location="cpu"
            )

            best_mean_inside_psnr = float(
                best_checkpoint.get(
                    "mean_inside_psnr",
                    checkpoint.get(
                        "mean_inside_psnr",
                        -float("inf")
                    )
                )
            )

        else:

            best_mean_inside_psnr = float(
                checkpoint.get(
                    "mean_inside_psnr",
                    -float("inf")
                )
            )

        if start_iteration > args.iterations:
            raise ValueError(
                "Resume checkpoint is already at or beyond "
                "the requested final iteration.\n"
                f"Checkpoint iteration : {completed_iteration}\n"
                f"Target iterations    : {args.iterations}"
            )

        print()
        print(
            "========================================"
        )
        print(
            "RESUME MODE"
        )
        print(
            "========================================"
        )
        print(
            f"Checkpoint       : "
            f"{resume_path.resolve()}"
        )
        print(
            f"Completed iter   : "
            f"{completed_iteration}"
        )
        print(
            f"Resume from iter : "
            f"{start_iteration}"
        )
        print(
            f"Target iter      : "
            f"{args.iterations}"
        )
        print(
            "========================================"
        )

    start_time = time.time()

    # ========================================================
    # Iteration 0 / resume-state evaluation
    # ========================================================

    if not args.resume:

        model.eval()

        (
            initial_per_sample,
            initial_mean_overall,
            initial_mean_inside,
            initial_min_inside
        ) = evaluate_all_samples(
            model=model,
            samples=samples,
            tile_positions=tile_positions,
            patch_size=args.patch_size,
            scale_num=scale_num,
            scale_den=scale_den,
            iteration=0,
            sample_dir=sample_dir,
            save_images=True
        )

        print()
        print(
            f"Iter 000000 | "
            f"Mean PSNR "
            f"{initial_mean_overall:.4f} | "
            f"Mean inside "
            f"{initial_mean_inside:.4f} | "
            f"Min inside "
            f"{initial_min_inside:.4f}"
        )

    model.train()

    # ========================================================
    # Training
    # ========================================================

    for iteration in range(
            start_iteration,
            args.iterations + 1
    ):

        job_index = (
            (
                iteration
                - 1
            )
            % len(
                training_jobs
            )
        )

        (
            sample_index,
            crop_x,
            crop_y
        ) = training_jobs[
            job_index
        ]

        sample = samples[
            sample_index
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

        stage1 = sample[
                     "stage1_full"
                 ][
                 :,
                 :,
                 crop_y:
                 crop_y + args.patch_size,
                 crop_x:
                 crop_x + args.patch_size
                 ]

        mosaic = sample[
                     "mosaic_full"
                 ][
                 :,
                 :,
                 crop_y:
                 crop_y + args.patch_size,
                 crop_x:
                 crop_x + args.patch_size
                 ]

        hr = sample[
            "hr_full"
        ][
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

            sr = model(
                stage1,
                mosaic
            )

            loss_pixel = (
                weighted_charbonnier_loss(
                    sr,
                    hr,
                    weight_map
                )
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

        should_print = (
            iteration == 1
            or
            iteration
            % args.log_every
            == 0
            or
            iteration
            == args.iterations
        )

        should_evaluate = (
            iteration
            % args.eval_every
            == 0
            or
            iteration
            == args.iterations
        )

        if should_print:

            print(
                f"Train "
                f"{iteration:06d}/"
                f"{args.iterations:06d} | "
                f"Stem "
                f"{sample['stem']} | "
                f"Loss "
                f"{loss_total.item():.6f}"
            )

        if should_evaluate:

            model.eval()

            (
                per_sample_results,
                mean_overall,
                mean_inside,
                min_inside
            ) = evaluate_all_samples(
                model=model,
                samples=samples,
                tile_positions=tile_positions,
                patch_size=args.patch_size,
                scale_num=scale_num,
                scale_den=scale_den,
                iteration=iteration,
                sample_dir=sample_dir,
                save_images=True
            )

            elapsed = (
                time.time()
                - start_time
            )

            print()
            print(
                f"Eval {iteration:06d} | "
                f"Mean PSNR "
                f"{mean_overall:.4f} | "
                f"Mean inside "
                f"{mean_inside:.4f} | "
                f"Min inside "
                f"{min_inside:.4f}"
            )

            for result in (
                    per_sample_results
            ):

                print(
                    "  "
                    f"{result['stem']} | "
                    f"PSNR "
                    f"{result['overall_psnr']:.4f} | "
                    f"Inside "
                    f"{result['inside_psnr']:.4f} | "
                    f"Gain "
                    f"{result['inside_gain_vs_bicubic']:+.4f}"
                )

                append_log(
                    per_stem_log_path,
                    {
                        "iteration":
                            iteration,

                        "stem":
                            result[
                                "stem"
                            ],

                        "overall_psnr":
                            result[
                                "overall_psnr"
                            ],

                        "inside_psnr":
                            result[
                                "inside_psnr"
                            ],

                        "inside_gain_vs_bicubic":
                            result[
                                "inside_gain_vs_bicubic"
                            ],

                        "outside_psnr":
                            result[
                                "outside_psnr"
                            ],

                        "seconds":
                            elapsed,
                    }
                )

            append_log(
                summary_log_path,
                {
                    "iteration":
                        iteration,

                    "train_loss":
                        loss_total.item(),

                    "mean_overall_psnr":
                        mean_overall,

                    "mean_inside_psnr":
                        mean_inside,

                    "min_inside_psnr":
                        min_inside,

                    "seconds":
                        elapsed,
                }
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

                "stems":
                    stems,

                "mode":
                    "multi_image_overfit_6ch_stage1_mosaic",

                "patch_size":
                    args.patch_size,

                "tile_stride":
                    args.tile_stride,

                "tile_count_per_image":
                    len(
                        tile_positions
                    ),

                "jobs_per_cycle":
                    len(
                        training_jobs
                    ),

                "mean_overall_psnr":
                    mean_overall,

                "mean_inside_psnr":
                    mean_inside,

                "min_inside_psnr":
                    min_inside,

                "args":
                    vars(
                        args
                    ),
            }

            torch.save(
                checkpoint,
                checkpoint_dir
                / (
                    f"iter_"
                    f"{iteration:06d}"
                    f".pth"
                )
            )

            if (
                    mean_inside
                    >
                    best_mean_inside_psnr
            ):

                best_mean_inside_psnr = (
                    mean_inside
                )

                torch.save(
                    checkpoint,
                    checkpoint_dir
                    / "best_mean_inside_psnr.pth"
                )

            model.train()

    print()
    print(
        "========================================"
    )

    print(
        "Multi-image overfit completed"
    )

    print(
        "========================================"
    )

    print(
        f"Best mean inside PSNR : "
        f"{best_mean_inside_psnr:.4f} dB"
    )

    print(
        f"Output dir            : "
        f"{checkpoint_dir.resolve()}"
    )

    print(
        f"Samples               : "
        f"{sample_dir.resolve()}"
    )

    print(
        f"Summary log           : "
        f"{summary_log_path.resolve()}"
    )

    print(
        f"Per-stem log          : "
        f"{per_stem_log_path.resolve()}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
