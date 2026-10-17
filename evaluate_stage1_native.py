import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.stage2_dataset import (
    load_manifest,
    split_records,
    split_records_by_sequence,
    get_sequence_id_from_stem,
    Stage2SRDataset,
)


# ============================================================
# Basic metrics
# ============================================================

def calculate_psnr(
        prediction,
        target
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

    mse = float(
        (
            prediction
            - target
        )
        .pow(2)
        .mean()
        .item()
    )

    if mse <= 1e-12:
        return 99.0

    return (
        10.0
        * math.log10(
            1.0 / mse
        )
    )


def downsample_clean_to_stage1(
        hr,
        stage1
):
    """
    Stage2 dataset:
        stage1 = train_A
        clean  = train_B

    train_B is higher resolution than train_A, so the clean target is
    bicubic-downsampled to the exact Stage1 output size for native
    Stage1 evaluation.
    """

    target_size = (
        stage1.shape[-2],
        stage1.shape[-1]
    )

    clean_stage1 = F.interpolate(
        hr.float(),
        size=target_size,
        mode="bicubic",
        align_corners=False,
        antialias=True
    )

    return clean_stage1.clamp(
        0.0,
        1.0
    )


def calculate_stage1_metrics(
        stage1,
        clean_stage1
):
    stage1 = (
        stage1
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    clean_stage1 = (
        clean_stage1
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
    )

    difference = (
        stage1
        - clean_stage1
    )

    absolute_difference = (
        difference.abs()
    )

    # RGB平均の1画素エラー
    pixel_absolute_error = (
        absolute_difference.mean(
            dim=1
        )
    )

    flat_pixel_error = (
        pixel_absolute_error
        .reshape(
            -1
        )
    )

    mse = float(
        (
            difference
            * difference
        )
        .mean()
        .item()
    )

    mae = float(
        absolute_difference
        .mean()
        .item()
    )

    rmse = float(
        math.sqrt(
            max(
                mse,
                0.0
            )
        )
    )

    psnr = (
        99.0
        if mse <= 1e-12
        else
        (
            10.0
            * math.log10(
                1.0 / mse
            )
        )
    )

    error_fraction_gt_0p05 = float(
        (
            flat_pixel_error
            > 0.05
        )
        .float()
        .mean()
        .item()
    )

    error_fraction_gt_0p10 = float(
        (
            flat_pixel_error
            > 0.10
        )
        .float()
        .mean()
        .item()
    )

    p95_error = float(
        torch.quantile(
            flat_pixel_error,
            0.95
        )
        .item()
    )

    p99_error = float(
        torch.quantile(
            flat_pixel_error,
            0.99
        )
        .item()
    )

    pixel_count = int(
        flat_pixel_error.numel()
    )

    top1_count = max(
        1,
        int(
            math.ceil(
                pixel_count
                * 0.01
            )
        )
    )

    top5_count = max(
        1,
        int(
            math.ceil(
                pixel_count
                * 0.05
            )
        )
    )

    top1_error_mae = float(
        torch.topk(
            flat_pixel_error,
            k=top1_count,
            largest=True,
            sorted=False
        )
        .values
        .mean()
        .item()
    )

    top5_error_mae = float(
        torch.topk(
            flat_pixel_error,
            k=top5_count,
            largest=True,
            sorted=False
        )
        .values
        .mean()
        .item()
    )

    return {
        "stage1_psnr":
            psnr,

        "stage1_mae":
            mae,

        "stage1_rmse":
            rmse,

        "error_fraction_gt_0p05":
            error_fraction_gt_0p05,

        "error_fraction_gt_0p10":
            error_fraction_gt_0p10,

        "p95_pixel_error":
            p95_error,

        "p99_pixel_error":
            p99_error,

        "top1_error_mae":
            top1_error_mae,

        "top5_error_mae":
            top5_error_mae,
    }


# ============================================================
# Image utility
# ============================================================

def tensor_to_pil(
        tensor
):
    tensor = (
        tensor
        .detach()
        .float()
        .clamp(
            0.0,
            1.0
        )
        .cpu()
    )

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
    )

    array = np.clip(
        array,
        0.0,
        255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="RGB"
    )


def save_comparison_image(
        stage1,
        clean_stage1,
        save_path,
        title_text
):
    stage1_image = tensor_to_pil(
        stage1
    )

    clean_image = tensor_to_pil(
        clean_stage1
    )

    # 1画素ごとのRGB平均絶対誤差
    error = (
        (
            stage1.detach().float()
            -
            clean_stage1.detach().float()
        )
        .abs()
        .mean(
            dim=0
        )
        .clamp(
            0.0,
            1.0
        )
        .cpu()
        .numpy()
    )

    # 0.10以上を白として表示。
    # 強い残存誤差を目視しやすくするための表示用スケール。
    error_visual = np.clip(
        error / 0.10,
        0.0,
        1.0
    )

    error_uint8 = (
        error_visual
        * 255.0
    ).astype(
        np.uint8
    )

    error_image = Image.fromarray(
        error_uint8,
        mode="L"
    ).convert(
        "RGB"
    )

    width, height = (
        stage1_image.size
    )

    header_height = 44

    canvas = Image.new(
        "RGB",
        (
            width * 3,
            height + header_height
        ),
        (
            0,
            0,
            0
        )
    )

    canvas.paste(
        stage1_image,
        (
            0,
            header_height
        )
    )

    canvas.paste(
        clean_image,
        (
            width,
            header_height
        )
    )

    canvas.paste(
        error_image,
        (
            width * 2,
            header_height
        )
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            8,
            4
        ),
        title_text,
        fill=(
            255,
            255,
            255
        )
    )

    draw.text(
        (
            8,
            22
        ),
        "Stage1 output",
        fill=(
            255,
            255,
            255
        )
    )

    draw.text(
        (
            width + 8,
            22
        ),
        "Clean downsampled",
        fill=(
            255,
            255,
            255
        )
    )

    draw.text(
        (
            width * 2 + 8,
            22
        ),
        "Abs error (0.10 -> white)",
        fill=(
            255,
            255,
            255
        )
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    canvas.save(
        save_path
    )


# ============================================================
# Summary helpers
# ============================================================

def finite_values(
        rows,
        key
):
    return np.array(
        [
            row[
                key
            ]
            for row in rows
            if math.isfinite(
                row[
                    key
                ]
            )
        ],
        dtype=np.float64
    )


def print_group_summary(
        label,
        rows
):
    psnr_values = finite_values(
        rows,
        "stage1_psnr"
    )

    mae_values = finite_values(
        rows,
        "stage1_mae"
    )

    err05_values = finite_values(
        rows,
        "error_fraction_gt_0p05"
    )

    err10_values = finite_values(
        rows,
        "error_fraction_gt_0p10"
    )

    p95_values = finite_values(
        rows,
        "p95_pixel_error"
    )

    p99_values = finite_values(
        rows,
        "p99_pixel_error"
    )

    top1_values = finite_values(
        rows,
        "top1_error_mae"
    )

    top5_values = finite_values(
        rows,
        "top5_error_mae"
    )

    print()
    print(
        label
    )

    print(
        f"  N                       : "
        f"{len(rows)}"
    )

    print(
        f"  Stage1 PSNR mean        : "
        f"{psnr_values.mean():.4f} dB"
    )

    print(
        f"  Stage1 PSNR median      : "
        f"{np.median(psnr_values):.4f} dB"
    )

    print(
        f"  Stage1 PSNR range       : "
        f"{psnr_values.min():.4f} - "
        f"{psnr_values.max():.4f} dB"
    )

    print(
        f"  Stage1 MAE mean         : "
        f"{mae_values.mean():.6f}"
    )

    print(
        f"  Error > 0.05            : "
        f"{err05_values.mean() * 100.0:.3f} %"
    )

    print(
        f"  Error > 0.10            : "
        f"{err10_values.mean() * 100.0:.3f} %"
    )

    print(
        f"  P95 pixel error         : "
        f"{p95_values.mean():.6f}"
    )

    print(
        f"  P99 pixel error         : "
        f"{p99_values.mean():.6f}"
    )

    print(
        f"  Top 5% error mean       : "
        f"{top5_values.mean():.6f}"
    )

    print(
        f"  Top 1% error mean       : "
        f"{top1_values.mean():.6f}"
    )


def print_sequence_summary(
        rows
):
    print()
    print(
        "========================================"
    )

    print(
        "Stage1 native statistics by sequence"
    )

    print(
        "========================================"
    )

    sequence_ids = sorted(
        {
            row[
                "sequence_id"
            ]
            for row in rows
        }
    )

    for sequence_id in sequence_ids:

        sequence_rows = [
            row
            for row in rows
            if (
                row[
                    "sequence_id"
                ]
                == sequence_id
            )
        ]

        print_group_summary(
            f"Sequence {sequence_id}",
            sequence_rows
        )


def print_stage1_quartiles(
        rows
):
    ordered = sorted(
        rows,
        key=lambda row:
        row[
            "stage1_psnr"
        ]
    )

    groups = np.array_split(
        np.array(
            ordered,
            dtype=object
        ),
        4
    )

    print()
    print(
        "========================================"
    )

    print(
        "Stage1 native quality quartiles"
    )

    print(
        "Q1 = hardest / lowest Stage1 PSNR"
    )

    print(
        "Q4 = easiest / highest Stage1 PSNR"
    )

    print(
        "========================================"
    )

    for group_index, group in enumerate(
            groups,
            start=1
    ):

        group_rows = list(
            group
        )

        psnr_values = finite_values(
            group_rows,
            "stage1_psnr"
        )

        print()
        print(
            f"Q{group_index} | "
            f"Stage1 PSNR "
            f"{psnr_values.min():.4f} - "
            f"{psnr_values.max():.4f} dB"
        )

        print_group_summary(
            "  Summary",
            group_rows
        )


# ============================================================
# Sample selection
# ============================================================

def select_visual_samples(
        rows,
        sample_count
):
    """
    Per sequence:
        worst N
        median-nearest N
        best N
    """

    selected = []

    sequence_ids = sorted(
        {
            row[
                "sequence_id"
            ]
            for row in rows
        }
    )

    for sequence_id in sequence_ids:

        sequence_rows = sorted(
            [
                row
                for row in rows
                if (
                    row[
                        "sequence_id"
                    ]
                    == sequence_id
                )
            ],
            key=lambda row:
            row[
                "stage1_psnr"
            ]
        )

        if not sequence_rows:
            continue

        count = min(
            sample_count,
            len(
                sequence_rows
            )
        )

        # Worst
        for rank, row in enumerate(
                sequence_rows[
                    :count
                ],
                start=1
        ):
            selected.append(
                (
                    sequence_id,
                    "worst",
                    rank,
                    row
                )
            )

        # Best
        best_rows = list(
            reversed(
                sequence_rows[
                    -count:
                ]
            )
        )

        for rank, row in enumerate(
                best_rows,
                start=1
        ):
            selected.append(
                (
                    sequence_id,
                    "best",
                    rank,
                    row
                )
            )

        # Median-nearest
        median_psnr = float(
            np.median(
                [
                    row[
                        "stage1_psnr"
                    ]
                    for row in sequence_rows
                ]
            )
        )

        middle_rows = sorted(
            sequence_rows,
            key=lambda row:
            abs(
                row[
                    "stage1_psnr"
                ]
                - median_psnr
            )
        )[
            :count
        ]

        for rank, row in enumerate(
                middle_rows,
                start=1
        ):
            selected.append(
                (
                    sequence_id,
                    "middle",
                    rank,
                    row
                )
            )

    return selected


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Stage2 checkpoint used only to recover "
            "the same val_ratio / seed / split_mode."
        )
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./datasets/stage2_1280x720"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=3,
        help=(
            "Number of worst / middle / best images "
            "saved per validation sequence."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "Optional output directory. "
            "If omitted, outputs are saved next to "
            "the Stage2 checkpoint."
        )
    )

    parser.add_argument(
        "--no_save_samples",
        action="store_true"
    )

    return parser.parse_args()


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

    checkpoint_path = Path(
        args.checkpoint
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    checkpoint_args = checkpoint.get(
        "args",
        {}
    )

    val_ratio = checkpoint_args.get(
        "val_ratio",
        0.10
    )

    seed = checkpoint_args.get(
        "seed",
        1234
    )

    split_mode = checkpoint_args.get(
        "split_mode",
        "frame"
    )

    lr_patch = checkpoint_args.get(
        "lr_patch",
        192
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Reference checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Split mode: {split_mode}"
    )

    print(
        f"Val ratio : {val_ratio}"
    )

    print(
        f"Seed      : {seed}"
    )

    # ========================================================
    # Dataset / same validation split as Stage2
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
    # Output
    # ========================================================

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

    else:

        output_dir = (
            checkpoint_path.parent
            / "stage1_native_analysis"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_dir = (
        output_dir
        / "samples"
    )

    csv_path = (
        output_dir
        / (
            f"{checkpoint_path.stem}"
            f"_stage1_native.csv"
        )
    )

    fieldnames = [
        "stem",
        "sequence_id",

        "stage1_width",
        "stage1_height",

        "clean_width",
        "clean_height",

        "stage1_psnr",
        "stage1_mae",
        "stage1_rmse",

        "error_fraction_gt_0p05",
        "error_fraction_gt_0p10",

        "p95_pixel_error",
        "p99_pixel_error",

        "top5_error_mae",
        "top1_error_mae",
    ]

    print()
    print(
        f"CSV output: "
        f"{csv_path.resolve()}"
    )

    # CSVファイルを評価開始前に作成する。
    # 各画像の評価直後に1行ずつ追記するため、
    # 実行途中でもCSVを確認できる。
    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    csv_file.flush()

    # ========================================================
    # Evaluate
    # ========================================================

    rows = []

    # Save tensors only for selected visual samples later.
    # Full tensors are not retained for all 450 images.
    stem_to_dataset_index = {
        record[
            "stem"
        ]:
            index
        for index, record
        in enumerate(
            val_records
        )
    }

    for batch_index, batch in enumerate(
            val_loader,
            start=1
    ):

        stem = (
            batch[
                "stem"
            ][0]
        )

        sequence_id = (
            get_sequence_id_from_stem(
                stem
            )
        )

        stage1 = (
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

        clean_stage1 = (
            downsample_clean_to_stage1(
                hr=hr,
                stage1=stage1
            )
        )

        metrics = (
            calculate_stage1_metrics(
                stage1=stage1,
                clean_stage1=clean_stage1
            )
        )

        row = {
            "stem":
                stem,

            "sequence_id":
                sequence_id,

            "stage1_width":
                int(
                    stage1.shape[-1]
                ),

            "stage1_height":
                int(
                    stage1.shape[-2]
                ),

            "clean_width":
                int(
                    hr.shape[-1]
                ),

            "clean_height":
                int(
                    hr.shape[-2]
                ),

            **metrics,
        }

        rows.append(
            row
        )

        # 評価直後にCSVへ保存
        writer.writerow(
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
            csv_file.flush()

            print(
                f"Evaluate "
                f"{batch_index:04d}/"
                f"{len(val_loader):04d}"
            )

            print(
                f"  CSV rows saved: "
                f"{batch_index}"
            )

    csv_file.flush()
    csv_file.close()

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "Stage1 native analysis"
    )

    print(
        "========================================"
    )

    print_group_summary(
        "Overall",
        rows
    )

    print_sequence_summary(
        rows
    )

    print_stage1_quartiles(
        rows
    )

    # ========================================================
    # Worst 10
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "Worst 10 Stage1 native images"
    )

    print(
        "========================================"
    )

    worst_rows = sorted(
        rows,
        key=lambda row:
        row[
            "stage1_psnr"
        ]
    )[
        :10
    ]

    for row in worst_rows:

        print(
            f"{row['stem']} | "
            f"PSNR "
            f"{row['stage1_psnr']:.4f} dB | "
            f"MAE "
            f"{row['stage1_mae']:.6f} | "
            f">0.05 "
            f"{row['error_fraction_gt_0p05'] * 100.0:.2f}% | "
            f">0.10 "
            f"{row['error_fraction_gt_0p10'] * 100.0:.2f}% | "
            f"P99 "
            f"{row['p99_pixel_error']:.6f} | "
            f"Top1% "
            f"{row['top1_error_mae']:.6f}"
        )

    # ========================================================
    # Visual samples
    # ========================================================

    if not args.no_save_samples:

        selected = (
            select_visual_samples(
                rows=rows,
                sample_count=max(
                    1,
                    args.sample_count
                )
            )
        )

        print()
        print(
            "========================================"
        )

        print(
            "Saving visual samples"
        )

        print(
            "========================================"
        )

        for (
                sequence_id,
                category,
                rank,
                row
        ) in selected:

            stem = row[
                "stem"
            ]

            dataset_index = (
                stem_to_dataset_index[
                    stem
                ]
            )

            item = val_dataset[
                dataset_index
            ]

            stage1 = (
                item[
                    "lr"
                ]
                .unsqueeze(
                    0
                )
                .to(
                    device
                )
            )

            hr = (
                item[
                    "hr"
                ]
                .unsqueeze(
                    0
                )
                .to(
                    device
                )
            )

            clean_stage1 = (
                downsample_clean_to_stage1(
                    hr=hr,
                    stage1=stage1
                )
            )

            save_path = (
                sample_dir
                / sequence_id
                / (
                    f"{category}_"
                    f"{rank:02d}_"
                    f"{stem}.png"
                )
            )

            title_text = (
                f"{stem} | "
                f"{category} | "
                f"PSNR "
                f"{row['stage1_psnr']:.4f} dB"
            )

            save_comparison_image(
                stage1=stage1[0],
                clean_stage1=clean_stage1[0],
                save_path=save_path,
                title_text=title_text
            )

        print(
            f"Sample directory: "
            f"{sample_dir}"
        )

    print()
    print(
        f"CSV saved: "
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()
