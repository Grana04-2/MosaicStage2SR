import argparse
import math
from pathlib import Path
import csv
import numpy as np
import torch
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
)


# ============================================================
# PSNR from accumulated squared error
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

    # mask:
    # [B, 1, H, W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    mask = mask.to(
        device=prediction.device,
        dtype=prediction.dtype
    )

    # RGB 3chへ展開
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
# Add stats
# ============================================================

def add_stats(
        stats,
        key,
        prediction,
        target,
        mask
):

    error_sum, count = (
        masked_error_stats(
            prediction=prediction,
            target=target,
            mask=mask
        )
    )

    stats[
        key + "_error"
    ] += error_sum

    stats[
        key + "_count"
    ] += count


# ============================================================
# arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "./checkpoints/"
            "residual_sr_v2_region/"
            "baseline_epoch_010.pth"
        )
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
        "--num_workers",
        type=int,
        default=0
    )

    return parser.parse_args()


# ============================================================
# main
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
    # Load checkpoint
    # ========================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
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
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    # ========================================================
    # Recover training parameters
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

    print()
    print(
        "Region parameters"
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

    # full-size画像なので batch_size=1 を推奨
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
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
    # Accumulators
    # ========================================================

    stats = {
        "sr_region_error": 0.0,
        "sr_region_count": 0.0,

        "bicubic_region_error": 0.0,
        "bicubic_region_count": 0.0,

        "sr_outside_error": 0.0,
        "sr_outside_count": 0.0,

        "bicubic_outside_error": 0.0,
        "bicubic_outside_count": 0.0,
    }

    # ========================================================
    # Per-image results
    # ========================================================

    per_image_rows = []

    region_pixel_count = 0.0
    total_pixel_count = 0.0

    # ========================================================
    # Evaluate
    # ========================================================

    for batch_index, batch in enumerate(
            val_loader,
            start=1
    ):

        lr = (
            batch[
                "lr"
            ].to(
                device,
                non_blocking=True
            )
        )

        hr = (
            batch[
                "hr"
            ].to(
                device,
                non_blocking=True
            )
        )

        # ----------------------------------------------------
        # SR
        # ----------------------------------------------------

        sr = (
            model(
                lr
            )
            .clamp(
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        bicubic = (
            model.upscale_bicubic(
                lr
            )
            .clamp(
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # Same Region score used during training
        # ----------------------------------------------------

        (
            _,
            score_map
        ) = build_region_weight(
            bicubic=bicubic,
            target=hr,
            threshold=region_threshold,
            change_range=region_range,
            region_boost=region_boost,
            smooth_kernel=region_smooth
        )

        # ----------------------------------------------------
        # Binary evaluation mask
        #
        # score >= eval_threshold -> Region内
        # score <  eval_threshold -> Region外
        # ----------------------------------------------------

        region_mask = (
            score_map
            >= args.eval_threshold
        )

        outside_mask = (
            score_map
            < args.eval_threshold
        )

        # ====================================================
        # Per-image Region PSNR
        # ====================================================

        stem = batch[
            "stem"
        ][0]

        # ----------------------------------------------------
        # Region
        # ----------------------------------------------------

        (
            sr_region_error,
            sr_region_count
        ) = masked_error_stats(
            prediction=sr,
            target=hr,
            mask=region_mask
        )

        (
            bicubic_region_error,
            bicubic_region_count
        ) = masked_error_stats(
            prediction=bicubic,
            target=hr,
            mask=region_mask
        )

        # ----------------------------------------------------
        # Outside
        # ----------------------------------------------------

        (
            sr_outside_error,
            sr_outside_count
        ) = masked_error_stats(
            prediction=sr,
            target=hr,
            mask=outside_mask
        )

        (
            bicubic_outside_error,
            bicubic_outside_count
        ) = masked_error_stats(
            prediction=bicubic,
            target=hr,
            mask=outside_mask
        )

        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        sr_region_psnr_image = (
            psnr_from_stats(
                sr_region_error,
                sr_region_count
            )
        )

        bicubic_region_psnr_image = (
            psnr_from_stats(
                bicubic_region_error,
                bicubic_region_count
            )
        )

        sr_outside_psnr_image = (
            psnr_from_stats(
                sr_outside_error,
                sr_outside_count
            )
        )

        bicubic_outside_psnr_image = (
            psnr_from_stats(
                bicubic_outside_error,
                bicubic_outside_count
            )
        )

        region_gain_image = (
            sr_region_psnr_image
            - bicubic_region_psnr_image
        )

        outside_gain_image = (
            sr_outside_psnr_image
            - bicubic_outside_psnr_image
        )

        # ----------------------------------------------------
        # Region coverage
        # ----------------------------------------------------

        region_pixels = (
            region_mask
            .float()
            .sum()
            .item()
        )

        total_pixels = (
            region_mask
            .numel()
        )

        region_coverage_image = (
            100.0
            * region_pixels
            / max(
                total_pixels,
                1
            )
        )

        # ----------------------------------------------------
        # Save per-image result
        # ----------------------------------------------------

        per_image_rows.append(
            {
                "stem":
                    stem,

                "region_coverage_percent":
                    region_coverage_image,

                "bicubic_region_psnr":
                    bicubic_region_psnr_image,

                "sr_region_psnr":
                    sr_region_psnr_image,

                "region_gain":
                    region_gain_image,

                "bicubic_outside_psnr":
                    bicubic_outside_psnr_image,

                "sr_outside_psnr":
                    sr_outside_psnr_image,

                "outside_gain":
                    outside_gain_image,
            }
        )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        region_pixel_count += (
            region_mask
            .float()
            .sum()
            .item()
        )

        total_pixel_count += (
            region_mask
            .numel()
        )

        # ----------------------------------------------------
        # Region
        # ----------------------------------------------------

        add_stats(
            stats=stats,
            key="sr_region",
            prediction=sr,
            target=hr,
            mask=region_mask
        )

        add_stats(
            stats=stats,
            key="bicubic_region",
            prediction=bicubic,
            target=hr,
            mask=region_mask
        )

        # ----------------------------------------------------
        # Outside Region
        # ----------------------------------------------------

        add_stats(
            stats=stats,
            key="sr_outside",
            prediction=sr,
            target=hr,
            mask=outside_mask
        )

        add_stats(
            stats=stats,
            key="bicubic_outside",
            prediction=bicubic,
            target=hr,
            mask=outside_mask
        )

        if (
                batch_index % 50 == 0
                or
                batch_index == len(
                    val_loader
                )
        ):
            print(
                f"Evaluate "
                f"{batch_index:04d}/"
                f"{len(val_loader):04d}"
            )

    # ========================================================
    # PSNR
    # ========================================================

    sr_region_psnr = (
        psnr_from_stats(
            stats[
                "sr_region_error"
            ],
            stats[
                "sr_region_count"
            ]
        )
    )

    bicubic_region_psnr = (
        psnr_from_stats(
            stats[
                "bicubic_region_error"
            ],
            stats[
                "bicubic_region_count"
            ]
        )
    )

    sr_outside_psnr = (
        psnr_from_stats(
            stats[
                "sr_outside_error"
            ],
            stats[
                "sr_outside_count"
            ]
        )
    )

    bicubic_outside_psnr = (
        psnr_from_stats(
            stats[
                "bicubic_outside_error"
            ],
            stats[
                "bicubic_outside_count"
            ]
        )
    )

    region_gain = (
        sr_region_psnr
        - bicubic_region_psnr
    )

    outside_gain = (
        sr_outside_psnr
        - bicubic_outside_psnr
    )

    region_coverage = (
        100.0
        * region_pixel_count
        / max(
            total_pixel_count,
            1.0
        )
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "Region PSNR evaluation"
    )

    print(
        "========================================"
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Eval threshold   : "
        f"{args.eval_threshold:.3f}"
    )

    print(
        f"Region coverage  : "
        f"{region_coverage:.3f} %"
    )

    print()

    print(
        "Region inside"
    )

    print(
        f"  Bicubic : "
        f"{bicubic_region_psnr:.4f} dB"
    )

    print(
        f"  SR      : "
        f"{sr_region_psnr:.4f} dB"
    )

    print(
        f"  Gain    : "
        f"{region_gain:+.4f} dB"
    )

    print()

    print(
        "Region outside"
    )

    print(
        f"  Bicubic : "
        f"{bicubic_outside_psnr:.4f} dB"
    )

    print(
        f"  SR      : "
        f"{sr_outside_psnr:.4f} dB"
    )

    print(
        f"  Gain    : "
        f"{outside_gain:+.4f} dB"
    )

    print(
        "========================================"
    )

    # ========================================================
    # Per-image CSV
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
            f"_region_eval_"
            f"{threshold_text}.csv"
        )
    )

    with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "stem",
            "region_coverage_percent",
            "bicubic_region_psnr",
            "sr_region_psnr",
            "region_gain",
            "bicubic_outside_psnr",
            "sr_outside_psnr",
            "outside_gain",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in per_image_rows:
            writer.writerow(
                row
            )

    print()
    print(
        f"Per-image CSV saved: "
        f"{csv_path}"
    )

    # ========================================================
    # Region Gain distribution
    # ========================================================

    valid_rows = [
        row
        for row in per_image_rows
        if (
            math.isfinite(
                row[
                    "region_gain"
                ]
            )
        )
    ]

    region_gains = np.array(
        [
            row[
                "region_gain"
            ]
            for row in valid_rows
        ],
        dtype=np.float64
    )

    coverages = np.array(
        [
            row[
                "region_coverage_percent"
            ]
            for row in valid_rows
        ],
        dtype=np.float64
    )

    if len(
            region_gains
    ) > 0:

        improved_count = int(
            np.sum(
                region_gains > 0.0
            )
        )

        degraded_count = int(
            np.sum(
                region_gains < 0.0
            )
        )

        total_valid = len(
            region_gains
        )

        total_images = len(
            per_image_rows
        )

        no_region_count = (
            total_images
            - total_valid
        )

        improved_percent = (
            100.0
            * improved_count
            / total_valid
        )

        degraded_percent = (
            100.0
            * degraded_count
            / total_valid
        )

        print()
        print(
            "========================================"
        )

        print(
            "Per-image Region Gain distribution"
        )

        print(
            "========================================"
        )

        print(
            f"Valid images : "
            f"{total_valid}"
        )

        print(
            f"No-region    : "
            f"{no_region_count}"
        )

        print(
            f"Improved     : "
            f"{improved_count} "
            f"({improved_percent:.2f} %)"
        )

        print(
            f"Degraded     : "
            f"{degraded_count} "
            f"({degraded_percent:.2f} %)"
        )

        print()

        print(
            f"Mean Gain    : "
            f"{np.mean(region_gains):+.4f} dB"
        )

        print(
            f"Median Gain  : "
            f"{np.median(region_gains):+.4f} dB"
        )

        print(
            f"P10 Gain     : "
            f"{np.percentile(region_gains, 10):+.4f} dB"
        )

        print(
            f"P25 Gain     : "
            f"{np.percentile(region_gains, 25):+.4f} dB"
        )

        print(
            f"P75 Gain     : "
            f"{np.percentile(region_gains, 75):+.4f} dB"
        )

        print(
            f"P90 Gain     : "
            f"{np.percentile(region_gains, 90):+.4f} dB"
        )

        print(
            f"Min Gain     : "
            f"{np.min(region_gains):+.4f} dB"
        )

        print(
            f"Max Gain     : "
            f"{np.max(region_gains):+.4f} dB"
        )

        print()

        print(
            f"Mean image coverage: "
            f"{np.mean(coverages):.3f} %"
        )

        print(
            "========================================"
        )

        # ====================================================
        # Worst / Best images
        # ====================================================

        sorted_rows = sorted(
            valid_rows,
            key=lambda row:
                row[
                    "region_gain"
                ]
        )

        print()
        print(
            "Worst 10 Region Gain"
        )

        for row in sorted_rows[:10]:

            print(
                f"{row['stem']} | "
                f"Gain "
                f"{row['region_gain']:+.4f} dB | "
                f"Coverage "
                f"{row['region_coverage_percent']:.3f} %"
            )

        print()

        print(
            "Best 10 Region Gain"
        )

        for row in sorted_rows[-10:][::-1]:

            print(
                f"{row['stem']} | "
                f"Gain "
                f"{row['region_gain']:+.4f} dB | "
                f"Coverage "
                f"{row['region_coverage_percent']:.3f} %"
            )


if __name__ == "__main__":
    main()