import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.stage2_dataset import (
    Stage2SRDataset,
    get_sequence_id_from_stem,
    load_manifest,
    split_records,
    split_records_by_sequence,
)

from models.mosaic_gate import (
    MosaicGateNet,
)


# ============================================================
# Metrics
# ============================================================

def empty_metric_totals():

    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "pred_positive": 0.0,
        "target_positive": 0.0,
        "pixels": 0.0,
    }


def accumulate_binary_metrics(
        totals,
        prediction,
        target
):

    prediction = (
        prediction
        >= 0.5
    )

    target_binary = (
        target
        >= 0.5
    )

    totals[
        "tp"
    ] += (
        prediction
        & target_binary
    ).sum().item()

    totals[
        "fp"
    ] += (
        prediction
        & ~target_binary
    ).sum().item()

    totals[
        "fn"
    ] += (
        ~prediction
        & target_binary
    ).sum().item()

    totals[
        "tn"
    ] += (
        ~prediction
        & ~target_binary
    ).sum().item()

    totals[
        "pred_positive"
    ] += (
        prediction.sum().item()
    )

    totals[
        "target_positive"
    ] += (
        target_binary.sum().item()
    )

    totals[
        "pixels"
    ] += (
        target_binary.numel()
    )


def finalize_metrics(
        totals
):

    epsilon = 1e-8

    tp = totals[
        "tp"
    ]

    fp = totals[
        "fp"
    ]

    fn = totals[
        "fn"
    ]

    tn = totals[
        "tn"
    ]

    pixels = max(
        totals[
            "pixels"
        ],
        1.0
    )

    precision = (
        tp
        / (
            tp
            + fp
            + epsilon
        )
    )

    recall = (
        tp
        / (
            tp
            + fn
            + epsilon
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + epsilon
        )
    )

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + epsilon
        )
    )

    accuracy = (
        (
            tp
            + tn
        )
        / pixels
    )

    pred_coverage = (
        totals[
            "pred_positive"
        ]
        / pixels
    )

    target_coverage = (
        totals[
            "target_positive"
        ]
        / pixels
    )

    return {
        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "accuracy":
            accuracy,

        "pred_coverage":
            pred_coverage,

        "target_coverage":
            target_coverage,
    }


# ============================================================
# Single-image metrics
# ============================================================

def calculate_single_metrics(
        prediction,
        target
):

    totals = (
        empty_metric_totals()
    )

    accumulate_binary_metrics(
        totals=totals,
        prediction=prediction,
        target=target
    )

    return finalize_metrics(
        totals
    )


# ============================================================
# Visualization
# ============================================================

def mask_to_pil(
        tensor
):

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

        tensor = tensor[
            0,
            0
        ]

    elif tensor.dim() == 3:

        tensor = tensor[
            0
        ]

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
        bar_height=28
):

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + bar_height
        ),
        "white"
    )

    canvas.paste(
        image,
        (
            0,
            bar_height
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


def save_threshold_panel(
        target,
        probability,
        thresholds,
        save_path
):

    panels = []

    panels.append(
        add_title(
            mask_to_pil(
                target
            ),
            "Target train_S"
        )
    )

    panels.append(
        add_title(
            mask_to_pil(
                probability
            ),
            "Pred probability"
        )
    )

    for threshold in thresholds:

        prediction = (
            probability
            >= threshold
        ).float()

        panels.append(
            add_title(
                mask_to_pil(
                    prediction
                ),
                (
                    f"Pred >= "
                    f"{threshold:.2f}"
                )
            )
        )

    total_width = sum(
        panel.width
        for panel
        in panels
    )

    max_height = max(
        panel.height
        for panel
        in panels
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            max_height
        ),
        "white"
    )

    x = 0

    for panel in panels:

        canvas.paste(
            panel,
            (
                x,
                0
            )
        )

        x += (
            panel.width
        )

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    canvas.save(
        save_path
    )


# ============================================================
# CSV
# ============================================================

def write_csv(
        path,
        rows
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        return

    with open(
            path,
            "w",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.50,
            0.60,
            0.70,
            0.80,
        ]
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=10
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
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

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
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

    # ========================================================
    # Training configuration from checkpoint
    # ========================================================

    if args.dataset_root:

        dataset_root = (
            args.dataset_root
        )

    else:

        dataset_root = (
            checkpoint_args.get(
                "dataset_root",
                (
                    "./datasets/"
                    "stage2_lp21_sigma6_mask67_1280x720"
                )
            )
        )

    val_ratio = (
        checkpoint_args.get(
            "val_ratio",
            0.10
        )
    )

    split_mode = (
        checkpoint_args.get(
            "split_mode",
            "sequence"
        )
    )

    seed = (
        checkpoint_args.get(
            "seed",
            1234
        )
    )

    lr_patch = (
        checkpoint_args.get(
            "lr_patch",
            192
        )
    )

    base_channels = (
        checkpoint_args.get(
            "base_channels",
            32
        )
    )

    thresholds = sorted(
        {
            float(
                threshold
            )
            for threshold
            in args.thresholds
        }
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
            / (
                "threshold_eval_"
                + checkpoint_path.stem
            )
        )

    sample_dir = (
        output_dir
        / "samples"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "MosaicGate threshold evaluation"
    )

    print(
        "========================================"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Checkpoint   : "
        f"{checkpoint_path}"
    )

    print(
        f"Epoch        : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Dataset root : "
        f"{dataset_root}"
    )

    print(
        f"Split mode   : "
        f"{split_mode}"
    )

    print(
        f"Val ratio    : "
        f"{val_ratio}"
    )

    print(
        f"Seed         : "
        f"{seed}"
    )

    print(
        f"Thresholds   : "
        f"{thresholds}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    records = load_manifest(
        dataset_root
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

    train_sequence_ids = sorted(
        {
            get_sequence_id_from_stem(
                record[
                    "stem"
                ]
            )
            for record
            in train_records
        }
    )

    val_sequence_ids = sorted(
        {
            get_sequence_id_from_stem(
                record[
                    "stem"
                ]
            )
            for record
            in val_records
        }
    )

    overlap = (
        set(
            train_sequence_ids
        )
        &
        set(
            val_sequence_ids
        )
    )

    print()
    print(
        "Validation split"
    )

    print(
        f"  train frames    : "
        f"{len(train_records)}"
    )

    print(
        f"  val frames      : "
        f"{len(val_records)}"
    )

    print(
        f"  train sequences : "
        f"{len(train_sequence_ids)}"
    )

    print(
        f"  val sequences   : "
        f"{len(val_sequence_ids)}"
    )

    print(
        f"  overlap         : "
        f"{len(overlap)}"
    )

    print(
        f"  val sequence IDs: "
        f"{', '.join(val_sequence_ids)}"
    )

    if overlap:

        raise RuntimeError(
            "Sequence leakage detected: "
            f"{sorted(overlap)}"
        )

    val_dataset = Stage2SRDataset(
        dataset_root=dataset_root,
        records=val_records,
        train=False,
        lr_patch_size=lr_patch,
        horizontal_flip=False,
        mosaic_dir_name="train_M",
        supervision_dir_name="train_S"
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
    # Model
    # ========================================================

    model = MosaicGateNet(
        input_channels=6,
        base_channels=base_channels
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

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print()
    print(
        f"Parameters    : "
        f"{parameter_count:,}"
    )

    # ========================================================
    # Threshold metric containers
    # ========================================================

    metric_totals = {
        threshold:
            empty_metric_totals()
        for threshold
        in thresholds
    }

    per_image_rows = []

    # ========================================================
    # Evaluation
    # ========================================================

    for batch_index, batch in enumerate(
            val_loader
    ):

        stage1 = (
            batch[
                "lr"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        mosaic = (
            batch[
                "mosaic"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        target = (
            batch[
                "supervision"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        logits = model(
            stage1,
            mosaic
        )

        probability = (
            torch.sigmoid(
                logits.float()
            )
        )

        stem = (
            batch[
                "stem"
            ][
                0
            ]
        )

        # ====================================================
        # All thresholds from same probability map
        # ====================================================

        for threshold in thresholds:

            prediction = (
                probability
                >= threshold
            ).float()

            accumulate_binary_metrics(
                totals=metric_totals[
                    threshold
                ],
                prediction=prediction,
                target=target
            )

            image_metrics = (
                calculate_single_metrics(
                    prediction=prediction,
                    target=target
                )
            )

            per_image_rows.append(
                {
                    "stem":
                        stem,

                    "threshold":
                        threshold,

                    "iou":
                        image_metrics[
                            "iou"
                        ],

                    "precision":
                        image_metrics[
                            "precision"
                        ],

                    "recall":
                        image_metrics[
                            "recall"
                        ],

                    "f1":
                        image_metrics[
                            "f1"
                        ],

                    "pred_coverage":
                        image_metrics[
                            "pred_coverage"
                        ],

                    "target_coverage":
                        image_metrics[
                            "target_coverage"
                        ],
                }
            )

        # ====================================================
        # Visualization
        # ====================================================

        if (
                batch_index
                < args.sample_count
        ):

            save_threshold_panel(
                target=target[
                    0
                ],
                probability=probability[
                    0
                ],
                thresholds=thresholds,
                save_path=(
                    sample_dir
                    / (
                        stem
                        + "_thresholds.png"
                    )
                )
            )

        if (
                (
                    batch_index
                    + 1
                )
                % 100
                == 0
                or
                (
                    batch_index
                    + 1
                )
                == len(
                    val_loader
                )
        ):

            print(
                f"Eval "
                f"{batch_index + 1:04d}/"
                f"{len(val_loader):04d}"
            )

    # ========================================================
    # Summary
    # ========================================================

    summary_rows = []

    print()
    print(
        "========================================"
    )

    print(
        "Threshold summary"
    )

    print(
        "========================================"
    )

    for threshold in thresholds:

        metrics = finalize_metrics(
            metric_totals[
                threshold
            ]
        )

        row = {
            "threshold":
                threshold,

            "iou":
                metrics[
                    "iou"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "pred_coverage":
                metrics[
                    "pred_coverage"
                ],

            "target_coverage":
                metrics[
                    "target_coverage"
                ],
        }

        summary_rows.append(
            row
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"IoU {metrics['iou']:.4f} | "
            f"Precision {metrics['precision']:.4f} | "
            f"Recall {metrics['recall']:.4f} | "
            f"F1 {metrics['f1']:.4f} | "
            f"PredCov {metrics['pred_coverage']:.4f} | "
            f"TargetCov {metrics['target_coverage']:.4f}"
        )

    # ========================================================
    # Save CSV
    # ========================================================

    summary_csv = (
        output_dir
        / "threshold_summary.csv"
    )

    per_image_csv = (
        output_dir
        / "threshold_per_image.csv"
    )

    write_csv(
        summary_csv,
        summary_rows
    )

    write_csv(
        per_image_csv,
        per_image_rows
    )

    print()
    print(
        "========================================"
    )

    print(
        "Completed"
    )

    print(
        "========================================"
    )

    print(
        f"Summary CSV  : "
        f"{summary_csv.resolve()}"
    )

    print(
        f"Per-image CSV: "
        f"{per_image_csv.resolve()}"
    )

    print(
        f"Samples      : "
        f"{sample_dir.resolve()}"
    )


if __name__ == "__main__":
    main()