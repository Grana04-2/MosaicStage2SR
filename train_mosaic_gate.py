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
# Seed
# ============================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Loss
# ============================================================


def soft_dice_loss(
        logits,
        target,
        epsilon=1e-6,
):
    probability = torch.sigmoid(
        logits
    )

    dims = (
        1,
        2,
        3,
    )

    intersection = (
        probability
        * target
    ).sum(
        dim=dims
    )

    denominator = (
        probability.sum(
            dim=dims
        )
        + target.sum(
            dim=dims
        )
    )

    dice = (
        2.0 * intersection
        + epsilon
    ) / (
        denominator
        + epsilon
    )

    return (
        1.0
        - dice.mean()
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


@torch.no_grad()
def accumulate_metrics(
        totals,
        logits,
        target,
        threshold,
):
    probability = torch.sigmoid(
        logits.float()
    )

    prediction = (
        probability
        >= threshold
    )

    target_binary = (
        target
        >= 0.5
    )

    totals["tp"] += (
        prediction
        & target_binary
    ).sum().item()

    totals["fp"] += (
        prediction
        & ~target_binary
    ).sum().item()

    totals["fn"] += (
        ~prediction
        & target_binary
    ).sum().item()

    totals["tn"] += (
        ~prediction
        & ~target_binary
    ).sum().item()

    totals["pred_positive"] += (
        prediction.sum().item()
    )

    totals["target_positive"] += (
        target_binary.sum().item()
    )

    totals["pixels"] += (
        target_binary.numel()
    )


def finalize_metrics(totals):
    epsilon = 1e-8

    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    tn = totals["tn"]
    pixels = max(
        totals["pixels"],
        1.0,
    )

    precision = (
        tp
        / (tp + fp + epsilon)
    )

    recall = (
        tp
        / (tp + fn + epsilon)
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
        (tp + tn)
        / pixels
    )

    pred_coverage = (
        totals["pred_positive"]
        / pixels
    )

    target_coverage = (
        totals["target_positive"]
        / pixels
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "pred_coverage": pred_coverage,
        "target_coverage": target_coverage,
    }


# ============================================================
# Visualization
# ============================================================


def tensor_rgb_to_pil(tensor):
    tensor = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    array = (
        tensor
        .permute(
            1,
            2,
            0,
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
        array,
        mode="RGB",
    )


def tensor_mask_to_pil(tensor):
    tensor = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    if tensor.dim() == 3:
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
        mode="L",
    ).convert(
        "RGB"
    )


def add_title(
        image,
        title,
        bar_height=28,
):
    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + bar_height,
        ),
        "white",
    )

    canvas.paste(
        image,
        (
            0,
            bar_height,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            8,
            6,
        ),
        title,
        fill="black",
    )

    return canvas


def save_sample_panel(
        stage1,
        mosaic,
        target,
        logits,
        threshold,
        save_path,
):
    probability = torch.sigmoid(
        logits
    )

    prediction = (
        probability
        >= threshold
    ).float()

    panels = [
        add_title(
            tensor_rgb_to_pil(
                stage1
            ),
            "Stage1",
        ),
        add_title(
            tensor_rgb_to_pil(
                mosaic
            ),
            "Original Mosaic",
        ),
        add_title(
            tensor_mask_to_pil(
                target
            ),
            "Target train_S",
        ),
        add_title(
            tensor_mask_to_pil(
                probability
            ),
            "Pred probability",
        ),
        add_title(
            tensor_mask_to_pil(
                prediction
            ),
            f"Pred binary >= {threshold:.2f}",
        ),
    ]

    total_width = sum(
        panel.width
        for panel in panels
    )

    max_height = max(
        panel.height
        for panel in panels
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            max_height,
        ),
        "white",
    )

    x = 0

    for panel in panels:
        canvas.paste(
            panel,
            (
                x,
                0,
            ),
        )

        x += panel.width

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        save_path
    )


# ============================================================
# Train
# ============================================================


def train_one_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device,
        use_amp,
        bce_pos_weight,
        dice_weight,
        threshold,
):
    model.train()

    total_loss_sum = 0.0
    bce_loss_sum = 0.0
    dice_loss_sum = 0.0
    image_count = 0

    metric_totals = empty_metric_totals()

    pos_weight_tensor = torch.tensor(
        [bce_pos_weight],
        dtype=torch.float32,
        device=device,
    )

    for batch_index, batch in enumerate(
            loader,
            start=1,
    ):
        stage1 = batch[
            "lr"
        ].to(
            device,
            non_blocking=True,
        )

        mosaic = batch[
            "mosaic"
        ].to(
            device,
            non_blocking=True,
        )

        target = batch[
            "supervision"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
                enabled=use_amp,
        ):
            logits = model(
                stage1,
                mosaic,
            )

            if logits.shape != target.shape:
                raise RuntimeError(
                    "Gate output / target shape mismatch:\n"
                    f"logits = {tuple(logits.shape)}\n"
                    f"target = {tuple(target.shape)}"
                )

            loss_bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight_tensor,
            )

            loss_dice = soft_dice_loss(
                logits,
                target,
            )

            loss_total = (
                loss_bce
                + float(dice_weight)
                * loss_dice
            )

        scaler.scale(
            loss_total
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = stage1.shape[0]
        image_count += batch_size

        total_loss_sum += (
            loss_total.item()
            * batch_size
        )

        bce_loss_sum += (
            loss_bce.item()
            * batch_size
        )

        dice_loss_sum += (
            loss_dice.item()
            * batch_size
        )

        accumulate_metrics(
            totals=metric_totals,
            logits=logits.detach(),
            target=target,
            threshold=threshold,
        )

        if (
                batch_index % 100 == 0
                or
                batch_index == len(loader)
        ):
            current_metrics = finalize_metrics(
                metric_totals
            )

            print(
                "Train "
                f"{batch_index:04d}/"
                f"{len(loader):04d} | "
                f"Loss {loss_total.item():.6f} | "
                f"BCE {loss_bce.item():.6f} | "
                f"DiceLoss {loss_dice.item():.6f} | "
                f"IoU {current_metrics['iou']:.4f} | "
                f"Recall {current_metrics['recall']:.4f} | "
                f"PredCov {current_metrics['pred_coverage']:.4f}"
            )

    image_count = max(
        1,
        image_count,
    )

    metrics = finalize_metrics(
        metric_totals
    )

    return {
        "loss": total_loss_sum / image_count,
        "bce": bce_loss_sum / image_count,
        "dice_loss": dice_loss_sum / image_count,
        **metrics,
    }


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def validate(
        model,
        loader,
        device,
        bce_pos_weight,
        dice_weight,
        threshold,
        sample_dir,
        epoch,
        sample_count,
):
    model.eval()

    total_loss_sum = 0.0
    bce_loss_sum = 0.0
    dice_loss_sum = 0.0
    image_count = 0

    metric_totals = empty_metric_totals()

    pos_weight_tensor = torch.tensor(
        [bce_pos_weight],
        dtype=torch.float32,
        device=device,
    )

    epoch_sample_dir = (
        Path(sample_dir)
        / f"epoch_{epoch:03d}"
    )

    epoch_sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for batch_index, batch in enumerate(
            loader,
    ):
        stage1 = batch[
            "lr"
        ].to(
            device,
        )

        mosaic = batch[
            "mosaic"
        ].to(
            device,
        )

        target = batch[
            "supervision"
        ].to(
            device,
        )

        logits = model(
            stage1,
            mosaic,
        )

        loss_bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight_tensor,
        )

        loss_dice = soft_dice_loss(
            logits,
            target,
        )

        loss_total = (
            loss_bce
            + float(dice_weight)
            * loss_dice
        )

        batch_size = stage1.shape[0]
        image_count += batch_size

        total_loss_sum += (
            loss_total.item()
            * batch_size
        )

        bce_loss_sum += (
            loss_bce.item()
            * batch_size
        )

        dice_loss_sum += (
            loss_dice.item()
            * batch_size
        )

        accumulate_metrics(
            totals=metric_totals,
            logits=logits,
            target=target,
            threshold=threshold,
        )

        if batch_index < sample_count:
            stem = batch[
                "stem"
            ][0]

            save_sample_panel(
                stage1=stage1[0],
                mosaic=mosaic[0],
                target=target[0],
                logits=logits[0],
                threshold=threshold,
                save_path=(
                    epoch_sample_dir
                    / f"{stem}_gate.png"
                ),
            )

    image_count = max(
        1,
        image_count,
    )

    metrics = finalize_metrics(
        metric_totals
    )

    return {
        "loss": total_loss_sum / image_count,
        "bce": bce_loss_sum / image_count,
        "dice_loss": dice_loss_sum / image_count,
        **metrics,
    }


# ============================================================
# Checkpoint / log
# ============================================================


def save_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        epoch,
        best_iou,
        args,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_iou": best_iou,
        "args": vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )


def append_log(
        log_path,
        row,
):
    log_path = Path(
        log_path
    )

    exists = log_path.exists()

    with open(
            log_path,
            "a",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# Arguments
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./datasets/stage2_lp21_sigma6_mask67_1280x720",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="mosaic_gate_v1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lr_patch",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--split_mode",
        type=str,
        choices=[
            "frame",
            "sequence",
        ],
        default="sequence",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--dice_weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(
        f"Device: {device}"
    )

    print(
        f"AMP: {use_amp}"
    )

    records = load_manifest(
        args.dataset_root
    )

    if args.split_mode == "sequence":
        (
            train_records,
            val_records,
        ) = split_records_by_sequence(
            records,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    else:
        (
            train_records,
            val_records,
        ) = split_records(
            records,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    train_sequence_ids = sorted(
        {
            get_sequence_id_from_stem(
                record["stem"]
            )
            for record in train_records
        }
    )

    val_sequence_ids = sorted(
        {
            get_sequence_id_from_stem(
                record["stem"]
            )
            for record in val_records
        }
    )

    sequence_overlap = (
        set(train_sequence_ids)
        & set(val_sequence_ids)
    )

    print()
    print("Dataset split")
    print(f"  split mode      : {args.split_mode}")
    print(f"  train frames    : {len(train_records)}")
    print(f"  val frames      : {len(val_records)}")
    print(f"  train sequences : {len(train_sequence_ids)}")
    print(f"  val sequences   : {len(val_sequence_ids)}")
    print(f"  overlap         : {len(sequence_overlap)}")
    print(f"  val sequence IDs: {', '.join(val_sequence_ids)}")

    if (
            args.split_mode == "sequence"
            and sequence_overlap
    ):
        raise RuntimeError(
            "Sequence leakage detected: "
            f"{sorted(sequence_overlap)}"
        )

    train_dataset = Stage2SRDataset(
        dataset_root=args.dataset_root,
        records=train_records,
        train=True,
        lr_patch_size=args.lr_patch,
        mosaic_dir_name="train_M",
        supervision_dir_name="train_S",
    )

    val_dataset = Stage2SRDataset(
        dataset_root=args.dataset_root,
        records=val_records,
        train=False,
        lr_patch_size=args.lr_patch,
        horizontal_flip=False,
        mosaic_dir_name="train_M",
        supervision_dir_name="train_S",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = MosaicGateNet(
        input_channels=6,
        base_channels=args.base_channels,
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.99,
        ),
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

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
        exist_ok=True,
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        checkpoint_dir
        / "training_log.csv"
    )

    start_epoch = 1
    best_iou = -1.0

    if args.resume:
        resume_path = Path(
            args.resume
        )

        print(
            f"Loading checkpoint: {resume_path}"
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if "scaler" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        best_iou = checkpoint.get(
            "best_iou",
            -1.0,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

        print(
            f"Resume from epoch {start_epoch}"
        )

    for epoch in range(
            start_epoch,
            args.epochs + 1,
    ):
        epoch_start = time.time()

        print()
        print("========================================")
        print(f"Epoch {epoch}/{args.epochs}")
        print("========================================")

        train_results = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            bce_pos_weight=args.bce_pos_weight,
            dice_weight=args.dice_weight,
            threshold=args.threshold,
        )

        val_results = validate(
            model=model,
            loader=val_loader,
            device=device,
            bce_pos_weight=args.bce_pos_weight,
            dice_weight=args.dice_weight,
            threshold=args.threshold,
            sample_dir=sample_dir,
            epoch=epoch,
            sample_count=args.sample_count,
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print()
        print("----------------------------------------")
        print(
            "Train | "
            f"Loss {train_results['loss']:.6f} | "
            f"IoU {train_results['iou']:.4f} | "
            f"F1 {train_results['f1']:.4f} | "
            f"Recall {train_results['recall']:.4f} | "
            f"PredCov {train_results['pred_coverage']:.4f} | "
            f"TargetCov {train_results['target_coverage']:.4f}"
        )

        print(
            "Val   | "
            f"Loss {val_results['loss']:.6f} | "
            f"IoU {val_results['iou']:.4f} | "
            f"Precision {val_results['precision']:.4f} | "
            f"Recall {val_results['recall']:.4f} | "
            f"F1 {val_results['f1']:.4f} | "
            f"PredCov {val_results['pred_coverage']:.4f} | "
            f"TargetCov {val_results['target_coverage']:.4f}"
        )

        print(
            f"Time: {elapsed:.1f} sec"
        )

        is_best = (
            val_results["iou"]
            > best_iou
        )

        if is_best:
            best_iou = val_results["iou"]

        epoch_path = (
            checkpoint_dir
            / f"epoch_{epoch:03d}.pth"
        )

        save_checkpoint(
            path=epoch_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_iou=best_iou,
            args=args,
        )

        latest_path = (
            checkpoint_dir
            / "latest.pth"
        )

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_iou=best_iou,
            args=args,
        )

        if is_best:
            best_path = (
                checkpoint_dir
                / "best_iou.pth"
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_iou=best_iou,
                args=args,
            )

            print(
                f"New best IoU: {best_iou:.4f}"
            )

        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_results["loss"],
                "train_bce": train_results["bce"],
                "train_dice_loss": train_results["dice_loss"],
                "train_iou": train_results["iou"],
                "train_precision": train_results["precision"],
                "train_recall": train_results["recall"],
                "train_f1": train_results["f1"],
                "train_pred_coverage": train_results["pred_coverage"],
                "train_target_coverage": train_results["target_coverage"],
                "val_loss": val_results["loss"],
                "val_bce": val_results["bce"],
                "val_dice_loss": val_results["dice_loss"],
                "val_iou": val_results["iou"],
                "val_precision": val_results["precision"],
                "val_recall": val_results["recall"],
                "val_f1": val_results["f1"],
                "val_accuracy": val_results["accuracy"],
                "val_pred_coverage": val_results["pred_coverage"],
                "val_target_coverage": val_results["target_coverage"],
                "seconds": elapsed,
            },
        )

    print()
    print("========================================")
    print("MosaicGateNet training completed")
    print("========================================")
    print(f"Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
