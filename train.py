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
import matplotlib.cm as cm

from data.stage2_dataset import (
    load_manifest,
    split_records,
    split_records_by_sequence,
    get_sequence_id_from_stem,
    Stage2SRDataset,
)

from models.residual_sr import (
    ResidualSRNet
)

from models.mosaic_gate import (
    MosaicGateNet
)

from losses import (
    CharbonnierLoss,
    gradient_loss,
    laplacian_loss,

    build_region_weight,
    build_hard_region_weight,

    weighted_charbonnier_loss,
    weighted_gradient_loss,
    weighted_laplacian_loss,
)


# ============================================================
# seed
# ============================================================

def set_seed(seed):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# tensor -> PIL
# ============================================================

def tensor_to_pil(tensor):
    tensor = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0
        )
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
            + 0.5
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array
    )


# ============================================================
# PSNR
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


@torch.no_grad()
def build_safe_mosaic(
        gate_model,
        lr,
        mosaic,
        gate_low=0.60,
        gate_high=0.80,
        gate_strength=1.0
):
    """
    Frozen MosaicGateNetを使って、
    Original MosaicをStage1へsoft blendする。

    probability <= gate_low:
        Original Mosaicをそのまま使用

    probability >= gate_high:
        Stage1へ完全置換

    その間:
        soft transition
    """

    if gate_high <= gate_low:
        raise ValueError(
            "gate_high must be greater than gate_low. "
            f"Got low={gate_low}, high={gate_high}"
        )

    logits = gate_model(
        lr,
        mosaic
    )

    probability = torch.sigmoid(
        logits.float()
    )

    gate = (
                   probability
                   - float(
               gate_low
           )
           ) / (
                   float(
                       gate_high
                   )
                   - float(
               gate_low
           )
           )

    gate = torch.clamp(
        gate,
        0.0,
        1.0
    )

    gate = gate.to(
        dtype=mosaic.dtype
    )

    if (
            gate_strength < 0.0
            or
            gate_strength > 1.0
    ):
        raise ValueError(
            "gate_strength must be between 0.0 and 1.0. "
            f"Got {gate_strength}"
        )

    effective_gate = (
            float(
                gate_strength
            )
            * gate
    )

    safe_mosaic = (
            (
                    1.0
                    - effective_gate
            )
            * mosaic
            +
            effective_gate
            * lr
    )

    return (
        safe_mosaic,
        probability,
        effective_gate
    )


# ============================================================
# comparison image
# ============================================================

def add_title(
        image,
        title,
        title_height=30
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
            8
        ),
        title,
        fill="black"
    )

    return canvas


def save_comparison(
        lr,
        sr,
        hr,
        model,
        save_path
):
    with torch.no_grad():
        bicubic = (
            model.upscale_bicubic(
                lr.unsqueeze(0)
            )[0]
        )

    bicubic_image = add_title(
        tensor_to_pil(
            bicubic
        ),
        "Bicubic"
    )

    sr_image = add_title(
        tensor_to_pil(
            sr
        ),
        "Residual SR"
    )

    hr_image = add_title(
        tensor_to_pil(
            hr
        ),
        "Target"
    )

    width = (
            bicubic_image.width
            + sr_image.width
            + hr_image.width
    )

    height = max(
        bicubic_image.height,
        sr_image.height,
        hr_image.height
    )

    output = Image.new(
        "RGB",
        (
            width,
            height
        ),
        "white"
    )

    x = 0

    for image in [
        bicubic_image,
        sr_image,
        hr_image
    ]:
        output.paste(
            image,
            (
                x,
                0
            )
        )

        x += image.width

    output.save(
        save_path
    )


def tensor_to_uint8_rgb(tensor):
    x = (
        tensor.detach()
        .float()
        .cpu()
    )

    if x.dim() == 4:
        x = x[0]

    x = x.clamp(
        0.0,
        1.0
    )

    x = (
        x.permute(
            1,
            2,
            0
        )
        .numpy()
    )

    x = (
            x * 255.0
            + 0.5
    ).astype(
        np.uint8
    )

    return x


def single_map_to_heatmap_uint8(
        tensor,
        vmin=None,
        vmax=None,
        cmap_name="turbo"
):
    x = (
        tensor.detach()
        .float()
        .cpu()
    )

    if x.dim() == 4:
        x = x[0, 0]

    elif x.dim() == 3:
        x = x[0]

    arr = x.numpy()

    if vmin is None:
        vmin = float(
            arr.min()
        )

    if vmax is None:
        vmax = float(
            arr.max()
        )

    denom = max(
        vmax - vmin,
        1e-8
    )

    arr_norm = np.clip(
        (arr - vmin) / denom,
        0.0,
        1.0
    )

    cmap = cm.get_cmap(
        cmap_name
    )

    rgb = (
        cmap(
            arr_norm
        )[..., :3]
    )

    rgb = (
            rgb * 255.0
            + 0.5
    ).astype(
        np.uint8
    )

    return rgb


def add_title_bar(
        img_array,
        title
):
    img = Image.fromarray(
        img_array
    )

    bar_h = 28

    canvas = Image.new(
        "RGB",
        (
            img.width,
            img.height + bar_h
        ),
        "white"
    )

    canvas.paste(
        img,
        (
            0,
            bar_h
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


def save_fixed_sample_region_panel(
        save_path,
        bicubic_tensor,
        sr_tensor,
        target_tensor,
        score_map,
        weight_map,
        region_boost
):
    bicubic_img = add_title_bar(
        tensor_to_uint8_rgb(
            bicubic_tensor
        ),
        "Bicubic"
    )

    sr_img = add_title_bar(
        tensor_to_uint8_rgb(
            sr_tensor
        ),
        "Residual SR"
    )

    target_img = add_title_bar(
        tensor_to_uint8_rgb(
            target_tensor
        ),
        "Target"
    )

    score_img = add_title_bar(
        single_map_to_heatmap_uint8(
            score_map,
            vmin=0.0,
            vmax=1.0,
            cmap_name="turbo"
        ),
        "Region score"
    )

    weight_min = float(
        weight_map.min()
        .detach()
        .cpu()
    )

    weight_max = float(
        weight_map.max()
        .detach()
        .cpu()
    )

    weight_img = add_title_bar(
        single_map_to_heatmap_uint8(
            weight_map,
            vmin=1.0,
            vmax=(
                    1.0
                    + float(
                region_boost
            )
            ),
            cmap_name="turbo"
        ),
        (
            f"Weight "
            f"{weight_min:.3f}-"
            f"{weight_max:.3f}"
        )
    )

    panels = [
        bicubic_img,
        sr_img,
        target_img,
        score_img,
        weight_img
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

        x += panel.width

    canvas.save(
        save_path
    )


# ============================================================
# train
# ============================================================

def train_one_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device,
        pixel_weight,
        use_amp
):
    model.train()

    totals = {
        "total": 0.0,
        "pixel": 0.0,
    }

    image_count = 0

    for batch_index, batch in enumerate(
            loader,
            start=1
    ):

        # ================================================
        # Input
        #
        # lr:
        #   train_A
        #
        # mosaic:
        #   train_M
        # ================================================

        lr = batch[
            "lr"
        ].to(
            device,
            non_blocking=True
        )

        mosaic = batch[
            "mosaic"
        ].to(
            device,
            non_blocking=True
        )

        hr = batch[
            "hr"
        ].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
                enabled=use_amp
        ):

            # ================================================
            # 6ch Stage2
            # ================================================

            sr = model(
                lr,
                mosaic
            )

            # ================================================
            # Full-area Pixel-only
            #
            # overfit_multi_image と同じ意味:
            # 全画素 weight = 1.0
            # ================================================

            weight_map = torch.ones(
                (
                    hr.shape[0],
                    1,
                    hr.shape[-2],
                    hr.shape[-1]
                ),
                device=hr.device,
                dtype=hr.dtype
            )

            loss_pixel = (
                weighted_charbonnier_loss(
                    sr,
                    hr,
                    weight_map
                )
            )

            loss_total = (
                    float(
                        pixel_weight
                    )
                    * loss_pixel
            )

        # ================================================
        # Backward
        # ================================================

        scaler.scale(
            loss_total
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # ================================================
        # Statistics
        # ================================================

        batch_size = (
            lr.shape[0]
        )

        image_count += (
            batch_size
        )

        totals[
            "total"
        ] += (
                loss_total.item()
                * batch_size
        )

        totals[
            "pixel"
        ] += (
                loss_pixel.item()
                * batch_size
        )

        # ================================================
        # Progress
        # ================================================

        if (
                batch_index % 100 == 0
                or
                batch_index == len(loader)
        ):
            print(
                "Train "
                f"{batch_index:04d}/"
                f"{len(loader):04d} | "
                f"Total "
                f"{loss_total.item():.6f} | "
                f"Pixel "
                f"{loss_pixel.item():.6f}"
            )

    # ================================================
    # Epoch average
    # ================================================

    image_count = max(
        1,
        image_count
    )

    for key in totals:
        totals[
            key
        ] /= (
            image_count
        )

    return totals


# ============================================================
# validation
# ============================================================

@torch.no_grad()
def validate(
        model,
        loader,
        pixel_loss_fn,
        device,
        sample_dir,
        epoch,
        sample_count=4
):
    model.eval()

    sr_psnr_total = 0.0
    bicubic_psnr_total = 0.0

    pixel_total = 0.0
    gradient_total = 0.0
    laplacian_total = 0.0

    count = 0

    epoch_sample_dir = (
            sample_dir
            / f"epoch_{epoch:03d}"
    )

    epoch_sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for batch_index, batch in enumerate(
            loader
    ):

        lr = batch[
            "lr"
        ].to(
            device
        )

        mosaic = batch[
            "mosaic"
        ].to(
            device
        )

        hr = batch[
            "hr"
        ].to(
            device
        )

        sr = model(
            lr,
            mosaic
        )

        sr_clamped = sr.clamp(
            0.0,
            1.0
        )

        bicubic = (
            model.upscale_bicubic(
                lr
            ).clamp(
                0.0,
                1.0
            )
        )

        sr_psnr = calculate_psnr(
            sr_clamped,
            hr
        )

        bicubic_psnr = (
            calculate_psnr(
                bicubic,
                hr
            )
        )

        loss_pixel = pixel_loss_fn(
            sr,
            hr
        )

        loss_gradient = gradient_loss(
            sr,
            hr
        )

        loss_laplacian = laplacian_loss(
            sr,
            hr
        )

        sr_psnr_total += (
            sr_psnr
        )

        bicubic_psnr_total += (
            bicubic_psnr
        )

        pixel_total += (
            loss_pixel.item()
        )

        gradient_total += (
            loss_gradient.item()
        )

        laplacian_total += (
            loss_laplacian.item()
        )

        count += 1

        # ================================================
        # visual samples
        # ================================================

        if (
                batch_index
                < sample_count
        ):
            stem = batch[
                "stem"
            ][0]

            save_path = (
                    epoch_sample_dir
                    / f"{stem}_compare.png"
            )

            save_comparison(
                lr=lr[0],
                sr=sr_clamped[0],
                hr=hr[0],
                model=model,
                save_path=save_path
            )

    count = max(
        1,
        count
    )

    results = {
        "sr_psnr":
            sr_psnr_total
            / count,

        "bicubic_psnr":
            bicubic_psnr_total
            / count,

        "pixel":
            pixel_total
            / count,

        "gradient":
            gradient_total
            / count,

        "laplacian":
            laplacian_total
            / count,
    }

    return results


@torch.no_grad()
def save_fixed_sample(
        model,
        dataset_root,
        records,
        target_stem,
        device,
        sample_dir,
        epoch,
        lr_patch_size,
        region_boost,
        region_threshold,
        region_range,
        region_smooth
):
    if not target_stem:
        return

    target_stem = Path(
        target_stem
    ).stem

    target_record = None

    for record in records:

        if (
                record["stem"]
                == target_stem
        ):
            target_record = record
            break

    if target_record is None:
        print(
            f"[Fixed sample] "
            f"Stem not found: "
            f"{target_stem}"
        )

        return

    # ================================================
    # Full-size fixed sample
    # ================================================

    fixed_dataset = Stage2SRDataset(
        dataset_root=dataset_root,
        records=[
            target_record
        ],
        train=False,
        lr_patch_size=lr_patch_size,
        horizontal_flip=False,
        mosaic_dir_name="train_M"
    )

    item = fixed_dataset[0]

    lr = (
        item["lr"]
        .unsqueeze(0)
        .to(
            device
        )
    )

    mosaic = (
        item["mosaic"]
        .unsqueeze(0)
        .to(
            device
        )
    )

    hr = (
        item["hr"]
        .unsqueeze(0)
        .to(
            device
        )
    )

    model.eval()

    # ================================================
    # SR
    # ================================================

    sr = model(
        lr,
        mosaic
    ).clamp(
        0.0,
        1.0
    )

    # ================================================
    # Bicubic
    # ================================================

    bicubic = (
        model.upscale_bicubic(
            lr
        ).clamp(
            0.0,
            1.0
        )
    )

    # ================================================
    # IMPORTANT:
    # 学習と全く同じ Region map を使用
    #
    # build_region_weight の戻り値は
    #   weight_map, score_map
    # の順番
    # ================================================

    (
        weight_map,
        score_map
    ) = build_region_weight(
        bicubic=bicubic,
        target=hr,
        threshold=region_threshold,
        change_range=region_range,
        region_boost=region_boost,
        smooth_kernel=region_smooth
    )

    # ================================================
    # Epoch output folder
    # ================================================

    epoch_sample_dir = (
            sample_dir
            / f"epoch_{epoch:03d}"
    )

    epoch_sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ================================================
    # Existing 3-panel comparison
    # ================================================

    compare_path = (
            epoch_sample_dir
            / f"{target_stem}_compare.png"
    )

    save_comparison(
        lr=lr[0],
        sr=sr[0],
        hr=hr[0],
        model=model,
        save_path=compare_path
    )

    # ================================================
    # New 5-panel Region visualization
    # ================================================

    region_panel_path = (
            epoch_sample_dir
            / f"{target_stem}_region_panel.png"
    )

    save_fixed_sample_region_panel(
        save_path=region_panel_path,
        bicubic_tensor=bicubic,
        sr_tensor=sr,
        target_tensor=hr,
        score_map=score_map,
        weight_map=weight_map,
        region_boost=region_boost
    )

    # ================================================
    # PSNR
    # ================================================

    bicubic_psnr = calculate_psnr(
        bicubic,
        hr
    )

    sr_psnr = calculate_psnr(
        sr,
        hr
    )

    # ================================================
    # Region diagnostics
    # ================================================

    score_mean = (
        score_map.mean().item()
    )

    score_max = (
        score_map.max().item()
    )

    weight_mean = (
        weight_map.mean().item()
    )

    weight_max = (
        weight_map.max().item()
    )

    print(
        f"[Fixed sample] "
        f"{target_stem} | "
        f"Bicubic "
        f"{bicubic_psnr:.4f} dB | "
        f"SR "
        f"{sr_psnr:.4f} dB | "
        f"Gain "
        f"{sr_psnr - bicubic_psnr:+.4f} dB"
    )

    print(
        f"[Fixed sample region] "
        f"Score mean "
        f"{score_mean:.4f} | "
        f"max "
        f"{score_max:.4f} | "
        f"Weight mean "
        f"{weight_mean:.4f} | "
        f"max "
        f"{weight_max:.4f}"
    )

    print(
        f"[Fixed sample] Saved: "
        f"{compare_path}"
    )

    print(
        f"[Fixed sample] "
        f"Saved region panel: "
        f"{region_panel_path}"
    )


# ============================================================
# checkpoint
# ============================================================

def save_checkpoint(
        save_path,
        model,
        optimizer,
        scaler,
        epoch,
        best_psnr,
        args
):
    checkpoint = {
        "epoch":
            epoch,

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "scaler":
            scaler.state_dict(),

        "best_psnr":
            best_psnr,

        "args":
            vars(args),
    }

    torch.save(
        checkpoint,
        save_path
    )


# ============================================================
# csv log
# ============================================================

def append_log(
        log_path,
        row
):
    exists = log_path.exists()

    with open(
            log_path,
            "a",
            newline="",
            encoding="utf-8"
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
# arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fixed_sample",
        type=str,
        default=""
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./datasets/stage2_1280x720"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="residual_sr_v1"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--lr_patch",
        type=int,
        default=192
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234
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
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=""
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=4
    )

    parser.add_argument(
        "--load_model",
        type=str,
        default=""
    )

    parser.add_argument(
        "--region_boost",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--supervision_boost",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--gate_checkpoint",
        type=str,
        default=""
    )

    parser.add_argument(
        "--gate_low",
        type=float,
        default=0.60
    )

    parser.add_argument(
        "--gate_high",
        type=float,
        default=0.80
    )

    parser.add_argument(
        "--gate_strength",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--hard_region_boost",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--hard_gradient_threshold",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--hard_gradient_range",
        type=float,
        default=0.04
    )

    parser.add_argument(
        "--hard_gradient_smooth",
        type=int,
        default=3
    )

    parser.add_argument(
        "--region_threshold",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--region_range",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--region_smooth",
        type=int,
        default=5
    )

    parser.add_argument(
        "--split_mode",
        type=str,
        choices=[
            "frame",
            "sequence",
        ],
        default="frame",
        help=(
            "frame: conventional random frame split, "
            "sequence: keep each sequence entirely "
            "in train or validation"
        )
    )

    parser.add_argument(
        "--overfit_stem",
        type=str,
        default=""
    )

    parser.add_argument(
        "--overfit_repeat",
        type=int,
        default=400
    )

    return parser.parse_args()


# ============================================================
# main
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
            and
            device.type == "cuda"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = (
            True
        )

    print(
        f"Device: {device}"
    )

    print(
        f"AMP: {use_amp}"
    )

    # ================================================
    # Dataset
    # ================================================

    records = load_manifest(
        args.dataset_root
    )

    if args.split_mode == "sequence":

        (
            train_records,
            val_records
        ) = split_records_by_sequence(
            records,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

    else:

        (
            train_records,
            val_records
        ) = split_records(
            records,
            val_ratio=args.val_ratio,
            seed=args.seed
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

    sequence_overlap = (
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
        "Dataset split"
    )

    print(
        f"  split mode      : "
        f"{args.split_mode}"
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
        f"{len(sequence_overlap)}"
    )

    if (
            args.split_mode == "sequence"
            and
            sequence_overlap
    ):
        raise RuntimeError(
            "Sequence leakage detected in train.py: "
            f"{sorted(sequence_overlap)}"
        )

    print(
        f"  val sequence IDs: "
        f"{', '.join(val_sequence_ids)}"
    )

    print(
        f"Total pairs : {len(records)}"
    )

    print(
        f"Train pairs : {len(train_records)}"
    )

    print(
        f"Val pairs   : {len(val_records)}"
    )

    # ================================================
    # One-image overfit diagnostic
    # ================================================

    if args.overfit_stem:

        overfit_stem = Path(
            args.overfit_stem
        ).stem

        matched_records = [
            record
            for record in records
            if record["stem"] == overfit_stem
        ]

        if len(matched_records) != 1:
            raise RuntimeError(
                "Overfit stem must match exactly one record.\n"
                f"Stem    : {overfit_stem}\n"
                f"Matches : {len(matched_records)}"
            )

        if args.overfit_repeat < args.batch_size:
            raise ValueError(
                "--overfit_repeat must be >= --batch_size."
            )

        overfit_record = matched_records[0]

        # 同じ1枚を多数回並べる。
        # Stage2SRDataset(train=True)なので、
        # 各getitemでrandom crop位置は変化する。
        train_records = [
            overfit_record
            for _ in range(
                args.overfit_repeat
            )
        ]

        # Validationも同じ1枚。
        val_records = [
            overfit_record
        ]

        print()
        print(
            "========================================"
        )
        print(
            "ONE IMAGE OVERFIT MODE"
        )
        print(
            "========================================"
        )
        print(
            f"Stem         : {overfit_stem}"
        )
        print(
            f"Train repeats: {len(train_records)}"
        )
        print(
            f"Val frames   : {len(val_records)}"
        )
        print(
            "========================================"
        )

    train_dataset = Stage2SRDataset(
        dataset_root=args.dataset_root,
        records=train_records,
        train=True,
        lr_patch_size=args.lr_patch,
        mosaic_dir_name="train_M"
    )

    val_dataset = Stage2SRDataset(
        dataset_root=args.dataset_root,
        records=val_records,
        train=False,
        lr_patch_size=args.lr_patch,
        horizontal_flip=False,
        mosaic_dir_name="train_M"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
                device.type == "cuda"
        ),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
                device.type == "cuda"
        )
    )

    # ================================================
    # Model
    # ================================================

    model = ResidualSRNet(
        input_channels=6,
        channels=args.channels,
        num_blocks=args.blocks,
        block_res_scale=0.10,
        output_res_scale=1.00,
        scale_num=5,
        scale_den=3
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # ================================================
    # Loss / optimizer
    # ================================================

    pixel_loss_fn = CharbonnierLoss()

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

    # ================================================
    # output
    # ================================================

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

    log_path = (
            checkpoint_dir
            / "training_log.csv"
    )

    # ================================================
    # Resume / model-only initialization
    # ================================================

    start_epoch = 1
    best_psnr = -1.0

    if (
            args.resume
            and
            args.load_model
    ):
        raise ValueError(
            "--resume and --load_model "
            "cannot be used together."
        )

    # ================================================
    # Model weights only
    #
    # Loss semanticsを変更した新experiment用。
    # optimizer/scaler/epoch/bestは引き継がない。
    # ================================================

    if args.load_model:

        load_model_path = Path(
            args.load_model
        )

        print(
            f"Loading model weights only: "
            f"{load_model_path}"
        )

        checkpoint = torch.load(
            load_model_path,
            map_location=device
        )

        if (
                isinstance(
                    checkpoint,
                    dict
                )
                and
                "model" in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model"
                ]
            )

            source_epoch = (
                checkpoint.get(
                    "epoch",
                    "unknown"
                )
            )

        else:

            state_dict = (
                checkpoint
            )

            source_epoch = (
                "unknown"
            )

        model.load_state_dict(
            state_dict,
            strict=True
        )

        # optimizerは新規作成済みのものをそのまま使う
        # scalerも新品
        # v2 epochは1から
        start_epoch = 1
        best_psnr = -1.0

        print(
            "Model weights loaded."
        )

        print(
            f"Source epoch: "
            f"{source_epoch}"
        )

        print(
            "Optimizer: fresh"
        )

        print(
            "AMP scaler: fresh"
        )

        print(
            "Experiment epoch starts from 1"
        )


    # ================================================
    # True resume
    # ================================================

    elif args.resume:

        resume_path = Path(
            args.resume
        )

        print(
            f"Loading checkpoint: "
            f"{resume_path}"
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint[
                "model"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

        if "scaler" in checkpoint:
            scaler.load_state_dict(
                checkpoint[
                    "scaler"
                ]
            )

        start_epoch = (
                checkpoint[
                    "epoch"
                ]
                + 1
        )

        best_psnr = checkpoint.get(
            "best_psnr",
            -1.0
        )

        for param_group in (
                optimizer.param_groups
        ):
            param_group[
                "lr"
            ] = args.lr

        print(
            f"Resume from epoch "
            f"{start_epoch}"
        )

    # ================================================
    # Training loop
    # ================================================

    for epoch in range(
            start_epoch,
            args.epochs + 1
    ):

        epoch_start = time.time()

        print()
        print(
            "========================================"
        )

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print(
            "========================================"
        )

        train_results = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            pixel_weight=args.pixel_weight,
            use_amp=use_amp
        )

        val_results = validate(
            model=model,
            loader=val_loader,
            pixel_loss_fn=pixel_loss_fn,
            device=device,
            sample_dir=sample_dir,
            epoch=epoch,
            sample_count=args.sample_count
        )

        save_fixed_sample(
            model=model,
            dataset_root=args.dataset_root,
            records=records,
            target_stem=args.fixed_sample,
            device=device,
            sample_dir=sample_dir,
            epoch=epoch,
            lr_patch_size=args.lr_patch,

            region_boost=args.region_boost,
            region_threshold=args.region_threshold,
            region_range=args.region_range,
            region_smooth=args.region_smooth
        )

        elapsed = (
                time.time()
                - epoch_start
        )

        improvement = (
                val_results[
                    "sr_psnr"
                ]
                -
                val_results[
                    "bicubic_psnr"
                ]
        )

        print()
        print(
            "----------------------------------------"
        )

        print(
            f"Train total: "
            f"{train_results['total']:.6f}"
        )

        print(
            f"Train pixel: "
            f"{train_results['pixel']:.6f}"
        )

        print(
            f"Validation PSNR | "
            f"Bicubic "
            f"{val_results['bicubic_psnr']:.4f} dB | "
            f"SR "
            f"{val_results['sr_psnr']:.4f} dB | "
            f"Gain "
            f"{improvement:+.4f} dB"
        )

        print(
            f"Validation loss | "
            f"Pixel "
            f"{val_results['pixel']:.6f} | "
            f"Grad "
            f"{val_results['gradient']:.6f} | "
            f"Lap "
            f"{val_results['laplacian']:.6f}"
        )

        print(
            f"Time: "
            f"{elapsed:.1f} sec"
        )

        # ============================================
        # checkpoint
        # ============================================

        epoch_path = (
                checkpoint_dir
                / f"epoch_{epoch:03d}.pth"
        )

        save_checkpoint(
            save_path=epoch_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_psnr=best_psnr,
            args=args
        )

        latest_path = (
                checkpoint_dir
                / "latest.pth"
        )

        save_checkpoint(
            save_path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_psnr=best_psnr,
            args=args
        )

        # ============================================
        # best
        # ============================================

        if (
                val_results[
                    "sr_psnr"
                ]
                > best_psnr
        ):
            best_psnr = (
                val_results[
                    "sr_psnr"
                ]
            )

            best_path = (
                    checkpoint_dir
                    / "best_psnr.pth"
            )

            save_checkpoint(
                save_path=best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_psnr=best_psnr,
                args=args
            )

            print(
                f"New best PSNR: "
                f"{best_psnr:.4f} dB"
            )

        # ============================================
        # CSV
        # ============================================

        append_log(
            log_path,
            {
                "epoch":
                    epoch,

                "train_total":
                    train_results[
                        "total"
                    ],

                "train_pixel":
                    train_results[
                        "pixel"
                    ],

                "bicubic_psnr":
                    val_results[
                        "bicubic_psnr"
                    ],

                "sr_psnr":
                    val_results[
                        "sr_psnr"
                    ],

                "psnr_gain":
                    improvement,

                "val_pixel":
                    val_results[
                        "pixel"
                    ],

                "val_gradient":
                    val_results[
                        "gradient"
                    ],

                "val_laplacian":
                    val_results[
                        "laplacian"
                    ],

                "seconds":
                    elapsed,

            }
        )

    print()
    print("========================================")
    print("Training completed")
    print("========================================")

    print(
        f"Best PSNR: "
        f"{best_psnr:.4f} dB"
    )


if __name__ == "__main__":
    main()
