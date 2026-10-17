import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# Fixed experiment parameters
# ============================================================

EXPECTED_COUNT = 3900

OUTPUT_WIDTH = 768
EXPECTED_HEIGHT = 432

CHANGE_THRESHOLD = 0.03
CHANGE_RANGE = 0.15

SEED_THRESHOLD = 0.10
SEED_DILATE_KERNEL = 3

DENSITY_KERNEL = 9
DENSITY_MIN = 0.50

CLOSE_KERNEL = 5


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Utilities
# ============================================================

def get_bicubic():

    if hasattr(
            Image,
            "Resampling"
    ):
        return Image.Resampling.BICUBIC

    return Image.BICUBIC


def resize_scale_width(
        image,
        output_width=OUTPUT_WIDTH
):
    """
    Stage1 の resize_or_crop=scale_width 相当。

    1280x720
        ->
    768x432
    """

    width, height = image.size

    if (
            width <= 0
            or
            height <= 0
    ):
        raise RuntimeError(
            f"Invalid image size: {image.size}"
        )

    if width == output_width:
        return image.copy()

    new_height = int(
        output_width
        * height
        / width
    )

    return image.resize(
        (
            output_width,
            new_height
        ),
        get_bicubic()
    )


def pil_rgb_to_minus1_plus1_tensor(
        image,
        device
):
    """
    RGB PIL
        ->
    [1, 3, H, W]
    float32
    range [-1, 1]

    Stage1 change_score と同じ値域に合わせる。
    """

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = (
        array
        / 255.0
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    )

    tensor = (
        tensor * 2.0
        - 1.0
    )

    return tensor.to(
        device=device,
        dtype=torch.float32
    )


# ============================================================
# Source mosaic map
# ============================================================

def build_source_map(
        source_dir
):
    source_dir = Path(
        source_dir
    )

    if not source_dir.exists():

        raise FileNotFoundError(
            "Original Mosaic directory not found:\n"
            f"{source_dir.resolve()}"
        )

    source_map = {}

    for path in sorted(
            source_dir.iterdir()
    ):

        if (
                not path.is_file()
                or
                path.suffix.lower()
                not in IMAGE_EXTENSIONS
        ):
            continue

        stem = path.stem

        if stem in source_map:

            raise RuntimeError(
                "Duplicate Original Mosaic stem:\n"
                f"{stem}\n"
                f"{source_map[stem]}\n"
                f"{path}"
            )

        source_map[
            stem
        ] = path

    if not source_map:

        raise RuntimeError(
            "No Original Mosaic images found:\n"
            f"{source_dir.resolve()}"
        )

    return source_map


# ============================================================
# Manifest
# ============================================================

def load_jobs(
        manifest_path,
        source_map,
        clean_dir,
        expected_count
):
    manifest_path = Path(
        manifest_path
    )

    clean_dir = Path(
        clean_dir
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            "Manifest not found:\n"
            f"{manifest_path.resolve()}"
        )

    if not clean_dir.exists():

        raise FileNotFoundError(
            "Clean Target directory not found:\n"
            f"{clean_dir.resolve()}"
        )

    jobs = []

    used_outputs = set()

    missing_mosaic = []
    missing_clean = []

    with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        required_fields = {
            "stem",
            "stage1_file",
            "clean_file",
        }

        if reader.fieldnames is None:

            raise RuntimeError(
                "manifest.csv has no header."
            )

        missing_fields = (
            required_fields
            - set(
                reader.fieldnames
            )
        )

        if missing_fields:

            raise RuntimeError(
                "manifest.csv is missing fields:\n"
                f"{sorted(missing_fields)}"
            )

        for line_number, row in enumerate(
                reader,
                start=2
        ):

            stem = (
                row["stem"]
                .strip()
            )

            stage1_file = (
                row["stage1_file"]
                .strip()
            )

            clean_file = (
                row["clean_file"]
                .strip()
            )

            if not stem:

                raise RuntimeError(
                    "Empty stem at manifest line "
                    f"{line_number}"
                )

            if not stage1_file:

                raise RuntimeError(
                    "Empty stage1_file at manifest line "
                    f"{line_number}"
                )

            if not clean_file:

                raise RuntimeError(
                    "Empty clean_file at manifest line "
                    f"{line_number}"
                )

            if stem not in source_map:

                missing_mosaic.append(
                    stem
                )

                continue

            clean_relative = Path(
                clean_file
            )

            output_relative = Path(
                stage1_file
            )

            if (
                    clean_relative.is_absolute()
                    or
                    ".." in clean_relative.parts
            ):

                raise RuntimeError(
                    "Unsafe clean_file path:\n"
                    f"{clean_file}"
                )

            if (
                    output_relative.is_absolute()
                    or
                    ".." in output_relative.parts
            ):

                raise RuntimeError(
                    "Unsafe stage1_file path:\n"
                    f"{stage1_file}"
                )

            clean_path = (
                clean_dir
                / clean_relative
            )

            if not clean_path.exists():

                missing_clean.append(
                    str(
                        clean_path
                    )
                )

                continue

            output_key = str(
                output_relative
            ).lower()

            if output_key in used_outputs:

                raise RuntimeError(
                    "Duplicate train_S output path:\n"
                    f"{output_relative}"
                )

            used_outputs.add(
                output_key
            )

            jobs.append(
                {
                    "stem":
                        stem,

                    "mosaic_path":
                        source_map[
                            stem
                        ],

                    "clean_path":
                        clean_path,

                    "output_relative":
                        output_relative,
                }
            )

    if missing_mosaic:

        preview = "\n".join(
            "  " + stem
            for stem
            in missing_mosaic[:20]
        )

        raise RuntimeError(
            "Original Mosaic files are missing.\n"
            f"Count: {len(missing_mosaic)}\n"
            f"First entries:\n"
            f"{preview}"
        )

    if missing_clean:

        preview = "\n".join(
            "  " + path
            for path
            in missing_clean[:20]
        )

        raise RuntimeError(
            "Clean Target files are missing.\n"
            f"Count: {len(missing_clean)}\n"
            f"First entries:\n"
            f"{preview}"
        )

    if (
            expected_count > 0
            and
            len(jobs)
            != expected_count
    ):

        raise RuntimeError(
            "Unexpected pair count.\n"
            f"Expected: {expected_count}\n"
            f"Actual  : {len(jobs)}"
        )

    return jobs


# ============================================================
# train_S generation
# ============================================================

@torch.no_grad()
def build_train_s_mask(
        mosaic_image,
        clean_image,
        device
):
    # --------------------------------------------------------
    # Stage1 tensor range
    # --------------------------------------------------------

    mosaic_tensor = (
        pil_rgb_to_minus1_plus1_tensor(
            mosaic_image,
            device
        )
    )

    clean_tensor = (
        pil_rgb_to_minus1_plus1_tensor(
            clean_image,
            device
        )
    )

    # ========================================================
    # 1. change_score
    #
    # Exactly the same semantics as Stage1:
    #
    # pair_diff =
    #     mean(abs(input - target), RGB)
    #
    # score =
    #     (pair_diff - 0.03) / 0.15
    # ========================================================

    pair_diff = torch.abs(
        mosaic_tensor
        - clean_tensor
    ).mean(
        dim=1,
        keepdim=True
    )

    change_score = (
        pair_diff
        - CHANGE_THRESHOLD
    ) / CHANGE_RANGE

    change_score = torch.clamp(
        change_score,
        0.0,
        1.0
    )

    # ========================================================
    # 2. Binary seed
    # ========================================================

    seed = (
        change_score
        > SEED_THRESHOLD
    ).float()

    # ========================================================
    # 3. 3x3 seed dilation
    # ========================================================

    seed_dilated = F.max_pool2d(
        seed,
        kernel_size=SEED_DILATE_KERNEL,
        stride=1,
        padding=SEED_DILATE_KERNEL // 2
    )

    # ========================================================
    # 4. 9x9 local density
    # ========================================================

    density = F.avg_pool2d(
        seed_dilated,
        kernel_size=DENSITY_KERNEL,
        stride=1,
        padding=DENSITY_KERNEL // 2
    )

    # ========================================================
    # 5. density >= 0.50
    # ========================================================

    binary = (
        density
        >= DENSITY_MIN
    ).float()

    # ========================================================
    # 6. 5x5 closing
    #
    # dilation -> erosion
    # Same formulation used in diagnostic experiment.
    # ========================================================

    binary_dilated = F.max_pool2d(
        binary,
        kernel_size=CLOSE_KERNEL,
        stride=1,
        padding=CLOSE_KERNEL // 2
    )

    binary_closed = (
        1.0
        - F.max_pool2d(
            1.0 - binary_dilated,
            kernel_size=CLOSE_KERNEL,
            stride=1,
            padding=CLOSE_KERNEL // 2
        )
    )

    binary_closed = torch.clamp(
        binary_closed,
        0.0,
        1.0
    )

    stats = {
        "change_score_mean":
            change_score.mean().item(),

        "seed_coverage":
            seed.mean().item(),

        "seed_dilated_coverage":
            seed_dilated.mean().item(),

        "density_mean":
            density.mean().item(),

        "density_max":
            density.max().item(),

        "train_s_coverage":
            binary_closed.mean().item(),
    }

    return (
        binary_closed,
        stats
    )


# ============================================================
# Save binary mask
# ============================================================

def save_binary_mask(
        mask,
        output_path
):
    """
    mask:
        [1, 1, H, W]

    保存値:
        0   = Non-Mosaic
        255 = Mosaic supervision region
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    array = (
        mask
        .detach()
        .float()
        .cpu()[0, 0]
        .clamp(
            0.0,
            1.0
        )
        .numpy()
    )

    array = (
        array
        >= 0.5
    ).astype(
        np.uint8
    ) * 255

    image = Image.fromarray(
        array,
        mode="L"
    )

    # Binary supervision must be lossless.
    #
    # stage1_file is expected to be PNG in this dataset.
    # To avoid JPEG corruption, always encode as PNG.
    image.save(
        output_path,
        format="PNG"
    )


# ============================================================
# Export one pair
# ============================================================

@torch.no_grad()
def export_one(
        job,
        output_dir,
        device,
        output_width,
        expected_height,
        overwrite
):
    output_path = (
        Path(output_dir)
        / job[
            "output_relative"
        ]
    )

    if (
            output_path.exists()
            and
            not overwrite
    ):

        raise FileExistsError(
            "Output already exists:\n"
            f"{output_path}\n"
            "Use --overwrite only when intentionally rebuilding train_S."
        )

    # --------------------------------------------------------
    # Original Mosaic
    # --------------------------------------------------------

    with Image.open(
            job[
                "mosaic_path"
            ]
    ) as image:

        mosaic_image = (
            image
            .convert(
                "RGB"
            )
        )

        mosaic_image = resize_scale_width(
            mosaic_image,
            output_width=output_width
        )

    # --------------------------------------------------------
    # Clean Target
    # --------------------------------------------------------

    with Image.open(
            job[
                "clean_path"
            ]
    ) as image:

        clean_image = (
            image
            .convert(
                "RGB"
            )
        )

        clean_image = resize_scale_width(
            clean_image,
            output_width=output_width
        )

    # --------------------------------------------------------
    # Size safety
    # --------------------------------------------------------

    expected_size = (
        output_width,
        expected_height
    )

    if mosaic_image.size != expected_size:

        raise RuntimeError(
            "Unexpected Mosaic resized size.\n"
            f"Stem     : {job['stem']}\n"
            f"Expected : {expected_size}\n"
            f"Actual   : {mosaic_image.size}"
        )

    if clean_image.size != expected_size:

        raise RuntimeError(
            "Unexpected Clean resized size.\n"
            f"Stem     : {job['stem']}\n"
            f"Expected : {expected_size}\n"
            f"Actual   : {clean_image.size}"
        )

    if mosaic_image.size != clean_image.size:

        raise RuntimeError(
            "Mosaic / Clean size mismatch.\n"
            f"Stem   : {job['stem']}\n"
            f"Mosaic : {mosaic_image.size}\n"
            f"Clean  : {clean_image.size}"
        )

    # --------------------------------------------------------
    # train_S
    # --------------------------------------------------------

    (
        train_s,
        stats
    ) = build_train_s_mask(
        mosaic_image=mosaic_image,
        clean_image=clean_image,
        device=device
    )

    save_binary_mask(
        train_s,
        output_path
    )

    return {
        "stem":
            job["stem"],

        "mosaic_path":
            str(
                job[
                    "mosaic_path"
                ]
            ),

        "clean_path":
            str(
                job[
                    "clean_path"
                ]
            ),

        "output_path":
            str(
                output_path
            ),

        "width":
            output_width,

        "height":
            expected_height,

        **stats,
    }


# ============================================================
# CSV
# ============================================================

def save_log(
        log_path,
        rows
):
    log_path = Path(
        log_path
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "stem",
        "mosaic_path",
        "clean_path",
        "output_path",
        "width",
        "height",
        "change_score_mean",
        "seed_coverage",
        "seed_dilated_coverage",
        "density_mean",
        "density_max",
        "train_s_coverage",
    ]

    with open(
            log_path,
            "w",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build Stage2 train_S binary supervision masks "
            "from Original Mosaic / Clean Target pairs."
        )
    )

    parser.add_argument(
        "--mosaic_dir",
        type=str,
        default=(
            "../DeepMosaics/"
            "train/"
            "pix2pixHD/"
            "datasets/"
            "original_4000/"
            "train_A"
        )
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=(
            "./datasets/"
            "stage2_lp21_sigma6_mask67_1280x720"
        )
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=""
    )

    parser.add_argument(
        "--clean_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--log_csv",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_width",
        type=int,
        default=OUTPUT_WIDTH
    )

    parser.add_argument(
        "--expected_height",
        type=int,
        default=EXPECTED_HEIGHT
    )

    parser.add_argument(
        "--expected_count",
        type=int,
        default=EXPECTED_COUNT
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    dataset_root = Path(
        args.dataset_root
    )

    mosaic_dir = Path(
        args.mosaic_dir
    )

    if args.manifest:

        manifest_path = Path(
            args.manifest
        )

    else:

        manifest_path = (
            dataset_root
            / "manifest.csv"
        )

    if args.clean_dir:

        clean_dir = Path(
            args.clean_dir
        )

    else:

        clean_dir = (
            dataset_root
            / "train_B"
        )

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

    else:

        output_dir = (
            dataset_root
            / "train_S"
        )

    if args.log_csv:

        log_path = Path(
            args.log_csv
        )

    else:

        log_path = (
            dataset_root
            / "train_S_export_log.csv"
        )

    device = torch.device(
        args.device
    )

    if (
            device.type == "cuda"
            and
            not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not dataset_root.exists():

        raise FileNotFoundError(
            "Dataset root not found:\n"
            f"{dataset_root.resolve()}"
        )

    if not manifest_path.exists():

        raise FileNotFoundError(
            "Manifest not found:\n"
            f"{manifest_path.resolve()}"
        )

    if (
            output_dir.exists()
            and
            any(
                output_dir.iterdir()
            )
            and
            not args.overwrite
    ):

        raise RuntimeError(
            "train_S already contains files:\n"
            f"{output_dir.resolve()}\n"
            "Use --overwrite only when intentionally rebuilding train_S."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "Build Stage2 train_S"
    )
    print(
        "========================================"
    )

    print(
        f"Device              : {device}"
    )

    print(
        f"Original Mosaic dir : {mosaic_dir.resolve()}"
    )

    print(
        f"Clean Target dir    : {clean_dir.resolve()}"
    )

    print(
        f"Dataset root        : {dataset_root.resolve()}"
    )

    print(
        f"Manifest            : {manifest_path.resolve()}"
    )

    print(
        f"Output dir          : {output_dir.resolve()}"
    )

    print(
        f"Expected count      : {args.expected_count}"
    )

    print(
        f"Output size         : "
        f"{args.output_width}"
        f"x"
        f"{args.expected_height}"
    )

    print(
        f"Change threshold    : {CHANGE_THRESHOLD}"
    )

    print(
        f"Change range        : {CHANGE_RANGE}"
    )

    print(
        f"Seed threshold      : {SEED_THRESHOLD}"
    )

    print(
        f"Seed dilation       : {SEED_DILATE_KERNEL}"
    )

    print(
        f"Density kernel      : {DENSITY_KERNEL}"
    )

    print(
        f"Density min         : {DENSITY_MIN}"
    )

    print(
        f"Closing kernel      : {CLOSE_KERNEL}"
    )

    print(
        "Output              : binary 0 / 255 PNG"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Source map / jobs
    # --------------------------------------------------------

    source_map = build_source_map(
        mosaic_dir
    )

    print(
        f"Original Mosaic files: "
        f"{len(source_map)}"
    )

    jobs = load_jobs(
        manifest_path=manifest_path,
        source_map=source_map,
        clean_dir=clean_dir,
        expected_count=args.expected_count
    )

    print(
        f"Export jobs         : {len(jobs)}"
    )

    print()

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    rows = []

    total = len(
        jobs
    )

    for index, job in enumerate(
            jobs,
            start=1
    ):

        row = export_one(
            job=job,
            output_dir=output_dir,
            device=device,
            output_width=args.output_width,
            expected_height=args.expected_height,
            overwrite=args.overwrite
        )

        rows.append(
            row
        )

        if (
                index % 100 == 0
                or
                index == total
        ):

            print(
                f"Export "
                f"{index:04d}/"
                f"{total:04d} | "
                f"{job['stem']} | "
                f"coverage "
                f"{row['train_s_coverage']:.4f}"
            )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_log(
        log_path=log_path,
        rows=rows
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    output_count = 0
    invalid_size_count = 0
    invalid_value_count = 0

    for job in jobs:

        output_path = (
            output_dir
            / job[
                "output_relative"
            ]
        )

        if not output_path.exists():
            continue

        output_count += 1

        with Image.open(
                output_path
        ) as image:

            if image.size != (
                    args.output_width,
                    args.expected_height
            ):

                invalid_size_count += 1

            array = np.asarray(
                image
            )

            unique_values = np.unique(
                array
            )

            if not set(
                    unique_values.tolist()
            ).issubset(
                {
                    0,
                    255,
                }
            ):

                invalid_value_count += 1

    if output_count != total:

        raise RuntimeError(
            "Output count mismatch.\n"
            f"Expected: {total}\n"
            f"Actual  : {output_count}"
        )

    if invalid_size_count > 0:

        raise RuntimeError(
            "Invalid train_S image sizes found.\n"
            f"Count: {invalid_size_count}"
        )

    if invalid_value_count > 0:

        raise RuntimeError(
            "Non-binary train_S images found.\n"
            f"Count: {invalid_value_count}"
        )

    # --------------------------------------------------------
    # Coverage summary
    # --------------------------------------------------------

    coverages = np.array(
        [
            row[
                "train_s_coverage"
            ]
            for row in rows
        ],
        dtype=np.float64
    )

    empty_count = int(
        np.sum(
            coverages == 0.0
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "train_S completed"
    )
    print(
        "========================================"
    )

    print(
        f"Pairs           : {total}"
    )

    print(
        f"Output files    : {output_count}"
    )

    print(
        f"Output size     : "
        f"{args.output_width}"
        f"x"
        f"{args.expected_height}"
    )

    print(
        f"Binary values   : 0 / 255"
    )

    print(
        f"Coverage mean   : "
        f"{coverages.mean():.6f}"
    )

    print(
        f"Coverage median : "
        f"{np.median(coverages):.6f}"
    )

    print(
        f"Coverage min    : "
        f"{coverages.min():.6f}"
    )

    print(
        f"Coverage max    : "
        f"{coverages.max():.6f}"
    )

    print(
        f"Empty masks     : "
        f"{empty_count}"
    )

    print(
        f"Output dir      : "
        f"{output_dir.resolve()}"
    )

    print(
        f"Log CSV         : "
        f"{log_path.resolve()}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()