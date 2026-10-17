import argparse
import csv
from pathlib import Path

from PIL import Image


# ============================================================
# Defaults
# ============================================================

EXPECTED_COUNT = 3900
LOAD_WIDTH = 768
EXPECTED_HEIGHT = 432

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
# Build source image map
# ============================================================

def build_source_map(
        source_dir
):
    source_dir = Path(
        source_dir
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            "Source directory not found:\n"
            f"{source_dir.resolve()}"
        )

    if not source_dir.is_dir():
        raise RuntimeError(
            "Source path is not a directory:\n"
            f"{source_dir.resolve()}"
        )

    image_paths = sorted(
        path
        for path
        in source_dir.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            "No source images found:\n"
            f"{source_dir.resolve()}"
        )

    source_map = {}

    for path in image_paths:

        stem = path.stem

        if stem in source_map:
            raise RuntimeError(
                "Duplicate source stem:\n"
                f"{stem}\n"
                f"{source_map[stem]}\n"
                f"{path}"
            )

        source_map[stem] = path

    return source_map


# ============================================================
# resize_or_crop = scale_width equivalent
# ============================================================

def resize_scale_width(
        image,
        load_width=LOAD_WIDTH
):
    """
    Keep aspect ratio and resize width to load_width.

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

    if width == load_width:

        return image.copy()

    new_height = int(
        load_width
        * height
        / width
    )

    if hasattr(
            Image,
            "Resampling"
    ):
        bicubic = (
            Image.Resampling.BICUBIC
        )

    else:
        bicubic = (
            Image.BICUBIC
        )

    return image.resize(
        (
            load_width,
            new_height
        ),
        bicubic
    )


# ============================================================
# Load jobs from existing Stage2 manifest
# ============================================================

def load_jobs(
        manifest_path,
        source_map,
        expected_count
):
    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest not found:\n"
            f"{manifest_path.resolve()}"
        )

    jobs = []
    missing_stems = []
    used_output_names = set()

    with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                "manifest.csv has no header."
            )

        required_fields = {
            "stem",
            "stage1_file",
        }

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

            if not stem:
                raise RuntimeError(
                    "Empty stem in manifest "
                    f"line {line_number}"
                )

            if not stage1_file:
                raise RuntimeError(
                    "Empty stage1_file in manifest "
                    f"line {line_number}"
                )

            if stem not in source_map:

                missing_stems.append(
                    stem
                )

                continue

            output_relative = Path(
                stage1_file
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

            normalized_output = (
                str(
                    output_relative
                )
                .lower()
            )

            if (
                    normalized_output
                    in used_output_names
            ):
                raise RuntimeError(
                    "Duplicate stage1_file in manifest:\n"
                    f"{stage1_file}"
                )

            used_output_names.add(
                normalized_output
            )

            jobs.append(
                {
                    "stem":
                        stem,

                    "source_path":
                        source_map[
                            stem
                        ],

                    "output_relative":
                        output_relative,
                }
            )

    if missing_stems:

        preview = "\n".join(
            "  " + stem
            for stem
            in missing_stems[:20]
        )

        raise RuntimeError(
            "Some manifest stems do not exist "
            "in source_dir.\n"
            f"Missing count: {len(missing_stems)}\n"
            f"First missing stems:\n"
            f"{preview}"
        )

    if (
            expected_count > 0
            and
            len(jobs)
            != expected_count
    ):
        raise RuntimeError(
            "Unexpected job count.\n"
            f"Expected: {expected_count}\n"
            f"Actual  : {len(jobs)}"
        )

    return jobs


# ============================================================
# Save image
# ============================================================

def save_image(
        image,
        output_path
):
    output_path = Path(
        output_path
    )

    suffix = (
        output_path
        .suffix
        .lower()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if suffix in {
        ".jpg",
        ".jpeg",
    }:

        image.save(
            output_path,
            quality=100,
            subsampling=0
        )

    else:

        image.save(
            output_path
        )


# ============================================================
# Export one image
# ============================================================

def export_one(
        job,
        output_dir,
        load_width,
        expected_height,
        overwrite
):
    source_path = (
        job["source_path"]
    )

    output_path = (
        Path(output_dir)
        / job["output_relative"]
    )

    if (
            output_path.exists()
            and
            not overwrite
    ):
        raise FileExistsError(
            "Output already exists:\n"
            f"{output_path}\n"
            "Use --overwrite only if intentional."
        )

    with Image.open(
            source_path
    ) as source_image:

        source_image = (
            source_image
            .convert(
                "RGB"
            )
        )

        original_size = (
            source_image.size
        )

        resized = resize_scale_width(
            source_image,
            load_width=load_width
        )

    if (
            resized.size[0]
            != load_width
    ):
        raise RuntimeError(
            "Unexpected output width.\n"
            f"Stem     : {job['stem']}\n"
            f"Source   : {source_path}\n"
            f"Expected : {load_width}\n"
            f"Actual   : {resized.size[0]}"
        )

    if (
            expected_height > 0
            and
            resized.size[1]
            != expected_height
    ):
        raise RuntimeError(
            "Unexpected output height.\n"
            f"Stem     : {job['stem']}\n"
            f"Source   : {source_path}\n"
            f"Expected : {expected_height}\n"
            f"Actual   : {resized.size[1]}"
        )

    save_image(
        resized,
        output_path
    )

    return {
        "stem":
            job["stem"],

        "source_path":
            str(
                source_path
            ),

        "output_path":
            str(
                output_path
            ),

        "source_width":
            original_size[0],

        "source_height":
            original_size[1],

        "output_width":
            resized.size[0],

        "output_height":
            resized.size[1],
    }


# ============================================================
# Log
# ============================================================

def save_log(
        log_path,
        rows
):
    fieldnames = [
        "stem",
        "source_path",
        "output_path",
        "source_width",
        "source_height",
        "output_width",
        "output_height",
    ]

    log_path = Path(
        log_path
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
            "Build train_M for "
            "2-input MosaicStage2SR."
        )
    )

    parser.add_argument(
        "--source_dir",
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
        "--load_width",
        type=int,
        default=LOAD_WIDTH
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
        "--overwrite",
        action="store_true"
    )

    parser.add_argument(
        "--log_csv",
        type=str,
        default=""
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    source_dir = Path(
        args.source_dir
    )

    dataset_root = Path(
        args.dataset_root
    )

    output_dir = (
        dataset_root
        / "train_M"
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

    if args.log_csv:

        log_path = Path(
            args.log_csv
        )

    else:

        log_path = (
            dataset_root
            / "train_M_export_log.csv"
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
            "train_M already contains files:\n"
            f"{output_dir.resolve()}\n"
            "Use --overwrite only if rebuilding "
            "train_M intentionally."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "========================================"
    )
    print(
        "Build Stage2 train_M"
    )
    print(
        "========================================"
    )

    print(
        f"Source dir      : "
        f"{source_dir.resolve()}"
    )

    print(
        f"Dataset root    : "
        f"{dataset_root.resolve()}"
    )

    print(
        f"Manifest        : "
        f"{manifest_path.resolve()}"
    )

    print(
        f"Output dir      : "
        f"{output_dir.resolve()}"
    )

    print(
        f"Expected count  : "
        f"{args.expected_count}"
    )

    print(
        f"Output size     : "
        f"{args.load_width}"
        f"x"
        f"{args.expected_height}"
    )

    print(
        "Processing      : Bicubic resize only"
    )

    print(
        "Blur            : none"
    )

    print(
        "Mask            : none"
    )

    print(
        f"Overwrite       : "
        f"{args.overwrite}"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Source images
    # --------------------------------------------------------

    source_map = build_source_map(
        source_dir
    )

    print(
        f"Source images   : "
        f"{len(source_map)}"
    )

    # --------------------------------------------------------
    # Exact manifest jobs
    # --------------------------------------------------------

    jobs = load_jobs(
        manifest_path=manifest_path,
        source_map=source_map,
        expected_count=(
            args.expected_count
        )
    )

    print(
        f"Export jobs     : "
        f"{len(jobs)}"
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
            load_width=(
                args.load_width
            ),
            expected_height=(
                args.expected_height
            ),
            overwrite=(
                args.overwrite
            )
        )

        rows.append(
            row
        )

        if (
                index % 100
                == 0
                or
                index == total
        ):

            print(
                f"Export "
                f"{index:04d}/"
                f"{total:04d} | "
                f"{job['stem']}"
            )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    save_log(
        log_path=log_path,
        rows=rows
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    output_count = 0
    invalid_sizes = []

    for job in jobs:

        output_path = (
            output_dir
            / job["output_relative"]
        )

        if not output_path.exists():
            continue

        output_count += 1

        with Image.open(
                output_path
        ) as image:

            if (
                    image.size
                    != (
                        args.load_width,
                        args.expected_height
                    )
            ):

                invalid_sizes.append(
                    (
                        job["stem"],
                        image.size
                    )
                )

    if (
            output_count
            != total
    ):
        raise RuntimeError(
            "Output count mismatch.\n"
            f"Expected: {total}\n"
            f"Actual  : {output_count}"
        )

    if invalid_sizes:
        raise RuntimeError(
            "Invalid output sizes found.\n"
            f"Count: {len(invalid_sizes)}\n"
            f"First entries: "
            f"{invalid_sizes[:20]}"
        )

    print()
    print(
        "========================================"
    )
    print(
        "train_M completed"
    )
    print(
        "========================================"
    )

    print(
        f"Pairs        : {total}"
    )

    print(
        f"Output files : {output_count}"
    )

    print(
        f"Output size  : "
        f"{args.load_width}"
        f"x"
        f"{args.expected_height}"
    )

    print(
        f"Output dir   : "
        f"{output_dir.resolve()}"
    )

    print(
        f"Log CSV      : "
        f"{log_path.resolve()}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()