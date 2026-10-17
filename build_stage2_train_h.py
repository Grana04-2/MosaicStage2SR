import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_COUNT = 3900
EXPECTED_WIDTH = 768
EXPECTED_HEIGHT = 432


# ============================================================
# Manifest
# ============================================================

def load_jobs(
        manifest_path,
        train_a_dir,
        train_m_dir,
        train_s_dir,
        expected_count
):
    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path.resolve()}"
        )

    jobs = []

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
                row[
                    "stem"
                ].strip()
            )

            stage1_file = (
                row[
                    "stage1_file"
                ].strip()
            )

            if not stem:
                raise RuntimeError(
                    f"Empty stem at line {line_number}"
                )

            if not stage1_file:
                raise RuntimeError(
                    f"Empty stage1_file at line {line_number}"
                )

            relative_path = Path(
                stage1_file
            )

            if (
                    relative_path.is_absolute()
                    or
                    ".." in relative_path.parts
            ):
                raise RuntimeError(
                    "Unsafe stage1_file path:\n"
                    f"{stage1_file}"
                )

            stage1_path = (
                Path(
                    train_a_dir
                )
                / relative_path
            )

            mosaic_path = (
                Path(
                    train_m_dir
                )
                / relative_path
            )

            mask_path = (
                Path(
                    train_s_dir
                )
                / relative_path
            )

            for label, path in [
                (
                    "train_A",
                    stage1_path
                ),
                (
                    "train_M",
                    mosaic_path
                ),
                (
                    "train_S",
                    mask_path
                ),
            ]:

                if not path.exists():
                    raise FileNotFoundError(
                        f"{label} file not found:\n"
                        f"Stem: {stem}\n"
                        f"Path: {path}"
                    )

            jobs.append(
                {
                    "stem":
                        stem,

                    "relative_path":
                        relative_path,

                    "stage1_path":
                        stage1_path,

                    "mosaic_path":
                        mosaic_path,

                    "mask_path":
                        mask_path,
                }
            )

    if (
            expected_count > 0
            and
            len(
                jobs
            )
            != expected_count
    ):
        raise RuntimeError(
            "Unexpected pair count.\n"
            f"Expected: {expected_count}\n"
            f"Actual  : {len(jobs)}"
        )

    return jobs


# ============================================================
# Load
# ============================================================

def load_rgb(
        path
):
    with Image.open(
            path
    ) as image:

        return (
            image
            .convert(
                "RGB"
            )
            .copy()
        )


def load_mask(
        path
):
    with Image.open(
            path
    ) as image:

        return (
            image
            .convert(
                "L"
            )
            .copy()
        )


# ============================================================
# Build Hybrid
# ============================================================

def build_hybrid(
        stage1_image,
        mosaic_image,
        mask_image
):
    if (
            stage1_image.size
            != mosaic_image.size
    ):
        raise RuntimeError(
            "Stage1 / Mosaic size mismatch:\n"
            f"Stage1 = {stage1_image.size}\n"
            f"Mosaic = {mosaic_image.size}"
        )

    if (
            stage1_image.size
            != mask_image.size
    ):
        raise RuntimeError(
            "Stage1 / train_S size mismatch:\n"
            f"Stage1 = {stage1_image.size}\n"
            f"train_S = {mask_image.size}"
        )

    stage1 = np.asarray(
        stage1_image,
        dtype=np.float32
    )

    mosaic = np.asarray(
        mosaic_image,
        dtype=np.float32
    )

    mask_raw = np.asarray(
        mask_image,
        dtype=np.uint8
    )

    # ========================================================
    # train_S must be binary
    # ========================================================

    unique_values = np.unique(
        mask_raw
    )

    if not set(
            unique_values.tolist()
    ).issubset(
        {
            0,
            255,
        }
    ):
        raise RuntimeError(
            "train_S contains non-binary values:\n"
            f"{unique_values.tolist()[:20]}"
        )

    # [H, W, 1]
    mask = (
        mask_raw.astype(
            np.float32
        )
        / 255.0
    )[
        ...,
        None
    ]

    # ========================================================
    # Hybrid
    #
    # train_S = 0:
    #   Original Mosaic
    #
    # train_S = 1:
    #   Stage1 Blur
    # ========================================================

    hybrid = (
        (
            1.0
            - mask
        )
        * mosaic
        +
        mask
        * stage1
    )

    hybrid = np.clip(
        hybrid,
        0.0,
        255.0
    )

    hybrid = (
        hybrid
        + 0.5
    ).astype(
        np.uint8
    )

    coverage = float(
        mask.mean()
    )

    return (
        Image.fromarray(
            hybrid,
            mode="RGB"
        ),
        coverage
    )


# ============================================================
# Export one
# ============================================================

def export_one(
        job,
        output_dir,
        expected_width,
        expected_height,
        overwrite
):
    output_path = (
        Path(
            output_dir
        )
        / job[
            "relative_path"
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
            "Use --overwrite only when intentionally rebuilding train_H."
        )

    stage1_image = load_rgb(
        job[
            "stage1_path"
        ]
    )

    mosaic_image = load_rgb(
        job[
            "mosaic_path"
        ]
    )

    mask_image = load_mask(
        job[
            "mask_path"
        ]
    )

    expected_size = (
        expected_width,
        expected_height
    )

    if (
            stage1_image.size
            != expected_size
    ):
        raise RuntimeError(
            "Unexpected train_A size.\n"
            f"Stem     : {job['stem']}\n"
            f"Expected : {expected_size}\n"
            f"Actual   : {stage1_image.size}"
        )

    if (
            mosaic_image.size
            != expected_size
    ):
        raise RuntimeError(
            "Unexpected train_M size.\n"
            f"Stem     : {job['stem']}\n"
            f"Expected : {expected_size}\n"
            f"Actual   : {mosaic_image.size}"
        )

    if (
            mask_image.size
            != expected_size
    ):
        raise RuntimeError(
            "Unexpected train_S size.\n"
            f"Stem     : {job['stem']}\n"
            f"Expected : {expected_size}\n"
            f"Actual   : {mask_image.size}"
        )

    (
        hybrid_image,
        coverage
    ) = build_hybrid(
        stage1_image=stage1_image,
        mosaic_image=mosaic_image,
        mask_image=mask_image
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Stage2 dataset is expected to use PNG.
    if (
            output_path.suffix.lower()
            != ".png"
    ):
        raise RuntimeError(
            "train_H output must be PNG.\n"
            f"Unexpected filename: {output_path.name}"
        )

    hybrid_image.save(
        output_path,
        format="PNG"
    )

    return {
        "stem":
            job[
                "stem"
            ],

        "stage1_path":
            str(
                job[
                    "stage1_path"
                ]
            ),

        "mosaic_path":
            str(
                job[
                    "mosaic_path"
                ]
            ),

        "mask_path":
            str(
                job[
                    "mask_path"
                ]
            ),

        "output_path":
            str(
                output_path
            ),

        "width":
            expected_width,

        "height":
            expected_height,

        "train_s_coverage":
            coverage,
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
        "stage1_path",
        "mosaic_path",
        "mask_path",
        "output_path",
        "width",
        "height",
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
            "Build Stage2 train_H Oracle hybrid input.\n"
            "Outside train_S: Original Mosaic.\n"
            "Inside train_S : Stage1 Blur."
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
        "--train_a",
        type=str,
        default=""
    )

    parser.add_argument(
        "--train_m",
        type=str,
        default=""
    )

    parser.add_argument(
        "--train_s",
        type=str,
        default=""
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=""
    )

    parser.add_argument(
        "--log_csv",
        type=str,
        default=""
    )

    parser.add_argument(
        "--expected_count",
        type=int,
        default=EXPECTED_COUNT
    )

    parser.add_argument(
        "--expected_width",
        type=int,
        default=EXPECTED_WIDTH
    )

    parser.add_argument(
        "--expected_height",
        type=int,
        default=EXPECTED_HEIGHT
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

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found:\n"
            f"{dataset_root.resolve()}"
        )

    train_a_dir = (
        Path(
            args.train_a
        )
        if args.train_a
        else
        dataset_root
        / "train_A"
    )

    train_m_dir = (
        Path(
            args.train_m
        )
        if args.train_m
        else
        dataset_root
        / "train_M"
    )

    train_s_dir = (
        Path(
            args.train_s
        )
        if args.train_s
        else
        dataset_root
        / "train_S"
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else
        dataset_root
        / "train_H"
    )

    manifest_path = (
        Path(
            args.manifest
        )
        if args.manifest
        else
        dataset_root
        / "manifest.csv"
    )

    log_path = (
        Path(
            args.log_csv
        )
        if args.log_csv
        else
        dataset_root
        / "train_H_export_log.csv"
    )

    for label, directory in [
        (
            "train_A",
            train_a_dir
        ),
        (
            "train_M",
            train_m_dir
        ),
        (
            "train_S",
            train_s_dir
        ),
    ]:

        if not directory.exists():
            raise FileNotFoundError(
                f"{label} directory not found:\n"
                f"{directory.resolve()}"
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
            "train_H already contains files:\n"
            f"{output_dir.resolve()}\n"
            "Use --overwrite only when intentionally rebuilding train_H."
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
        "Build Stage2 train_H"
    )
    print(
        "========================================"
    )

    print(
        f"Dataset root : "
        f"{dataset_root.resolve()}"
    )

    print(
        f"train_A      : "
        f"{train_a_dir.resolve()}"
    )

    print(
        f"train_M      : "
        f"{train_m_dir.resolve()}"
    )

    print(
        f"train_S      : "
        f"{train_s_dir.resolve()}"
    )

    print(
        f"train_H      : "
        f"{output_dir.resolve()}"
    )

    print(
        f"Manifest     : "
        f"{manifest_path.resolve()}"
    )

    print(
        f"Expected     : "
        f"{args.expected_count}"
    )

    print(
        f"Output size  : "
        f"{args.expected_width}"
        f"x"
        f"{args.expected_height}"
    )

    print(
        "Blend rule   : "
        "S=0 -> train_M, "
        "S=1 -> train_A"
    )

    print(
        "========================================"
    )

    jobs = load_jobs(
        manifest_path=manifest_path,
        train_a_dir=train_a_dir,
        train_m_dir=train_m_dir,
        train_s_dir=train_s_dir,
        expected_count=args.expected_count
    )

    print(
        f"Export jobs  : "
        f"{len(jobs)}"
    )

    print()

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
            expected_width=args.expected_width,
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

    save_log(
        log_path=log_path,
        rows=rows
    )

    # ========================================================
    # Final verification
    # ========================================================

    output_count = 0
    size_errors = []

    for job in jobs:

        output_path = (
            output_dir
            / job[
                "relative_path"
            ]
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
                        args.expected_width,
                        args.expected_height
                    )
            ):

                size_errors.append(
                    (
                        job[
                            "stem"
                        ],
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

    if size_errors:

        raise RuntimeError(
            "Invalid train_H sizes found:\n"
            f"{size_errors[:20]}"
        )

    coverages = np.array(
        [
            row[
                "train_s_coverage"
            ]
            for row
            in rows
        ],
        dtype=np.float64
    )

    print()
    print(
        "========================================"
    )

    print(
        "train_H completed"
    )

    print(
        "========================================"
    )

    print(
        f"Pairs           : "
        f"{total}"
    )

    print(
        f"Output files    : "
        f"{output_count}"
    )

    print(
        f"Output size     : "
        f"{args.expected_width}"
        f"x"
        f"{args.expected_height}"
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