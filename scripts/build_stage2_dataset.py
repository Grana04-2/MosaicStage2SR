from pathlib import Path
import csv
import shutil

from PIL import Image


# ============================================================
# 設定
# ============================================================

# Stage 1 pix2pixHD project
STAGE1_PROJECT = Path(
    r"C:\Users\MSDev\PycharmProjects\DeepMosaics\train\pix2pixHD"
)

# Stage 1 epoch7 final outputs
STAGE1_OUTPUT_DIR = (
    STAGE1_PROJECT
    / "stage2_export"
    / "stage1_epoch7"
)

# 元clean 1280x720
CLEAN_DIR = (
    STAGE1_PROJECT
    / "datasets"
    / "original_4000"
    / "train_B"
)

# Stage 2 project
STAGE2_PROJECT = Path(
    r"C:\Users\MSDev\PycharmProjects\MosaicStage2SR"
)

OUTPUT_ROOT = (
    STAGE2_PROJECT
    / "datasets"
    / "stage2_1280x720"
)

OUTPUT_A_DIR = (
    OUTPUT_ROOT
    / "train_A"
)

OUTPUT_B_DIR = (
    OUTPUT_ROOT
    / "train_B"
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "manifest.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


EXPECTED_LR_SIZE = (
    768,
    432
)

EXPECTED_HR_SIZE = (
    1280,
    720
)


# ============================================================
# helper
# ============================================================

def make_image_map(folder):

    image_map = {}

    for path in folder.iterdir():

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        stem = path.stem

        if stem in image_map:

            raise RuntimeError(
                f"Duplicate stem: {stem}\n"
                f"{image_map[stem]}\n"
                f"{path}"
            )

        image_map[stem] = path

    return image_map


# ============================================================
# main
# ============================================================

def main():

    print("========================================")
    print("Build Stage 2 dataset")
    print("========================================")

    if not STAGE1_OUTPUT_DIR.exists():

        raise FileNotFoundError(
            STAGE1_OUTPUT_DIR
        )

    if not CLEAN_DIR.exists():

        raise FileNotFoundError(
            CLEAN_DIR
        )

    stage1_map = make_image_map(
        STAGE1_OUTPUT_DIR
    )

    clean_map = make_image_map(
        CLEAN_DIR
    )

    paired_stems = sorted(
        set(stage1_map.keys())
        &
        set(clean_map.keys())
    )

    missing_clean = (
        set(stage1_map.keys())
        -
        set(clean_map.keys())
    )

    missing_stage1 = (
        set(clean_map.keys())
        -
        set(stage1_map.keys())
    )

    print(
        f"Stage1 outputs : "
        f"{len(stage1_map)}"
    )

    print(
        f"Clean images   : "
        f"{len(clean_map)}"
    )

    print(
        f"Pairs          : "
        f"{len(paired_stems)}"
    )

    print(
        f"Missing clean  : "
        f"{len(missing_clean)}"
    )

    print(
        f"Missing Stage1 : "
        f"{len(missing_stage1)}"
    )

    if missing_clean:

        raise RuntimeError(
            "Stage1 output に対応する"
            " clean画像がありません。"
        )

    if missing_stage1:

        raise RuntimeError(
            "一部clean画像に対応する"
            " Stage1 outputがありません。"
        )

    # --------------------------------------------------------
    # output
    # --------------------------------------------------------

    OUTPUT_A_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_B_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if any(OUTPUT_A_DIR.iterdir()):

        raise RuntimeError(
            f"train_A が空ではありません:\n"
            f"{OUTPUT_A_DIR}"
        )

    if any(OUTPUT_B_DIR.iterdir()):

        raise RuntimeError(
            f"train_B が空ではありません:\n"
            f"{OUTPUT_B_DIR}"
        )

    rows = []

    for index, stem in enumerate(
            paired_stems,
            start=1
    ):

        stage1_path = (
            stage1_map[stem]
        )

        clean_path = (
            clean_map[stem]
        )

        # ----------------------------------------------------
        # size check
        # ----------------------------------------------------

        with Image.open(
                stage1_path
        ) as img_a:

            size_a = img_a.size

        with Image.open(
                clean_path
        ) as img_b:

            size_b = img_b.size

        if size_a != EXPECTED_LR_SIZE:

            raise RuntimeError(
                f"Stage1 size mismatch: "
                f"{stage1_path.name}\n"
                f"{size_a} "
                f"!= {EXPECTED_LR_SIZE}"
            )

        if size_b != EXPECTED_HR_SIZE:

            raise RuntimeError(
                f"Clean size mismatch: "
                f"{clean_path.name}\n"
                f"{size_b} "
                f"!= {EXPECTED_HR_SIZE}"
            )

        # ----------------------------------------------------
        # Stage 2 A
        #
        # Stage1 outputはPNGで保存
        # ----------------------------------------------------

        dst_a = (
            OUTPUT_A_DIR
            / f"{stem}.png"
        )

        shutil.copy2(
            stage1_path,
            dst_a
        )

        # ----------------------------------------------------
        # Stage 2 B
        #
        # 元画像は再圧縮しない。
        # 拡張子をそのまま維持。
        # ----------------------------------------------------

        dst_b = (
            OUTPUT_B_DIR
            / clean_path.name
        )

        shutil.copy2(
            clean_path,
            dst_b
        )

        # ----------------------------------------------------
        # manifest
        # ----------------------------------------------------

        scale_x = (
            size_b[0]
            / size_a[0]
        )

        scale_y = (
            size_b[1]
            / size_a[1]
        )

        rows.append(
            {
                "stem": stem,
                "stage1_file":
                    dst_a.name,
                "clean_file":
                    dst_b.name,
                "lr_width":
                    size_a[0],
                "lr_height":
                    size_a[1],
                "hr_width":
                    size_b[0],
                "hr_height":
                    size_b[1],
                "scale_x":
                    scale_x,
                "scale_y":
                    scale_y,
            }
        )

        if (
                index % 100 == 0
                or
                index == len(paired_stems)
        ):

            print(
                f"Copied "
                f"{index}/"
                f"{len(paired_stems)}"
            )

    # --------------------------------------------------------
    # manifest.csv
    # --------------------------------------------------------

    with open(
            MANIFEST_PATH,
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "stage1_file",
                "clean_file",
                "lr_width",
                "lr_height",
                "hr_width",
                "hr_height",
                "scale_x",
                "scale_y",
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print("========================================")
    print("Stage 2 dataset completed")
    print("========================================")

    print(
        f"Pairs: "
        f"{len(rows)}"
    )

    print(
        f"Dataset: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()