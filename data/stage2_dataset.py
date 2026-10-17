import csv
import random
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# image utility
# ============================================================

def pil_to_tensor(image):
    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = (
            array / 255.0
    )

    array = np.transpose(
        array,
        (
            2,
            0,
            1
        )
    )

    return torch.from_numpy(
        array
    )


def pil_mask_to_tensor(
        image
):
    """
    Grayscale binary mask
        [H, W]
    ->
        [1, H, W]

    0   -> 0.0
    255 -> 1.0
    """

    array = np.array(
        image,
        dtype=np.float32,
        copy=True
    )

    array = (
            array / 255.0
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return torch.from_numpy(
        array
    )


# ============================================================
# manifest
# ============================================================

def load_manifest(
        dataset_root
):
    dataset_root = Path(
        dataset_root
    )

    manifest_path = (
            dataset_root
            / "manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found:\n"
            f"{manifest_path}"
        )

    records = []

    with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            records.append(
                row
            )

    if not records:
        raise RuntimeError(
            "manifest.csv is empty."
        )

    return records


# ============================================================
# train / val split
# ============================================================

def split_records(
        records,
        val_ratio=0.10,
        seed=1234
):
    indices = list(
        range(
            len(records)
        )
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        indices
    )

    val_count = max(
        1,
        int(
            len(indices)
            * val_ratio
        )
    )

    val_indices = set(
        indices[
        :val_count
        ]
    )

    train_records = []
    val_records = []

    for index, record in enumerate(
            records
    ):

        if index in val_indices:

            val_records.append(
                record
            )

        else:

            train_records.append(
                record
            )

    return (
        train_records,
        val_records
    )


# ============================================================
# Sequence ID
# ============================================================

def get_sequence_id_from_stem(
        stem
):
    # Example:
    #
    # 00115_00005
    #     ->
    # 00115

    if "_" not in stem:
        return stem

    sequence_id, _ = stem.rsplit(
        "_",
        1
    )

    return sequence_id


# ============================================================
# Sequence-level train / validation split
# ============================================================

def split_records_by_sequence(
        records,
        val_ratio=0.10,
        seed=1234
):
    # --------------------------------------------------------
    # Group records by sequence
    # --------------------------------------------------------

    sequence_to_records = {}

    for record in records:

        stem = record[
            "stem"
        ]

        sequence_id = (
            get_sequence_id_from_stem(
                stem
            )
        )

        if sequence_id not in sequence_to_records:
            sequence_to_records[
                sequence_id
            ] = []

        sequence_to_records[
            sequence_id
        ].append(
            record
        )

    sequence_ids = list(
        sequence_to_records.keys()
    )

    if len(
            sequence_ids
    ) < 2:
        raise RuntimeError(
            "Sequence-level split requires "
            "at least 2 sequences."
        )

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    rng = random.Random(
        seed
    )

    rng.shuffle(
        sequence_ids
    )

    # --------------------------------------------------------
    # Number of validation sequences
    #
    # Never split one sequence just to obtain an exact
    # validation frame count.
    # --------------------------------------------------------

    val_sequence_count = max(
        1,
        int(
            math.ceil(
                len(sequence_ids)
                * val_ratio
            )
        )
    )

    # trainも最低1系列残す
    val_sequence_count = min(
        val_sequence_count,
        len(sequence_ids) - 1
    )

    val_sequence_ids = set(
        sequence_ids[
        :val_sequence_count
        ]
    )

    train_sequence_ids = set(
        sequence_ids[
        val_sequence_count:
        ]
    )

    # --------------------------------------------------------
    # Expand groups back into record lists
    # --------------------------------------------------------

    train_records = []
    val_records = []

    for record in records:

        sequence_id = (
            get_sequence_id_from_stem(
                record[
                    "stem"
                ]
            )
        )

        if sequence_id in val_sequence_ids:

            val_records.append(
                record
            )

        else:

            train_records.append(
                record
            )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    overlap = (
            train_sequence_ids
            &
            val_sequence_ids
    )

    if len(
            overlap
    ) > 0:
        raise RuntimeError(
            "Sequence leakage detected: "
            f"{sorted(overlap)}"
        )

    return (
        train_records,
        val_records
    )


# ============================================================
# dataset
# ============================================================

class Stage2SRDataset(Dataset):

    def __init__(
            self,
            dataset_root,
            records,
            train=True,
            lr_patch_size=192,
            scale_num=5,
            scale_den=3,
            horizontal_flip=True,
            lr_dir_name="train_A",
            mosaic_dir_name=None,
            supervision_dir_name=None
    ):

        self.dataset_root = Path(
            dataset_root
        )

        self.records = records

        self.train = train

        self.lr_patch_size = (
            lr_patch_size
        )

        self.scale_num = (
            scale_num
        )

        self.scale_den = (
            scale_den
        )

        self.horizontal_flip = (
            horizontal_flip
        )

        if (
                lr_patch_size
                % scale_den
                != 0
        ):
            raise ValueError(
                "lr_patch_size must be "
                f"divisible by {scale_den}"
            )

        self.hr_patch_size = (
                lr_patch_size
                * scale_num
                // scale_den
        )

        self.lr_dir = (
                self.dataset_root
                / lr_dir_name
        )

        if not self.lr_dir.exists():
            raise FileNotFoundError(
                "LR input directory not found:\n"
                f"{self.lr_dir}"
            )

        self.hr_dir = (
                self.dataset_root
                / "train_B"
        )

        self.mosaic_dir = (
            None
            if mosaic_dir_name is None
            else
            (
                    self.dataset_root
                    / mosaic_dir_name
            )
        )

        if (
                self.mosaic_dir is not None
                and
                not self.mosaic_dir.exists()
        ):
            raise FileNotFoundError(
                "Mosaic input directory not found:\n"
                f"{self.mosaic_dir}"
            )

        self.supervision_dir = (
            None
            if supervision_dir_name is None
            else
            (
                self.dataset_root
                / supervision_dir_name
            )
        )

        if (
                self.supervision_dir is not None
                and
                not self.supervision_dir.exists()
        ):
            raise FileNotFoundError(
                "Supervision directory not found:\n"
                f"{self.supervision_dir}"
            )

    def __len__(self):

        return len(
            self.records
        )

    def _load_pair(
            self,
            record
    ):

        lr_path = (
                self.lr_dir
                / record[
                    "stage1_file"
                ]
        )

        hr_path = (
                self.hr_dir
                / record[
                    "clean_file"
                ]
        )

        mosaic_path = None
        supervision_path = None

        if self.mosaic_dir is not None:
            mosaic_path = (
                    self.mosaic_dir
                    / record[
                        "stage1_file"
                    ]
            )

        if self.supervision_dir is not None:
            supervision_path = (
                    self.supervision_dir
                    / record[
                        "stage1_file"
                    ]
            )

        if not lr_path.exists():
            raise FileNotFoundError(
                lr_path
            )

        if not hr_path.exists():
            raise FileNotFoundError(
                hr_path
            )

        if (
                mosaic_path is not None
                and
                not mosaic_path.exists()
        ):
            raise FileNotFoundError(
                mosaic_path
            )

        if (
                supervision_path is not None
                and
                not supervision_path.exists()
        ):
            raise FileNotFoundError(
                supervision_path
            )

        lr_image = Image.open(
            lr_path
        ).convert(
            "RGB"
        )

        hr_image = Image.open(
            hr_path
        ).convert(
            "RGB"
        )

        # ================================================
        # Optional Original Mosaic
        # ================================================

        mosaic_image = None

        if mosaic_path is not None:

            mosaic_image = Image.open(
                mosaic_path
            ).convert(
                "RGB"
            )

            if (
                    mosaic_image.size
                    != lr_image.size
            ):
                raise RuntimeError(
                    "Stage1 / Mosaic size mismatch:\n"
                    f"Stage1 = {lr_image.size}\n"
                    f"Mosaic = {mosaic_image.size}\n"
                    f"Stage1 path = {lr_path}\n"
                    f"Mosaic path = {mosaic_path}"
                )

        # ================================================
        # Optional Mosaic supervision mask
        # ================================================

        supervision_image = None

        if supervision_path is not None:

            supervision_image = Image.open(
                supervision_path
            ).convert(
                "L"
            )

            if (
                    supervision_image.size
                    != lr_image.size
            ):
                raise RuntimeError(
                    "Stage1 / Supervision size mismatch:\n"
                    f"Stage1      = {lr_image.size}\n"
                    f"Supervision = {supervision_image.size}\n"
                    f"Stage1 path = {lr_path}\n"
                    f"Supervision path = {supervision_path}"
                )

        return (
            lr_image,
            mosaic_image,
            supervision_image,
            hr_image,
            lr_path,
            mosaic_path,
            supervision_path,
            hr_path
        )

    def _check_ratio(
            self,
            lr_image,
            hr_image
    ):

        lr_width, lr_height = (
            lr_image.size
        )

        hr_width, hr_height = (
            hr_image.size
        )

        expected_hr_width = (
                lr_width
                * self.scale_num
                // self.scale_den
        )

        expected_hr_height = (
                lr_height
                * self.scale_num
                // self.scale_den
        )

        if (
                expected_hr_width
                != hr_width
                or
                expected_hr_height
                != hr_height
        ):
            raise RuntimeError(
                "LR/HR ratio mismatch:\n"
                f"LR = {lr_image.size}\n"
                f"HR = {hr_image.size}\n"
                f"Expected HR = "
                f"({expected_hr_width}, "
                f"{expected_hr_height})"
            )

    def _random_aligned_crop(
            self,
            lr_image,
            mosaic_image,
            supervision_image,
            hr_image
    ):

        lr_width, lr_height = (
            lr_image.size
        )

        patch = (
            self.lr_patch_size
        )

        if (
                lr_width < patch
                or
                lr_height < patch
        ):
            raise RuntimeError(
                "LR image is smaller than patch."
            )

        max_x = (
                lr_width
                - patch
        )

        max_y = (
                lr_height
                - patch
        )

        # ================================================
        # 5/3対応のため
        # crop開始位置も3の倍数に限定
        # ================================================

        x_lr = (
                random.randint(
                    0,
                    max_x
                    // self.scale_den
                )
                * self.scale_den
        )

        y_lr = (
                random.randint(
                    0,
                    max_y
                    // self.scale_den
                )
                * self.scale_den
        )

        x_hr = (
                x_lr
                * self.scale_num
                // self.scale_den
        )

        y_hr = (
                y_lr
                * self.scale_num
                // self.scale_den
        )

        # ================================================
        # Stage1
        # ================================================

        lr_patch = lr_image.crop(
            (
                x_lr,
                y_lr,
                x_lr + patch,
                y_lr + patch
            )
        )

        # ================================================
        # Original Mosaic
        # ================================================

        mosaic_patch = None

        if mosaic_image is not None:
            mosaic_patch = mosaic_image.crop(
                (
                    x_lr,
                    y_lr,
                    x_lr + patch,
                    y_lr + patch
                )
            )

        # ================================================
        # Mosaic supervision
        #
        # train_S is LR size, so use exactly the same
        # crop coordinates as train_A / train_M.
        # ================================================

        supervision_patch = None

        if supervision_image is not None:
            supervision_patch = (
                supervision_image.crop(
                    (
                        x_lr,
                        y_lr,
                        x_lr + patch,
                        y_lr + patch
                    )
                )
            )

        # ================================================
        # HR Target
        # ================================================

        hr_patch_size = (
            self.hr_patch_size
        )

        hr_patch = hr_image.crop(
            (
                x_hr,
                y_hr,
                x_hr + hr_patch_size,
                y_hr + hr_patch_size
            )
        )

        return (
            lr_patch,
            mosaic_patch,
            supervision_patch,
            hr_patch
        )

    def __getitem__(
            self,
            index
    ):

        record = (
            self.records[
                index
            ]
        )

        (
            lr_image,
            mosaic_image,
            supervision_image,
            hr_image,
            lr_path,
            mosaic_path,
            supervision_path,
            hr_path
        ) = self._load_pair(
            record
        )

        self._check_ratio(
            lr_image,
            hr_image
        )

        if self.train:

            (
                lr_image,
                mosaic_image,
                supervision_image,
                hr_image
            ) = self._random_aligned_crop(
                lr_image,
                mosaic_image,
                supervision_image,
                hr_image
            )

            # ============================================
            # synchronized horizontal flip
            # ============================================

            if (
                    self.horizontal_flip
                    and
                    random.random()
                    < 0.5
            ):

                lr_image = (
                    lr_image.transpose(
                        Image.Transpose.FLIP_LEFT_RIGHT
                    )
                )

                if mosaic_image is not None:
                    mosaic_image = (
                        mosaic_image.transpose(
                            Image.Transpose.FLIP_LEFT_RIGHT
                        )
                    )

                if supervision_image is not None:
                    supervision_image = (
                        supervision_image.transpose(
                            Image.Transpose.FLIP_LEFT_RIGHT
                        )
                    )

                hr_image = (
                    hr_image.transpose(
                        Image.Transpose.FLIP_LEFT_RIGHT
                    )
                )

        # ================================================
        # Tensor conversion
        # ================================================

        lr_tensor = pil_to_tensor(
            lr_image
        )

        hr_tensor = pil_to_tensor(
            hr_image
        )

        output = {
            "lr":
                lr_tensor,

            "hr":
                hr_tensor,

            "stem":
                record[
                    "stem"
                ],

            "lr_path":
                str(
                    lr_path
                ),

            "hr_path":
                str(
                    hr_path
                )
        }

        # ================================================
        # Optional Original Mosaic
        # ================================================

        if mosaic_image is not None:
            mosaic_tensor = pil_to_tensor(
                mosaic_image
            )

            output[
                "mosaic"
            ] = mosaic_tensor

            output[
                "mosaic_path"
            ] = str(
                mosaic_path
            )

        # ================================================
        # Optional Mosaic supervision mask
        #
        # shape:
        #   full       [1, 432, 768]
        #   train crop [1, 192, 192]
        #
        # values:
        #   0.0 / 1.0
        # ================================================

        if supervision_image is not None:
            supervision_tensor = (
                pil_mask_to_tensor(
                    supervision_image
                )
            )

            output[
                "supervision"
            ] = supervision_tensor

            output[
                "supervision_path"
            ] = str(
                supervision_path
            )

        return output
