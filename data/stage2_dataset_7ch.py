import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# RGB -> Tensor
# ============================================================

def pil_to_tensor(
        image
):
    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1
    )

    return tensor


# ============================================================
# Grayscale mask -> [1, H, W]
# ============================================================

def mask_to_tensor(
        image
):
    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).unsqueeze(
        0
    )

    return tensor


# ============================================================
# 7ch Dataset
# ============================================================

class Stage2SRDataset7ch(
        Dataset
):

    def __init__(
            self,
            dataset_root,
            records,
            train=True,
            lr_patch_size=192,
            scale_num=5,
            scale_den=3,
            horizontal_flip=True,
            mosaic_dir_name="train_M",
            stage1_mask_dir_name="train_stage1_mask"
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
            / "train_A"
        )

        self.hr_dir = (
            self.dataset_root
            / "train_B"
        )

        self.mosaic_dir = (
            self.dataset_root
            / mosaic_dir_name
        )

        self.stage1_mask_dir = (
            self.dataset_root
            / stage1_mask_dir_name
        )

        for path in [
            self.lr_dir,
            self.hr_dir,
            self.mosaic_dir,
            self.stage1_mask_dir,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    path
                )

    def __len__(
            self
    ):
        return len(
            self.records
        )

    # ========================================================
    # Load
    # ========================================================

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

        mosaic_path = (
            self.mosaic_dir
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

        mask_relative_path = Path(
            record[
                "stage1_file"
            ]
        ).with_suffix(
            ".png"
        )

        stage1_mask_path = (
            self.stage1_mask_dir
            / mask_relative_path
        )

        for path in [
            lr_path,
            mosaic_path,
            hr_path,
            stage1_mask_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    path
                )

        lr_image = Image.open(
            lr_path
        ).convert(
            "RGB"
        )

        mosaic_image = Image.open(
            mosaic_path
        ).convert(
            "RGB"
        )

        hr_image = Image.open(
            hr_path
        ).convert(
            "RGB"
        )

        stage1_mask_image = Image.open(
            stage1_mask_path
        ).convert(
            "L"
        )

        if (
                mosaic_image.size
                != lr_image.size
        ):
            raise RuntimeError(
                "Stage1 / Mosaic size mismatch:\n"
                f"Stage1 = {lr_image.size}\n"
                f"Mosaic = {mosaic_image.size}"
            )

        if (
                stage1_mask_image.size
                != lr_image.size
        ):
            raise RuntimeError(
                "Stage1 / Mask size mismatch:\n"
                f"Stage1 = {lr_image.size}\n"
                f"Mask   = {stage1_mask_image.size}"
            )

        return (
            lr_image,
            mosaic_image,
            stage1_mask_image,
            hr_image,
            lr_path,
            mosaic_path,
            stage1_mask_path,
            hr_path
        )

    # ========================================================
    # Ratio
    # ========================================================

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

    # ========================================================
    # Synchronized crop
    # ========================================================

    def _random_aligned_crop(
            self,
            lr_image,
            mosaic_image,
            stage1_mask_image,
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

        lr_patch = lr_image.crop(
            (
                x_lr,
                y_lr,
                x_lr + patch,
                y_lr + patch
            )
        )

        mosaic_patch = (
            mosaic_image.crop(
                (
                    x_lr,
                    y_lr,
                    x_lr + patch,
                    y_lr + patch
                )
            )
        )

        mask_patch = (
            stage1_mask_image.crop(
                (
                    x_lr,
                    y_lr,
                    x_lr + patch,
                    y_lr + patch
                )
            )
        )

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
            mask_patch,
            hr_patch
        )

    # ========================================================
    # Get item
    # ========================================================

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
            stage1_mask_image,
            hr_image,
            lr_path,
            mosaic_path,
            stage1_mask_path,
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
                stage1_mask_image,
                hr_image
            ) = self._random_aligned_crop(
                lr_image,
                mosaic_image,
                stage1_mask_image,
                hr_image
            )

            if (
                    self.horizontal_flip
                    and
                    random.random()
                    < 0.5
            ):
                lr_image = lr_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                mosaic_image = mosaic_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                stage1_mask_image = (
                    stage1_mask_image.transpose(
                        Image.Transpose.FLIP_LEFT_RIGHT
                    )
                )

                hr_image = hr_image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

        output = {
            "lr":
                pil_to_tensor(
                    lr_image
                ),

            "mosaic":
                pil_to_tensor(
                    mosaic_image
                ),

            "stage1_mask":
                mask_to_tensor(
                    stage1_mask_image
                ),

            "hr":
                pil_to_tensor(
                    hr_image
                ),

            "stem":
                record[
                    "stem"
                ],

            "lr_path":
                str(
                    lr_path
                ),

            "mosaic_path":
                str(
                    mosaic_path
                ),

            "stage1_mask_path":
                str(
                    stage1_mask_path
                ),

            "hr_path":
                str(
                    hr_path
                ),
        }

        return output