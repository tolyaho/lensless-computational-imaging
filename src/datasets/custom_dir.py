from pathlib import Path

import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils.digicam_preprocess import align_lensed, rotate_lensless
from src.utils.image_io import load_image
from src.utils.psf import mask_vals_to_psf


class CustomDirDataset(BaseDataset):

    def __init__(self, root, limit=None, shuffle_index=False, instance_transforms=None):
        self.root = Path(root)
        self.lensless_dir = self.root / "lensless"
        self.masks_dir = self.root / "masks"
        self.lensed_dir = self.root / "lensed"

        if not self.lensless_dir.exists():
            raise FileNotFoundError(f"missing lensless dir: {self.lensless_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"missing masks dir: {self.masks_dir}")

        lensless_files = sorted(self.lensless_dir.glob("*.png"))

        if not lensless_files:
            raise RuntimeError(f"no png images found in {self.lensless_dir}")

        index = []
        for lensless_path in lensless_files:
            image_id = lensless_path.stem
            mask_path = self.masks_dir / f"{image_id}.npy"
            lensed_path = self.lensed_dir / f"{image_id}.png"

            if not mask_path.exists():
                raise FileNotFoundError(f"missing mask file for {image_id}: {mask_path}")

            item = {
                "image_id": image_id,
                "lensless": lensless_path,
                "mask": mask_path,
            }

            if lensed_path.exists():
                item["lensed"] = lensed_path

            index.append(item)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def __getitem__(self, index):
        item = self._index[index]

        lensless = rotate_lensless(load_image(item["lensless"]))
        mask = mask_vals_to_psf(np.load(item["mask"]))

        sample = {
            "image_id": item["image_id"],
            "lensless": lensless,
            "mask": mask,
        }

        if "lensed" in item:
            sample["lensed"] = align_lensed(load_image(item["lensed"]), lensless.shape[-2:])

        return self.preprocess_data(sample)

    @staticmethod
    def _assert_index_is_valid(index):
        for item in index:
            assert "image_id" in item
            assert "lensless" in item
            assert "mask" in item
