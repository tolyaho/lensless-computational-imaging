from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.image_io import load_image, load_mask


class CustomDirDataset(BaseDataset):
    """small folder dataset used by the demo or inference path."""

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

        sample = {
            "image_id": item["image_id"],
            "lensless": load_image(item["lensless"]),
            "mask": load_mask(item["mask"]),
        }

        if "lensed" in item:
            sample["lensed"] = load_image(item["lensed"])

        return self.preprocess_data(sample)

    @staticmethod
    def _assert_index_is_valid(index):
        for item in index:
            assert "image_id" in item
            assert "lensless" in item
            assert "mask" in item