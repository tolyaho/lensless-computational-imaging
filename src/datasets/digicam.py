from pathlib import Path

from datasets import load_dataset
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.utils.image_io import image_to_tensor, load_mask

DEFAULT_DATASET_NAME = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


class DigiCamDataset(BaseDataset):
    """huggingface digicam train or test split."""

    def __init__(
        self,
        dataset_name=None,
        split="train",
        masks_root=None,
        require_lensed=True,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        if dataset_name is None:
            dataset_name = DEFAULT_DATASET_NAME
        if masks_root is None:
            raise ValueError("masks_root must be set to the directory with mask_*.npy files")

        self.masks_root = Path(masks_root)
        if not self.masks_root.exists():
            raise FileNotFoundError(f"missing masks root: {self.masks_root}")

        self.require_lensed = require_lensed
        self.hf_dataset = load_dataset(dataset_name, split=split)
        self._mask_cache = {}

        index = []
        for idx in range(len(self.hf_dataset)):
            index.append({"image_id": f"{idx:06d}", "idx": idx})

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _load_mask_by_label(self, mask_label):
        label = int(mask_label)
        if label not in self._mask_cache:
            mask_path = self.masks_root / f"mask_{label}.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"missing mask file for label {label}: {mask_path}")
            self._mask_cache[label] = load_mask(mask_path)
        return self._mask_cache[label]

    @staticmethod
    def _row_image(row, field):
        value = row[field]
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            return value["image"]
        raise TypeError(f"unexpected {field} type: {type(value)}")

    def __getitem__(self, index):
        item = self._index[index]
        row = self.hf_dataset[item["idx"]]

        sample = {
            "image_id": item["image_id"],
            "lensless": image_to_tensor(self._row_image(row, "lensless")),
            "mask": self._load_mask_by_label(row["mask_label"]).clone(),
        }

        lensed = row.get("lensed")
        if lensed is not None:
            sample["lensed"] = image_to_tensor(self._row_image(row, "lensed"))
        elif self.require_lensed:
            raise FileNotFoundError(f"missing lensed image for {item['image_id']}")

        return self.preprocess_data(sample)

    @staticmethod
    def _assert_index_is_valid(index):
        for item in index:
            assert "image_id" in item
            assert "idx" in item
