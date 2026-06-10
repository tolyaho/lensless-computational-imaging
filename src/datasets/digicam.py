from src.datasets.base_dataset import BaseDataset


class DigiCamDataset(BaseDataset):
    """huggingface digicam train/test loader."""

    def __init__(
        self,
        root,
        split="train",
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        raise NotImplementedError
