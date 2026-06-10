from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    """small folder dataset used by the demo or inference path."""

    def __init__(self, root, limit=None, shuffle_index=False, instance_transforms=None):
        self.root = Path(root)
        raise NotImplementedError
