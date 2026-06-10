from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)
    datasets = instantiate(config.datasets)
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]
        assert config.dataloader.batch_size <= len(
            dataset
        ), f"The batch size ({config.dataloader.batch_size}) cannot be larger than the dataset length ({len(dataset)})"
        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=dataset_partition == "train",
            shuffle=dataset_partition == "train",
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader
    return (dataloaders, batch_transforms)
