def collate_fn(dataset_items: list[dict]):
    """stack lensless batches for the trainer/inferencer."""
    raise NotImplementedError
