import torch


def collate_fn(dataset_items: list[dict]):
    out = {
        "image_id": [x["image_id"] for x in dataset_items],
        "lensless": torch.stack([x["lensless"] for x in dataset_items]),
        "mask": torch.stack([x["mask"] for x in dataset_items]),
    }

    if all("lensed" in x for x in dataset_items):
        out["lensed"] = torch.stack([x["lensed"] for x in dataset_items])

    return out
