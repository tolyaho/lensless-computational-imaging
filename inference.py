import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.image_io import save_image

warnings.filterwarnings("ignore", category=UserWarning)


def get_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    device = get_device(config.get("device", "auto"))
    print(f"using device: {device}")

    dataset = instantiate(config.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    print(f"dataset size: {len(dataset)}")

    model = instantiate(config.model).to(device)
    model.eval()

    output_dir = Path(to_absolute_path(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="inference"):
            image_ids = batch["image_id"]

            model_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            output = model(**model_batch)
            recon = output["recon"].detach().cpu()

            for i, image_id in enumerate(image_ids):
                save_image(recon[i], output_dir / f"{image_id}.png")

    print(f"saved reconstructions to {output_dir}")


if __name__ == "__main__":
    main()
