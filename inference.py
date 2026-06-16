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


def load_checkpoint(model, checkpoint_path, device, model_config=None):
    path = Path(to_absolute_path(checkpoint_path))
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    print(f"loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and checkpoint.get("state_dict") is not None:
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch")
        if epoch is not None:
            print(f"checkpoint epoch: {epoch}")
        saved_config = checkpoint.get("config")
        if (
            model_config is not None
            and saved_config is not None
            and saved_config.get("model") != model_config
        ):
            print(
                "warning: model config differs from checkpoint; "
                "state_dict load may fail if architectures do not match"
            )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


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

    checkpoint_path = config.get("checkpoint")
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device, model_config=config.model)
    else:
        print("no checkpoint provided, using initialized model weights")

    model.eval()

    output_dir = Path(to_absolute_path(config.output_dir))
    recon_dir = output_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    save_targets = config.get("save_targets", False)
    target_dir = output_dir / "lensed"
    if save_targets:
        target_dir.mkdir(parents=True, exist_ok=True)

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
                save_image(recon[i].clamp(0, 1), recon_dir / f"{image_id}.png")

                if save_targets and "lensed" in batch:
                    save_image(
                        batch["lensed"][i].detach().cpu().clamp(0, 1),
                        target_dir / f"{image_id}.png",
                    )

    print(f"saved reconstructions to {recon_dir}")
    if save_targets and target_dir.exists():
        print(f"saved targets to {target_dir}")


if __name__ == "__main__":
    main()
