import json
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from tqdm.auto import tqdm

from src.utils.image_io import load_image

warnings.filterwarnings("ignore", category=UserWarning)


def image_files(root):
    root = Path(root)

    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(root.glob(ext))

    return {path.stem: path for path in sorted(files)}


def load_pair(pred_path, target_path, device):
    pred = load_image(pred_path).unsqueeze(0).to(device)
    target = load_image(target_path).unsqueeze(0).to(device)
    return pred, target


def build_metrics(config):
    return {
        name: instantiate(metric_config)
        for name, metric_config in config.metrics.items()
    }


@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    device = config.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    print(f"using device: {device}")

    pred_files = image_files(to_absolute_path(config.pred_dir))
    target_files = image_files(to_absolute_path(config.target_dir))

    image_ids = sorted(set(pred_files.keys()) & set(target_files.keys()))

    if not image_ids:
        raise RuntimeError("no matching image ids found")

    print(f"found {len(image_ids)} matching image ids")

    metrics = build_metrics(config)
    values = {name: [] for name in metrics.keys()}

    for image_id in tqdm(image_ids, desc="metrics"):
        pred, target = load_pair(pred_files[image_id], target_files[image_id], device)

        batch = {
            "recon": pred,
            "lensed": target,
        }

        for name, metric in metrics.items():
            value = metric(**batch)
            values[name].append(value)

    results = {name: sum(vals) / len(vals) for name, vals in values.items()}
    results["num_images"] = len(image_ids)

    for name, value in results.items():
        print(f"{name}: {value}")

    output_path = config.get("output_path")
    if output_path is not None:
        output_path = Path(to_absolute_path(output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
