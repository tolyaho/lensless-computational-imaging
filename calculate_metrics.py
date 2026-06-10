import json
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from tqdm.auto import tqdm

from src.utils.image_io import load_image

warnings.filterwarnings("ignore", category=UserWarning)


def require_dir(path, name):
    if path is None:
        raise ValueError(f"{name} is required")
    path = Path(to_absolute_path(path))
    if not path.is_dir():
        raise FileNotFoundError(f"missing {name} directory: {path}")
    return path


def image_files(root):
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


def match_image_ids(pred_files, gt_files, match_by):
    if match_by != "image_id":
        raise ValueError(f"unsupported match_by: {match_by}")

    pred_only = sorted(set(pred_files) - set(gt_files))
    gt_only = sorted(set(gt_files) - set(pred_files))
    if pred_only:
        print(f"warning: {len(pred_only)} predictions without ground truth, skipped")
    if gt_only:
        print(
            f"warning: {len(gt_only)} ground truth images without predictions, skipped"
        )

    return sorted(set(pred_files.keys()) & set(gt_files.keys()))


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

    pred_dir = require_dir(config.pred_dir, "pred_dir")
    gt_dir = require_dir(config.gt_dir, "gt_dir")

    pred_files = image_files(pred_dir)
    gt_files = image_files(gt_dir)

    image_ids = match_image_ids(pred_files, gt_files, config.match_by)

    if not image_ids:
        raise RuntimeError("no matching image ids found")

    print(f"found {len(image_ids)} matching image ids")

    metrics = build_metrics(config)
    values = {name: [] for name in metrics.keys()}

    for image_id in tqdm(image_ids, desc="metrics"):
        pred, target = load_pair(pred_files[image_id], gt_files[image_id], device)

        batch = {
            "recon": pred,
            "lensed": target,
        }

        for name, metric in metrics.items():
            values[name].append(metric(**batch))

    results = {name: sum(vals) / len(vals) for name, vals in values.items()}
    results["num_images"] = len(image_ids)

    for name in metrics.keys():
        print(f"{name}: {results[name]}")
    print(f"num_images: {results['num_images']}")

    output_path = config.get("output_path")
    if output_path is not None:
        output_path = Path(to_absolute_path(output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
