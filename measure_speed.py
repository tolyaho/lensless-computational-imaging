from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader

from inference import get_device, load_checkpoint
from src.datasets.collate import collate_fn

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG_DIR = Path(__file__).resolve().parent / "src" / "configs"


@dataclass(frozen=True)
class BenchmarkSpec:
    method: str
    model: str
    config_name: str
    checkpoint: str | None
    overrides: tuple[str, ...] = ()
    required: bool = True
    optional: bool = False


CHECKPOINT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "checkpoints/flatnet_lite.pth": (
        "outputs/train_flatnet_lite/flatnet_lite/checkpoint-epoch10.pth",
    ),
}

BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        method="ADMM-100 + bg sub",
        model="ADMM",
        config_name="inference",
        checkpoint=None,
    ),
    BenchmarkSpec(
        method="LeADMM-20",
        model="UnrolledADMM",
        config_name="inference_leadmm20",
        checkpoint="checkpoints/leadmm20.pth",
    ),
    BenchmarkSpec(
        method="LeADMM-5 + DRUNet pre",
        model="modular_leadmm5_pre_drunet8m",
        config_name="inference_modular_pre",
        checkpoint="checkpoints/leadmm5_pre_drunet8m.pth",
        overrides=("model=modular_leadmm5_pre_drunet8m",),
    ),
    BenchmarkSpec(
        method="LeADMM-5 + DRUNet post",
        model="modular_leadmm5_post_drunet8m",
        config_name="inference_modular_post",
        checkpoint="checkpoints/leadmm5_post_drunet8m.pth",
        overrides=("model=modular_leadmm5_post_drunet8m",),
    ),
    BenchmarkSpec(
        method="LeADMM-5 + DRUNet pre+post",
        model="modular_leadmm5_prepost_drunet8m",
        config_name="inference_modular_prepost",
        checkpoint="checkpoints/leadmm5_prepost_drunet8m.pth",
        overrides=("model=modular_leadmm5_prepost_drunet8m",),
    ),
    BenchmarkSpec(
        method="FlatNet-lite",
        model="flatnet_lite",
        config_name="inference_flatnet",
        checkpoint="checkpoints/flatnet_lite.pth",
        required=False,
        optional=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure lensless reconstruction forward-pass speed."
    )
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-path", type=str, default="reports/tables/speed.json")
    parser.add_argument("--csv-path", type=str, default=None)
    return parser.parse_args()


def resolve_checkpoint(path: str | None) -> Path | None:
    if path is None:
        return None

    candidates = (path, *CHECKPOINT_FALLBACKS.get(path, ()))
    for candidate in candidates:
        resolved = Path(to_absolute_path(candidate))
        if resolved.is_file():
            if candidate != path:
                print(f"note: using checkpoint fallback {candidate}")
            return resolved

    return Path(to_absolute_path(path))


def load_hydra_config(
    config_name: str,
    num_images: int,
    overrides: tuple[str, ...],
    checkpoint: str | None,
):
    all_overrides = [
        f"dataset.limit={num_images}",
        "dataset.shuffle_index=false",
        "dataset.require_lensed=true",
        "batch_size=1",
        "num_workers=0",
        *overrides,
    ]

    if checkpoint is not None:
        all_overrides.append(f"checkpoint={checkpoint}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=all_overrides)


def prepare_batches(config, device: torch.device, num_images: int) -> list[dict]:
    dataset = instantiate(config.dataset)

    if len(dataset) < num_images:
        raise RuntimeError(f"dataset has {len(dataset)} images, need {num_images}")

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    batches = []
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batches.append(batch)

        if len(batches) >= num_images:
            break

    return batches


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(config, device: torch.device):
    model = instantiate(config.model).to(device)
    checkpoint = config.get("checkpoint")

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, device, model_config=config.model)

    return model


def benchmark_model(
    model,
    batches: list[dict],
    device: torch.device,
    warmup: int,
) -> tuple[float, int]:
    if not batches:
        raise ValueError("no batches to benchmark")

    model.eval()

    with torch.no_grad():
        for i in range(warmup):
            model(**batches[i % len(batches)])

        synchronize(device)
        start = time.perf_counter()

        for batch in batches:
            model(**batch)

        synchronize(device)

    return time.perf_counter() - start, len(batches)


def format_result(
    spec: BenchmarkSpec,
    num_images: int,
    device: torch.device,
    seconds_total: float,
) -> dict:
    return {
        "method": spec.method,
        "model": spec.model,
        "num_images": num_images,
        "device": str(device),
        "seconds_total": round(seconds_total, 4),
        "seconds_per_image": round(seconds_total / num_images, 6),
        "images_per_second": round(num_images / seconds_total, 6)
        if seconds_total > 0
        else 0.0,
    }


def run_benchmark(
    spec: BenchmarkSpec,
    num_images: int,
    warmup: int,
    device: torch.device,
) -> dict | None:
    checkpoint = resolve_checkpoint(spec.checkpoint)

    if spec.checkpoint is not None and (checkpoint is None or not checkpoint.is_file()):
        message = f"checkpoint not found for {spec.method}: {spec.checkpoint}"

        if spec.optional:
            print(f"warning: {message} — skipping optional benchmark")
            return None

        raise FileNotFoundError(message)

    print(f"\n=== {spec.method} ===")

    checkpoint_override = None
    if checkpoint is not None and checkpoint.is_file():
        try:
            checkpoint_override = str(checkpoint.relative_to(Path.cwd()))
        except ValueError:
            checkpoint_override = str(checkpoint)

    config = load_hydra_config(
        config_name=spec.config_name,
        num_images=num_images,
        overrides=spec.overrides,
        checkpoint=checkpoint_override,
    )

    batches = prepare_batches(config, device, num_images)
    model = build_model(config, device)

    seconds_total, counted = benchmark_model(model, batches, device, warmup)

    print(
        f"timed {counted} forward passes in {seconds_total:.3f}s "
        f"({seconds_total / counted:.4f}s/image)"
    )

    return format_result(spec, counted, device, seconds_total)


def write_csv(path: Path, results: list[dict]) -> None:
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_summary_table(results: list[dict]) -> None:
    if not results:
        print("no benchmark results")
        return

    headers = ["method", "seconds/image", "images/sec", "total (s)", "n", "device"]

    rows = [
        [
            row["method"],
            f"{row['seconds_per_image']:.4f}",
            f"{row['images_per_second']:.4f}",
            f"{row['seconds_total']:.2f}",
            str(row["num_images"]),
            row["device"],
        ]
        for row in results
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print()
    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))

    for row in rows:
        print(fmt(row))

    print()


def main() -> int:
    args = parse_args()

    if args.num_images < 1:
        raise SystemExit("--num-images must be >= 1")

    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    device = get_device(args.device)

    print(f"device: {device}")
    print(f"num_images: {args.num_images}, warmup: {args.warmup}")

    results = []

    for spec in BENCHMARKS:
        try:
            result = run_benchmark(spec, args.num_images, args.warmup, device)
        except FileNotFoundError as exc:
            if spec.required:
                raise SystemExit(str(exc)) from exc

            print(f"warning: {exc} — skipping")
            continue
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if result is not None:
            results.append(result)

    output_path = Path(to_absolute_path(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    csv_path = Path(to_absolute_path(args.csv_path)) if args.csv_path else output_path.with_suffix(".csv")
    write_csv(csv_path, results)

    print(f"\nwrote {output_path}")
    print(f"wrote {csv_path}")

    print_summary_table(results)

    required = {spec.method for spec in BENCHMARKS if spec.required}
    got = {row["method"] for row in results}
    missing = required - got

    if missing:
        raise SystemExit(f"missing required benchmark results: {sorted(missing)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())