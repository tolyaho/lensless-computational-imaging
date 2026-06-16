from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

GT_DIR = REPO_ROOT / "outputs/final_leadmm5_drunet8m_prepost/lensed"

COLUMNS = [
    ("GT", GT_DIR, False),
    ("ADMM-100", REPO_ROOT / "outputs/final_admm100_bgsub/recon", False),
    ("LeADMM-20", REPO_ROOT / "outputs/final_leadmm20/recon", False),
    ("Pre", REPO_ROOT / "outputs/final_leadmm5_drunet8m_pre/recon", False),
    ("Post", REPO_ROOT / "outputs/final_leadmm5_drunet8m_post/recon", False),
    ("Pre+Post", REPO_ROOT / "outputs/final_leadmm5_drunet8m_prepost/recon", False),
    ("FlatNet-lite", REPO_ROOT / "outputs/final_flatnet_lite/recon", True),
]

COMPACT_COLUMNS = ("GT", "ADMM-100", "LeADMM-20", "Pre+Post")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-stems", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="reports/assets/qualitative")
    return parser.parse_args()


def index_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}

    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resolve_columns() -> list[tuple[str, dict[str, Path]]]:
    columns = []

    for title, folder, optional in COLUMNS:
        index = index_images(folder)

        if not index:
            if optional:
                print(f"[skip] optional column missing or empty: {title}")
                continue
            raise FileNotFoundError(f"missing required images for {title}: {folder}")

        columns.append((title, index))

    return columns


def get_shared_stems(columns: list[tuple[str, dict[str, Path]]]) -> list[str]:
    stem_sets = [set(index) for _, index in columns]
    shared = set.intersection(*stem_sets)

    if not shared:
        raise RuntimeError("no shared image stems across selected folders")

    return sorted(shared)


def pick_stems(
    columns: list[tuple[str, dict[str, Path]]],
    num_samples: int,
    sample_stems: list[str] | None,
) -> list[str]:
    shared = get_shared_stems(columns)

    if sample_stems:
        missing = [stem for stem in sample_stems if stem not in shared]
        if missing:
            raise ValueError(f"sample stems not found everywhere: {missing}")
        return sample_stems

    if num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    if len(shared) < num_samples:
        raise ValueError(f"only {len(shared)} shared stems, requested {num_samples}")

    ids = np.linspace(0, len(shared) - 1, num_samples, dtype=int)
    return [shared[i] for i in ids]


def select_columns(
    columns: list[tuple[str, dict[str, Path]]],
    titles: tuple[str, ...],
) -> list[tuple[str, dict[str, Path]]]:
    by_title = dict(columns)

    missing = [title for title in titles if title not in by_title]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    return [(title, by_title[title]) for title in titles]


def save_grid(
    columns: list[tuple[str, dict[str, Path]]],
    stems: list[str],
    output_path: Path,
) -> None:
    rows, cols = len(stems), len(columns)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.8 * cols, 2.8 * rows),
        squeeze=False,
    )

    for col, (title, index) in enumerate(columns):
        axes[0, col].set_title(title, fontsize=11, pad=6)

        for row, stem in enumerate(stems):
            ax = axes[row, col]
            ax.imshow(load_image(index[stem]))
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(stem, fontsize=9, rotation=0, labelpad=36, va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output_path}")


def main() -> int:
    args = parse_args()

    columns = resolve_columns()
    stems = pick_stems(columns, args.num_samples, args.sample_stems)
    output_dir = (REPO_ROOT / args.output_dir).resolve()

    print("samples:", ", ".join(stems))

    save_grid(columns, stems, output_dir / "final_comparison_grid.png")
    save_grid(
        select_columns(columns, COMPACT_COLUMNS),
        stems,
        output_dir / "final_comparison_grid_compact.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
