import argparse
import os
import shutil
from pathlib import Path

HF_REPO_ID = "tolyho/lensless-computational-imaging-checkpoints"

CHECKPOINTS = {
    "leadmm20.pth": "leadmm20_model_best.pth",
    "leadmm5_pre_drunet8m.pth": "leadmm5_drunet8m_pre_model_best.pth",
    "leadmm5_post_drunet8m.pth": "leadmm5_drunet8m_post_model_best.pth",
    "leadmm5_prepost_drunet8m.pth": "leadmm5_drunet8m_prepost_model_best.pth",
    "flatnet_lite.pth": "flatnet_lite_model_best.pth",
}

REQUIRED_CHECKPOINTS = {
    name for name in CHECKPOINTS if name != "flatnet_lite.pth"
}

DEFAULT_CHECKPOINT = "leadmm5_prepost_drunet8m.pth"
MIN_BYTES = 1_000_000


def download_from_hf(repo_file: str, dest: Path) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        print(f"[skip] {dest} already present")
        return True

    print(f"[download] {HF_REPO_ID}/{repo_file}")
    print(f"[dest]     {dest}")

    try:
        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            filename=repo_file,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False

    shutil.copyfile(cached, dest)

    size = dest.stat().st_size
    if size < MIN_BYTES:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"download failed or file too small ({size} bytes): {dest.name}")

    print(f"[done] wrote {size} bytes")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model checkpoints.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint filename to download (see CHECKPOINTS keys).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every required checkpoint listed in CHECKPOINTS.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        names = list(REQUIRED_CHECKPOINTS)
        if "flatnet_lite.pth" in CHECKPOINTS:
            names.append("flatnet_lite.pth")
    else:
        if args.name not in CHECKPOINTS:
            known = ", ".join(CHECKPOINTS)
            raise SystemExit(f"unknown checkpoint {args.name!r}; known: {known}")
        names = [args.name]

    for name in names:
        repo_file = CHECKPOINTS[name]
        ok = download_from_hf(repo_file, output_dir / name)
        if ok:
            continue
        if name == "flatnet_lite.pth" and args.all:
            print(f"[skip] {name} not on hugging face")
            continue
        raise SystemExit(f"checkpoint not found on hugging face: {repo_file}")


if __name__ == "__main__":
    main()
