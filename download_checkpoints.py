import argparse
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


CHECKPOINTS = {
    "leadmm5_prepost_drunet8m.pth": {
        "url": (
            "https://huggingface.co/tolyaho/lensless-checkpoints/resolve/main/"
            "leadmm5_prepost_drunet8m.pth"
        ),
        "min_bytes": 1_000_000,
    },
    "leadmm5_post_drunet8m.pth": {
        "url": (
            "https://huggingface.co/tolyaho/lensless-checkpoints/resolve/main/"
            "leadmm5_post_drunet8m.pth"
        ),
        "min_bytes": 1_000_000,
    },
    "leadmm5_pre_drunet8m.pth": {
        "url": (
            "https://huggingface.co/tolyaho/lensless-checkpoints/resolve/main/"
            "leadmm5_pre_drunet8m.pth"
        ),
        "min_bytes": 1_000_000,
    },
    "leadmm20.pth": {
        "url": (
            "https://huggingface.co/tolyaho/lensless-checkpoints/resolve/main/"
            "leadmm20.pth"
        ),
        "min_bytes": 1_000_000,
    },
}

DEFAULT_CHECKPOINT = "leadmm5_prepost_drunet8m.pth"


def _is_drive_url(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def _download_http(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        dest.write_bytes(response.read())


def _download_curl(url: str, dest: Path) -> None:
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5", "-o", str(dest), url],
        check=True,
    )


def _download_drive(url: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("gdown is required for google drive urls: pip install gdown") from exc
    gdown.download(url, str(dest), fuzzy=True)


def download(url: str, dest: Path, min_bytes: int = 1) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[skip] {dest} already present")
        return

    if "<" in url or "YOUR_" in url:
        raise SystemExit(
            f"placeholder url for {dest.name} — edit CHECKPOINTS in "
            "download_checkpoints.py or pass --url / CHECKPOINT_URL"
        )

    print(f"[download] {url}")
    print(f"[dest]     {dest}")

    if _is_drive_url(url):
        _download_drive(url, dest)
    else:
        try:
            _download_http(url, dest)
        except (urllib.error.HTTPError, urllib.error.URLError):
            _download_curl(url, dest)

    size = dest.stat().st_size
    if size < min_bytes:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"download failed or file too small ({size} bytes): {dest.name}")

    print(f"[done] wrote {size} bytes")


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
        help="Download every checkpoint listed in CHECKPOINTS.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Override download url (only with a single --name).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        names = list(CHECKPOINTS.keys())
    else:
        if args.name not in CHECKPOINTS and args.url is None:
            known = ", ".join(CHECKPOINTS)
            raise SystemExit(f"unknown checkpoint {args.name!r}; known: {known}")
        names = [args.name]

    for name in names:
        spec = CHECKPOINTS.get(name, {})
        url = args.url if len(names) == 1 and args.url else os.environ.get("CHECKPOINT_URL")
        if url is None:
            url = spec.get("url")
        if url is None:
            raise SystemExit(f"no url configured for {name}")

        min_bytes = spec.get("min_bytes", 1)
        download(url, output_dir / name, min_bytes=min_bytes)


if __name__ == "__main__":
    main()
