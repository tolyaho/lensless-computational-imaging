from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

from huggingface_hub import hf_hub_download


def main():
    parser = ArgumentParser()
    parser.add_argument("--repo-id", default="bezzam/DigiCam-Mirflickr-MultiMask-10K")
    parser.add_argument("--out-dir", default="data/digicam_masks")
    parser.add_argument("--num-masks", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_masks):
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=f"masks/mask_{i}.npy",
        )
        copyfile(path, out_dir / f"mask_{i}.npy")

    print(f"saved {len(list(out_dir.glob('mask_*.npy')))} masks to {out_dir}")


if __name__ == "__main__":
    main()