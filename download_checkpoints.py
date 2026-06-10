import argparse


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints.",
    )
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: download checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
