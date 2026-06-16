from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "reports" / "tables"

FIELDNAMES = [
    "rank",
    "method",
    "hydra_model",
    "checkpoint",
    "params",
    "psnr",
    "ssim",
    "mse",
    "lpips",
    "split",
    "n_images",
    "notes",
]

METHOD_SPECS = [
    {
        "method": "ADMM-100 + bg sub",
        "hydra_model": "ADMM",
        "checkpoint": "",
        "params": 0,
        "metrics_path": "outputs/final_admm100_bgsub/metrics.json",
        "optional": False,
        "notes": "",
    },
    {
        "method": "LeADMM-20",
        "hydra_model": "UnrolledADMM",
        "checkpoint": "checkpoints/leadmm20.pth",
        "params": 40,
        "metrics_path": "outputs/final_leadmm20/metrics.json",
        "optional": False,
        "notes": "",
    },
    {
        "method": "LeADMM-5 + DRUNet pre",
        "hydra_model": "modular_leadmm5_pre_drunet8m",
        "checkpoint": "checkpoints/leadmm5_pre_drunet8m.pth",
        "params": 8_337_293,
        "metrics_path": "outputs/final_leadmm5_drunet8m_pre/metrics.json",
        "optional": False,
        "notes": "",
    },
    {
        "method": "LeADMM-5 + DRUNet post",
        "hydra_model": "modular_leadmm5_post_drunet8m",
        "checkpoint": "checkpoints/leadmm5_post_drunet8m.pth",
        "params": 8_337_293,
        "metrics_path": "outputs/final_leadmm5_drunet8m_post/metrics.json",
        "optional": False,
        "notes": "",
    },
    {
        "method": "LeADMM-5 + DRUNet pre+post",
        "hydra_model": "modular_leadmm5_prepost_drunet8m",
        "checkpoint": "checkpoints/leadmm5_prepost_drunet8m.pth",
        "params": 8_512_016,
        "metrics_path": "outputs/final_leadmm5_drunet8m_prepost/metrics.json",
        "optional": False,
        "notes": "",
    },
    {
        "method": "FlatNet-lite",
        "hydra_model": "flatnet_lite",
        "checkpoint": "checkpoints/flatnet_lite.pth",
        "params": 2_725_132,
        "metrics_path": "outputs/final_flatnet_lite/metrics.json",
        "optional": True,
        "notes": "bonus",
    },
]


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "psnr": float(data["PSNR"]),
        "ssim": float(data["SSIM"]),
        "mse": float(data["MSE"]),
        "lpips": float(data["LPIPS"]),
        "n_images": int(data.get("num_images", 1500)),
    }


def make_row(spec: dict, metrics: dict) -> dict:
    psnr = metrics["psnr"]
    return {
        "rank": 0,
        "method": spec["method"],
        "hydra_model": spec["hydra_model"],
        "checkpoint": spec["checkpoint"],
        "params": spec["params"],
        "psnr": round(psnr, 4),
        "ssim": round(metrics["ssim"], 4),
        "mse": round(metrics["mse"], 4),
        "lpips": round(metrics["lpips"], 4),
        "split": "test",
        "n_images": metrics["n_images"],
        "notes": spec.get("notes", ""),
    }


def resolve_row(spec: dict) -> dict | None:
    path = REPO_ROOT / spec["metrics_path"]

    if not path.is_file():
        if spec.get("optional"):
            print(f"[skip] optional method, no metrics: {spec['method']}")
            return None
        raise FileNotFoundError(f"missing metrics: {path}")

    return make_row(spec, load_metrics(path))


def assign_ranks(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["psnr"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    headers = ["rank", "method", "psnr", "ssim", "mse", "lpips", "split"]
    table_rows = [
        [
            str(row["rank"]),
            row["method"],
            f"{row['psnr']:.2f}",
            f"{row['ssim']:.3f}",
            f"{row['mse']:.3f}",
            f"{row['lpips']:.3f}",
            row["split"],
        ]
        for row in rows
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print()
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in table_rows:
        print(fmt(row))
    print()


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("comet", "qualitative"):
        (REPO_ROOT / "reports" / "assets" / name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dirs()

    rows = []
    for spec in METHOD_SPECS:
        row = resolve_row(spec)
        if row is not None:
            rows.append(row)

    rows = assign_ranks(rows)

    json_path = TABLES_DIR / "final_metrics.json"
    csv_path = TABLES_DIR / "final_metrics.csv"

    write_json(json_path, rows)
    write_csv(csv_path, rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print_table(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())