#!/usr/bin/env python3
"""
Generate screenshots from a product CSV (e.g. products_ace_bb_market_share.csv).

Loads the CSV, adds experiment_label and experiment_number, then reuses ACES
screenshot collection to capture one screenshot per unique (query, experiment_label, experiment_number).
Screenshots are saved under {csv_parent}/screenshots/{dataset_name}/. A new CSV is written
with all default/added columns plus a screenshot_path column referencing each screenshot.

Usage:
    python experiments/product_csv_to_screenshots.py experiments/local_datasets/products_ace_bb_market_share.csv
    python experiments/product_csv_to_screenshots.py products.csv --num-workers 8 --output-csv output.csv
"""
import sys
from pathlib import Path

# Ensure ACES root is on path so experiments/ and sandbox/ import correctly when run from any directory
_aces_root = Path(__file__).resolve().parent.parent
if str(_aces_root) not in sys.path:
    sys.path.insert(0, str(_aces_root))

import argparse
from pathlib import Path

import pandas as pd
from rich import print as _print

from experiments.data_loader import experiments_iter
from experiments.utils.dataset_ops import get_dataset_name
from experiments.utils.screenshot_collector import collect_screenshots_parallel


EXPERIMENT_LABEL = "master_experiment"
EXPERIMENT_NUMBER = 0

def get_screenshot_path(csv_path: Path, dataset_name: str, query: str) -> Path:
    """Path where screenshot is saved for this experiment."""
    dataset_dir = csv_path.parent
    screenshots_dir = dataset_dir / "screenshots" / dataset_name
    filename = f"{query}_{EXPERIMENT_LABEL}_{EXPERIMENT_NUMBER}.png"
    return screenshots_dir / query / EXPERIMENT_LABEL / filename


def prepare_experiment_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add experiment columns and ensure sandbox-compatible columns."""
    df = df.copy()
    df["experiment_label"] = EXPERIMENT_LABEL
    df["experiment_number"] = EXPERIMENT_NUMBER
    df["assigned_position"] = df.groupby("query").cumcount()

    # Sandbox template expects rating_count and id
    if "number_of_reviews" in df.columns and "rating_count" not in df.columns:
        df["rating_count"] = df["number_of_reviews"]
    if "sku" in df.columns and "id" not in df.columns:
        df["id"] = df["sku"]

    # Optional template fields (default False if missing)
    for col in ("sponsored", "overall_pick", "best_seller", "limited_time", "discounted", "low_stock"):
        if col not in df.columns:
            df[col] = False
    if "stock_quantity" not in df.columns and "low_stock" in df.columns:
        df["stock_quantity"] = 100

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate screenshots from a product CSV using ACES sandbox.",
    )
    parser.add_argument(
        "product_csv",
        type=str,
        help="Path to product CSV (query, sku, title, url, image_url, price, rating, number_of_reviews)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV with screenshot paths. Default: {input_stem}_with_screenshots.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.product_csv)
    if not csv_path.exists():
        _print(f"[red]Error: file not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if "query" not in df.columns:
        _print("[red]Error: CSV must have a 'query' column")
        return 1

    df = prepare_experiment_df(df)
    dataset_name = get_dataset_name(str(csv_path))

    experiments = list(experiments_iter(df, dataset_name))
    _print(f"[blue]Loaded {len(experiments)} experiments from {len(df)} product rows")

    collect_screenshots_parallel(
        experiments,
        str(csv_path),
        num_workers=args.num_workers,
        verbose=True,
    )

    # Add screenshot_path to each row and write output CSV
    df["screenshot_path"] = df["query"].apply(
        lambda q: str(get_screenshot_path(csv_path, dataset_name, q))
    )
    output_path = Path(args.output_csv) if args.output_csv else csv_path.parent / f"{csv_path.stem}_with_screenshots.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    _print(f"[green]Wrote {len(df)} rows with screenshot paths to {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
