#!/usr/bin/env python3
"""
Generate screenshots from a product CSV using the ACES sandbox.

Loads the CSV, adds experiment_label and experiment_number, then reuses ACES
screenshot collection to capture one screenshot per unique (query, experiment_label, experiment_number).
When the CSV lives under ``local_datasets/csvs/<catalog>/<query>/<variant>.csv``, screenshots are saved under
``local_datasets/screenshots/<catalog>/<query>/<variant>/`` as
``{query}_{experiment_label}_{experiment_number}.png``. Otherwise PNGs use
``{csv_parent}/screenshots/{dataset_name}/{query}/{experiment_label}/``. A new CSV is written with a ``screenshot_path`` column.

Resolves the input CSV from a catalog root (``local_datasets/csvs/<catalog>/``), query, and label::

    {csv_base}/{sanitized_query}/{experiment_label}.csv

Usage:
    python experiments/product_csv_to_screenshots.py \\
        --csv-base experiments/local_datasets/csvs/products_ace_bb_market_share \\
        --query toilet_paper --experiment-label demo
    python experiments/product_csv_to_screenshots.py \\
        --csv-base experiments/local_datasets/csvs/products_ace_bb_market_share \\
        --query toilet_paper --experiment-label demo --num-workers 8 --output-csv out.csv
"""
import sys
from pathlib import Path

# Ensure ACES root is on path so experiments/ and sandbox/ import correctly when run from any directory
_aces_root = Path(__file__).resolve().parent.parent
if str(_aces_root) not in sys.path:
    sys.path.insert(0, str(_aces_root))

import argparse
from typing import Optional

import pandas as pd
from rich import print as _print

from experiments.data_loader import experiments_iter
from experiments.utils.dataset_ops import get_dataset_name, get_experiment_screenshot_png_path
from experiments.utils.screenshot_collector import collect_screenshots_parallel


EXPERIMENT_LABEL = "baseline"
EXPERIMENT_NUMBER = 0


def _sanitize_experiment_label_filename(s: str) -> str:
    """Match test.py / hf_to_product_csv path segments for {experiment_label}.csv."""
    out = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))
    return out[:80] or "experiment"


def resolve_csv_under_query_dir(query_dir: Path, experiment_label: str) -> Optional[Path]:
    """Resolve ``{experiment_label}.csv`` under ``query_dir`` (sanitized stem, then raw)."""
    query_dir = query_dir.expanduser().resolve()
    if not query_dir.is_dir():
        _print(f"[red]Error: not a directory: {query_dir}")
        return None
    label_stem = _sanitize_experiment_label_filename(experiment_label)
    for name in (f"{label_stem}.csv", f"{experiment_label}.csv"):
        candidate = (query_dir / name).resolve()
        if candidate.is_file():
            return candidate
    _print(
        f"[red]Error: no CSV in {query_dir} named {label_stem}.csv or {experiment_label}.csv"
    )
    return None


def resolve_csv_from_base(csv_base: Path, query: str, experiment_label: str) -> Optional[Path]:
    """
    Resolve ``{csv_base}/{sanitized_query}/{experiment_label}.csv`` (same filename rules as
    ``resolve_csv_under_query_dir``).
    """
    csv_base = csv_base.expanduser().resolve()
    if not csv_base.is_dir():
        _print(f"[red]Error: csv base is not a directory: {csv_base}")
        return None
    query_dir = csv_base / _sanitize_experiment_label_filename(query)
    return resolve_csv_under_query_dir(query_dir, experiment_label)


def get_screenshot_path(
    csv_path: Path,
    query: str,
    experiment_label: str,
    experiment_number: int,
) -> Path:
    """Path where screenshot is saved for this experiment."""
    return get_experiment_screenshot_png_path(
        csv_path, query, experiment_label, experiment_number
    )


def prepare_experiment_df(
    df: pd.DataFrame,
    *,
    default_experiment_label: str = EXPERIMENT_LABEL,
    default_experiment_number: int = EXPERIMENT_NUMBER,
) -> pd.DataFrame:
    """Add experiment columns and ensure sandbox-compatible columns.

    If the CSV already includes ``experiment_label`` / ``experiment_number`` (e.g. from a
    variant workflow), those values are preserved; missing cells are filled from defaults.
    """
    df = df.copy()
    if "experiment_label" not in df.columns:
        df["experiment_label"] = default_experiment_label
    else:
        df["experiment_label"] = df["experiment_label"].fillna(default_experiment_label)

    if "experiment_number" not in df.columns:
        df["experiment_number"] = default_experiment_number
    else:
        df["experiment_number"] = (
            pd.to_numeric(df["experiment_number"], errors="coerce")
            .fillna(default_experiment_number)
            .astype(int)
        )

    if "assigned_position" not in df.columns:
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
        "--csv-base",
        type=str,
        default="local_datasets/csvs/products_ace_bb_market_share",
        help="Catalog root (e.g. local_datasets/csvs/products_ace_bb_market_share). "
        "Resolves {csv_base}/{query}/{experiment_label}.csv.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query subdirectory under csv-base (sanitized) and filter rows after load.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate screenshots even when a valid PNG already exists.",
    )
    parser.add_argument(
        "--experiment-label",
        type=str,
        default=EXPERIMENT_LABEL,
        help="Selects {label}.csv under the query directory; fills missing experiment_label in the dataframe.",
    )
    parser.add_argument(
        "--experiment-number",
        type=int,
        default=EXPERIMENT_NUMBER,
        help="Default experiment_number if the CSV omits that column (default: 0).",
    )
    args = parser.parse_args()

    csv_path = resolve_csv_from_base(Path(args.csv_base), args.query, args.experiment_label)
    if csv_path is None:
        return 1
    if not csv_path.is_file():
        _print(f"[red]Error: file not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if "query" not in df.columns:
        _print("[red]Error: CSV must have a 'query' column")
        return 1

    df = df[df["query"] == args.query].copy()
    if df.empty:
        _print(f"[red]Error: no rows for query={args.query!r}")
        return 1
    _print(f"[dim]Filtered to query={args.query!r} ({len(df)} rows)")

    df = prepare_experiment_df(
        df,
        default_experiment_label=args.experiment_label,
        default_experiment_number=args.experiment_number,
    )
    dataset_name = get_dataset_name(str(csv_path))

    experiments = list(experiments_iter(df, dataset_name))
    _print(f"[blue]Loaded {len(experiments)} experiments from {len(df)} product rows")

    collect_screenshots_parallel(
        experiments,
        str(csv_path),
        num_workers=args.num_workers,
        verbose=True,
        force_regenerate=args.force,
    )

    exp_label = str(df["experiment_label"].iloc[0])
    exp_num = int(df["experiment_number"].iloc[0])

    # Add screenshot_path to each row and write output CSV
    df["screenshot_path"] = df["query"].apply(
        lambda q: str(get_screenshot_path(csv_path, q, exp_label, exp_num))
    )
    output_path = Path(args.output_csv) if args.output_csv else csv_path.parent / f"{csv_path.stem}_with_screenshots.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    _print(f"[green]Wrote {len(df)} rows with screenshot paths to {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
