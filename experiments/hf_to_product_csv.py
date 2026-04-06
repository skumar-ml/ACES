#!/usr/bin/env python3
"""
Standalone script to extract a baseline product catalog from a HuggingFace ACES dataset (e.g. ACE-BB).

The HF dataset has many permutations of the same products (different prices, ratings, positions).
This script extracts one row per unique product per query, so you can permute the baseline
yourself for future experiments.

Writes under ``local_datasets/csvs/``, mirroring how screenshots use a top-level ``screenshots/`` tree::

    local_datasets/csvs/products_<dataset>_<subset>/<sanitized_query>/baseline.csv

Usage:
    python experiments/hf_to_product_csv.py --dataset My-Custom-AI/ACE-BB --subset choice_behavior
    python experiments/hf_to_product_csv.py --dataset My-Custom-AI/ACE-BB --subset market_share --limit 100
    python experiments/hf_to_product_csv.py ... --output-dir local_datasets/csvs/my_catalog
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# Column renames for ACES compatibility (HF name -> experiment CSV name)
COLUMN_RENAMES = {
    "product": "title",
    "rating_count": "number_of_reviews",
    "asin": "sku",
    "id": "sku",  # use id as sku if sku not present
}

# Output columns in order
OUTPUT_COLUMNS = ["query", "sku", "title", "url", "image_url", "price", "rating", "number_of_reviews"]

# Columns to exclude from output (HF-specific, not experiment data)
EXCLUDE_COLUMNS = {"screenshot", "image"}  # screenshot is Image type, not serializable to CSV

BASELINE_FILENAME = "baseline.csv"
CSVS_SUBDIR = "csvs"


def sanitize_path_segment(s: str) -> str:
    """Safe directory name for a query (aligned with ecommerce-attack test.py)."""
    out = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))
    return out[:80] or "query"


def expand_hf_row(row: dict) -> list[dict]:
    """
    Expand a single HuggingFace dataset row (one experiment) into a list of product row dicts.

    HF format: scalar columns (query, experiment_label, experiment_number) + sequence columns (lists).
    Scalar columns like brand, color are repeated for each product.
    """
    query = row["query"]

    sequence_columns = {}
    scalar_columns = {}
    for k, v in row.items():
        if k in ("query", "experiment_label", "experiment_number") or k in EXCLUDE_COLUMNS:
            continue
        if isinstance(v, list):
            sequence_columns[k] = v
        else:
            scalar_columns[k] = v

    if not sequence_columns:
        return []

    num_products = len(next(iter(sequence_columns.values())))
    rows = []

    for i in range(num_products):
        r = {"query": query}
        for col_name, col_values in sequence_columns.items():
            value = col_values[i] if i < len(col_values) else None
            out_col = COLUMN_RENAMES.get(col_name, col_name)
            r[out_col] = value
        for col_name, value in scalar_columns.items():
            out_col = COLUMN_RENAMES.get(col_name, col_name)
            r[out_col] = value
        rows.append(r)

    return rows


def deduplicate_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per unique (query, product), using first occurrence."""
    if "sku" in df.columns:
        subset = ["query", "sku"]
    elif "title" in df.columns:
        subset = ["query", "title"]
    elif "id" in df.columns:
        subset = ["query", "id"]
    else:
        return df
    return df.drop_duplicates(subset=subset, keep="first").copy()


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order output columns, filling missing with defaults."""
    result = pd.DataFrame()
    for col in OUTPUT_COLUMNS:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = "" if col in ("query", "sku", "title", "url", "image_url") else 0
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract baseline product catalog from HuggingFace ACES dataset (one row per unique product per query).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="My-Custom-AI/ACE-BB",
        help="HuggingFace dataset name (default: My-Custom-AI/ACE-BB)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/config (e.g. choice_behavior, market_share for ACE-BB)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="data",
        help="Dataset split to load (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Catalog root: per-query folders with baseline.csv live directly under this path "
        "(default: local_datasets/csvs/products_<dataset>_<subset>/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of experiments (for debugging)",
    )
    args = parser.parse_args()

    dataset_short = args.dataset.split("/")[-1].lower().replace("-", "_")
    subset_suffix = f"_{args.subset}" if args.subset else ""
    catalog_name = f"products_{dataset_short}{subset_suffix}"
    default_output_dir = Path("local_datasets") / CSVS_SUBDIR / catalog_name
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir

    # Load dataset
    if args.subset:
        dataset = load_dataset(args.dataset, args.subset, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    # Expand each row to product rows
    all_rows = []
    experiment_count = 0
    for i, row in enumerate(dataset):
        if args.limit is not None and i >= args.limit:
            break
        all_rows.extend(expand_hf_row(row))
        experiment_count += 1

    if not all_rows:
        print("No data extracted. Check dataset structure.")
        return 1

    df = pd.DataFrame(all_rows)
    # Exclude specific products (by query) from output
    df = df[df["query"].str.lower().str.replace(" ", "_", regex=False) != "usb_cable"]
    df = deduplicate_to_baseline(df)
    df = select_output_columns(df)

    # Sort for consistency
    sort_cols = ["query"]
    if "title" in df.columns:
        sort_cols.append("title")
    elif "sku" in df.columns:
        sort_cols.append("sku")
    df.sort_values(by=sort_cols, inplace=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_written = []
    for query_value, group in df.groupby("query", sort=False):
        query_dir = sanitize_path_segment(query_value)
        query_path = output_dir / query_dir
        query_path.mkdir(parents=True, exist_ok=True)
        out_file = query_path / BASELINE_FILENAME
        group.to_csv(out_file, index=False)
        queries_written.append(str(query_path / BASELINE_FILENAME))

    print(
        f"Wrote {len(queries_written)} query baseline(s) under {output_dir.resolve()} "
        f"({len(df)} unique products, {experiment_count} experiments processed)"
    )

    return 0


if __name__ == "__main__":
    exit(main())
