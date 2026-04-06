from pathlib import Path
from typing import Optional


def get_local_datasets_root(dataset_csv_path: str | Path) -> Optional[Path]:
    """Return the ``local_datasets`` directory ancestor of ``dataset_csv``, or None."""
    p = Path(dataset_csv_path).resolve()
    parts = p.parts
    try:
        idx = parts.index("local_datasets")
    except ValueError:
        return None
    return Path(*parts[: idx + 1])


def parse_csvs_catalog_query_variant(
    dataset_csv_path: str | Path,
) -> Optional[tuple[str, str, str]]:
    """
    If ``dataset_csv_path`` is ``.../local_datasets/csvs/<catalog>/<query>/<file>.csv``
    (or legacy ``.../local_datasets/<catalog>/csvs/<query>/<file>.csv``),
    return ``(catalog, query_dir, variant_stem)`` where ``variant_stem`` is the CSV filename stem.

    Otherwise return None.
    """
    p = Path(dataset_csv_path).resolve()
    parts = p.parts
    try:
        idx = parts.index("local_datasets")
    except ValueError:
        return None
    tail = parts[idx + 1 :]
    filename: str | None = None
    catalog: str | None = None
    query_dir: str | None = None
    if len(tail) == 4 and str(tail[0]).lower() == "csvs":
        _csvs, catalog, query_dir, filename = tail
    elif len(tail) == 4 and str(tail[1]).lower() == "csvs":
        catalog, _csvs, query_dir, filename = tail
    elif len(tail) == 3:
        catalog, query_dir, filename = tail
    else:
        return None
    if not filename or not str(filename).lower().endswith(".csv"):
        return None
    variant = Path(filename).stem.replace("_dataset", "")
    return (catalog, query_dir, variant)


def get_experiment_screenshot_png_path(
    dataset_csv_path: str | Path,
    query: str,
    experiment_label: str,
    experiment_number: int,
) -> Path:
    """
    Absolute path to the PNG for one experiment.

    Under ``local_datasets/csvs/<catalog>/<query>/<variant>.csv``::

        local_datasets/screenshots/<catalog>/<query>/<variant>/{query}_{label}_{n}.png

    Otherwise (flat / non-csvs layouts)::

        {csv_parent}/screenshots/{dataset_name}/{query}/{experiment_label}/...

    where ``dataset_name`` comes from :func:`get_dataset_name`.
    """
    p = Path(dataset_csv_path).resolve()
    parsed = parse_csvs_catalog_query_variant(p)
    filename_png = f"{query}_{experiment_label}_{experiment_number}.png"
    if parsed:
        catalog, query_dir, variant = parsed
        root = get_local_datasets_root(p)
        assert root is not None
        return root / "screenshots" / catalog / query_dir / variant / filename_png
    dataset_name = get_dataset_name(str(p))
    return (
        p.parent / "screenshots" / dataset_name / query / experiment_label / filename_png
    )


def get_screenshots_dataset_root(dataset_csv_path: str | Path, dataset_name: str) -> Path:
    """
    Deprecated intermediate for callers that still expect a *directory prefix* before
    ``query/experiment_label/``. Prefer :func:`get_experiment_screenshot_png_path`.

    For csvs layouts, returns the variant directory (PNG lives directly inside).
    """
    p = Path(dataset_csv_path).resolve()
    parsed = parse_csvs_catalog_query_variant(p)
    if parsed:
        catalog, query_dir, variant = parsed
        root = get_local_datasets_root(p)
        assert root is not None
        return root / "screenshots" / catalog / query_dir / variant
    return Path(dataset_csv_path).resolve().parent / "screenshots" / dataset_name


def get_screenshots_loader_base(dataset_csv_path: str | Path) -> Path:
    """
    Base directory passed to ``ExperimentData.get_local_screenshot_path`` (includes ``dataset_name`` segment).

    Under ``local_datasets``: ``local_datasets/screenshots``. Else ``{csv_parent}/screenshots``.
    """
    root = get_local_datasets_root(dataset_csv_path)
    if root is not None:
        return root / "screenshots"
    return Path(dataset_csv_path).resolve().parent / "screenshots"


def get_dataset_name(local_dataset_path: str) -> str:
    """
    Derive ACES dataset name from a local CSV path.

    Nested layouts under ``experiments/local_datasets``:

    - ``csvs/<catalog>/<query>/<file>.csv`` (four segments) — preferred; catalog lives under
      a top-level ``csvs/`` directory (parallel to ``screenshots/``).
    - ``<catalog>/csvs/<query>/<file>.csv`` (four segments) — older layout; still supported.
    - ``<catalog>/<query>/<file>.csv`` (three segments) — legacy, still supported.

    In all nested cases returns ``{catalog}__{query}__{stem}`` so short leaf names stay unique.

    Otherwise returns the filename stem (with ``_dataset`` removed), preserving flat layouts.
    """
    p = Path(local_dataset_path).resolve()
    stem = p.stem.replace("_dataset", "")
    parts = p.parts
    try:
        idx = parts.index("local_datasets")
    except ValueError:
        return stem
    tail = parts[idx + 1 :]
    if len(tail) == 4 and str(tail[0]).lower() == "csvs":
        _csvs, catalog, query, filename = tail
        if not str(filename).lower().endswith(".csv"):
            return stem
        return f"{catalog}__{query}__{stem}"
    if len(tail) == 4 and str(tail[1]).lower() == "csvs":
        catalog, _csvs, query, filename = tail
        if not str(filename).lower().endswith(".csv"):
            return stem
        return f"{catalog}__{query}__{stem}"
    if len(tail) == 3:
        catalog, query, filename = tail
        if not str(filename).lower().endswith(".csv"):
            return stem
        return f"{catalog}__{query}__{stem}"
    return stem
