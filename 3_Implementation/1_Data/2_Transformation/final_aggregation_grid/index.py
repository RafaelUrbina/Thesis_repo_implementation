import geopandas as gpd
import pandas as pd

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]
sys.path.insert(0, str(project_root))

from Utils.paths import (GRID_PATH)

gpkg_path = GRID_PATH / 'h310_grid_final_datacube.gpkg'


def list_gpkg_layers(path):
    """List layers in a GeoPackage trying multiple backends."""
    if hasattr(gpd, "list_layers"):
        try:
            return gpd.list_layers(path)["name"].tolist()
        except Exception:
            pass

    try:
        import pyogrio
        return [row[0] for row in pyogrio.list_layers(path)]
    except Exception:
        pass

    try:
        import fiona
        return fiona.listlayers(path)
    except Exception:
        pass

    import sqlite3
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("SELECT table_name FROM gpkg_contents;")
        return [row[0] for row in cur.fetchall()]
    finally:
        con.close()


def classify_dtype(dtype):
    """Human-readable dtype classification."""
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "string/object"
    if pd.api.types.is_categorical_dtype(dtype):
        return "categorical"
    return "other"


def inspect_layer(path, layer_name):
    """Read a layer and print a full column/dtype report."""
    print("\n" + "=" * 100)
    print(f"LAYER: {layer_name}")
    print("=" * 100)

    try:
        gdf = gpd.read_file(path, layer=layer_name)
    except Exception as e:
        print(f"  Could not read layer: {e}")
        return None

    n_rows, n_cols = len(gdf), len(gdf.columns)
    geom_col = gdf.geometry.name if gdf.geometry is not None else None

    print(f"Rows: {n_rows}")
    print(f"Columns: {n_cols}")
    print(f"Geometry column: {geom_col}")
    if gdf.crs is not None:
        print(f"CRS: {gdf.crs}")
    else:
        print("CRS: (none — non-spatial table?)")

    print("\n--- Column details ---")
    rows = []
    for col in gdf.columns:
        s = gdf[col]
        dtype = s.dtype
        kind = classify_dtype(dtype)
        n_missing = int(s.isna().sum())
        missing_pct = round(n_missing / n_rows * 100, 2) if n_rows else 0.0

        # Uniqueness — cheap, and useful for categorizing
        try:
            n_unique = int(s.nunique(dropna=True))
        except TypeError:
            n_unique = None  # unhashable types

        # A tiny sample (first non-null) to eyeball values
        try:
            sample = s.dropna().iloc[0] if s.notna().any() else None
        except Exception:
            sample = None

        rows.append({
            "column": col,
            "dtype": str(dtype),
            "kind": kind,
            "missing": n_missing,
            "missing_pct": missing_pct,
            "unique": n_unique,
            "sample": str(sample)[:40] if sample is not None else "",
        })

    report = pd.DataFrame(rows)
    # Make sure we can print wide tables
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 45):
        print(report.to_string(index=False))

    # Grouped summary
    print("\n--- Column counts by kind ---")
    print(report["kind"].value_counts().to_string())

    return gdf


def main():
    try:
        layers = list_gpkg_layers(gpkg_path)
    except Exception as e:
        print(f"Failed to list layers: {e}")
        return

    if not layers:
        print(f"No layers found in: {gpkg_path}")
        return

    print(f"Found {len(layers)} layer(s):")
    for i, name in enumerate(layers, 1):
        print(f"  {i}. {name}")

    # Inspect every layer
    for layer_name in layers:
        inspect_layer(gpkg_path, layer_name)


if __name__ == "__main__":
    main()