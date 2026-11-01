"""Add administrative, soil and bivariate-LISA layers to the causal GeoPackage."""

from __future__ import annotations

import sqlite3
import shutil
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.relative_import import add_project_root

add_project_root()
from Utils.paths import GPKG_PATH, GRID_PATH, PLOTS_SPATIAL_AUTOCORRELATION_PATH


SOURCE_GPKG = GPKG_PATH / "resources" / "pedologgia.gpkg"
OUTPUT_GPKG = REPO_ROOT / "3_Implementation" / "4_Results" / "map_contents" / (
    "h310_grid_final_datacube_causal.gpkg"
)
GRID_CANDIDATES = (
    OUTPUT_GPKG,
    GRID_PATH / "h310_grid_final_datacube_causal.gpkg",
    GRID_PATH / "h310_grid_final_datacube.gpkg",
)
STATIC_LAYERS = (
    "provcm01012026_wgs84",
    "com01012026_wgs84",
    "fertilita_chimica_dellorizzonte_superficiale",
    "capacita_duso_e_fertilita_dei_suoli",
)
LISA_FILES = (
    "report_bivariate_lisa_census_pop_vs_inflow_total.gpkg",
    "report_bivariate_lisa_idw_rain_mm_avg_monthly_vs_idw_wind_speed_max_95p.gpkg",
    "report_bivariate_lisa_idw_temp_thermal_range_vs_dtmidcnt_mean.gpkg",
)


def _layers(path: Path) -> list[str]:
    try:
        if hasattr(gpd, "list_layers"):
            return list(gpd.list_layers(path)["name"])
        import fiona

        return list(fiona.listlayers(path))
    except (ImportError, OSError, ValueError):
        return []


def _read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    if layer not in _layers(path):
        raise ValueError(f"Layer {layer!r} was not found in {path}")
    return gpd.read_file(path, layer=layer)


def _find_grid_layer(path: Path) -> str:
    for layer_name in _layers(path):
        try:
            if "causal_socioeconomic_index" in _read_layer(path, layer_name).columns:
                return layer_name
        except (OSError, ValueError, ImportError):
            continue
    raise KeyError(f"No layer containing causal_socioeconomic_index found in {path}")


def _remove_layer(path: Path, layer: str) -> None:
    """Remove one existing GeoPackage layer while retaining other layers."""
    if not path.exists() or layer not in _layers(path):
        return
    with sqlite3.connect(path) as connection:
        for table in ("gpkg_geometry_columns", "gpkg_contents"):
            connection.execute(f"DELETE FROM {table} WHERE table_name = ?", (layer,))
        connection.execute(f'DROP TABLE IF EXISTS "{layer.replace(chr(34), chr(34) * 2)}"')
        connection.commit()


def _write_layer(frame: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    _remove_layer(path, layer)
    frame.to_file(path, layer=layer, driver="GPKG")


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("No causal H3 GeoPackage was found in the configured grid paths.")


def _municipality_key(frame: gpd.GeoDataFrame) -> str:
    candidates = ("PRO_COM", "COD_COM", "COM", "com", "comune", "index")
    key = next((candidate for candidate in candidates if candidate in frame.columns), None)
    if key is None:
        raise KeyError(f"Could not identify a municipality key in columns: {list(frame.columns)}")
    return key


def _aggregate_socioeconomic_index(
    municipalities: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    index_column = "causal_socioeconomic_index"
    if index_column not in grid.columns:
        raise KeyError(f"Grid layer does not contain {index_column!r}.")
    if municipalities.crs is None or grid.crs is None:
        raise ValueError("Both municipality and grid layers must define a CRS.")

    municipality_key = _municipality_key(municipalities)
    grid_for_join = grid[[index_column, "geometry"]].copy()
    grid_for_join[index_column] = pd.to_numeric(grid_for_join[index_column], errors="coerce")
    grid_for_join = grid_for_join.dropna(subset=[index_column])
    if grid_for_join.empty:
        municipalities[index_column] = pd.Series(index=municipalities.index, dtype="float64")
        return municipalities

    grid_for_join = grid_for_join.to_crs(municipalities.crs)
    points = grid_for_join.copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(
        points,
        municipalities[[municipality_key, "geometry"]],
        how="left",
        predicate="within",
    )
    averages = joined.groupby(municipality_key, dropna=True)[index_column].mean()
    updated = municipalities.copy()
    updated[index_column] = updated[municipality_key].map(averages)
    return updated


def _report_layer_name(path: Path) -> str:
    layers = _layers(path)
    if not layers:
        raise ValueError(f"No readable layer found in report GeoPackage {path}")
    return layers[0]


def main() -> None:
    if not SOURCE_GPKG.exists():
        raise FileNotFoundError(f"Static GeoPackage not found: {SOURCE_GPKG}")
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    grid_path = _first_existing(GRID_CANDIDATES)
    if grid_path != OUTPUT_GPKG and not OUTPUT_GPKG.exists():
        shutil.copy2(grid_path, OUTPUT_GPKG)
    grid_layer = _find_grid_layer(grid_path)
    grid = _read_layer(grid_path, grid_layer)

    for layer_name in STATIC_LAYERS:
        static_layer = _read_layer(SOURCE_GPKG, layer_name)
        if layer_name == "com01012026_wgs84":
            static_layer = _aggregate_socioeconomic_index(static_layer, grid)
        _write_layer(static_layer, OUTPUT_GPKG, layer_name)

    for report_name in LISA_FILES:
        report_path = PLOTS_SPATIAL_AUTOCORRELATION_PATH / "report" / report_name
        if not report_path.exists():
            raise FileNotFoundError(f"Bivariate LISA report not found: {report_path}")
        report_layer = _report_layer_name(report_path)
        report = _read_layer(report_path, report_layer)
        _write_layer(report, OUTPUT_GPKG, Path(report_name).stem[:63])

    print(f"Updated GeoPackage: {OUTPUT_GPKG}")
    print(f"Added layers: {', '.join((*STATIC_LAYERS, *(Path(name).stem[:63] for name in LISA_FILES)))}")


if __name__ == "__main__":
    main()
