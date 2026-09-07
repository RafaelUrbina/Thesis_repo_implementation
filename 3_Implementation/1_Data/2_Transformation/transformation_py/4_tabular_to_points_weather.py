from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve()
for candidate in [REPO_ROOT, *REPO_ROOT.parents]:
    if (candidate / "Utils" / "paths.py").exists():
        REPO_ROOT = candidate
        break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MIDPOINTS_PATH

WEATHER_LAYER_MAP: Dict[str, Tuple[str, str]] = {
    "sir_temperature_toscana_datasets": ("termometri_stations_firenze", "temperature"),
    "sir_wind_toscana_datasets": ("anemometri_stations_firenze", "wind"),
    "sir_rain_toscana_datasets": ("cf_pluviometri", "rain"),
    "sir_idrometry_toscana_datasets": ("idrometri_stations_firenze", "idrometry"),
}

METRIC_RENAMES: Dict[str, Dict[str, str]] = {
    "temperature": {
        "max_c": "temp_max_c",
        "min_c": "temp_min_c",
    },
    "wind": {
        "vel_med_m_s": "wind_speed_mean_m_s",
        "vel_max_m_s": "wind_speed_max_m_s",
    },
    "rain": {
        "precipitazione_mm": "rain_mm",
    },
    "idrometry": {
        "livello_m": "water_level_m",
    },
}


def normalize_station_id(value: object) -> str:
    return str(value).strip().strip('"').upper()


def parse_numeric(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).strip().strip('"')
    if not text or text.lower() in {"nan", "none", "@", ""}:
        return None

    cleaned = re.sub(r"\s+", "", text)
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    cleaned = re.sub(r"[^0-9\-\.eE]", "", cleaned)
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(value: str) -> pd.Timestamp:
    return pd.to_datetime(value.strip().strip('"'), format="%d/%m/%Y", errors="coerce")


def extract_station_id(csv_path: Path, lines: List[str]) -> str | None:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Codice;"):
            parts = [part.strip().strip('"') for part in stripped.split(";") if part.strip()]
            if len(parts) >= 2:
                return normalize_station_id(parts[1])
            return None

    stem = csv_path.stem
    if "_" in stem:
        return normalize_station_id(stem.split("_", 1)[0])
    return normalize_station_id(stem)


def parse_monthly_weather_rows(csv_path: Path) -> List[Dict[str, object]]:
    lines = csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    station_id = extract_station_id(csv_path, lines)
    if station_id is None:
        raise ValueError(f"No station identifier found in {csv_path.name}")

    header_idx = None
    for idx, line in enumerate(lines):
        if "gg/mm/aaaa" in line:
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError(f"No data header found in {csv_path.name}")

    header_parts = [part.strip().strip('"') for part in lines[header_idx].split(";") if part.strip()]
    metric_names = header_parts[1:]

    rows: List[Dict[str, object]] = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith(";;"):
            continue

        parts = [part.strip().strip('"') for part in stripped.split(";")]
        if len(parts) < 2:
            continue

        date_value = parse_date(parts[0])
        if pd.isna(date_value):
            continue

        for idx, metric_name in enumerate(metric_names):
            if idx + 1 >= len(parts):
                continue
            if metric_name.lower().startswith("tipo") or metric_name.lower().startswith("dir"):
                continue

            numeric_value = parse_numeric(parts[idx + 1])
            if numeric_value is None:
                continue

            rows.append(
                {
                    "station_id": station_id,
                    "date": date_value.to_pydatetime(),
                    "metric": metric_name,
                    "value": numeric_value,
                    "source_file": csv_path.name,
                }
            )

    return rows


def build_weather_geodataframe(folder: Path, layer_name: str, layer_label: str) -> gpd.GeoDataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    records: List[Dict[str, object]] = []
    for csv_file in csv_files:
        rows = parse_monthly_weather_rows(csv_file)
        records.extend(rows)

    if not records:
        raise ValueError(f"No valid weather rows found in {folder}")

    weather_df = pd.DataFrame(records)
    weather_df["station_id"] = weather_df["station_id"].astype(str).str.strip().str.upper()
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    stations = gpd.read_file(GPKG_PATH / "merged_pedologgia_static.gpkg", layer=layer_name)

    station_id_col = None
    for col_candidate in ["id_stazion", "id_stazio", "station_id"]:
        if col_candidate in stations.columns:
            station_id_col = col_candidate
            break

    if station_id_col is None:
        raise KeyError(f"Could not find a station ID column in layer '{layer_name}'. "
                       f"Checked for ['id_stazion', 'id_stazio', 'station_id']. "
                       f"Available columns: {list(stations.columns)}")

    stations = stations[[station_id_col, "comune", "quota", "zona", "geometry"]].copy()
    stations.rename(columns={station_id_col: "station_id"}, inplace=True)
    stations["station_id"] = stations["station_id"].astype(str).str.strip().str.upper()

    merged = weather_df.merge(stations, on="station_id", how="left")
    merged = merged[merged["geometry"].notna()].copy()

    wide = (
        merged.pivot_table(
            index=["station_id", "date", "comune", "quota", "zona", "geometry"],
            columns="metric",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    wide = wide.rename(columns=lambda col: re.sub(r"[^0-9A-Za-z]+", "_", str(col)).strip("_").lower())
    rename_map = METRIC_RENAMES.get(layer_label, {})
    wide = wide.rename(columns={col: rename_map.get(col, col) for col in wide.columns})
    wide["dataset"] = layer_label
    gdf = gpd.GeoDataFrame(wide, geometry="geometry", crs=stations.crs)
    gdf = gdf.sort_values(["station_id", "date"]).reset_index(drop=True)
    return gdf


def export_weather_to_gpkg(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    layer_names = list(WEATHER_LAYER_MAP.items())
    for index, (folder_name, (layer_name, layer_label)) in enumerate(layer_names):
        folder_path = MIDPOINTS_PATH / folder_name
        if not folder_path.exists():
            print(f"Skipping missing folder: {folder_path}")
            continue

        print(f"Joining {folder_name} to layer {layer_name}")
        gdf = build_weather_geodataframe(folder_path, layer_name, layer_label)
        mode = "w" if index == 0 else "a"
        gdf.to_file(output_path, layer=layer_label, driver="GPKG", mode=mode)
        print(f"Wrote {len(gdf)} rows to layer {layer_label}")


if __name__ == "__main__":
    export_weather_to_gpkg(MIDPOINTS_PATH / "weather_aggregates" / "weather_station_monthly.gpkg")
