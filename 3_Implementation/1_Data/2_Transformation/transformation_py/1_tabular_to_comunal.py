"""Join tabular municipal datasets to the comunal geometry layer.

This script reads the requested CSV files under the master data folder,
filters them to the 2018-2023 window, keeps one REF_AREA and one TIME_PERIOD,
pivots the long-format dimensions into wide columns with source suffixes,
and joins the resulting municipality-year attributes to the geometry layer
`com01012026_wgs84` from `merged_pedologgia_static.gpkg`.

Output: a GeoPackage and a shapefile written to MIDPOINTS_PATH.
"""
from pathlib import Path
from typing import Tuple
import csv
import logging
import re
import sys

import geopandas as gpd
import pandas as pd

# ensure repo root is on sys.path so `from Utils.paths import ...` works when
# the script is executed directly
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MASTER_DATA_PATH, MIDPOINTS_PATH

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_csv_smart(path: Path) -> pd.DataFrame:
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8", dtype=str)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, engine="python", dtype=str, encoding="utf-8", errors="replace")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)


def ensure_column(df: pd.DataFrame, target: str, aliases: Tuple[str, ...]) -> pd.DataFrame:
    lowered = {c.strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            col = lowered[alias.lower()]
            if col != target:
                df = df.rename(columns={col: target})
            break
    return df


def filter_year_window(df: pd.DataFrame) -> pd.DataFrame:
    if "TIME_PERIOD" not in df.columns:
        return df
    df = df.copy()
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    return df[(df["TIME_PERIOD"] >= 2018) & (df["TIME_PERIOD"] <= 2023)].copy()


def sanitize_column_token(value: object) -> str:
    if pd.isna(value):
        return "NA"
    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9]+", "_", token)
    token = token.strip("_")
    return token or "VAL"


def build_short_column_mapping(columns: list[str], max_length: int = 10) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    explicit_names = {
        "REF_AREA": "mun_code",
        "TIME_PERIOD": "year",
        "PRO_COM": "pro_com",
        "PRO_COM_T": "pro_com_t",
        "COMUNE": "comune",
        "COD_REG": "cod_reg",
        "COD_PROV": "cod_prov",
        "CC_UTS": "cc_uts",
        "Shape_Leng": "shape_len",
        "Shape_Area": "shape_area",
        "geometry": "geometry",
    }

    def make_candidate(original: str) -> str:
        if original in explicit_names:
            return explicit_names[original]
        if original.startswith("Osservazione_census_"):
            return "census_pop"
        if original.startswith("OBS_VALUE_economical_princ_"):
            token = original[len("OBS_VALUE_economical_princ_"):]
            return f"epr_{sanitize_column_token(token)[:8]}"
        if original.startswith("OBS_VALUE_economical_reddito_"):
            token = original[len("OBS_VALUE_economical_reddito_"):]
            return f"erd_{sanitize_column_token(token)[:8]}"
        if original.startswith("Osservazione_economical_distrib_"):
            token = original[len("Osservazione_economical_distrib_"):]
            return f"edst_{sanitize_column_token(token)[:8]}"
        if original.startswith("OBS_VALUE_economical_indice_comp_"):
            token = original[len("OBS_VALUE_economical_indice_comp_"):]
            return f"eidx_{sanitize_column_token(token)[:8]}"
        return sanitize_column_token(original)

    for column in columns:
        if column == "geometry":
            mapping[column] = "geometry"
            continue

        base = re.sub(r"[^A-Za-z0-9_]", "_", make_candidate(column)).strip("_") or "col"
        base = base[:max_length]
        candidate = base
        idx = 1
        while candidate in used:
            candidate = (base[: max_length - 2] + f"{idx:02d}")[:max_length]
            idx += 1
        used.add(candidate)
        mapping[column] = candidate

    return mapping

def pivot_long_to_wide(
    df: pd.DataFrame,
    pivot_key: str,
    value_col: str,
    value_prefix: str,
    suffix: str,
    ref_col: str = "REF_AREA",
    time_col: str = "TIME_PERIOD",
) -> pd.DataFrame:
    if pivot_key not in df.columns or value_col not in df.columns:
        return df[[ref_col, time_col]].copy()

    out = df[[ref_col, time_col, pivot_key, value_col]].copy()
    out[pivot_key] = out[pivot_key].astype(str).str.strip()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    wide = (
        out.pivot_table(index=[ref_col, time_col], columns=pivot_key, values=value_col, aggfunc="first")
        .reset_index()
    )
    wide.columns = [
        col if col in {ref_col, time_col} else f"{value_prefix}_{suffix}_{sanitize_column_token(col)}"
        for col in wide.columns
    ]
    return wide


def prepare_census_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_column(df, "REF_AREA", ("REF_AREA", "ref_area"))
    df = ensure_column(df, "TIME_PERIOD", ("TIME_PERIOD", "time_period"))
    df = ensure_column(df, "Indicatore", ("Indicatore", "indicatore"))
    df = ensure_column(df, "Osservazione", ("Osservazione", "osservazione", "OBS_VALUE", "obs_value"))

    df = filter_year_window(df)
    df = df[["REF_AREA", "TIME_PERIOD", "Indicatore", "Osservazione"]].copy()
    df["REF_AREA"] = df["REF_AREA"].astype(str).str.strip()
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")

    # Keep one observation per municipality-year; the indicator is constant in this file.
    wide = df.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="Indicatore", values="Osservazione", aggfunc="first")
    wide = wide.reset_index()
    wide.columns = [
        col if col in {"REF_AREA", "TIME_PERIOD"} else f"Osservazione_census_{sanitize_column_token(col)}"
        for col in wide.columns
    ]
    return wide


def prepare_economical_princ_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_column(df, "REF_AREA", ("REF_AREA", "ref_area"))
    df = ensure_column(df, "TIME_PERIOD", ("TIME_PERIOD", "time_period"))
    df = ensure_column(df, "DATA_TYPE", ("DATA_TYPE", "data_type"))
    df = ensure_column(df, "OBS_VALUE", ("OBS_VALUE", "obs_value"))

    df = filter_year_window(df)
    return pivot_long_to_wide(df, pivot_key="DATA_TYPE", value_col="OBS_VALUE", value_prefix="OBS_VALUE", suffix="economical_princ")


def prepare_economical_reddito_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_column(df, "REF_AREA", ("REF_AREA", "ref_area"))
    df = ensure_column(df, "TIME_PERIOD", ("TIME_PERIOD", "time_period"))
    df = ensure_column(df, "AMOUNT_CLASS", ("AMOUNT_CLASS", "amount_class"))
    df = ensure_column(df, "OBS_VALUE", ("OBS_VALUE", "obs_value"))

    df = filter_year_window(df)
    return pivot_long_to_wide(df, pivot_key="AMOUNT_CLASS", value_col="OBS_VALUE", value_prefix="OBS_VALUE", suffix="economical_reddito")


def prepare_economical_distrib_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_column(df, "REF_AREA", ("REF_AREA", "ref_area"))
    df = ensure_column(df, "TIME_PERIOD", ("TIME_PERIOD", "time_period"))
    df = ensure_column(df, "DATA_TYPE", ("DATA_TYPE", "data_type"))
    df = ensure_column(df, "Osservazione", ("Osservazione", "osservazione", "OBS_VALUE", "obs_value"))

    df = filter_year_window(df)
    df = df[["REF_AREA", "TIME_PERIOD", "DATA_TYPE", "Osservazione"]].copy()
    df["REF_AREA"] = df["REF_AREA"].astype(str).str.strip()
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    df["DATA_TYPE"] = df["DATA_TYPE"].astype(str).str.strip()
    df["DATA_TYPE"] = df["DATA_TYPE"].replace({"ACQ_IMM": "acq_imm", "ACQ_EROG": "acq_erog"})

    wide = (
        df.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="DATA_TYPE", values="Osservazione", aggfunc="first")
        .reset_index()
    )
    wide.columns = [
        col if col in {"REF_AREA", "TIME_PERIOD"} else f"Osservazione_economical_distrib_{sanitize_column_token(col)}"
        for col in wide.columns
    ]
    return wide


def prepare_economical_indice_comp_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_column(df, "REF_AREA", ("REF_AREA", "ref_area"))
    df = ensure_column(df, "TIME_PERIOD", ("TIME_PERIOD", "time_period"))
    df = ensure_column(df, "DATA_TYPE", ("DATA_TYPE", "data_type"))
    df = ensure_column(df, "OBS_VALUE", ("OBS_VALUE", "obs_value"))

    df = filter_year_window(df)
    return pivot_long_to_wide(df, pivot_key="DATA_TYPE", value_col="OBS_VALUE", value_prefix="OBS_VALUE", suffix="economical_indice_comp")


def prepare_tables_by_source(master: Path) -> list[tuple[str, pd.DataFrame]]:
    files = {p.name: p for p in master.rglob("*.csv")}

    def find_file(keyword: str) -> Path | None:
        for name, path in files.items():
            if keyword.lower() in name.lower():
                return path
        return None

    prepared_tables: list[tuple[str, pd.DataFrame]] = []

    census_file = find_file("Popolazione residente")
    if census_file is not None:
        LOG.info("Processing census file: %s", census_file.name)
        census_df = read_csv_smart(census_file)
        prepared_tables.append(("census", prepare_census_table(census_df)))
    else:
        LOG.warning("Census CSV not found")

    princ_file = find_file("REDDITIIRPEF_COM_1")
    if princ_file is not None:
        LOG.info("Processing economical_princ file: %s", princ_file.name)
        df = read_csv_smart(princ_file)
        prepared_tables.append(("economical_princ", prepare_economical_princ_table(df)))
    else:
        LOG.warning("economical_princ CSV not found")

    reddito_file = find_file("REDDITIIRPEF_COM_2")
    if reddito_file is not None:
        LOG.info("Processing economical_reddito file: %s", reddito_file.name)
        df = read_csv_smart(reddito_file)
        prepared_tables.append(("economical_reddito", prepare_economical_reddito_table(df)))
    else:
        LOG.warning("economical_reddito CSV not found")

    acqua_file = find_file("CONSACQUA") or find_file("Distribuzione di acqua potabile")
    if acqua_file is not None:
        LOG.info("Processing economical_distrib file: %s", acqua_file.name)
        df = read_csv_smart(acqua_file)
        prepared_tables.append(("economical_distrib", prepare_economical_distrib_table(df)))
    else:
        LOG.warning("economical_distrib CSV not found")

    fragility_file = find_file("COMP_FRA_IND") or find_file("DF_COMP_FRA")
    if fragility_file is not None:
        LOG.info("Processing economical_indice_comp file: %s", fragility_file.name)
        df = read_csv_smart(fragility_file)
        prepared_tables.append(("economical_indice_comp", prepare_economical_indice_comp_table(df)))
    else:
        LOG.warning("economical_indice_comp CSV not found")

    if not prepared_tables:
        raise RuntimeError("No input tables were found or parsed")

    return prepared_tables


def normalize_codes_for_join(attr_df: pd.DataFrame, gdf: gpd.GeoDataFrame, ref_col: str = "REF_AREA", gid_col: str = "PRO_COM") -> Tuple[pd.DataFrame, gpd.GeoDataFrame, str]:
    attr_df = attr_df.copy()
    attr_df[ref_col] = attr_df[ref_col].astype(str).str.strip()

    gdf = gdf.copy()
    gdf[gid_col] = gdf[gid_col].astype(str).str.strip()

    attr_max = attr_df[ref_col].str.len().max() or 0
    gdf_max = gdf[gid_col].str.len().max() or 0
    width = int(max(attr_max, gdf_max, 6))

    attr_df["REF_AREA_norm"] = attr_df[ref_col].astype(str).str.zfill(width)
    gdf["PRO_COM_norm"] = gdf[gid_col].astype(str).str.zfill(width)
    return attr_df, gdf, "REF_AREA_norm"


def convert_to_numeric_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns to numeric types."""
    df_out = df.copy()
    for col in df_out.select_dtypes(include=["object"]).columns:
        # Attempt to convert to numeric, coercing errors to NaN
        numeric_col = pd.to_numeric(df_out[col], errors="coerce")
        # If the conversion was successful (not all values became NaN), update the column
        if not numeric_col.isnull().all():
            # Downcast to integer if possible (no decimals, no NaNs)
            df_out[col] = pd.to_numeric(numeric_col.dropna(), downcast="integer")
    return df_out


def write_shapefile_with_short_fields(gdf_out: gpd.GeoDataFrame, out_path: Path) -> None:
    mapping = build_short_column_mapping(list(gdf_out.columns), max_length=10)
    gdf_shp = gdf_out.rename(columns=mapping)
    gdf_shp = convert_to_numeric_if_possible(gdf_shp)
    gdf_shp.to_file(out_path)


def main() -> None:
    master = MASTER_DATA_PATH
    LOG.info("Master data path: %s", master)
    prepared_tables = prepare_tables_by_source(master)

    gpkg_file = GPKG_PATH / "merged_pedologgia_static.gpkg"
    layer_name = "com01012026_wgs84"
    LOG.info("Reading layer %s from %s", layer_name, gpkg_file)
    gdf = gpd.read_file(gpkg_file, layer=layer_name)

    for source_name, attrs in prepared_tables:
        if {"REF_AREA", "TIME_PERIOD"}.issubset(attrs.columns):
            attrs, gdf_norm, key = normalize_codes_for_join(attrs, gdf, ref_col="REF_AREA", gid_col="PRO_COM")
            LOG.info("Merging %d %s attribute rows with %d geometry rows", len(attrs), source_name, len(gdf_norm))

            gdf_out = gdf_norm.merge(attrs, left_on="PRO_COM_norm", right_on=key, how="inner")
            gdf_out = gdf_out.drop(columns=["PRO_COM_norm", "REF_AREA_norm"])

            gpkg_mapping = build_short_column_mapping(list(gdf_out.columns), max_length=24)
            gdf_out = gdf_out.rename(columns=gpkg_mapping)

            # Convert data types before saving to ensure numeric columns are not text
            gdf_out = convert_to_numeric_if_possible(gdf_out)

            out_name = f"tabular_comunal_{source_name}"
            out_gpkg = MIDPOINTS_PATH / "socio_economic" / f"{out_name}.gpkg"
            LOG.info("Writing GeoPackage to %s", out_gpkg)
            gdf_out.to_file(out_gpkg, layer=out_name, driver="GPKG")
        else:
            LOG.warning("Skipping %s because required join columns are missing", source_name)

    LOG.info("Done.")


if __name__ == "__main__":
    main()
