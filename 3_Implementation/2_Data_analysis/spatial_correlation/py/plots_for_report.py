import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
import pandas as pd
from esda.moran import Moran_Local_BV
from splot.esda import lisa_cluster
from statsmodels.stats.multitest import multipletests

# Dynamically add repository root to sys.path to resolve Utils imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import project paths from Utils.paths
from Utils.paths import GRID_PATH, PLOTS_SPATIAL_AUTOCORRELATION_PATH

# Output directory configuration
OUTPUT_DIR = PLOTS_SPATIAL_AUTOCORRELATION_PATH
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map each variable pair to a custom plot title
PAIR_TITLES = {
    ("census_pop", "inflow_total"): "Spatial Association: Population and Inflow",
    ("idw_rain_mm_avg_monthly", "idw_wind_speed_max_95p"): "Spatial Association: Rainfall and Wind Speed",
    ("idw_temp_thermal_range", "dtmidcnt_mean"): "Spatial Association: Thermal Range and Elevation",
}


def prepare_variable(gdf: gpd.GeoDataFrame, var_name: str) -> pd.Series | None:
    """Handles missing values and verifies non-zero standard deviation."""
    y = pd.to_numeric(gdf[var_name], errors="coerce").copy()
    if y.isnull().any():
        mean_val = y.mean()
        y.fillna(mean_val, inplace=True)

    if y.std() == 0:
        print(f"Skipping '{var_name}': Variable has zero variance.")
        return None
    return y


def apply_fdr_correction(local_stat_model):
    """Applies Benjamini-Hochberg FDR correction to local p-values."""
    reject, p_corrected, _, _ = multipletests(
        local_stat_model.p_sim, alpha=0.05, method="fdr_bh"
    )
    local_stat_model.p_sim = p_corrected
    return local_stat_model


def run_bivariate_lisa(
    gdf: gpd.GeoDataFrame,
    weights: libpysal.weights.W,
    var1: str,
    var2: str,
    custom_title: str,
    output_dir: Path,
):
    """Calculates Bivariate LISA and generates cluster maps with a custom title."""
    print(f"\nRunning Bivariate LISA for pair: {var1} vs {var2}")

    y1 = prepare_variable(gdf, var1)
    y2 = prepare_variable(gdf, var2)

    if y1 is None or y2 is None:
        return

    # Calculate Bivariate Local Moran's I
    bivariate_lisa = Moran_Local_BV(y1, y2, weights, seed=42)
    apply_fdr_correction(bivariate_lisa)

    # Plot and save cluster map
    fig, ax = plt.subplots(figsize=(12, 10))
    lisa_cluster(bivariate_lisa, gdf, ax=ax, legend=True)
    
    # Apply custom title
    ax.set_title(
        f"{custom_title}\n(FDR Corrected, p < 0.05)",
        fontsize=12,
    )
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()

    out_file = output_dir / f"report_bivariate_lisa_{var1}_vs_{var2}.png"
    plt.savefig(out_file)
    plt.close(fig)
    print(f"Saved LISA cluster map to: {out_file.name}")


def main():
    # 1. Load spatial dataset
    gpkg_file = GRID_PATH / "h310_grid_final_datacube.gpkg"
    print(f"Loading layer from {gpkg_file}...")
    gdf = gpd.read_file(gpkg_file)

    # 2. Build Queen spatial weights matrix
    print("Creating spatial weights matrix (Queen)...")
    weights = libpysal.weights.Queen.from_dataframe(gdf)
    weights.transform = "r"

    # 3. Process each pair with its custom title
    for (var1, var2), custom_title in PAIR_TITLES.items():
        if var1 in gdf.columns and var2 in gdf.columns:
            run_bivariate_lisa(gdf, weights, var1, var2, custom_title, OUTPUT_DIR)
        else:
            missing = [v for v in (var1, var2) if v not in gdf.columns]
            print(f"Skipping pair ({var1}, {var2}): Columns missing {missing}")


if __name__ == "__main__":
    main()