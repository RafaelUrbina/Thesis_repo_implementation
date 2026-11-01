import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
import pandas as pd
from esda.moran import Moran_Local, Moran_Local_BV
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

# Map each variable pair to a custom plot title (Bivariate)
PAIR_TITLES = {
    ("census_pop", "inflow_total"): "Spatial Association: Population and Inflow",
    ("idw_rain_mm_avg_monthly", "idw_wind_speed_max_95p"): "Spatial Association: Rainfall and Wind Speed",
    ("idw_temp_thermal_range", "dtmidcnt_mean"): "Spatial Association: Thermal Range and Elevation",
}

# Map each univariate variable to a custom plot title
UNIVARIATE_TITLES = {
    "census_pop": "Spatial Association: Population",
    "inflow_total": "Spatial Association: Inflow Total",
    "idw_rain_mm_avg_monthly": "Spatial Association: Monthly Average Rainfall",
    "idw_wind_speed_max_95p": "Spatial Association: 95th Percentile Max Wind Speed",
    "idw_temp_thermal_range": "Spatial Association: Thermal Range",
    "dtmidcnt_mean": "Spatial Association: Mean Elevation (dtmidcnt)",
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


def run_univariate_lisa(
    gdf: gpd.GeoDataFrame,
    weights: libpysal.weights.W,
    var_name: str,
    custom_title: str,
    output_dir: Path,
):
    """Calculates Univariate LISA and generates cluster maps with a custom title."""
    print(f"\nRunning Univariate LISA for: {var_name}")

    y = prepare_variable(gdf, var_name)
    if y is None:
        return

    # Calculate Univariate Local Moran's I
    univariate_lisa = Moran_Local(y, weights, seed=42)
    apply_fdr_correction(univariate_lisa)

    # Plot and save cluster map
    fig, ax = plt.subplots(figsize=(12, 10))
    lisa_cluster(univariate_lisa, gdf, ax=ax, legend=True)
    
    # Apply custom title
    ax.set_title(
        f"{custom_title}\n(FDR Corrected, p < 0.05)",
        fontsize=12,
    )
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()

    out_file = output_dir / f"report_univariate_lisa_{var_name}.png"
    plt.savefig(out_file)
    plt.close(fig)
    print(f"Saved univariate LISA cluster map to: {out_file.name}")


def run_bivariate_lisa(
    gdf: gpd.GeoDataFrame,
    weights: libpysal.weights.W,
    var1: str,
    var2: str,
    custom_title: str,
    output_dir: Path,
):
    """Calculates Bivariate LISA, generates cluster maps, and exports filtered GIS layers."""
    print(f"\nRunning Bivariate LISA for pair: {var1} vs {var2}")

    y1 = prepare_variable(gdf, var1)
    y2 = prepare_variable(gdf, var2)

    if y1 is None or y2 is None:
        return

    # 1. Calculate Bivariate Local Moran's I
    bivariate_lisa = Moran_Local_BV(y1, y2, weights, seed=42)
    apply_fdr_correction(bivariate_lisa)

    # 2. Extract GIS Attributes & Build Cluster Labels
    quad_labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    sig_mask = bivariate_lisa.p_sim < 0.05

    cluster_col = []
    for q, is_sig in zip(bivariate_lisa.q, sig_mask):
        if not is_sig:
            cluster_col.append("Not Significant")
        else:
            cluster_col.append(quad_labels.get(q, "Unknown"))

    prefix = f"bv_{var1}_{var2}"

    # 3. Create a clean export GeoDataFrame containing ONLY geometry, index, and bv_ columns
    gdf_export = gpd.GeoDataFrame(
        {
            f"{prefix}_moran_i": bivariate_lisa.Is,
            f"{prefix}_p_value": bivariate_lisa.p_sim,
            f"{prefix}_quadrant": bivariate_lisa.q,
            f"{prefix}_cluster": cluster_col,
        },
        geometry=gdf.geometry,
        crs=gdf.crs,
        index=gdf.index,
    )

    # Define the 'report' subfolder path and ensure it exists
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 4. Export the filtered GeoPackage (.gpkg)
    out_gpkg = report_dir / f"report_bivariate_lisa_{var1}_vs_{var2}.gpkg"
    layer_name = f"bivariate_{var1}_{var2}".replace(".", "_")[:63]
    gdf_export.to_file(out_gpkg, layer=layer_name, driver="GPKG")
    print(f"Saved filtered bivariate GIS layer to: {out_gpkg} (Layer: {layer_name})")

    # 5. Plot and save cluster map image to the report subfolder
    fig, ax = plt.subplots(figsize=(12, 10))
    lisa_cluster(bivariate_lisa, gdf, ax=ax, legend=True)
    
    ax.set_title(
        f"{custom_title}\n(FDR Corrected, p < 0.05)",
        fontsize=12,
    )
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()

    out_img = report_dir / f"report_bivariate_lisa_{var1}_vs_{var2}.png"
    plt.savefig(out_img)
    plt.close(fig)
    print(f"Saved bivariate LISA cluster map image to: {out_img}")


def main():
    # 1. Load spatial dataset
    gpkg_file = GRID_PATH / "h310_grid_final_datacube.gpkg"
    print(f"Loading layer from {gpkg_file}...")
    gdf = gpd.read_file(gpkg_file)

    # 2. Build Queen spatial weights matrix
    print("Creating spatial weights matrix (Queen)...")
    weights = libpysal.weights.Queen.from_dataframe(gdf)
    weights.transform = "r"

    # 3. Process each univariate variable with its custom title
    print("--- Processing Univariate LISA ---")
    for var_name, custom_title in UNIVARIATE_TITLES.items():
        if var_name in gdf.columns:
            print("skipping univariate variable '{var_name}': manual skip.")
            #run_univariate_lisa(gdf, weights, var_name, custom_title, OUTPUT_DIR)
        else:
            print(f"Skipping univariate variable '{var_name}': Column missing from dataset.")

    # 4. Process each bivariate pair with its custom title
    print("\n--- Processing Bivariate LISA ---")
    for (var1, var2), custom_title in PAIR_TITLES.items():
        if var1 in gdf.columns and var2 in gdf.columns:
            run_bivariate_lisa(gdf, weights, var1, var2, custom_title, OUTPUT_DIR)
        else:
            missing = [v for v in (var1, var2) if v not in gdf.columns]
            print(f"Skipping pair ({var1}, {var2}): Columns missing {missing}")


if __name__ == "__main__":
    main()