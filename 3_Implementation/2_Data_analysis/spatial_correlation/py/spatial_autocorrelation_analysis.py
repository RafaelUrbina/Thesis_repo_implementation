"""
General pipeline for spatial autocorrelation analysis on a static spatial dataset.

This script provides a comprehensive workflow for exploring spatial patterns in a
geospatial dataset. It systematically identifies which variables exhibit spatial
clustering and where those clusters are located.

The analysis is divided into several key parts:

1.  **Data Loading and Preparation**:
    - Loads the master grid GeoDataFrame.
    - Defines and separates variables into numerical and categorical types.
    - Enforces correct data types to prevent errors during analysis.
    - Creates a single Queen contiguity spatial weights matrix, which defines
      "neighborhoods" for all subsequent spatial calculations.

2.  **Univariate Numerical Analysis**:
    - Calculates Global Moran's I for all numerical variables to measure the
      overall degree of spatial clustering (positive, negative, or random).
    - Ranks variables by their Moran's I value to identify the most and least
      spatially autocorrelated variables.
    - For these top variables, it performs detailed local analyses:
        - **LISA (Local Moran's I)**: Identifies hotspots, coldspots, and spatial
          outliers (e.g., a high value surrounded by low values).
        - **Getis-Ord Gi***: Specifically identifies statistically significant
          hotspots (clusters of high values) and coldspots (clusters of low values).

3.  **Bivariate Numerical Analysis**:
    - For a predefined subset of interesting numerical variables, it calculates
      the Bivariate Local Moran's I. This reveals how a variable at a location
      relates to a *different* variable in neighboring locations (e.g., where
      high landslide counts are spatially correlated with low population).

4.  **Categorical Analysis**:
    - **Join-Count Statistics**: For top categorical variables, this univariate
      test determines if areas of the same category are more clustered together
      than would be expected by chance.
    - **Spatial Cramér's V**: A bivariate measure that assesses the spatial
      association between pairs of categorical variables (e.g., do certain
      land use types tend to be neighbors with specific soil types?).

5.  **Multivariate Clustering**:
    - Performs a non-spatial K-Means clustering using both numerical and
      categorical variables to identify locations with similar overall profiles.
    - The results are mapped to show the spatial distribution of these
      multivariate profile clusters.
"""

""" Available columns:
['index', 'total_intervention_cost_1ring', 'class_intervention_1ring', 'active_fountains_count_1ring', 
'total_fountains_count_1ring', 'landslide_point_count_1ring', 'tipo_movimento_<lambda_0>_1ring', 
'max_peak_elevation_1ring', 'is_locality_1ring', 'locality_name_1ring', 'seismic_event_count_1ring', 
'max_seismic_magnitude_1ring', 'avg_seismic_magnitude_1ring', 'road_density_m_per_m2', 'm_per_hex', 
'dominant_highway', 'dominant_surface', 'dominant_tunnel', 'dominant_bridge', 'dist_to_waterway_m', 
'rooting_depth_class', 'surface_stoniness_class', 'usda_hydrologic_group', 'landslide_surface_class', 
'pai_landslide_hazard_level', 'avg_descending_soil_speed', 'descending_soil_presence', 
'avg_ascending_soil_speed', 'ascending_soil_presence', 'hydraulic_hazard_level', 'dominant_building_1', 
'dominant_building_2', 'dominant_building_3', 'census_pop', 'epr_NTAXP', 'epr_TAXABINC', 'epr_CADINCR', 
'epr_CADINCF', 'epr_SUBEMPTR', 'epr_PENSINCR', 'epr_PENSINCF', 'epr_ENTROAIN', 'epr_ENTROAIN01', 
'erd_E0_10000', 'erd_E10000_1', 'erd_E15000_2', 'erd_E26000_5', 'erd_E55000_7', 'erd_E75000_1', 
'erd_E_GE1200', 'edst_acq_imm', 'edst_acq_erog', 'eidx_COMP_FRA', 'eidx_LAND_CON', 'eidx_EMPL_RAT', 
'eidx_POP_25_6', 'eidx_POP_DEPE', 'eidx_INDEX_AC', 'eidx_PERSEMP', 'inflow_total', 
'nearest_hydro_distance_m', 'nearest_hydro_river_stage_max_m', 'nearest_hydro_river_stage_mean_m', 
'nearest_hydro_river_stage_std_m', 'nearest_hydro_quota', 'idw_temp_max_peak', 'idw_temp_min_nadir', 
'idw_temp_thermal_range', 'idw_rain_mm_sum_annual', 'idw_rain_mm_max_monthly', 'idw_rain_mm_min_monthly', 
'idw_rain_mm_avg_monthly', 'idw_rain_mm_std', 'idw_wind_speed_max_max',
 'idw_wind_speed_max_95p', 'idw_wind_speed_avg_mean', 'dtmidcnt_mean', 'dtmidcnt_max', 'geometry'] """


""" 
IGNORED VARIABLES:

'index', 'geometry'

 """

""" 
CATEGORICAL VARIABLES (STRING):

'class_intervention_1ring', 'locality_name_1ring', 'dominant_highway', 'dominant_surface', 
'dominant_tunnel', 'dominant_bridge', 'usda_hydrologic_group', 'pai_landslide_hazard_level',
'dominant_building_1', 'dominant_building_2', 'dominant_building_3',

 """


""" 
NUMERICAL VARIABLES (FLOAT,INT):

'total_intervention_cost_1ring', 'max_peak_elevation_1ring', 'active_fountains_count_1ring', 
'total_fountains_count_1ring', 'landslide_point_count_1ring', 'seismic_event_count_1ring', 
'max_seismic_magnitude_1ring', 'avg_seismic_magnitude_1ring', 'road_density_m_per_m2', 'm_per_hex',
'dist_to_waterway_m', 'avg_descending_soil_speed', 'avg_ascending_soil_speed', 'census_pop', 'epr_NTAXP', 
'epr_TAXABINC', 'epr_CADINCR', 'epr_CADINCF', 'epr_SUBEMPTR', 'epr_PENSINCR', 'epr_PENSINCF', 'epr_ENTROAIN', 'epr_ENTROAIN01', 
'erd_E0_10000', 'erd_E10000_1', 'erd_E15000_2', 'erd_E26000_5', 'erd_E55000_7', 'erd_E75000_1', 
'erd_E_GE1200', 'edst_acq_imm', 'edst_acq_erog', 'eidx_COMP_FRA', 'eidx_LAND_CON', 'eidx_EMPL_RAT', 
'eidx_POP_25_6', 'eidx_POP_DEPE', 'eidx_INDEX_AC', 'eidx_PERSEMP', 'inflow_total', 
'nearest_hydro_distance_m', 'nearest_hydro_river_stage_max_m', 'nearest_hydro_river_stage_mean_m', 
'nearest_hydro_river_stage_std_m', 'nearest_hydro_quota', 'idw_temp_max_peak', 'idw_temp_min_nadir', 
'idw_temp_thermal_range', 'idw_rain_mm_sum_annual', 'idw_rain_mm_max_monthly', 'idw_rain_mm_min_monthly', 
'idw_rain_mm_avg_monthly', 'idw_rain_mm_std', 'idw_wind_speed_max_max',
 'idw_wind_speed_max_95p', 'idw_wind_speed_avg_mean', 'dtmidcnt_mean', 'dtmidcnt_max',

 """


""" 
CATEGORICAL VARIABLES (INT):

'tipo_movimento_<lambda_0>_1ring', 'is_locality_1ring', 'rooting_depth_class', 'surface_stoniness_class',
'landslide_surface_class', 'descending_soil_presence', 'ascending_soil_presence', 'hydraulic_hazard_level',

 """


import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
import numpy as np
import fiona
import pandas as pd # Added for categorical variable handling
from esda.join_counts import Join_Counts # Added for categorical variable handling
from libpysal.weights import lag_spatial # Added for spatial contingency analysis
from esda.moran import Moran, Moran_Local, Moran_Local_BV, Moran_BV
from esda.geary import Geary
from esda.getisord import G_Local
from itertools import combinations
from tqdm.auto import tqdm # Import tqdm for progress bars
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from splot.esda import moran_scatterplot, lisa_cluster
from scipy.stats import contingency
from statsmodels.stats.multitest import multipletests # Added for FDR/Bonferroni correction


# Add the project root to the Python path to allow for absolute imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GRID_PATH, PLOTS_SPATIAL_AUTOCORRELATION_PATH, TABLE_SPATIAL_AUTOCORRELATION_PATH

# Define a threshold for unique values to consider an integer column as numeric
# If a column has more unique values than this, it's likely continuous.
CARDINALITY_THRESHOLD = 25

def load_data(gpkg_path: Path) -> gpd.GeoDataFrame | None:
    """
    Loads the GeoPackage file into a GeoDataFrame.
    It automatically detects the layer name.

    Args:
        gpkg_path (Path): The path to the GeoPackage file.

    Returns:
        A GeoDataFrame if successful, otherwise None.
    """
    print(f"Loading data from: {gpkg_path}")
    try:
        # Discover layers in the GeoPackage
        layers = fiona.listlayers(gpkg_path)
        if not layers:
            print("Error: No layers found in the GeoPackage file.")
            return None
        
        layer_to_load = layers[0]
        print(f"Found layers: {layers}. Loading the first one: '{layer_to_load}'")
        gdf = gpd.read_file(gpkg_path, layer=layer_to_load)
        print("Data loaded successfully.")
        print(f"Shape of the dataset: {gdf.shape}")
        print("\nAvailable columns:")
        print(gdf.columns.tolist())
        return gdf
    except Exception as e:
        print(f"Error loading GeoPackage file: {e}")
        print("Please ensure the file exists and you have the necessary drivers (e.g., 'fiona').")
        return None

def create_weights(gdf: gpd.GeoDataFrame) -> libpysal.weights.W:
    """
    Creates a spatial weights matrix for the given GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame for which to create weights.

    Returns:
        A row-standardized Queen contiguity spatial weights matrix.
    """
    print("\nCreating spatial weights matrix (Queen Contiguity)...")
    weights = libpysal.weights.Queen.from_dataframe(gdf)
    weights.transform = 'r'  # Row-standardize the weights
    print("Weights matrix created.")
    return weights

def _prepare_variable(gdf: gpd.GeoDataFrame, variable_name: str) -> pd.Series | None:
    """Prepares a single variable for analysis by handling NaNs and checking variance."""
    y = gdf[variable_name].copy()
    if y.isnull().any():
        mean_val = y.mean()
        y.fillna(mean_val, inplace=True)
        print(f"  - Warning: Missing values in '{variable_name}'. Filled with mean ({mean_val:.2f}).")
    
    if y.std() == 0:
        print(f"  - Skipping '{variable_name}': Variable has zero variance.")
        return None
    return y

def _check_and_skip(plot_path: Path, analysis_name: str) -> bool:
    """Checks if a plot already exists and prints a skip message."""
    if plot_path.exists() and plot_path.stat().st_size > 0:
        print(f"  - Plot '{plot_path.name}' already exists. Skipping {analysis_name}.")
        return True
    return False

def _sanitize_filename(name: str) -> str:
    """Removes characters that are invalid for file paths."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

def _apply_fdr_correction(local_stat_model):
    """Applies FDR correction and returns the results."""
    reject, p_corrected, _, _ = multipletests(
        local_stat_model.p_sim, 
        alpha=0.05, 
        method='fdr_bh'
    )
    # Overwrite the model's p-values with the corrected ones
    local_stat_model.p_sim = p_corrected
    return reject, p_corrected # Return both reject and corrected p-values

def analyze_variable_detailed(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variable_name: str, output_dir: Path):
    """
    Runs a detailed spatial autocorrelation analysis (Global Moran Plot and LISA) for a single variable.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variable_name (str): The name of the column to analyze.
        output_dir (Path): The directory to save the output plots.
    """
    print(f"\n--- Running Detailed Univariate Analysis for: {variable_name} ---")

    y = _prepare_variable(gdf, variable_name)
    if y is None:
        return

    sanitized_name = _sanitize_filename(variable_name)
    # --- 1. Global Moran's I Scatterplot ---
    moran_plot_path = output_dir / f"moran_plot_{sanitized_name}.png"
    if not _check_and_skip(moran_plot_path, "Moran scatterplot"):
        moran = Moran(y, weights)
        fig, ax = moran_scatterplot(moran, aspect_equal=True)
        title = f"Global Moran's I for {variable_name}\nI={moran.I:.4f}, p-value=({moran.p_sim:.4f})"
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(moran_plot_path)
        plt.close(fig)
        print(f"  - Saved Moran scatterplot to: {moran_plot_path.name}")

    # --- 2. Local Moran's I (LISA) Cluster Map ---
    lisa_plot_path = output_dir / f"lisa_cluster_map_{sanitized_name}.png"
    if not _check_and_skip(lisa_plot_path, "LISA cluster map"):
        lisa = Moran_Local(y, weights, seed=42)
        _apply_fdr_correction(lisa) # This function modifies 'lisa' in-place

        fig, ax = plt.subplots(figsize=(12, 10))
        lisa_cluster(lisa, gdf, ax=ax, legend=True)
        ax.set_title(f"Local Moran's I (LISA) for {variable_name} (FDR Corrected, p<0.05)")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.tight_layout()
        plt.savefig(lisa_plot_path)
        plt.close(fig)
        print(f"  - Saved LISA cluster map to: {lisa_plot_path.name}")

def analyze_bivariate_detailed(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, var1: str, var2: str, output_dir: Path):
    """
    Runs a detailed Bivariate Local Moran's I analysis for a specific pair of variables.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.
        output_dir (Path): The directory to save the output plots.
    """
    print(f"\n--- Running Detailed Bivariate Analysis for: {var1} vs. {var2} ---")
    s_var1 = _sanitize_filename(var1)
    s_var2 = _sanitize_filename(var2)
    bv_plot_path = output_dir / f"bivariate_lisa_{s_var1}_vs_{s_var2}_fdr.png"
    if _check_and_skip(bv_plot_path, "Bivariate LISA"):
        return

    y1 = _prepare_variable(gdf, var1)
    y2 = _prepare_variable(gdf, var2)

    if y1 is None or y2 is None:
        print("  - Skipping pair: At least one variable had zero variance.")
        return

    bivariate_lisa = Moran_Local_BV(y1, y2, weights, seed=42)
    _apply_fdr_correction(bivariate_lisa)

    fig, ax = plt.subplots(figsize=(12, 10))
    lisa_cluster(bivariate_lisa, gdf, ax=ax, legend=True)
    ax.set_title(f"Bivariate LISA: {var1} vs. {var2} (FDR Corrected, p<0.05)")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(bv_plot_path)
    plt.close(fig)
    print(f"  - Saved Bivariate LISA cluster map to: {bv_plot_path.name}")

def analyze_getis_ord_gi(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variable_name: str, output_dir: Path):
    """
    Calculates and plots the Getis-Ord Gi* local statistic to identify hot and cold spots.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variable_name (str): The name of the column to analyze.
        output_dir (Path): The directory to save the output plot.
    """
    sanitized_name = _sanitize_filename(variable_name)
    plot_path = output_dir / f"getis_ord_gi_star_map_{sanitized_name}_fdr.png"
    print(f"\n--- Running Getis-Ord Gi* for: {variable_name} ---")
    if _check_and_skip(plot_path, "Getis-Ord Gi* map"):
        return

    y = _prepare_variable(gdf, variable_name)
    if y is None:
        return

    # Temporarily set weights to binary for Gi* calculation
    original_transform = weights.transform
    weights.transform = 'b'

    # Calculate Gi* statistic. star=True is essential.
    # The transform='R' here is for the internal calculation of the expected value, not the weights themselves.
    gi_star = G_Local(y, weights, transform='R', star=True, seed=42)

    # Apply FDR correction to the p-values
    reject, p_corrected = _apply_fdr_correction(gi_star)
    # Create labels based on corrected significance
    labels = pd.Series('Not significant', index=gdf.index)
    labels[ (gi_star.Zs > 0) & (p_corrected < 0.05) ] = 'Hot spot'
    labels[ (gi_star.Zs < 0) & (p_corrected < 0.05) ] = 'Cold spot'

    # Restore the original weights transformation for other analyses
    weights.transform = original_transform

    color_map = {
        'Not significant': 'grey',
        'Hot spot': 'red',
        'Cold spot': 'blue'
    }
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in color_map.items()
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.assign(cl=labels).plot(
        column='cl',
        categorical=True,
        color=[color_map.get(l, 'lightgrey') for l in labels],
        legend=False,
        ax=ax,
    )

    ax.legend(handles=legend_patches, title='Gi* Cluster Type', loc='upper right')
    ax.set_title(f"Getis-Ord Gi* Hot/Cold Spot Map for {variable_name}\n(FDR Corrected, p<0.05)")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  - Saved Getis-Ord Gi* map to: {plot_path.name}")

def analyze_categorical_join_counts(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variable_name: str, table_output_dir: Path):
    """
    Runs Join-Count Statistics for a categorical variable.
    For each unique category, it performs a Join-Count analysis treating that category
    as 'Black' and all others as 'White'.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variable_name (str): The name of the categorical column to analyze.
        table_output_dir (Path): The directory to save the output table.
    """
    sanitized_name = _sanitize_filename(variable_name)
    table_path = table_output_dir / f"join_counts_{sanitized_name}.csv"
    if _check_and_skip(table_path, f"Join-Count analysis for '{variable_name}'"):
        return

    print(f"\n{'='*20} Analyzing Categorical (Join-Counts): {variable_name} {'='*20}")

    original_weights_transform = weights.transform
    weights.transform = 'b' # Join-counts require binary weights style 'b'

    unique_categories = gdf[variable_name].dropna().unique()
    results = []

    if len(unique_categories) < 2:
        print(f"Skipping: Categorical variable '{variable_name}' has less than 2 unique categories or all are NaN.")
        weights.transform = original_weights_transform
        return

    for category in unique_categories:
        print(f"\n--- Join-Count for category: '{category}' in {variable_name} ---")
        # Create a binary variable: 1 if current category, 0 otherwise
        y_bin = (gdf[variable_name] == category).astype(int)

        # Check if there's enough variation for analysis
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            print(f"Skipping: Category '{category}' in '{variable_name}' is uniform (all 0s or all 1s).")
            continue

        jc = Join_Counts(y_bin, weights)
        expected_ww = jc.J - jc.mean_bb - jc.mean_bw

        results.append({
            'category': category,
            'observed_BB': jc.bb,
            'expected_BB': jc.mean_bb,
            'observed_WW': jc.ww,
            'expected_WW': expected_ww,
            'observed_BW': jc.bw,
            'expected_BW': jc.mean_bw,
            'p_value_chi2': jc.chi2_p
        })

        print(f"  Observed Black-Black joins (BB): {jc.bb}")
        print(f"  Expected BB under randomness: {jc.mean_bb:.2f}")
        print(f"  P-value (chi2): {jc.chi2_p:.4f}")
        print(f"  Observed White-White joins (WW): {jc.ww}")
        # The expected WW is not always a direct attribute, so we derive it.
        print(f"  Expected WW under randomness: {expected_ww:.2f}")
        print(f"  Observed Black-White joins (BW): {jc.bw}")
        print(f"  Expected BW under randomness: {jc.mean_bw:.2f}")

        if jc.chi2_p < 0.05:
            print(f"  Result: Statistically significant clustering for category '{category}'.")
        else:
            print(f"  Result: No statistically significant clustering for category '{category}'.")

    if results: # Only save if there are actual results
        results_df = pd.DataFrame(results)
        results_df.to_csv(table_path, index=False)
        print(f"\n  - Saved Join-Counts results to: {table_path}")
        
    weights.transform = original_weights_transform # Reset weights transform
def analyze_bivariate_categorical_association(
    gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, top_vars: list, table_output_dir: Path
) -> list:
    """
    Calculates a global measure of SPATIAL association (Cramér's V) for all pairs of top categorical variables.
    This is a SPATIAL BIVARIATE analysis that builds a contingency table of
    Var1 at a focal location vs. Var2 at neighboring locations.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        top_vars (list): The list of top categorical variable names to analyze.
        table_output_dir (Path): The directory to save the output table.

    Returns:
        list: A list of the top 10 variable pairs with the highest spatial Cramér's V values.
    """
    table_path = table_output_dir / "bivariate_spatial_association_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"\n--- Bivariate Spatial Association table '{table_path.name}' already exists. Skipping. ---")
        results_df = pd.read_csv(table_path)
        top_10_pairs = [tuple(x) for x in results_df.sort_values(by='cramers_v', ascending=False).head(10)[['variable_1', 'variable_2']].to_numpy()]
        print(f"Loaded existing results. Top 10 pairs by spatial Cramér's V: {top_10_pairs}")
        return top_10_pairs

    print(f"\n{'='*20} Analyzing Bivariate SPATIAL Association {'='*20}")

    results = []

    for var1, var2 in tqdm(list(combinations(top_vars, 2)), desc="  - Bivariate Spatial Association"):
        # Impute missing values for this pair
        y1 = gdf[var1].copy().fillna("not defined").astype(str)
        y2 = gdf[var2].copy().fillna("not defined").astype(str)

        # Get unique categories and create mapping for both variables
        cats1 = sorted(y1.unique())
        cat_map1 = {cat: i for i, cat in enumerate(cats1)}
        n_cats1 = len(cats1)

        cats2 = sorted(y2.unique())
        cat_map2 = {cat: i for i, cat in enumerate(cats2)}
        n_cats2 = len(cats2)

        # Initialize an empty contingency table
        contingency_table = np.zeros((n_cats1, n_cats2), dtype=int)

        # Build the spatial contingency table: focal var1 vs neighbor var2
        for i, focal_id in enumerate(weights.id_order):
            focal_cat_var1 = y1.loc[focal_id]
            focal_idx_var1 = cat_map1[focal_cat_var1]

            neighbor_ids = weights.neighbors[focal_id]
            for neighbor_id in neighbor_ids:
                neighbor_cat_var2 = y2.loc[neighbor_id]
                neighbor_idx_var2 = cat_map2[neighbor_cat_var2]
                
                contingency_table[focal_idx_var1, neighbor_idx_var2] += 1

        # Calculate Cramér's V
        try:
            cramers_v = contingency.association(contingency_table, method="cramer")
            results.append({'variable_1': var1, 'variable_2': var2, 'cramers_v': cramers_v})
            print(f"    - Pair: {var1} & {var2}, Cramér's V: {cramers_v:.4f}")
        except ValueError as e:
            print(f"    - Could not calculate Cramér's V for '{var1}' vs '{var2}': {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values(by='cramers_v', ascending=False)
        results_df.to_csv(table_path, index=False)
        print(f"\n  - Saved Bivariate Spatial Association results to: {table_path.name}")
        print("\n  - Top 10 most spatially associated variable pairs:")
        print(results_df.head(10))
        # Return a list of tuples for the top 10 pairs
        top_10_pairs = [tuple(x) for x in results_df.head(10)[['variable_1', 'variable_2']].to_numpy()]
    else:
        top_10_pairs = []

    print(f"\n{'='*20} Finished Bivariate Spatial Analysis {'='*20}")
    return top_10_pairs

def analyze_global_categorical_association(
    gdf: gpd.GeoDataFrame,
    weights: libpysal.weights.W,
    categorical_vars: list,
    table_output_dir: Path,
) -> list:
    """
    Calculates a global measure of spatial association (Cramér's V) for each categorical variable (univariate spatial).

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        categorical_vars (list): The list of categorical variable names to analyze.
        table_output_dir (Path): The directory to save the output table.

    Returns:
        list: A list of the top 10 variable names with the highest Cramér's V values.
    """
    table_path = table_output_dir / "univariate_spatial_association_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"\n--- Univariate Spatial Association table '{table_path.name}' already exists. Skipping. ---")
        results_df = pd.read_csv(table_path)
        top_10_vars = results_df.sort_values(by='cramers_v', ascending=False).head(10)['variable'].tolist()
        print(f"Loaded existing results. Top 10 variables by spatial Cramér's V: {top_10_vars}")
        return top_10_vars

    print(f"\n{'='*20} Analyzing Univariate Spatial Association {'='*20}")

    results = []

    for variable in tqdm(categorical_vars, desc="  - Global Categorical Association"):
        # Create a working copy and impute missing values
        y = gdf[variable].copy().fillna("not defined").astype(str)

        # Get the unique categories to build the contingency table
        unique_categories = sorted(y.unique())
        cat_map = {cat: i for i, cat in enumerate(unique_categories)}
        n_cats = len(unique_categories)

        if n_cats < 2:
            print(f"    - Skipping '{variable}': Not enough unique categories.")
            continue

        # Initialize an empty contingency table
        contingency_table = np.zeros((n_cats, n_cats), dtype=int)

        # Build the contingency table from neighbor pairs
        for i, focal_id in enumerate(weights.id_order):
            focal_cat = y.loc[focal_id]
            focal_idx = cat_map[focal_cat]
            
            neighbor_ids = weights.neighbors[focal_id]
            for neighbor_id in neighbor_ids:
                neighbor_cat = y.loc[neighbor_id]
                neighbor_idx = cat_map[neighbor_cat]
                
                # Increment count for the pair (focal, neighbor)
                contingency_table[focal_idx, neighbor_idx] += 1

        # Calculate Cramér's V using scipy
        # The 'association' function returns Cramér's V by default
        try:
            cramers_v = contingency.association(contingency_table, method="cramer")
            results.append({'variable': variable, 'cramers_v': cramers_v})
            print(f"    - Variable: {variable}, Cramér's V: {cramers_v:.4f}")
        except ValueError as e:
            print(f"    - Could not calculate Cramér's V for '{variable}': {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values(by='cramers_v', ascending=False)
        results_df.to_csv(table_path, index=False)
        print(f"\n  - Saved Univariate Spatial Association results to: {table_path.name}")
        top_10_vars = results_df.head(10)['variable'].tolist()
    else:
        top_10_vars = []

    print(f"\n{'='*20} Finished Univariate Spatial Analysis {'='*20}")
    return top_10_vars

def analyze_global_gearys_c(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variables_to_analyze: list, table_output_dir: Path):
    """
    Calculates Global Geary's C for a list of specified variables to confirm spatial patterns.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variables_to_analyze (list): A list of variable names to analyze.
        table_output_dir (Path): The directory to save the output table.
    """
    table_path = table_output_dir / "univariate_global_gearys_c_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"\n--- Global Geary's C table '{table_path.name}' already exists. Skipping. ---")
        return

    print(f"\n{'='*20} Analyzing Global Geary's C for Top Variables {'='*20}")
    geary_results = []
    for var in tqdm(variables_to_analyze, desc="  - Global Geary's C"):
        y = gdf[var].copy()
        if y.isnull().any():
            y.fillna(y.mean(), inplace=True)
        if y.std() > 0:
            geary = Geary(y, weights)
            geary_results.append({'variable': var, 'gearys_C': geary.C, 'p_value': geary.p_sim})
            print(f"    - Variable: {var}, Geary's C: {geary.C:.4f}, p-value: {geary.p_sim:.4f}")
        else:
            print(f"    - Skipping '{var}': Zero variance.")

    if geary_results:
        geary_df = pd.DataFrame(geary_results).sort_values(by='gearys_C', ascending=True) # Lower C indicates stronger positive autocorrelation
        geary_df.to_csv(table_path, index=False)
        print(f"\n  - Saved Global Geary's C results to: {table_path.name}")

    print(f"\n{'='*20} Finished Global Geary's C Analysis {'='*20}")


def analyze_multivariate_clusters(gdf: gpd.GeoDataFrame, numerical_vars: list, categorical_vars: list, output_dir: Path, n_clusters: int = 5):
    """
    Performs non-spatial clustering on a combination of numerical and
    one-hot encoded categorical variables, then maps the results.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        numerical_vars (list): The list of numerical variable names to include.
        categorical_vars (list): The list of categorical variable names to include.
        output_dir (Path): The directory to save the output plot.
        n_clusters (int): The number of clusters to create.
    """
    plot_path = output_dir / f"multivariate_kmeans_cluster_map_k{n_clusters}.png"
    if _check_and_skip(plot_path, "Multivariate Clustering analysis"):
        return
        
    print(f"\n{'='*20} Multivariate Clustering Analysis {'='*20}")

    # 1. Prepare numerical data: fill NaNs with column mean
    X_num = gdf[numerical_vars].fillna(gdf[numerical_vars].mean())

    # 2. Prepare categorical data: one-hot encode and handle potential NaNs
    if categorical_vars:
        # Filter for categorical variables that actually exist in the DataFrame
        valid_categorical_vars = [col for col in categorical_vars if col in gdf.columns]
        print(f"  - One-hot encoding {len(valid_categorical_vars)} valid categorical variables.")

        # Convert integer-based categories explicitly to strings so get_dummies processes all columns.
        # This prevents errors where get_dummies skips numeric-like columns.
        cat_df = gdf[valid_categorical_vars].astype(str)
        
        # Let pandas auto-assign prefixes based on column names and ensure output is float.
        X_cat = pd.get_dummies(cat_df, dummy_na=False, dtype=float)
    else:
        X_cat = pd.DataFrame(index=gdf.index) # Empty DataFrame if no categorical vars

    # 3. Combine numerical and one-hot encoded categorical data
    # Ensure indices are aligned before concatenation
    X_combined = pd.concat([X_num, X_cat], axis=1)
    
    # Drop any rows that might have become all NaN after combining (e.g., if original GDF had rows with only excluded columns)
    X_combined.dropna(inplace=True)

    if X_combined.empty:
        print("Skipping: No valid data points for multivariate clustering after NaN handling.")
        return
    
    # Ensure all columns are numeric after one-hot encoding
    X_combined = X_combined.select_dtypes(include=np.number)
    if X_combined.empty:
        print("Skipping: No numeric columns left for multivariate clustering after one-hot encoding.")
        return

    # 4. Standardize the combined data
    # StandardScaler expects 2D array, so ensure X_combined is not a Series if only one column
    if X_combined.shape[1] == 0:
        print("Skipping: No features available for clustering after preprocessing.")
        return
    
    X_scaled = StandardScaler().fit_transform(X_combined)

    # 5. Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Assign clusters back to the original GeoDataFrame, aligning by index
    gdf_filtered = gdf.loc[X_combined.index] # Filter gdf to match X_combined's index
    gdf_filtered['multivariate_cluster'] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_filtered.plot(column='multivariate_cluster', categorical=True, legend=True, figsize=(12, 10), aspect='equal', ax=ax)
    ax.set_title(f'Multivariate K-Means Clusters (k={n_clusters})')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  - Saved Multivariate K-Means cluster map to: {plot_path}")

def enforce_data_types_and_get_vars(gdf: gpd.GeoDataFrame, num_vars: list, cat_str_vars: list, cat_int_vars: list) -> tuple[list, list]:
    """Enforces data types and returns filtered lists of existing variables."""
    print("\n--- Enforcing Data Types ---")
    for var in num_vars:
        if var in gdf.columns:
            initial_nans = gdf[var].isnull().sum()
            gdf[var] = pd.to_numeric(gdf[var], errors='coerce')
            if gdf[var].isnull().sum() > initial_nans:
                print(f"  - Warning: Coerced non-numeric values to NaN in '{var}'.")
    print("Numerical variables enforced.")

    for var in cat_str_vars:
        if var in gdf.columns:
            gdf[var] = gdf[var].astype(str)
    print("String categorical variables enforced.")

    for var in cat_int_vars:
        if var in gdf.columns:
            initial_nans = gdf[var].isnull().sum()
            gdf[var] = pd.to_numeric(gdf[var], errors='coerce')
            if gdf[var].isnull().sum() > initial_nans:
                print(f"  - Warning: Coerced non-numeric values to NaN in '{var}'.")
            gdf[var] = gdf[var].astype('Int64')
    print("Integer categorical variables enforced.")
    print("--- Data Type Enforcement Complete ---")

    return [v for v in num_vars if v in gdf.columns], [v for v in (cat_str_vars + cat_int_vars) if v in gdf.columns]

def run_univariate_numerical_pipeline(gdf, weights, numerical_vars, output_plots_dir, output_tables_dir):
    """Orchestrates the entire univariate numerical analysis pipeline."""
    print(f"\n{'#'*30} Starting Univariate Numerical Analysis {'#'*30}")
    if not numerical_vars:
        print("No numerical variables to analyze.")
        return

    # Step 1: Calculate Global Moran's I for all variables
    table_path = output_tables_dir / "univariate_global_moran_I_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"\n--- Univariate Global Moran's I table '{table_path.name}' already exists. Skipping calculation. ---")
        moran_df = pd.read_csv(table_path)
    else:
        moran_results = []
        print("\n  - Calculating Global Moran's I for all numerical variables...")
        for var in tqdm(numerical_vars, desc="  - Univariate Global Moran's I"):
            y = _prepare_variable(gdf, var)
            if y is not None:
                moran = Moran(y, weights)
                moran_results.append({'variable': var, 'moran_I': moran.I, 'p_value': moran.p_sim})
                print(f"    - Variable: {var}, Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")
        
        moran_df = pd.DataFrame(moran_results).sort_values(by='moran_I', ascending=False) if moran_results else pd.DataFrame()
        if not moran_df.empty:
            moran_df.to_csv(table_path, index=False)
            print(f"\n  - Saved all Univariate Global Moran's I results to: {table_path.name}")

    if moran_df.empty:
        print("  - No valid Moran's I results to analyze for detailed plotting.")
        print(f"\n{'#'*30} Finished Univariate Numerical Analysis {'#'*30}")
        return

    # Step 2: Identify top variables for reporting and detailed analysis
    top_n = 6
    highest_moran_vars = moran_df.head(top_n)
    lowest_moran_vars = moran_df.tail(top_n)

    print(f"\n  - Top {top_n} variables with highest autocorrelation: {[f'{r.variable} (I={r.moran_I:.4f})' for r in highest_moran_vars.itertuples()]}")
    print(f"  - Top {top_n} variables with lowest autocorrelation: {[f'{r.variable} (I={r.moran_I:.4f})' for r in lowest_moran_vars.itertuples()]}")

    # Step 3: Create and save bar chart for ALL variables
    plot_path = output_plots_dir / "univariate_global_moran_I_barchart_all_variables.png"
    if not _check_and_skip(plot_path, "Moran's I summary bar chart for all variables"):
        moran_df_to_plot = moran_df.sort_values(by='moran_I', ascending=True).copy() # Sort for better visualization
        
        # Dynamically adjust figure height based on number of variables
        fig_height = max(8, len(moran_df_to_plot) * 0.3)
        plt.figure(figsize=(12, fig_height))
        
        plt.barh(moran_df_to_plot['variable'], moran_df_to_plot['moran_I'], color='skyblue')
        plt.xlabel("Global Moran's I")
        plt.ylabel("Variable")
        plt.title("Univariate Global Spatial Autocorrelation (All Variables)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.yticks(fontsize=8) # Adjust font size for readability if many variables
        plt.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved Moran's I summary bar chart for all variables to: {plot_path.name}")

    # Step 4: Run detailed local analyses on the top variables
    top_variables_for_confirmation = pd.concat([highest_moran_vars, lowest_moran_vars])['variable'].unique().tolist()

    analyze_global_gearys_c(gdf, weights, top_variables_for_confirmation, output_tables_dir)

    processed_vars = set()
    for var_name in top_variables_for_confirmation:
        if var_name not in processed_vars:
            analyze_variable_detailed(gdf, weights, var_name, output_plots_dir)
            analyze_getis_ord_gi(gdf, weights, var_name, output_plots_dir)
            processed_vars.add(var_name)
            
    print(f"\n{'#'*30} Finished Univariate Numerical Analysis {'#'*30}")

def run_bivariate_numerical_pipeline(gdf, weights, numerical_vars, output_plots_dir, output_tables_dir):
    """Orchestrates the entire bivariate numerical analysis pipeline."""
    top_n = 6  # Define how many top/bottom pairs to analyze in detail
    print(f"\n{'#'*30} Starting Bivariate Numerical Analysis {'#'*30}")
    
    bivariate_vars_subset = [
        'total_intervention_cost_1ring', 'max_peak_elevation_1ring', 'landslide_point_count_1ring',
        'seismic_event_count_1ring', 'avg_seismic_magnitude_1ring',
        'road_density_m_per_m2', 'dist_to_waterway_m', 'avg_descending_soil_speed',
        'avg_ascending_soil_speed', 'census_pop','idw_temp_thermal_range',
        'idw_rain_mm_sum_annual','idw_rain_mm_avg_monthly', 'idw_wind_speed_max_95p',
        'idw_wind_speed_avg_mean', 'inflow_total', 'dtmidcnt_mean', 'hydraulic_hazard_level'
    ]
    bivariate_vars_to_analyze = [v for v in bivariate_vars_subset if v in numerical_vars]

    if len(bivariate_vars_to_analyze) < 2:
        print("\n  - Skipping Bivariate Analysis: Fewer than 2 numerical variables available from the predefined subset.")
        return

    print(f"\n  - Selected {len(bivariate_vars_to_analyze)} variables for bivariate analysis.")

    # Step 1: Calculate Global Bivariate Moran's I for all pairs
    table_path = output_tables_dir / "bivariate_global_moran_I_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"\n--- Bivariate Global Moran's I table '{table_path.name}' already exists. Skipping calculation. ---")
        bivariate_df = pd.read_csv(table_path)
    else:
        bivariate_results = []
        print("\n  - Calculating Global Bivariate Moran's I for all numerical pairs...")
        
        # Standardize all variables first for efficiency
        gdf_std = gdf[bivariate_vars_to_analyze].copy()
        for col in gdf_std.columns:
            prepared_var = _prepare_variable(gdf, col)
            if prepared_var is not None and prepared_var.std() > 0:
                gdf_std[col] = (prepared_var - prepared_var.mean()) / prepared_var.std()
            else:
                gdf_std[col] = 0

        for var1, var2 in tqdm(list(combinations(bivariate_vars_to_analyze, 2)), desc="  - Bivariate Global Moran's I"):
            moran_bv = Moran_BV(gdf_std[var1], gdf_std[var2], weights)
            bivariate_results.append({'variable_1': var1, 'variable_2': var2, 'bivariate_moran_I': moran_bv.I, 'p_value': moran_bv.p_sim})
            print(f"    - Pair: {var1} & {var2}, Bivariate Moran's I: {moran_bv.I:.4f}, p-value: {moran_bv.p_sim:.4f}")

        bivariate_df = pd.DataFrame(bivariate_results).sort_values(by='bivariate_moran_I', ascending=False) if bivariate_results else pd.DataFrame()
        if not bivariate_df.empty:
            bivariate_df.to_csv(table_path, index=False)
            print(f"\n  - Saved all Bivariate Global Moran's I results to: {table_path.name}")

    if bivariate_df.empty:
        print("  - No valid Bivariate Moran's I results to analyze.")
        print(f"\n{'#'*30} Finished Bivariate Numerical Analysis {'#'*30}")
        return

    # Step 2: Create and save bar chart for ALL bivariate pairs
    plot_path_all_pairs = output_plots_dir / "bivariate_global_moran_I_barchart_all_pairs.png"
    if not _check_and_skip(plot_path_all_pairs, "Bivariate Moran's I bar chart for all pairs"):
        bivariate_df_to_plot = bivariate_df.copy()
        bivariate_df_to_plot['pair_label'] = bivariate_df_to_plot.apply(lambda row: f"{row['variable_1']} vs. {row['variable_2']}", axis=1)
        bivariate_df_to_plot = bivariate_df_to_plot.sort_values(by='bivariate_moran_I', ascending=True)

        # Dynamically adjust figure height based on number of pairs
        fig_height = max(10, len(bivariate_df_to_plot) * 0.05) # 0.05 inches per bar
        plt.figure(figsize=(14, fig_height))
        
        plt.barh(bivariate_df_to_plot['pair_label'], bivariate_df_to_plot['bivariate_moran_I'], color='lightcoral')
        plt.xlabel("Bivariate Global Moran's I")
        plt.ylabel("Variable Pair")
        plt.title("Bivariate Global Spatial Autocorrelation (All Pairs)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.yticks(fontsize=6) # Adjust font size for readability if many pairs
        plt.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(plot_path_all_pairs)
        plt.close()
        print(f"  - Saved Bivariate Moran's I bar chart for all pairs to: {plot_path_all_pairs.name}")

    # Step 2: Identify top pairs and run detailed local analysis ONLY on them
    highest_moran_pairs = bivariate_df.head(top_n)
    lowest_moran_pairs = bivariate_df.tail(top_n)
    top_pairs_for_analysis = pd.concat([highest_moran_pairs, lowest_moran_pairs])

    print(f"\n  - Top {top_n} pairs with highest spatial correlation: {[f'{r.variable_1} & {r.variable_2} (I_bv={r.bivariate_moran_I:.4f})' for r in highest_moran_pairs.itertuples()]}")
    print(f"  - Top {top_n} pairs with lowest spatial correlation: {[f'{r.variable_1} & {r.variable_2} (I_bv={r.bivariate_moran_I:.4f})' for r in lowest_moran_pairs.itertuples()]}")

    processed_pairs = set()
    for _, row in tqdm(top_pairs_for_analysis.iterrows(), total=len(top_pairs_for_analysis), desc="  - Detailed Bivariate Analysis"):
        var1, var2 = row['variable_1'], row['variable_2']
        pair_tuple = tuple(sorted((var1, var2)))
        if pair_tuple not in processed_pairs:
            analyze_bivariate_detailed(gdf, weights, var1, var2, output_plots_dir)
            processed_pairs.add(pair_tuple)

    print(f"\n{'#'*30} Finished Bivariate Numerical Analysis {'#'*30}")

def main():
    """
    Main function to execute the full analysis pipeline.
    """
    # --- 1. Load Data ---
    gpkg_path = GRID_PATH / "h310_grid_final_datacube.gpkg"
    gdf = load_data(gpkg_path)
    if gdf is None:
        return

    # --- 2. Define Variables & Enforce Types ---
    numerical_variables = [
        'total_intervention_cost_1ring', 'max_peak_elevation_1ring', 'active_fountains_count_1ring',
        'total_fountains_count_1ring', 'landslide_point_count_1ring', 'seismic_event_count_1ring',
        'avg_seismic_magnitude_1ring', 'road_density_m_per_m2','dist_to_waterway_m', 'avg_descending_soil_speed', 
        'avg_ascending_soil_speed', 'census_pop', 'epr_NTAXP',
        'epr_TAXABINC', 'epr_CADINCR', 'epr_CADINCF', 'epr_SUBEMPTR', 'epr_PENSINCR', 'epr_PENSINCF', 
        'epr_ENTROAIN', 'epr_ENTROAIN01', 'erd_E0_10000', 'erd_E10000_1', 'erd_E15000_2', 'erd_E26000_5', 
        'erd_E55000_7', 'erd_E75000_1', 'erd_E_GE1200', 'edst_acq_imm', 'edst_acq_erog', 'eidx_COMP_FRA', 
        'eidx_LAND_CON', 'eidx_EMPL_RAT', 'eidx_POP_25_6', 'eidx_POP_DEPE', 'eidx_INDEX_AC', 'eidx_PERSEMP', 
        'inflow_total', 'nearest_hydro_distance_m', 'nearest_hydro_river_stage_max_m', 'nearest_hydro_river_stage_mean_m', 
        'idw_temp_thermal_range', 'idw_rain_mm_sum_annual','idw_rain_mm_avg_monthly', 'idw_wind_speed_max_95p', 
        'idw_wind_speed_avg_mean', 'dtmidcnt_mean',
    ]

    categorical_variables = [
        'class_intervention_1ring', 'dominant_highway', 'dominant_surface',
        'dominant_tunnel', 'dominant_bridge', 'usda_hydrologic_group', 'pai_landslide_hazard_level',
        'dominant_building_1', 'dominant_building_2', 'dominant_building_3'
    ]

    categorical_int_variables = [
        'tipo_movimento_<lambda_0>_1ring', 'is_locality_1ring', 'rooting_depth_class', 'surface_stoniness_class',
        'landslide_surface_class', 'descending_soil_presence', 'ascending_soil_presence', 'hydraulic_hazard_level'
    ]

    numerical_vars, categorical_vars = enforce_data_types_and_get_vars(
        gdf, numerical_variables, categorical_variables, categorical_int_variables
    )

    if not numerical_vars and not categorical_vars:
        print("\nNo variables found to analyze after separation. Exiting.")
        return

    all_categorical_variables = categorical_variables + categorical_int_variables
    print(f"\nFound {len(numerical_variables)} numerical variables to analyze:")
    print(numerical_variables)
    print(f"\nFound {len(all_categorical_variables)} categorical variables to analyze:")
    print(all_categorical_variables)

    # --- 3. Ensure output directory exists ---
    PLOTS_SPATIAL_AUTOCORRELATION_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots will be saved to: {PLOTS_SPATIAL_AUTOCORRELATION_PATH}")
    TABLE_SPATIAL_AUTOCORRELATION_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Tables will be saved to: {TABLE_SPATIAL_AUTOCORRELATION_PATH}")

    # --- 4. Create Spatial Weights Matrix (once) ---
    weights = create_weights(gdf)

    # --- 5. Run Analysis Pipelines ---
    run_univariate_numerical_pipeline(gdf, weights, numerical_vars, PLOTS_SPATIAL_AUTOCORRELATION_PATH, TABLE_SPATIAL_AUTOCORRELATION_PATH)
    run_bivariate_numerical_pipeline(gdf, weights, numerical_vars, PLOTS_SPATIAL_AUTOCORRELATION_PATH, TABLE_SPATIAL_AUTOCORRELATION_PATH)

    # --- 6. Categorical Variable Pipeline ---
    if categorical_vars:
        print(f"\n{'#'*30} Starting Categorical Variable Analysis {'#'*30}")
        # Run the global association analysis first to get the top variables
        top_categorical_vars = analyze_global_categorical_association(gdf, weights, categorical_vars, TABLE_SPATIAL_AUTOCORRELATION_PATH)

        if top_categorical_vars:
            print(f"\n--- Proceeding with detailed analysis for the top {len(top_categorical_vars)} variables by Cramér's V ---")
            # Univariate Join-Counts analysis for each of the top variables
            for variable in top_categorical_vars:
                print(f"\n--- Analyzing Univariate Join-Counts for: {variable} ---")
                analyze_categorical_join_counts(gdf, weights, variable, TABLE_SPATIAL_AUTOCORRELATION_PATH)

            # Bivariate SPATIAL association analysis between the top variables
            top_10_pairs = analyze_bivariate_categorical_association(gdf, weights, top_categorical_vars, TABLE_SPATIAL_AUTOCORRELATION_PATH)
            print(f"\n--- Top 10 most spatially associated pairs for future local analysis: ---")
            print(top_10_pairs)
        else:
            print("\nNo top categorical variables found to analyze.")

        print(f"\n{'#'*30} Finished Categorical Variable Analysis {'#'*30}")
    else:
        print(f"\n{'#'*30} No Categorical Variables for Analysis {'#'*30}")

    # --- 7. Multivariate Clustering (Combined Numerical + Categorical) ---
    # Only run if there are any variables to cluster
    if numerical_vars or categorical_vars:
        print(f"\n{'#'*30} Starting Multivariate Clustering Analysis {'#'*30}")
        analyze_multivariate_clusters(gdf, numerical_vars, categorical_vars, PLOTS_SPATIAL_AUTOCORRELATION_PATH, n_clusters=10)
    else:
        print(f"\n{'#'*30} No Variables for Multivariate Clustering {'#'*30}")

    print(f"\n{'='*20} Pipeline Complete {'='*20}")


if __name__ == "__main__":
    main()