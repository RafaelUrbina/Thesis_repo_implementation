"""
General pipeline for spatial autocorrelation analysis on a static spatial dataset.

This script loads a geospatial dataset and performs both global and local
spatial autocorrelation tests to identify clustering patterns.

Steps:
1.  Load the final aggregated H3 grid data from the GeoPackage file.
2.  Define a variable of interest for the analysis.
3.  Create a spatial weights matrix to define neighborhood relationships.
4.  Calculate Global Moran's I to test for overall spatial clustering.
5.  Calculate Local Moran's I (LISA) to identify the location of specific
    hotspots, coldspots, and spatial outliers.
6.  Generate plots to visualize the results, including a Moran scatterplot and a
    LISA cluster map.
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
from splot.esda import moran_scatterplot, lisa_cluster
from itertools import combinations
from tqdm.auto import tqdm # Import tqdm for progress bars
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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

    # Create a working copy for this variable to avoid side effects
    y = gdf[variable_name].copy()

    # Define plot path for Moran Scatterplot
    moran_plot_path = output_dir / f"moran_plot_{variable_name}.png"

    if moran_plot_path.exists() and moran_plot_path.stat().st_size > 0:
            print(f"  - Plot '{moran_plot_path.name}' already exists. Skipping Moran scatterplot.")
    else:
        # For this analysis, we'll fill missing values with the mean.
        # You might consider other strategies like median or interpolation depending on your data.
        if y.isnull().any():
            mean_val = y.mean()
            y.fillna(mean_val, inplace=True)
            print(f"Warning: Missing values found. Filled with mean value ({mean_val:.2f}).")
    
        # Check for zero variance, which makes autocorrelation analysis impossible
        if y.std() == 0:
            print("Skipping: This variable has zero variance (all values are the same).")
            # We still need 'y' for the LISA part, so we don't return here if only the scatterplot exists
    
        moran = Moran(y, weights)
    
        # Define plot path for Moran Scatterplot
        moran_plot_path = output_dir / f"moran_plot_{variable_name}.png"
    
        
        # Plot the Moran Scatterplot
        fig, ax = moran_scatterplot(moran, aspect_equal=True)
        title = f"Global Moran's I for {variable_name}\nI={moran.I:.4f}, p-value=({moran.p_sim:.4f})"
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(moran_plot_path)
        plt.close(fig) # Close the figure to free up memory
        print(f"  - Saved Moran scatterplot to: {moran_plot_path.name}")

    # Define plot path for LISA Cluster Map
    lisa_plot_path = output_dir / f"lisa_cluster_map_{variable_name}.png"
    if lisa_plot_path.exists() and lisa_plot_path.stat().st_size > 0:
                print(f"  - Plot '{lisa_plot_path.name}' already exists. Skipping LISA cluster map.")
    else:
        # Re-calculate y if it wasn't calculated for the Moran plot
        if 'y' not in locals():
            y = gdf[variable_name].copy()
            if y.isnull().any():
                y.fillna(y.mean(), inplace=True)
        
        if y.std() == 0:
            print("Skipping LISA: This variable has zero variance.")
            return

        lisa = Moran_Local(y, weights)

        # Plot the LISA Cluster Map
        fig, ax = plt.subplots(figsize=(12, 10))
        lisa_cluster(lisa, gdf, ax=ax, legend=True)
        ax.set_title(f'Local Moran\'s I (LISA) for {variable_name}')
        ax.set_yticklabels([])
        ax.set_xticklabels([]) 
        plt.tight_layout()
        # Save the figure
        plt.savefig(lisa_plot_path)
        plt.close(fig) # Close the figure to free up memory
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

    bv_plot_path = output_dir / f"bivariate_lisa_{var1}_vs_{var2}.png"
    if bv_plot_path.exists() and bv_plot_path.stat().st_size > 0:
        print(f"  - Plot '{bv_plot_path.name}' already exists. Skipping.")
        return

    y1 = gdf[var1].copy()
    y2 = gdf[var2].copy()

    # Impute NaNs if necessary
    if y1.isnull().any(): y1.fillna(y1.mean(), inplace=True)
    if y2.isnull().any(): y2.fillna(y2.mean(), inplace=True)

    # Skip if either variable has zero variance
    if y1.std() == 0 or y2.std() == 0:
        print("  - Skipping pair: At least one variable has zero variance.")
        return

    # Tests if y1 at location i is correlated with y2 in neighbor locations
    bivariate_lisa = Moran_Local_BV(y1, y2, weights)

    # Plot the LISA Cluster Map for the bivariate case
    fig, ax = plt.subplots(figsize=(12, 10))
    lisa_cluster(bivariate_lisa, gdf, ax=ax, legend=True)
    ax.set_title(f'Bivariate LISA: {var1} vs. {var2}')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()

    # Save the figure
    plt.savefig(bv_plot_path)
    plt.close(fig)
    print(f"  - Saved Bivariate LISA cluster map to: {bv_plot_path.name}")

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
    table_path = table_output_dir / f"join_counts_{variable_name}.csv"
    # Check if the output table already exists and is not empty
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"  - Table '{table_path.name}' already exists. Skipping Join-Count analysis for '{variable_name}'.")
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

def analyze_spatial_contingency(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variable_name: str, table_output_dir: Path):
    """
    Analyzes spatial contingency for a single categorical variable by cross-tabulating
    the focal region's category against its neighbors' categories.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variable_name (str): The name of the categorical column to analyze.
        table_output_dir (Path): The directory to save the output table.
    """
    table_path = table_output_dir / f"spatial_contingency_{variable_name}.csv"
    # Check if the output table already exists and is not empty
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"  - Table '{table_path.name}' already exists. Skipping Spatial Contingency analysis for '{variable_name}'.")
        return

    print(f"\n{'='*20} Analyzing Categorical (Spatial Contingency): {variable_name} {'='*20}")

    temp_gdf = gdf[[variable_name]].copy()
    temp_gdf = temp_gdf.dropna(subset=[variable_name]) # Drop NaNs for factorize

    if temp_gdf.empty:
        print(f"Skipping: Categorical variable '{variable_name}' has no non-null values.")
        return

    codes, uniques = pd.factorize(temp_gdf[variable_name])
    temp_gdf['cat_code'] = codes

    # Calculate spatial lag of codes (mean of neighbors' codes)
    # This will be float, so we round it to get discrete neighbor categories
    # Note: lag_spatial expects a Series aligned with the weights object.
    # We need to ensure the index of temp_gdf matches the weights index.
    if not temp_gdf.index.equals(pd.Series(weights.id_order).index): # Check if indices match
        # If not, reindex temp_gdf to match weights.id_order
        # This is a common issue if rows were dropped or reordered.
        temp_gdf = temp_gdf.reindex(weights.id_order)
        codes, uniques = pd.factorize(temp_gdf[variable_name]) # Refactorize after reindexing
        temp_gdf['cat_code'] = codes
        temp_gdf = temp_gdf.dropna(subset=['cat_code']) # Drop NaNs again if reindexing introduced them

    if temp_gdf.empty:
        print(f"Skipping: Categorical variable '{variable_name}' has no non-null values after reindexing.")
        return

    # Ensure the series passed to lag_spatial is aligned with the weights
    y_for_lag = temp_gdf['cat_code'].reindex(weights.id_order).fillna(-1) # Fill NaNs with a placeholder if needed

    # Only include observations that are part of the weights matrix
    # This is important if some geometries were dropped due to no neighbors
    valid_indices = y_for_lag[y_for_lag != -1].index
    
    if valid_indices.empty:
        print(f"Skipping: No valid observations for '{variable_name}' to calculate spatial lag.")
        return

    # lag_spatial returns a numpy array. Convert it to a pandas Series with the correct index.
    neighbor_codes_raw_series = pd.Series(lag_spatial(weights, y_for_lag), index=y_for_lag.index)
    
    # Filter to only valid indices before rounding and crosstab
    neighbor_codes_filtered = neighbor_codes_raw_series.loc[valid_indices]
    neighbor_codes = np.round(neighbor_codes_filtered).astype(int)
    focal_codes = temp_gdf['cat_code'].loc[valid_indices].astype(int)

    # Cross-tabulate focal region vs. neighbor region
    # Use the original unique categories for row/column names for readability
    contingency = pd.crosstab(focal_codes, neighbor_codes)
    
    # Map codes back to original category names for display
    contingency.index = [uniques[i] for i in contingency.index]
    contingency.columns = [uniques[i] for i in contingency.columns]

    print(f"Spatial Contingency Table for {variable_name} (Focal vs. Neighbor):")
    print(contingency)
    print("\nInterpretation: Rows are focal unit categories, columns are neighbor categories.")
    print("Values indicate counts of how often a focal category is adjacent to a neighbor category.")

    if not contingency.empty: # Only save if there are actual results
        contingency.to_csv(table_path)
        print(f"  - Saved Spatial Contingency table to: {table_path}")


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
    # Check if the output plot already exists and is not empty
    if plot_path.exists() and plot_path.stat().st_size > 0:
        print(f"  - Plot '{plot_path.name}' already exists. Skipping Multivariate Clustering analysis.")
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

def main():
    """
    Main function to execute the full analysis pipeline.
    """
    # --- 1. Load Data ---
    gpkg_path = GRID_PATH / "h310_grid_final_datacube.gpkg"
    gdf = load_data(gpkg_path)
    if gdf is None:
        return

    # --- 2. Identify Numeric Variables to Analyze ---
    # --- Manual Variable Definition ---
    # Manually define your variable lists here.
    # Any column not in these lists or 'geometry' will be ignored.
    
    numerical_variables = [
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
        'idw_wind_speed_max_95p', 'idw_wind_speed_avg_mean', 'dtmidcnt_mean', 'dtmidcnt_max'
    ]

    categorical_variables = [
        'class_intervention_1ring', 'locality_name_1ring', 'dominant_highway', 'dominant_surface',
        'dominant_tunnel', 'dominant_bridge', 'usda_hydrologic_group', 'pai_landslide_hazard_level',
        'dominant_building_1', 'dominant_building_2', 'dominant_building_3'
    ]

    categorical_int_variables = [
        'tipo_movimento_<lambda_0>_1ring', 'is_locality_1ring', 'rooting_depth_class', 'surface_stoniness_class',
        'landslide_surface_class', 'descending_soil_presence', 'ascending_soil_presence', 'hydraulic_hazard_level'
    ]

    # Combine all categorical variables for analysis functions
    all_categorical_variables = categorical_variables + categorical_int_variables

    # --- Data Type Sanity Check and Enforcement ---
    print("\n--- Enforcing Data Types ---")
    for var in numerical_variables:
        if var in gdf.columns:
            # Check for errors during conversion by seeing if NaNs are introduced
            initial_nans = gdf[var].isnull().sum()
            # Convert to numeric, coercing errors to NaN
            gdf[var] = pd.to_numeric(gdf[var], errors='coerce')
            final_nans = gdf[var].isnull().sum()
            if final_nans > initial_nans:
                print(f"  - Warning: Errors found in variable '{var}' while converting to numeric. Non-numeric values were set to NaN.")
    print("Numerical variables converted to numeric types.")

    for var in categorical_variables:
        if var in gdf.columns:
            # .astype(str) is a robust conversion and typically does not produce errors,
            # as it can represent any value as a string.
            gdf[var] = gdf[var].astype(str)
    print("String-based categorical variables converted to string types.")

    for var in categorical_int_variables:
        if var in gdf.columns:
            # Check for errors during the initial numeric conversion
            initial_nans = gdf[var].isnull().sum()
            gdf[var] = pd.to_numeric(gdf[var], errors='coerce')
            final_nans = gdf[var].isnull().sum()
            if final_nans > initial_nans:
                print(f"  - Warning: Errors found in variable '{var}' while converting to integer. Non-numeric values were set to NaN.")
            # Using Int64 (capital I) to allow for NaNs in integer columns
            gdf[var] = gdf[var].astype('Int64')
    print("Integer-based categorical variables converted to nullable integer types.")
    print("--- Data Type Enforcement Complete ---")

    # Filter out variables that don't exist in the dataframe to prevent errors
    numerical_variables = [v for v in numerical_variables if v in gdf.columns]
    all_categorical_variables = [v for v in all_categorical_variables if v in gdf.columns]

    if not numerical_variables and not all_categorical_variables:
        print("\nNo variables found to analyze after separation. Exiting.")
        return

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

    # --- 5. Numerical Variable Pipeline ---
    """ if numerical_variables:
        print(f"\n{'#'*30} Starting Univariate Numerical Analysis {'#'*30}")
        
        # Step 1: Calculate Global Moran's I for all variables
        moran_results = []
        print("\n  - Calculating Global Moran's I for all numerical variables...")
        for var in tqdm(numerical_variables, desc="  - Univariate Global Moran's I"):
            y = gdf[var].copy()
            if y.isnull().any():
                y.fillna(y.mean(), inplace=True)
            if y.std() > 0:
                moran = Moran(y, weights)
                moran_results.append({'variable': var, 'moran_I': moran.I, 'p_value': moran.p_sim})
                print(f"    - Variable: {var}, Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")
            else:
                print(f"    - Skipping '{var}': Zero variance.")
        
        if moran_results:
            moran_df = pd.DataFrame(moran_results).sort_values(by='moran_I', ascending=False)
            
            # Save results to CSV
            table_path = TABLE_SPATIAL_AUTOCORRELATION_PATH / "univariate_global_moran_I_results.csv"
            moran_df.to_csv(table_path, index=False)
            print(f"\n  - Saved all Univariate Global Moran's I results to: {table_path.name}")

            # Create and save bar chart
            plot_path = PLOTS_SPATIAL_AUTOCORRELATION_PATH / "univariate_global_moran_I_barchart.png"
            plt.figure(figsize=(12, max(8, len(moran_df) * 0.3)))
            plt.barh(moran_df['variable'], moran_df['moran_I'], color='skyblue')
            plt.xlabel("Global Moran's I")
            plt.title("Univariate Global Spatial Autocorrelation")
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"  - Saved Moran's I summary bar chart to: {plot_path.name}")

            # Step 2: Identify top 6 highest and lowest Moran's I variables
            top_n = 6
            highest_moran_vars = moran_df.head(top_n)
            lowest_moran_vars = moran_df.tail(top_n)

            print(f"\n  - Top {top_n} variables with highest positive autocorrelation:")
            for _, row in highest_moran_vars.iterrows():
                print(f"    - {row['variable']} (I={row['moran_I']:.4f})")
            
            print(f"\n  - Top {top_n} variables with most negative autocorrelation:")
            for _, row in lowest_moran_vars.iterrows():
                print(f"    - {row['variable']} (I={row['moran_I']:.4f})")

            # Step 3: Run detailed analysis on these selected variables, avoiding duplicates
            processed_vars = set()
            print(f"\n  - Running detailed analysis for top {top_n} highest Moran's I variables:")
            for _, row in highest_moran_vars.iterrows():
                if row['variable'] not in processed_vars:
                    analyze_variable_detailed(gdf, weights, row['variable'], PLOTS_SPATIAL_AUTOCORRELATION_PATH)
                    processed_vars.add(row['variable'])
            
            print(f"\n  - Running detailed analysis for top {top_n} lowest Moran's I variables:")
            for _, row in lowest_moran_vars.iterrows():
                if row['variable'] not in processed_vars:
                    analyze_variable_detailed(gdf, weights, row['variable'], PLOTS_SPATIAL_AUTOCORRELATION_PATH)
                    processed_vars.add(row['variable'])

        else:
            print("  - No valid Moran's I results to analyze.")

        print(f"\n{'#'*30} Finished Univariate Numerical Analysis {'#'*30}")

        # --- Define Bivariate Variables of Interest ---
        # This list explicitly defines the variables for which bivariate Moran's I will be calculated.
        # This helps to focus the analysis and reduce computational load and output clutter.
        bivariate_vars_subset = [
            'total_intervention_cost_1ring', 'max_peak_elevation_1ring', 'landslide_point_count_1ring',
            'seismic_event_count_1ring', 'max_seismic_magnitude_1ring', 'avg_seismic_magnitude_1ring',
            'road_density_m_per_m2', 'dist_to_waterway_m', 'avg_descending_soil_speed',
            'avg_ascending_soil_speed', 'census_pop', 'idw_temp_max_peak', 'idw_temp_min_nadir',
            'idw_rain_mm_sum_annual', 'idw_rain_mm_max_monthly', 'idw_rain_mm_min_monthly',
            'idw_rain_mm_avg_monthly', 'idw_rain_mm_std', 'idw_wind_speed_max_95p',
            'idw_wind_speed_avg_mean', 'inflow_total', 'dtmidcnt_mean', 'hydraulic_hazard_level'
        ]

        # Filter the subset to only include variables that are present in the GeoDataFrame
        bivariate_vars_to_analyze = [v for v in bivariate_vars_subset if v in gdf.columns]
        print(f"\n  - Selected {len(bivariate_vars_to_analyze)} variables for bivariate analysis from the predefined subset.")

        # --- Bivariate Spatial Analysis Pipeline ---
        print(f"\n{'#'*30} Starting Bivariate Spatial Analysis {'#'*30}")
        if len(bivariate_vars_to_analyze) >= 2:
            # Step 1: Calculate Global Bivariate Moran's I for all pairs
            bivariate_results = []
            print("\n  - Calculating Global Bivariate Moran's I for all numerical pairs...")
            
            # Standardize all variables first for efficiency and correct calculation
            gdf_std = gdf[bivariate_vars_to_analyze].copy()
            for col in gdf_std.columns:
                if gdf_std[col].isnull().any():
                    gdf_std[col].fillna(gdf_std[col].mean(), inplace=True)
                if gdf_std[col].std() > 0:
                    gdf_std[col] = (gdf_std[col] - gdf_std[col].mean()) / gdf_std[col].std()
                else: # Handle zero variance columns
                    gdf_std[col] = 0 

            for var1, var2 in tqdm(list(combinations(bivariate_vars_to_analyze, 2)), desc="  - Bivariate Global Moran's I"):
                z1 = gdf_std[var1]
                z2 = gdf_std[var2]
                moran_bv = Moran_BV(z1, z2, weights)
                bivariate_results.append({'variable_1': var1, 'variable_2': var2, 'bivariate_moran_I': moran_bv.I, 'p_value': moran_bv.p_sim})
                print(f"    - Pair: {var1} & {var2}, Bivariate Moran's I: {moran_bv.I:.4f}, p-value: {moran_bv.p_sim:.4f}")

            if bivariate_results:
                bivariate_df = pd.DataFrame(bivariate_results).sort_values(by='bivariate_moran_I', ascending=False)
                
                # Save all results to CSV
                bivariate_table_path = TABLE_SPATIAL_AUTOCORRELATION_PATH / "bivariate_global_moran_I_results.csv"
                bivariate_df.to_csv(bivariate_table_path, index=False)
                print(f"\n  - Saved all Bivariate Global Moran's I results to: {bivariate_table_path.name}")

                # Step 2: Identify top 6 highest and lowest Bivariate Moran's I pairs
                top_n = 6
                highest_moran_pairs = bivariate_df.head(top_n)
                lowest_moran_pairs = bivariate_df.tail(top_n)

                print(f"\n  - Top {top_n} pairs with highest positive spatial correlation:")
                for _, row in highest_moran_pairs.iterrows():
                    print(f"    - {row['variable_1']} & {row['variable_2']} (I_bv={row['bivariate_moran_I']:.4f})")
                
                print(f"\n  - Top {top_n} pairs with most negative spatial correlation:")
                for _, row in lowest_moran_pairs.iterrows():
                    print(f"    - {row['variable_1']} & {row['variable_2']} (I_bv={row['bivariate_moran_I']:.4f})")

                # Step 3: Run detailed bivariate spatial analysis on these selected pairs, avoiding duplicates
                processed_pairs = set()
                print(f"\n  - Running detailed bivariate analysis for top {top_n} highest Bivariate Moran's I pairs:")
                for _, row in highest_moran_pairs.iterrows():
                    pair_tuple = tuple(sorted((row['variable_1'], row['variable_2'])))
                    if pair_tuple not in processed_pairs:
                        analyze_bivariate_detailed(gdf, weights, row['variable_1'], row['variable_2'], PLOTS_SPATIAL_AUTOCORRELATION_PATH)
                        processed_pairs.add(pair_tuple)
                
                print(f"\n  - Running detailed bivariate analysis for top {top_n} lowest Bivariate Moran's I pairs:")
                for _, row in lowest_moran_pairs.iterrows():
                    pair_tuple = tuple(sorted((row['variable_1'], row['variable_2'])))
                    if pair_tuple not in processed_pairs:
                        analyze_bivariate_detailed(gdf, weights, row['variable_1'], row['variable_2'], PLOTS_SPATIAL_AUTOCORRELATION_PATH)
                        processed_pairs.add(pair_tuple)
        else:
            print("\n  - Skipping Bivariate Analysis: Fewer than 2 numerical variables available.")
        print(f"\n{'#'*30} Finished Bivariate Spatial Analysis {'#'*30}")
    else:
        print(f"\n{'#'*30} No Numerical Variables for Analysis {'#'*30}") """


    # --- 6. Categorical Variable Pipeline ---
    if all_categorical_variables:
        print(f"\n{'#'*30} Starting Categorical Variable Analysis {'#'*30}")
        for variable in all_categorical_variables:
            print(f"\nAnalyzing Categorical Variable: {variable}")
            analyze_categorical_join_counts(gdf, weights, variable, TABLE_SPATIAL_AUTOCORRELATION_PATH)
            analyze_spatial_contingency(gdf, weights, variable, TABLE_SPATIAL_AUTOCORRELATION_PATH)
        print(f"\n{'#'*30} Finished Categorical Variable Analysis {'#'*30}")
    else:
        print(f"\n{'#'*30} No Categorical Variables for Analysis {'#'*30}")

    # --- 7. Multivariate Clustering (Combined Numerical + Categorical) ---
    # Only run if there are any variables to cluster
    if numerical_variables or all_categorical_variables:
        print(f"\n{'#'*30} Starting Multivariate Clustering Analysis {'#'*30}")
        analyze_multivariate_clusters(gdf, numerical_variables, all_categorical_variables, PLOTS_SPATIAL_AUTOCORRELATION_PATH)
    else:
        print(f"\n{'#'*30} No Variables for Multivariate Clustering {'#'*30}")

    print(f"\n{'='*20} Pipeline Complete {'='*20}")


if __name__ == "__main__":
    main()