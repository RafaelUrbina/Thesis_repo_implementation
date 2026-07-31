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
from esda.moran import Moran, Moran_Local, Moran_Local_BV
from splot.esda import moran_scatterplot, lisa_cluster
from itertools import combinations
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

def analyze_variable(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variable_name: str, output_dir: Path): # Renamed for clarity
    """
    Runs the spatial autocorrelation analysis for a single variable.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variable_name (str): The name of the column to analyze.
        output_dir (Path): The directory to save the output plots.
    """
    print(f"\n{'='*20} Analyzing: {variable_name} {'='*20}")

    # Create a working copy for this variable to avoid side effects
    y = gdf[variable_name].copy()

    # For this analysis, we'll fill missing values with the mean.
    # You might consider other strategies like median or interpolation depending on your data.
    if y.isnull().any():
        mean_val = y.mean()
        y.fillna(mean_val, inplace=True)
        print(f"Warning: Missing values found. Filled with mean value ({mean_val:.2f}).")

    # Check for zero variance, which makes autocorrelation analysis impossible
    if y.std() == 0:
        print("Skipping: This variable has zero variance (all values are the same).")
        return

    # --- Global Spatial Autocorrelation (Moran's I) ---
    print(f"--- Calculating Global Moran's I ---")
    # Moran's I tells us if there is a general pattern of clustering globally.
    # I > 0: Positive autocorrelation (clustering of similar values)
    # I < 0: Negative autocorrelation (checkerboard pattern)
    # I ~ 0: Random pattern
    moran = Moran(y, weights)

    print(f"Moran's I: {moran.I:.4f}")
    print(f"P-value: {moran.p_sim:.4f}")
    print(f"Z-score: {moran.z_sim:.4f}")

    if moran.p_sim < 0.05:
        print("Global Result: The pattern is statistically significant (p < 0.05).")
        if moran.I > 0:
            print("The data exhibits positive spatial autocorrelation (clustering).")
        else:
            print("The data exhibits negative spatial autocorrelation (dispersion).")
    else:
        print("Global Result: The pattern is not statistically significant (p >= 0.05). We cannot reject the null hypothesis of spatial randomness.")

    # Plot the Moran Scatterplot
    fig, ax = moran_scatterplot(moran, aspect_equal=True)
    plt.suptitle(f"Global Moran's I for {variable_name}", fontsize=14)
    plt.tight_layout()
    
    # Save the figure instead of showing it
    moran_plot_path = output_dir / f"moran_plot_{variable_name}.png"
    plt.savefig(moran_plot_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"  - Saved Moran scatterplot to: {moran_plot_path}")
    # --- Local Spatial Autocorrelation (LISA) ---
    print(f"\n--- Calculating Local Moran's I (LISA) ---")
    # LISA helps us identify the specific locations of clusters and outliers.
    lisa = Moran_Local(y, weights)

    # Plot the LISA Cluster Map
    # It classifies each location into categories like High-High (hotspot), Low-Low (coldspot), etc.
    fig, ax = plt.subplots(figsize=(12, 10))
    lisa_cluster(lisa, gdf, ax=ax, legend=True)
    ax.set_title(f'Local Moran\'s I (LISA) for {variable_name}')
    ax.set_yticklabels([])
    ax.set_xticklabels([]) 
    plt.tight_layout()
    
    # Save the figure instead of showing it
    lisa_plot_path = output_dir / f"lisa_cluster_map_{variable_name}.png"
    plt.savefig(lisa_plot_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"  - Saved LISA cluster map to: {lisa_plot_path}")

def analyze_bivariate(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variables: list, output_dir: Path):
    """
    Runs Bivariate Local Moran's I analysis for pairs of variables.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        weights (libpysal.weights.W): The pre-computed spatial weights matrix.
        variables (list): A list of variable names to analyze in pairs.
        output_dir (Path): The directory to save the output plots.
    """
    print(f"\n{'='*20} Bivariate Numerical Analysis (LISA) {'='*20}")

    # Create a copy to work with
    gdf_bv = gdf[variables].copy()

    # Fill NaNs with the mean for each column
    for col in gdf_bv.columns:
        if gdf_bv[col].isnull().any():
            mean_val = gdf_bv[col].mean()
            gdf_bv[col].fillna(mean_val, inplace=True)
            print(f"Filled NaNs in '{col}' with mean ({mean_val:.2f}) for bivariate analysis.")

    # Analyze spatial relationships between pairs of variables
    for var1, var2 in combinations(variables, 2):
        print(f"\n--- Analyzing pair: {var1} vs. {var2} ---")

        # Skip if either variable has zero variance
        if gdf_bv[var1].std() == 0 or gdf_bv[var2].std() == 0:
            print("Skipping pair: At least one variable has zero variance.")
            continue

        # Tests if var1 at location i is correlated with var2 in neighbor locations
        bivariate_lisa = Moran_Local_BV(gdf_bv[var1], gdf_bv[var2], weights)

        # Plot the LISA Cluster Map for the bivariate case
        fig, ax = plt.subplots(figsize=(12, 10))
        lisa_cluster(bivariate_lisa, gdf, ax=ax, legend=True)
        ax.set_title(f'Bivariate LISA: {var1} vs. {var2}')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.tight_layout()

        # Save the figure
        bv_plot_path = output_dir / f"bivariate_lisa_{var1}_vs_{var2}.png"
        plt.savefig(bv_plot_path)
        plt.close(fig)
        print(f"  - Saved Bivariate LISA cluster map to: {bv_plot_path}")

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

    results_df = pd.DataFrame(results)
    table_path = table_output_dir / f"join_counts_{variable_name}.csv"
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

    # Save the contingency table to a CSV file
    table_path = table_output_dir / f"spatial_contingency_{variable_name}.csv"
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

    # 6. Plot the combined multivariate clusters on a map
    plot_path = output_dir / f"multivariate_kmeans_cluster_map_k{n_clusters}.png"
    
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
        # Add your numerical variable names here, e.g., 'rainfall_mm', 'temperature'
    ]

    categorical_variables = [
        # Add your string-based categorical variable names here, e.g., 'land_use'
    ]

    categorical_int_variables = [
        # Add your integer-based categorical variable names here, e.g., 'soil_type_code'
    ]

    # Combine all categorical variables for analysis functions
    all_categorical_variables = categorical_variables + categorical_int_variables

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
    if numerical_variables:
        print(f"\n{'#'*30} Starting Numerical Variable Analysis {'#'*30}")
        for variable in numerical_variables:
            analyze_variable(gdf, weights, variable, PLOTS_SPATIAL_AUTOCORRELATION_PATH)

        # Bivariate Analysis for numerical variables
        # To keep the number of plots manageable, let's select a few interesting variables for pairing.
        # You can expand this list or use `numerical_variables` for all combinations.
        # Ensure there are at least two variables for bivariate analysis
        bivariate_vars_subset = [v for v in numerical_variables]
        if len(bivariate_vars_subset) < 2 and len(numerical_variables) >= 2:
            # If subset is too small, just take the first two numerical variables
            bivariate_vars_subset = numerical_variables[:2]
        elif len(bivariate_vars_subset) < 2:
            print("\nSkipping Bivariate Numerical Analysis: Less than 2 suitable numerical variables found.")
            bivariate_vars_subset = [] # Ensure it's empty if not enough vars

        if len(bivariate_vars_subset) >= 2:
            #print(f"\nPerforming Bivariate Numerical Analysis on variables: {bivariate_vars_subset}")
            analyze_bivariate(gdf, weights, bivariate_vars_subset, PLOTS_SPATIAL_AUTOCORRELATION_PATH)
        print(f"\n{'#'*30} Finished Numerical Variable Analysis {'#'*30}")
    else:
        print(f"\n{'#'*30} No Numerical Variables for Analysis {'#'*30}")


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