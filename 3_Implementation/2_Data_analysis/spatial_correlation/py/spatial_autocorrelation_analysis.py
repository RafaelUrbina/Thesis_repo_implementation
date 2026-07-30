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

import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
import numpy as np
from esda.moran import Moran, Moran_Local, Moran_Local_BV
from splot.esda import moran_scatterplot, lisa_cluster
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Add the project root to the Python path to allow for absolute imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GRID_PATH, PLOTS_SPATIAL_AUTOCORRELATION_PATH

def load_data(gpkg_path: Path, layer: str) -> gpd.GeoDataFrame | None:
    """
    Loads the GeoPackage file into a GeoDataFrame.

    Args:
        gpkg_path (Path): The path to the GeoPackage file.

    Returns:
        A GeoDataFrame if successful, otherwise None.s
    """
    print(f"Loading data from: {gpkg_path}")
    try:
        # It's good practice to specify the layer, especially if more are added later
        gdf = gpd.read_file(gpkg_path, layer=layer)
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

def analyze_variable(gdf: gpd.GeoDataFrame, weights: libpysal.weights.W, variable_name: str, output_dir: Path):
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
    print(f"\n{'='*20} Bivariate Analysis (LISA) {'='*20}")

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

def analyze_multivariate_clusters(gdf: gpd.GeoDataFrame, variables: list, output_dir: Path, n_clusters: int = 5):
    """
    Performs non-spatial clustering on all variables and maps the results.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data.
        variables (list): The list of variable names to include in the clustering.
        output_dir (Path): The directory to save the output plot.
        n_clusters (int): The number of clusters to create.
    """
    print(f"\n{'='*20} Multivariate Clustering Analysis {'='*20}")

    # 1. Select and scale all variables together
    X = gdf[variables].fillna(gdf[variables].mean()) # Fill NaNs with column mean
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Cluster across all variables simultaneously
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf['multivariate_cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Plot the combined multivariate clusters on a map
    plot_path = output_dir / "multivariate_kmeans_cluster_map.png"
    gdf.plot(column='multivariate_cluster', categorical=True, legend=True, figsize=(12, 10), aspect='equal')
    plt.title(f'Multivariate K-Means Clusters (k={n_clusters})')
    plt.savefig(plot_path)
    plt.close()
    print(f"  - Saved Multivariate K-Means cluster map to: {plot_path}")

def main():
    """
    Main function to execute the full analysis pipeline.
    """
    # --- 1. Load Data ---
    gpkg_path = GRID_PATH / "h310_grid_final_datacube.gpkg"
    gdf = load_data(gpkg_path, layer="h310_grid_final_datacube")
    if gdf is None:
        return

    # --- 2. Identify Numeric Variables to Analyze ---
    # Exclude any known non-data columns like 'index' or other identifiers.
    # The 'geometry' column is special and will be ignored by select_dtypes.
    numeric_cols = gdf.select_dtypes(include=np.number).columns.tolist()
    
    # Define columns to explicitly exclude from analysis
    # The user mentioned the 'id' is the h3 index, which is often named 'index' or 'h3_index'
    # Let's exclude common index names and identifiers.
    cols_to_exclude = ['index', 'h3_index', 'h310_index'] 
    
    variables_to_analyze = [col for col in numeric_cols if col not in cols_to_exclude]

    if not variables_to_analyze:
        print("\nNo numeric variables found to analyze. Exiting.")
        return

    print(f"\nFound {len(variables_to_analyze)} numeric variables to analyze:")
    print(variables_to_analyze)

    # --- 3. Ensure output directory exists ---
    PLOTS_SPATIAL_AUTOCORRELATION_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots will be saved to: {PLOTS_SPATIAL_AUTOCORRELATION_PATH}")

    # --- 4. Create Spatial Weights Matrix (once) ---
    weights = create_weights(gdf)

    # --- 5. Univariate Analysis ---
    for variable in variables_to_analyze:
        analyze_variable(gdf, weights, variable, PLOTS_SPATIAL_AUTOCORRELATION_PATH)

    # --- 6. Bivariate Analysis ---
    # To keep the number of plots manageable, let's select a few interesting variables for pairing.
    # You can expand this list or use `variables_to_analyze` for all combinations.
    bivariate_vars = [v for v in variables_to_analyze if 'rain' in v or 'temp' in v or 'river' in v or 'wind' in v][:4] # Example subset
    if len(bivariate_vars) >= 2:
        analyze_bivariate(gdf, weights, bivariate_vars, PLOTS_SPATIAL_AUTOCORRELATION_PATH)

    # --- 7. Multivariate Clustering ---
    analyze_multivariate_clusters(gdf, variables_to_analyze, PLOTS_SPATIAL_AUTOCORRELATION_PATH)

    print(f"\n{'='*20} Pipeline Complete {'='*20}")


if __name__ == "__main__":
    main()