"""
Advanced multivariate spatial and feature-space analysis pipeline.

This script implements several advanced statistical methods to uncover
complex relationships between multiple variables in a geospatial dataset.
It addresses two primary goals:
1.  Finding which variables act together (feature-space analysis).
2.  Identifying where they act together (spatial-domain analysis).

The pipeline includes:
1.  **Factor Analysis of Mixed Data (FAMD)**: To identify latent factors
    that group both numerical and categorical variables.
2.  **Association Rule Mining (Apriori)**: To find co-occurrence rules
    among categorical variables (e.g., {Condition A, Condition B} -> {Outcome C}).
3.  **Spatially Constrained Clustering (SKATER)**: To create geographically
    contiguous regions of areas with similar multivariate profiles.
4.  **Geographically Weighted PCA (GWPCA)**: To explore how the correlation
    structure between numerical variables changes across space.
"""
#C:\Program Files\R\R-4.6.1
import os
import sys
from pathlib import Path

# --- R Environment Setup for rpy2 ---
# This block programmatically sets the R_HOME and PATH environment variables
# to ensure rpy2 can find the R installation and all its DLLs on Windows.
# This must be done *before* rpy2 is imported.
r_home = r"C:\Program Files\R\R-4.6.1"
os.environ["R_HOME"] = r_home
os.environ["PATH"] = f"{r_home}\\bin\\x64;" + os.environ["PATH"]

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import pandas as pd
import prince
from libpysal.weights import Queen
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Add the project root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import (
    GRID_PATH,
    PLOTS_SPATIAL_AUTOCORRELATION_PATH,
    TABLE_SPATIAL_AUTOCORRELATION_PATH,
)


def load_data(gpkg_path: Path) -> gpd.GeoDataFrame | None:
    """Loads the first layer from a GeoPackage file."""
    print(f"Loading data from: {gpkg_path}")
    try:
        layers = fiona.listlayers(gpkg_path)
        if not layers:
            print("Error: No layers found in the GeoPackage file.")
            return None
        gdf = gpd.read_file(gpkg_path, layer=layers[0])
        print(f"Data loaded successfully. Shape: {gdf.shape}")
        return gdf
    except Exception as e:
        print(f"Error loading GeoPackage file: {e}")
        return None


def flag_and_drop_sparse_columns(gdf: gpd.GeoDataFrame, threshold: float = 0.9) -> gpd.GeoDataFrame:
    """
    Identifies, flags, and drops columns with missing data above a threshold.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.
        threshold (float): The missing value ratio threshold (e.g., 0.9 for 90%).

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with sparse columns removed.
    """
    print(f"\n--- Flagging and Dropping Sparse Columns (Threshold > {threshold*100}%) ---")
    missing_fraction = gdf.isnull().sum() / len(gdf)
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()

    if cols_to_drop:
        print(f"  - Found {len(cols_to_drop)} columns with more than {threshold*100}% missing data:")
        for col in cols_to_drop:
            print(f"    - {col} ({missing_fraction[col]:.2%})")
        
        print("  - Dropping these columns...")
        gdf = gdf.drop(columns=cols_to_drop)
        print(f"  - Columns dropped. New shape: {gdf.shape}")
    else:
        print("  - No columns found exceeding the missing data threshold.")
    return gdf


def enforce_data_types(
    gdf: gpd.GeoDataFrame,
    numerical_vars: list,
    categorical_vars: list,
    categorical_int_vars: list,
) -> tuple[gpd.GeoDataFrame, list, list]:
    """Enforces data types and returns a cleaned GeoDataFrame and variable lists."""
    print("\n--- Enforcing Data Types ---")
    for var in numerical_vars:
        if var in gdf.columns:
            initial_nans = gdf[var].isnull().sum()
            gdf[var] = pd.to_numeric(gdf[var], errors='coerce')
            final_nans = gdf[var].isnull().sum()
            if final_nans > initial_nans:
                print(f"  - Warning: Errors found in variable '{var}' while converting to numeric. Non-numeric values were set to NaN.")
    print("Numerical variables converted to numeric types.")

    for var in categorical_vars:
        if var in gdf.columns:
            gdf[var] = gdf[var].astype(str)
    print("String-based categorical variables converted to string types.")

    for var in categorical_int_vars:
        if var in gdf.columns:
            initial_nans = gdf[var].isnull().sum()
            gdf[var] = pd.to_numeric(gdf[var], errors='coerce')
            final_nans = gdf[var].isnull().sum()
            if final_nans > initial_nans:
                print(f"  - Warning: Errors found in variable '{var}' while converting to integer. Non-numeric values were set to NaN.")
            gdf[var] = gdf[var].astype('Int64')
    print("Integer-based categorical variables converted to nullable integer types.")
    print("--- Data Type Enforcement Complete ---")

    # Combine all categorical variables for analysis functions
    all_categorical_variables = categorical_vars + categorical_int_vars

    # Filter lists to only include columns that exist in the GDF
    valid_numerical = [v for v in numerical_vars if v in gdf.columns]
    valid_categorical = [v for v in all_categorical_variables if v in gdf.columns]

    return gdf, valid_numerical, valid_categorical


def analyze_famd(
    gdf: gpd.GeoDataFrame,
    numerical_vars: list,
    categorical_vars: list,
    output_dir: Path,
):
    """
    Performs Factor Analysis of Mixed Data (FAMD) to find latent components
    grouping both numerical and categorical variables.
    """
    print("\n--- 1. Running Factor Analysis of Mixed Data (FAMD) ---")
    all_vars = numerical_vars + categorical_vars
    famd_data = gdf[all_vars].copy()

    # Impute missing values for FAMD
    print("  - Imputing missing values for FAMD...")
    for col in numerical_vars:
        if col in famd_data.columns:
            famd_data[col] = famd_data[col].fillna(famd_data[col].mean())
    for col in categorical_vars:
        if col in famd_data.columns:
            # Convert to string and fill NaN
            famd_data[col] = famd_data[col].astype(str).fillna("Missing")

    if famd_data.empty:
        print("Skipping FAMD: No data left after imputation.")
        return

    famd = prince.FAMD(n_components=5, n_iter=3, random_state=42)
    famd = famd.fit(famd_data)

    # Plot the contribution of each variable to the first two components
    # The prince.FAMD.plot method returns an Altair chart.
    # We can customize it for better readability.
    base_chart = famd.plot(
        famd_data, 
        x_component=0, 
        y_component=1,
        show_row_labels=False,
        show_column_labels=True
    )

    # Customize the chart for better readability
    chart = base_chart.properties(
        width=2000,  # Increase width
        height=1600, # Increase height
        title="FAMD: Variable Factor Map & Sample Coordinates"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=20
    )

    plot_path = output_dir / "famd_row_coordinates.png"
    chart.save(plot_path, scale_factor=3.0) # Increase resolution
    print(f"  - Saved FAMD row coordinates plot to: {plot_path.name}")

    # Get and save variable contributions
    contributions = famd.column_contributions_
    contributions_path = TABLE_SPATIAL_AUTOCORRELATION_PATH / "famd_variable_contributions.csv"
    contributions.to_csv(contributions_path)
    print(f"  - Saved variable contributions to: {contributions_path.name}")


def analyze_association_rules(
    gdf: gpd.GeoDataFrame, categorical_vars: list, table_output_dir: Path
):
    """
    Performs Association Rule Mining (Apriori) on categorical data to find
    co-occurrence patterns.
    """
    print("\n--- 2. Running Association Rule Mining (Apriori) ---")
    # Discretize high-cardinality categorical variables for meaningful rules
    transactions_df = gdf[categorical_vars].copy()

    # Impute missing values for Apriori
    print("  - Imputing missing values for Apriori...")
    for col in transactions_df.columns:
        # Convert to string and fill NaN
        transactions_df[col] = transactions_df[col].astype(str).fillna("Missing")

    for col in transactions_df.columns:
        if transactions_df[col].nunique() > 20:  # Example threshold
            try:
                # Bin numerical-like categoricals
                transactions_df[col] = pd.qcut(
                    pd.to_numeric(transactions_df[col]),
                    q=4,
                    duplicates="drop",
                    labels=[f"{col}_Q1", f"{col}_Q2", f"{col}_Q3", f"{col}_Q4"],
                )
            except (ValueError, TypeError):
                # Keep as is if not convertible to numeric
                print(f"  - Warning: High-cardinality column '{col}' kept as is.")

    # Create transactions for Apriori
    transactions = transactions_df.apply(
        lambda row: [f"{col}={val}" for col, val in row.items()], axis=1
    ).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Run Apriori to find frequent itemsets
    frequent_itemsets = apriori(
        df_encoded, min_support=0.05, use_colnames=True
    )
    if frequent_itemsets.empty:
        print("  - No frequent itemsets found with min_support=0.05. Try lowering it.")
        return

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    if rules.empty:
        print("  - No association rules found with lift > 1.2.")
        return

    rules = rules.sort_values(by="lift", ascending=False)
    print("  - Top 10 Association Rules by Lift:")
    print(rules.head(10))

    rules_path = table_output_dir / "association_rules.csv"
    rules.to_csv(rules_path, index=False)
    print(f"  - Saved all association rules to: {rules_path.name}")


def analyze_skater(
    gdf: gpd.GeoDataFrame,
    numerical_vars: list,
    output_dir: Path,
    n_clusters: int = 6,
):
    """
    Performs SKATER spatially constrained clustering to create contiguous regions.
    """
    print("\n--- 3. Running Spatially Constrained Clustering (Ward) ---")
    
    # Prepare data: create a copy, fill NaNs, and scale numerical variables
    cluster_data = gdf[numerical_vars].copy()
    cluster_data.fillna(cluster_data.mean(), inplace=True)

    if cluster_data.empty:
        print("Skipping Spatial Clustering: No valid data.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data)

    # Create spatial weights
    print("  - Creating spatial weights matrix...")
    weights = Queen.from_dataframe(gdf)
    
    # Convert PySAL weights to a sparse connectivity matrix for scikit-learn
    # This is the key step for memory efficiency.
    print("  - Converting to sparse connectivity matrix...")
    sparse_connectivity = weights.to_sparse()

    # Run Agglomerative Clustering with the spatial constraint
    print(f"  - Running Agglomerative Clustering with k={n_clusters}...")
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=sparse_connectivity,
        linkage='ward'
    )
    
    labels = model.fit_predict(X_scaled)

    # Add cluster labels to the GeoDataFrame
    gdf["spatial_cluster"] = labels

    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(
        column="spatial_cluster",
        categorical=True,
        legend=True,
        ax=ax,
        edgecolor="k",
        linewidth=0.2,
    )
    ax.set_title(f"Spatially Constrained Ward Clusters (k={n_clusters})")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    
    plot_path = output_dir / f"spatial_ward_cluster_map_k{n_clusters}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  - Saved spatial cluster map to: {plot_path.name}")


def install_r_packages_if_needed(packages: list):
    """Checks if R packages are installed and installs them if not."""
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)  # Choose a default CRAN mirror

    for package in packages:
        if not ro.packages.isinstalled(package):
            print(f"Installing R package: {package}...")
            utils.install_packages(ro.StrVector([package]))
        else:
            print(f"R package '{package}' is already installed.")


def analyze_gwpca(
    gdf: gpd.GeoDataFrame, numerical_vars: list, output_dir: Path, table_output_dir: Path
):
    """
    Performs Geographically Weighted Principal Component Analysis (GWPCA)
    by calling the 'GWmodel' package in R via rpy2.
    """
    print("\n--- 4. Running Geographically Weighted PCA (GWPCA) ---")
    try:
        # 1. Install required R packages
        install_r_packages_if_needed(['sp', 'GWmodel'])
        sp = importr('sp')
        gwmodel = importr('GWmodel')

        # 2. Prepare data for R
        gwpca_vars = numerical_vars[:]
        if len(gwpca_vars) < 2:
            print("  - Skipping GWPCA: At least 2 numerical variables are required.")
            return
        print(f"  - Using variables: {gwpca_vars}")

        gwpca_data = gdf[gwpca_vars].copy()
        gwpca_data.fillna(gwpca_data.mean(), inplace=True)

        scaler = StandardScaler()
        gwpca_data_scaled = pd.DataFrame(scaler.fit_transform(gwpca_data), columns=gwpca_vars, index=gwpca_data.index)

        # Get coordinates
        coords = np.vstack([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).T
        coords_df = pd.DataFrame(coords, columns=['x', 'y'], index=gwpca_data.index)

        # Combine data and coordinates
        r_input_df = pd.concat([gwpca_data_scaled, coords_df], axis=1)

        # 3. Convert to R SpatialPointsDataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data_df = ro.conversion.py2rpy(r_input_df)
        
        ro.r.assign('r_data_df', r_data_df)
        ro.r('coordinates(r_data_df) <- ~x+y')

        # 4. Bandwidth Selection and GWPCA Execution
        # Optimization: If the dataset is large, estimate bandwidth on a sample.
        sample_size = 15000
        if len(gdf) > sample_size:
            print(f"  - Dataset is large ({len(gdf)} rows). Estimating bandwidth on a sample of {sample_size} rows.")
            
            # Create a random sample for bandwidth selection
            sample_df = r_input_df.sample(n=sample_size, random_state=42)
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_sample_df = ro.conversion.py2rpy(sample_df)
            
            ro.r.assign('r_sample_df', r_sample_df)
            ro.r('coordinates(r_sample_df) <- ~x+y')

            # Run bandwidth selection on the sample
            print("  - Selecting optimal bandwidth for GWPCA in R (on sample)...")
            bw = gwmodel.bw_gwpca(data=ro.r['r_sample_df'], vars=ro.StrVector(gwpca_vars), kernel='bisquare', adaptive=True)
            gwpca_bw = bw[0]
            print(f"  - Optimal adaptive bandwidth found from sample: {gwpca_bw} neighbors")
        else:
            # For smaller datasets, run on the full data
            print("  - Selecting optimal bandwidth for GWPCA in R (on full dataset)...")
            bw = gwmodel.bw_gwpca(data=ro.r['r_data_df'], vars=ro.StrVector(gwpca_vars), kernel='bisquare', adaptive=True)
            gwpca_bw = bw[0]
            print(f"  - Optimal adaptive bandwidth found: {gwpca_bw} neighbors")

        print("  - Running GWPCA on the full dataset...")
        gwpca_results = gwmodel.gwpca(data=ro.r['r_data_df'], vars=ro.StrVector(gwpca_vars), bw=gwpca_bw, k=1, kernel='bisquare', adaptive=True)

        # 5. Extract results from R object and convert back to Python
        # The result is a list-like object in rpy2. We access elements by index.
        # The main results are in a SpatialPointsDataFrame called 'SDF'.
        sdf_results = gwpca_results.rx2('SDF')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            results_df = ro.conversion.rpy2py(sdf_results)

        # --- Analyze and Plot GWPCA Results ---
        # 1. Local variance explained by the first component
        # In GWmodel, this is 'var' followed by the component number (e.g., 'var1')
        gdf["gwpca_local_var_explained"] = results_df["var1"]
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(column="gwpca_local_var_explained", cmap="viridis", legend=True, ax=ax)
        ax.set_title("GWPCA: Local Variance Explained by 1st Component")
        plot_path_var = output_dir / "gwpca_local_variance_explained.png"
        plt.savefig(plot_path_var)
        plt.close()
        print(f"  - Saved GWPCA local variance map to: {plot_path_var.name}")

        # 2. Local Component Loadings
        # In GWmodel, these are named like 'Comp1_varname'
        loading_cols = [f"Comp1_{var}" for var in gwpca_vars]
        loadings_df = results_df[loading_cols]
        loadings_df.columns = [f"{v}_loading" for v in gwpca_vars] # Rename for consistency

        loadings_path = table_output_dir / "gwpca_loadings_component1.csv"
        loadings_df.to_csv(loadings_path, index=False)
        print(f"  - Saved GWPCA component loadings to: {loadings_path.name}")

        # Plot the loading for the first variable as an example
        first_var_loading_col = f"{gwpca_vars[0]}_loading"
        gdf[first_var_loading_col] = loadings_df[first_var_loading_col]
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(column=first_var_loading_col, cmap="coolwarm", legend=True, ax=ax)
        ax.set_title(f"GWPCA: Loading of '{gwpca_vars[0]}' on 1st Component")
        plot_path_loading = output_dir / f"gwpca_loading_{gwpca_vars[0]}.png"
        plt.savefig(plot_path_loading)
        plt.close()
        print(f"  - Saved example GWPCA loading map to: {plot_path_loading.name}")

    except Exception as e:
        print(f"  - GWPCA analysis failed: {e}")
        print("  - Please ensure R is installed and in your system's PATH.")
        print("  - You may also need to install the 'sp' and 'GWmodel' R packages manually.")


def main():
    """Main function to execute the multivariate analysis pipeline."""
    # --- Configuration ---
    gpkg_path = GRID_PATH / "h310_grid_final_datacube.gpkg"
    output_plots_dir = PLOTS_SPATIAL_AUTOCORRELATION_PATH / "multivariate"
    output_tables_dir = TABLE_SPATIAL_AUTOCORRELATION_PATH / "multivariate"
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    gdf = load_data(gpkg_path)
    if gdf is None:
        return

    # --- Flag and Drop Sparse Columns ---
    gdf = flag_and_drop_sparse_columns(gdf, threshold=0.20)

    # --- Define Variables ---
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

    # --- Preprocessing ---
    gdf, num_vars, cat_vars = enforce_data_types(
        gdf, numerical_variables, categorical_variables, categorical_int_variables
    )

    # --- Run Analyses ---
    # 1. Feature-Space: Find which variables act together
    if num_vars and cat_vars:
        analyze_famd(gdf, num_vars, cat_vars, output_plots_dir)
    else:
        print("\nSkipping FAMD: Requires both numerical and categorical variables.")

    if cat_vars:
        analyze_association_rules(gdf, cat_vars, output_tables_dir)
    else:
        print("\nSkipping Association Rules: Requires categorical variables.")

    # 2. Spatial-Domain: Find where they act together
    if num_vars:
        analyze_skater(gdf, num_vars, output_plots_dir)
        analyze_gwpca(gdf, num_vars, output_plots_dir, output_tables_dir)
    else:
        print("\nSkipping SKATER and GWPCA: Requires numerical variables.")

    print(f"\n{'='*20} Multivariate Pipeline Complete {'='*20}")
    print(f"Outputs saved in:\n- {output_plots_dir}\n- {output_tables_dir}")


if __name__ == "__main__":
    main()