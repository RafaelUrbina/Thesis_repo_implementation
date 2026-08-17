"""
Comprehensive Multivariate Geospatial Analysis Pipeline

This script provides an advanced pipeline for conducting both feature-space and
spatial-domain multivariate analyses on a geospatial dataset. The primary goal
is to uncover complex, hidden relationships between a diverse set of variables
and to understand how these relationships manifest across geographic space.

The script is structured into two main analytical themes:
1.  **Feature-Space Analysis**: Investigates which variables tend to vary together,
    irrespective of their spatial location. This helps in understanding the
    underlying structure of the dataset.
2.  **Spatial-Domain Analysis**: Explores how these multivariate relationships
    are distributed geographically, identifying spatial patterns, clusters, and
    non-stationarity (i.e., relationships that change over space).

-------------------------------------------------------------------------------
SCRIPT LOGIC AND PIPELINE
-------------------------------------------------------------------------------

1.  **Data Loading and Preprocessing**:
    - The script begins by loading a master GeoDataFrame from a GeoPackage file.
    - It performs initial data cleaning by identifying and dropping columns that
      have a high percentage of missing values (e.g., >25%), as these columns
      are unlikely to be useful for analysis.
    - It then rigorously enforces data types for predefined lists of numerical
      and categorical variables. This step is crucial for preventing errors in
      downstream statistical functions. Non-numeric values in numerical columns
      are coerced to NaN with a warning.

2.  **Factor Analysis of Mixed Data (FAMD)** - `analyze_famd()`:
    - **Goal**: To reduce dimensionality and identify latent "factors" that
      group both numerical and categorical variables. This helps to see which
      variables contribute to the same underlying concepts.
    - **Process**: Missing values are imputed (mean for numerical, 'Missing'
      category for categorical). FAMD is run, and a variable factor map is
      plotted to visualize how variables align with the principal components.
    - **Output**: A plot of the FAMD components and a CSV file detailing the
      contribution of each variable to these components.

3.  **Association Rule Mining (Apriori)** - `analyze_association_rules()`:
    - **Goal**: To discover co-occurrence patterns among categorical variables,
      framed as "if-then" rules (e.g., if a hexagon has {Land Use A, Soil Type B},
      then it is likely to have {Hazard Level C}).
    - **Process**: High-cardinality variables are binned into quartiles to make
      rules more generalizable. Transactions are created for each location,
      and the Apriori algorithm finds frequent itemsets, from which association
      rules are generated based on a 'lift' metric. An initial filter is applied
      to keep only rules with multi-item antecedents and consequents.
    - **Output**: Two CSV files: one with rules after initial filtering and another with
      strictly filtered rules that exclude trivial or uninformative items.
      rules are generated based on a 'lift' metric.
    - **Output**: Two CSV files are produced:
      1. `association_rules.csv`: Contains rules filtered to keep only those with multi-item antecedents and consequents. These rules may still contain "uninformative" items (e.g., `variable=0`).
      2. `association_rules_strict_filter.csv`: A subset of the first file, where all rules containing any uninformative items have been completely removed.

4.  **Spatially Constrained Clustering** - `analyze_skater()`:
    - **Goal**: To partition the study area into a set of geographically
      contiguous regions where locations within each region are as similar as
      possible based on their numerical variable profiles.
    - **Process**: Numerical data is scaled, and a spatial weights matrix (Queen
      contiguity) is built to define neighborhoods. Spatially constrained
      Agglomerative Clustering (using a Ward linkage) is then performed.
    - **Output**: A map showing the resulting spatial clusters.

5.  **Geographically Weighted PCA (GWPCA)** - `analyze_gwpca()`:
    - **Goal**: To investigate spatial non-stationarity. Unlike standard PCA,
      which assumes one global correlation structure, GWPCA runs a separate
      PCA for each location using its neighbors, revealing how correlations
      between numerical variables change across space.
    - **Process**: The script interfaces with R's powerful 'GWmodel' package via
      `rpy2`. It prepares and transfers the data to R, automatically selects an
      optimal bandwidth (neighborhood size), runs GWPCA, and brings the
      results back into Python.
    - **Output**: Maps showing the local variance explained by the first
      principal component and the spatial variation of variable loadings. A CSV
      file of the loadings is also saved.

The `main` function orchestrates this entire pipeline, ensuring that each
analysis is run in a logical sequence and that outputs are saved to organized
directories.
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


def _to_numeric_with_warning(series: pd.Series, var_name: str, is_int: bool = False) -> pd.Series:
    """Converts a Series to a numeric type, warning if new NaNs are created."""
    initial_nans = series.isnull().sum()
    converted_series = pd.to_numeric(series, errors='coerce')
    if is_int:
        # Use nullable integer type
        converted_series = converted_series.astype('Int64')
    
    if converted_series.isnull().sum() > initial_nans:
        print(f"  - Warning: Coerced non-numeric values to NaN in '{var_name}'.")
    return converted_series


def enforce_data_types(
    gdf: gpd.GeoDataFrame,
    numerical_vars: list,
    categorical_vars: list,
    categorical_int_vars: list,
) -> tuple[gpd.GeoDataFrame, list, list]:
    """Enforces data types on the GeoDataFrame and returns validated variable lists."""
    print("\n--- Enforcing Data Types ---")
    
    existing_num_vars = [v for v in numerical_vars if v in gdf.columns]
    for var in existing_num_vars:
        gdf[var] = _to_numeric_with_warning(gdf[var], var)
    print("Numerical variables enforced.")

    existing_cat_str_vars = [v for v in categorical_vars if v in gdf.columns]
    for var in existing_cat_str_vars:
        gdf[var] = gdf[var].astype(str)
    print("String categorical variables enforced.")

    existing_cat_int_vars = [v for v in categorical_int_vars if v in gdf.columns]
    for var in existing_cat_int_vars:
        gdf[var] = _to_numeric_with_warning(gdf[var], var, is_int=True)
    print("Integer categorical variables enforced.")
    print("--- Data Type Enforcement Complete ---")

    # Combine all categorical variables for analysis functions
    all_categorical_variables = categorical_vars + categorical_int_vars

    # Filter lists to only include columns that exist in the GDF
    valid_numerical = [v for v in numerical_vars if v in gdf.columns]
    return gdf, valid_numerical, [v for v in all_categorical_variables if v in gdf.columns]

def _check_and_skip_analysis(output_paths: list[Path], analysis_name: str) -> bool:
    """Checks if all specified output files exist and are not empty, indicating the analysis can be skipped."""
    # Check if all paths exist and have a file size greater than 0 bytes.
    if all(p.exists() and p.stat().st_size > 0 for p in output_paths):
        print(f"\n--- Output for {analysis_name} already exists. Skipping analysis. ---")
        return True
    return False


def analyze_famd(
    gdf: gpd.GeoDataFrame,
    numerical_vars: list,
    categorical_vars: list,
    output_dir: Path,
    table_output_dir: Path,
):
    """
    Performs Factor Analysis of Mixed Data (FAMD) to find latent components
    grouping both numerical and categorical variables.
    """
    famd_plot_path = output_dir / "famd_row_coordinates.png"
    famd_contributions_path = table_output_dir / "famd_variable_contributions.csv"
    famd_mapping_path = output_dir / "famd_variable_name_mapping.csv"

    # Check if analysis can be skipped
    if _check_and_skip_analysis([famd_plot_path, famd_contributions_path, famd_mapping_path], "Factor Analysis of Mixed Data (FAMD)"):
        return

    print("\n--- 1. Running Factor Analysis of Mixed Data (FAMD) ---")
    all_vars = numerical_vars + categorical_vars
    famd_data = gdf[[v for v in all_vars if v in gdf.columns]].copy()

    # Impute missing values for FAMD
    print("  - Imputing missing values for FAMD...")
    for col in famd_data.select_dtypes(include=np.number).columns:
        famd_data[col] = famd_data[col].fillna(famd_data[col].mean())
    for col in categorical_vars:
        # Ensure all categorical columns are treated as strings for FAMD
        if col in famd_data.columns:
            famd_data[col] = famd_data[col].astype(str)

    for col in categorical_vars:
        if col in famd_data.columns:
            # Convert to string and fill NaN
            famd_data[col] = famd_data[col].astype(str).fillna("Missing") # Explicitly convert to string

    if famd_data.empty:
        print("Skipping FAMD: No data left after imputation.")
        return

    # Create short names for variables for better plot readability
    print("  - Creating short names for FAMD plot variables...")    
    original_name_map = {f"V{i:02d}": name for i, name in enumerate(famd_data.columns)}
    famd_data_renamed = famd_data.rename(columns={v: k for k, v in original_name_map.items()})

    # Save the mapping for user reference
    print("  - Saving variable name mapping...")
    mapping_df = pd.DataFrame(original_name_map.items(), columns=['Short_Name', 'Original_Name']).sort_values('Short_Name')
    mapping_path = output_dir / "famd_variable_name_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"  - Saved variable name mapping to: {mapping_path.name}")

    # Ensure categorical column names are a simple list to avoid indexing issues in prince
    famd_data_renamed.columns = famd_data_renamed.columns.tolist()

    famd = prince.FAMD(n_components=5, n_iter=3, random_state=42).fit(famd_data_renamed)

    # Plot the contribution of each variable to the first two components
    chart = famd.plot(
        famd_data_renamed,
        x_component=0, 
        y_component=1,
        show_row_labels=False,
        show_column_labels=True
    ).properties(
        width=2000,  # Increase width
        height=1600, # Increase height
        title="FAMD: Variable Factor Map & Sample Coordinates"
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(fontSize=20)

    plot_path = output_dir / "famd_row_coordinates.png"
    chart.save(str(plot_path), scale_factor=3.0)
    print(f"  - Saved FAMD row coordinates plot to: {plot_path.name}")

    # Get and save variable contributions
    # Select only component 0 and 1
    contributions = famd.column_contributions_[[0, 1]]
    
    # Rename the index from short codes back to original variable names for readability
    contributions_renamed = contributions.rename(index=original_name_map)

    contributions_path = table_output_dir / "famd_variable_contributions.csv"
    contributions_renamed.to_csv(contributions_path, float_format="%.4e")
    print(f"  - Saved variable contributions with original names to: {contributions_path.name}")


def analyze_association_rules(
    gdf: gpd.GeoDataFrame, categorical_vars: list, table_output_dir: Path
):
    """
    Performs Association Rule Mining (Apriori) on categorical data to find
    co-occurrence patterns.
    """    
    rules_path = table_output_dir / "association_rules.csv"
    strict_rules_path = table_output_dir / "association_rules_strict_filter.csv"

    # Check if analysis can be skipped. We check both files.
    if _check_and_skip_analysis([rules_path, strict_rules_path], "Association Rule Mining (Apriori)"):
        return

    print("\n--- 2. Running Association Rule Mining (Apriori) ---")
    # Discretize high-cardinality categorical variables for meaningful rules.
    transactions_df = gdf[[v for v in categorical_vars if v in gdf.columns]].copy()
    missing_mask = transactions_df.isnull() # Flag original missing values

    # Impute missing values for Apriori
    print("  - Imputing missing values for Apriori...")
    for col in transactions_df.columns:
        transactions_df[col] = transactions_df[col].astype(str).fillna("Missing") # Convert all to string for consistency

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

    # Create transactions, excluding items that were originally missing
    print("  - Creating transactions and filtering out imputed values...")
    transactions = transactions_df.apply(
        lambda row: [
            f"{col}={val}"
            for col, val in row.items()
            if not missing_mask.loc[row.name, col] # Exclude if originally NaN
        ],
        axis=1
    ).tolist()

    # Remove any empty lists that might have resulted from the filtering
    transactions = [t for t in transactions if t]
    print(f"  - Created {len(transactions)} valid transactions.")

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

    # --- Filter for multi-item antecedents and consequents ---
    print("\n  - Filtering for rules with multi-item antecedents and consequents...")
    initial_rule_count = len(rules)
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) > 1)) &
        (rules['consequents'].apply(lambda x: len(x) >= 2))
    ]
    print(f"  - Filtered out {initial_rule_count - len(rules)} rules. {len(rules)} rules remain.")


    def is_uninformative(item):
        """Checks if an item string is considered uninformative."""
        return item.endswith("=0") or item.endswith("=None") or "='not defined'" in item

    def count_uninformative(itemset):
        """Counts the number of uninformative items in an itemset."""
        return sum(1 for item in itemset if is_uninformative(item))

    # --- First Filter: Allow at most ONE uninformative item in total ---
    print("\n  - Applying first filter (max 1 uninformative item allowed)...")
    initial_rule_count = len(rules)
    
    rules['uninformative_count'] = rules.apply(
        lambda row: count_uninformative(row['antecedents']) + count_uninformative(row['consequents']),
        axis=1
    )
    rules = rules[rules['uninformative_count'] <= 1].drop(columns=['uninformative_count'])
    
    print(f"  - Filtered out {initial_rule_count - len(rules)} rules. {len(rules)} rules remain.")
    if rules.empty:
        print("  - No rules remain after the first filter.")
        return

    rules = rules.sort_values(by="lift", ascending=False)
    print("  - Top 10 Association Rules (after first filter):")
    print(rules.head(10))

    rules.to_csv(rules_path, index=False)
    print(f"  - Saved filtered association rules (max 1 uninformative) to: {rules_path.name}")

    # --- Second, Stricter Filter: Allow ZERO uninformative items ---
    print("\n  - Applying stricter filter (zero uninformative items allowed)...")
    rules_strict = rules[rules.apply(
        lambda row: (count_uninformative(row['antecedents']) + count_uninformative(row['consequents'])) == 0,
        axis=1
    )]

    print(f"  - Stricter filter resulted in {len(rules_strict)} rules.")

    if not rules_strict.empty:
        rules_strict.to_csv(strict_rules_path, index=False)
        print(f"  - Saved strictly filtered association rules to: {strict_rules_path.name}")


def analyze_skater(
    gdf: gpd.GeoDataFrame,
    numerical_vars: list,
    output_dir: Path,
    n_clusters: int = 6,
):
    """
    Performs SKATER spatially constrained clustering to create contiguous regions.
    """
    plot_path = output_dir / f"spatial_ward_cluster_map_k{n_clusters}.png"

    if _check_and_skip_analysis([plot_path], f"Spatially Constrained Clustering (k={n_clusters})"):
        return

    print("\n--- 3. Running Spatially Constrained Clustering (Agglomerative Ward) ---")
    
    # Prepare data: select vars, fill NaNs, and scale
    cluster_data = gdf[numerical_vars].copy()
    cluster_data = cluster_data.fillna(cluster_data.mean())

    if cluster_data.empty:
        print("Skipping Spatial Clustering: No valid data.")
        return

    X_scaled = StandardScaler().fit_transform(cluster_data)

    # Create spatial weights
    print("  - Creating spatial weights matrix...")
    weights = Queen.from_dataframe(gdf)    
    sparse_connectivity = weights.to_sparse()

    # Run Agglomerative Clustering with the spatial constraint
    print(f"  - Running Agglomerative Clustering with k={n_clusters}...")
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=sparse_connectivity,
        linkage='ward'
    )
    
    labels = model.fit_predict(X_scaled)
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
    ax.set_title(f"Spatially Constrained Agglomerative Clusters (k={n_clusters})")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    
    plt.savefig(plot_path)
    plt.close()
    print(f"  - Saved spatial cluster map to: {plot_path.name}")


def _create_gwpca_plot(gdf: gpd.GeoDataFrame, column: str, title: str, cmap: str, output_path: Path):
    """Helper function to generate and save a GWPCA map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(column=column, cmap=cmap, legend=True, ax=ax)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  - Saved GWPCA map to: {output_path.name}")

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
    gwpca_var_explained_plot = output_dir / "gwpca_local_variance_explained.png"
    gwpca_loadings_csv = table_output_dir / "gwpca_loadings_component1.csv"
    
    # Note: The third plot's name is dynamic, so we rely on the two stable outputs for skipping.
    if _check_and_skip_analysis([gwpca_var_explained_plot, gwpca_loadings_csv], "Geographically Weighted PCA (GWPCA)"):
        return

    print("\n--- 4. Running Geographically Weighted PCA (GWPCA) ---")
    try:
        # 1. Install required R packages
        install_r_packages_if_needed(['GWmodel'])
        gwmodel = importr('GWmodel')

        # 2. Prepare data for R
        # Use only existing numerical variables
        gwpca_vars = numerical_vars[:]
        if len(gwpca_vars) < 2:
            print("  - Skipping GWPCA: At least 2 numerical variables are required.")
            return
        print(f"  - Using variables: {gwpca_vars}")

        gwpca_data = gdf[gwpca_vars].copy()
        gwpca_data = gwpca_data.fillna(gwpca_data.mean())

        gwpca_data_scaled = pd.DataFrame(StandardScaler().fit_transform(gwpca_data), columns=gwpca_vars, index=gwpca_data.index)
        coords_df = pd.DataFrame({'x': gdf.geometry.centroid.x, 'y': gdf.geometry.centroid.y}, index=gwpca_data.index)

        # Combine data and coordinates
        r_input_df = pd.concat([gwpca_data_scaled, coords_df], axis=1)

        # 3. Convert to R SpatialPointsDataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data_df = ro.conversion.py2rpy(r_input_df)
        
        ro.r.assign('r_data_df', r_data_df)
        ro.r('coordinates(r_data_df) <- ~x+y')

        # 4. Bandwidth Selection and GWPCA Execution
        # Optimization: If the dataset is large, estimate bandwidth on a sample.
        sample_size = 10000
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
        # Plot local variance explained by the first component
        gdf["gwpca_local_var_explained"] = results_df["var1"]
        _create_gwpca_plot(gdf, "gwpca_local_var_explained", 
                           "GWPCA: Local Variance Explained by 1st Component", 
                           "viridis", 
                           output_dir / "gwpca_local_variance_explained.png")

        # 2. Local Component Loadings
        # In GWmodel, these are named like 'Comp1_varname'
        loading_cols = [f"Comp1_{var}" for var in gwpca_vars]
        loadings_df = results_df[loading_cols]
        loadings_df.columns = [f"{v}_loading" for v in gwpca_vars] # Rename for consistency

        loadings_path = table_output_dir / "gwpca_loadings_component1.csv"
        loadings_df.to_csv(loadings_path, index=False)
        print(f"  - Saved GWPCA component loadings to: {loadings_path.name}")

        # Plot the loading for the first variable as an example (this plot is not used for skipping logic)
        first_var_loading_col = f"{gwpca_vars[0]}_loading"
        gdf[first_var_loading_col] = loadings_df[first_var_loading_col]        
        _create_gwpca_plot(gdf, first_var_loading_col,
                           f"GWPCA: Loading of '{gwpca_vars[0]}' on 1st Component",
                           "coolwarm",
                           output_dir / f"gwpca_loading_{gwpca_vars[0]}.png")

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
    gdf = flag_and_drop_sparse_columns(gdf, threshold=0.25)

    # --- Define Variables ---
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

    # --- Preprocessing ---
    gdf, num_vars, cat_vars = enforce_data_types(
        gdf, numerical_variables, categorical_variables, categorical_int_variables
    )

    # --- Run Analyses ---
    # 1. Feature-Space: Find which variables act together
    if num_vars and cat_vars:
        analyze_famd(gdf, num_vars, cat_vars, output_plots_dir, output_tables_dir)
    else:
        print("\nSkipping FAMD: Requires both numerical and categorical variables.")

    if cat_vars:
        analyze_association_rules(gdf, cat_vars, output_tables_dir)
    else:
        print("\nSkipping Association Rules: Requires categorical variables.")

    # 2. Spatial-Domain: Find where they act together
    if num_vars:
        analyze_skater(gdf, num_vars, output_plots_dir, n_clusters=10)
        analyze_gwpca(gdf, num_vars, output_plots_dir, output_tables_dir)
    else:
        print("\nSkipping SKATER and GWPCA: Requires numerical variables.")

    print(f"\n{'='*20} Multivariate Pipeline Complete {'='*20}")
    print(f"Outputs saved in:\n- {output_plots_dir}\n- {output_tables_dir}")


if __name__ == "__main__":
    main()