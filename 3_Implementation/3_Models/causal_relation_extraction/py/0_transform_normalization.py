import os
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# ==========================================
# 1. Project Root & Imports Setup
# ==========================================
current_dir = Path(__file__).resolve().parent
# Adjust parents step if script location changes relative to repo root (e.g., parents[3])
project_root = current_dir.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import GRID_PATH, REPO_ROOT, CAUSAL_RELATION_MODEL_PATH

# ==========================================
# 2. Path Configurations
# ==========================================
INPUT_GPKG_PATH = GRID_PATH / "h310_grid_final_datacube.gpkg"
OUTPUT_GPKG_PATH = GRID_PATH / "h310_grid_transformed.gpkg"
OUTPUT_VECTORS_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "0_importances" / "feature_vectors.csv"

LAYER_NAME = 'final_data_cube'
OUTPUT_LAYER_NAME = 'transformed_data_cube'

# ==========================================
# 3. Variable Definitions
# ==========================================

# Variables to be completely dropped/excluded from GPKG and downstream processing
REMOVED_VARS = [
    'active_fountains_count_1ring', 'is_locality_1ring', 'road_density_m_per_m2',
    'idw_rain_mm_sum_annual', 'dtmidcnt_mean', 'epr_CADINCR', 'epr_CADINCF',
    'tipo_movimento_<lambda_0>_1ring', 'max_seismic_magnitude_1ring', 'idw_rain_mm_avg_monthly',
    'nearest_hydro_river_stage_max_m','total_intervention_cost_1ring',
    'total_fountains_count_1ring','max_peak_elevation_1ring',
    'seismic_event_count_1ring','rooting_depth_class',
    'surface_stoniness_class','landslide_surface_class',
    'epr_NTAXP','epr_TAXABINC',
    'epr_SUBEMPTR','epr_PENSINCR',
    'epr_PENSINCF','epr_ENTROAIN',
    'epr_ENTROAIN01','erd_E0_10000',
    'erd_E10000_1','erd_E15000_2',
    'erd_E26000_5','erd_E55000_7',
    'erd_E75000_1','erd_E_GE1200',
    'edst_acq_imm','edst_acq_erog',
    'eidx_LAND_CON','census_pop',
    'inflow_total'
]

# Categorical variables retained in GPKG for GIS toggling, but excluded from numeric feature vectors
CATEGORICAL_VARS = [
    'class_intervention_1ring', 'locality_name_1ring', 'dominant_highway', 
    'dominant_surface', 'dominant_tunnel', 'dominant_bridge', 
    'usda_hydrologic_group', 'pai_landslide_hazard_level', 
    'dominant_building_1', 'dominant_building_2', 'dominant_building_3'
]

# Variables for log1p transformation (non-negative, positively skewed)
LOG1P_VARS = [
        'idw_rain_mm_max_monthly',
        'idw_wind_speed_max_max',
        'dtmidcnt_max',
        'avg_seismic_magnitude_1ring',
        'idw_temp_thermal_range',
        'dist_to_waterway_m',
        'landslide_point_count_1ring',
        'hydraulic_hazard_level', 'm_per_hex'
]

# Variables for Yeo-Johnson transformation (can handle negative values, skewed)
YEOJOHNSON_VARS = [
    'avg_descending_soil_speed', 'avg_ascending_soil_speed'
]

# Binary variables retained as untransformed
BINARY_VARS = [
    'descending_soil_presence', 'ascending_soil_presence'
]

# Ensure removed variables do not leak into active variable groups
LOG1P_VARS = [v for v in LOG1P_VARS if v not in REMOVED_VARS]
YEOJOHNSON_VARS = [v for v in YEOJOHNSON_VARS if v not in REMOVED_VARS]
CATEGORICAL_VARS = [v for v in CATEGORICAL_VARS if v not in REMOVED_VARS]

# ==========================================
# 4. Helper Functions
# ==========================================
def safe_log1p(series):
    """Applies log1p transformation safely handling infinite/non-numeric values."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    finite_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    return np.log1p(finite_series)

def safe_yeojohnson(series, column_name):
    """Applies Yeo-Johnson transformation safely."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    finite_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    valid_data = finite_series.dropna()

    if valid_data.empty:
        print(f"Warning: No valid data for Yeo-Johnson transformation in column '{column_name}'. Returning NaNs.")
        return pd.Series([np.nan] * len(series), index=series.index)
    
    data_2d = valid_data.values.reshape(-1, 1)
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    
    try:
        transformed_data_valid = pt.fit_transform(data_2d)
        transformed_series = pd.Series([np.nan] * len(series), index=series.index)
        transformed_series.loc[valid_data.index] = transformed_data_valid.flatten()
        return transformed_series
    except Exception as e:
        print(f"Warning: Yeo-Johnson transformation failed for column '{column_name}' due to: {e}. Returning NaNs.")
        return pd.Series([np.nan] * len(series), index=series.index)

# ==========================================
# 5. Main Pipeline Execution
# ==========================================
def main():
    print("=== Phase 1: GeoPackage Data Transformation ===")
    
    try:
        print(f"Reading input GeoPackage: {INPUT_GPKG_PATH} (Layer: {LAYER_NAME})")
        gdf = gpd.read_file(INPUT_GPKG_PATH, layer=LAYER_NAME)
        print("GeoPackage loaded successfully.")

        # 1. Apply log1p transformations
        print("\nApplying log1p transformations...")
        transformed_feature_columns = []
        for col in LOG1P_VARS:
            if col in gdf.columns:
                new_col_name = f"{col}_log1p"
                gdf[new_col_name] = safe_log1p(gdf[col])
                transformed_feature_columns.append(new_col_name)
                print(f"  - Transformed '{col}' -> '{new_col_name}'")
            else:
                print(f"  - Warning: Column '{col}' not found. Skipping.")

        # 2. Apply Yeo-Johnson transformations
        print("\nApplying Yeo-Johnson transformations...")
        for col in YEOJOHNSON_VARS:
            if col in gdf.columns:
                new_col_name = f"{col}_yeojohnson"
                gdf[new_col_name] = safe_yeojohnson(gdf[col], col)
                transformed_feature_columns.append(new_col_name)
                print(f"  - Transformed '{col}' -> '{new_col_name}'")
            else:
                print(f"  - Warning: Column '{col}' not found. Skipping.")

        # 3. Log status for categorical and binary variables
        found_categorical = [c for c in CATEGORICAL_VARS if c in gdf.columns]
        print(f"\nCategorical variables retained in GeoPackage ({len(found_categorical)}/{len(CATEGORICAL_VARS)} found).")
        print("Binary variables are retained without transformation.")

        # 4. Remove Excluded Variables from GeoDataFrame
        if REMOVED_VARS:
            cols_to_drop = [col for col in REMOVED_VARS if col in gdf.columns]
            if cols_to_drop:
                gdf.drop(columns=cols_to_drop, inplace=True)
                print(f"\nRemoved {len(cols_to_drop)} explicitly excluded variables: {cols_to_drop}")

        # 5. Save Transformed GeoPackage
        OUTPUT_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving transformed GeoPackage to: {OUTPUT_GPKG_PATH} (Layer: {OUTPUT_LAYER_NAME})")
        gdf.to_file(OUTPUT_GPKG_PATH, layer=OUTPUT_LAYER_NAME, driver="GPKG")
        print("Transformed GeoPackage saved successfully.")

        print("\n=== Phase 2: Feature Vector & Percentile Generation ===")
        
        # 6. Identify H3 index column
        h3_index_column = None
        potential_h3_cols = ['h3_index', 'H3_Index', 'index', 'cell_id', 'H3ID']
        for col in gdf.columns:
            if col in potential_h3_cols:
                h3_index_column = col
                print(f"Identified H3 index column: '{h3_index_column}'")
                break
        
        if not h3_index_column:
            print("Warning: Could not automatically identify an H3 index column. Using DataFrame index.")

        # Ensure feature selection excludes categorical and explicitly removed variables
        transformed_feature_columns = [
            col for col in transformed_feature_columns 
            if col in gdf.columns and col not in REMOVED_VARS and col not in CATEGORICAL_VARS
        ]

        if not transformed_feature_columns:
            print("Error: No valid transformed numeric feature columns available. Skipping vector generation.")
            return

        # Prepare feature data
        feature_data_for_rank = gdf[transformed_feature_columns].copy()
        
        # Filter out rows where ALL transformed features are NaN
        valid_rows = feature_data_for_rank.dropna(how='all').index
        if len(valid_rows) < len(gdf):
            print(f"Dropped {len(gdf) - len(valid_rows)} rows with all NaNs in transformed feature columns.")
            feature_data_for_rank = feature_data_for_rank.loc[valid_rows]
            gdf_filtered = gdf.loc[valid_rows]
        else:
            gdf_filtered = gdf

        if feature_data_for_rank.empty:
            print("Error: No valid data remaining for feature vector generation.")
            return

        # Filter out zero-variance columns
        zero_variance_cols = feature_data_for_rank.columns[feature_data_for_rank.var(skipna=True) == 0]
        if not zero_variance_cols.empty:
            print(f"Warning: Excluding zero-variance columns from ranking: {list(zero_variance_cols)}")
            columns_for_ranking = [col for col in transformed_feature_columns if col not in zero_variance_cols]
            feature_data_for_rank = feature_data_for_rank[columns_for_ranking]
        else:
            columns_for_ranking = transformed_feature_columns

        if feature_data_for_rank.empty:
            print("Error: No non-zero-variance columns remaining to process.")
            return

        # 7. Calculate Exact Empirical Percentile Ranks via Pandas rank(pct=True)
        print(f"\nCalculating exact empirical percentile ranks for {len(columns_for_ranking)} columns using Pandas...")
        
        percentile_df = pd.DataFrame(index=gdf_filtered.index)
        for col in columns_for_ranking:
            # pct=True scales ranks from 0.0 to 1.0; method='average' cleanly handles zero-inflation ties
            percentile_df[f"{col}_pctl"] = feature_data_for_rank[col].rank(pct=True, method='average')

        # 8. Assemble and Export Final CSV
        if h3_index_column and h3_index_column in gdf_filtered.columns:
            h3_ids_df = gdf_filtered[[h3_index_column]].copy()
            feature_vectors_df = pd.concat([h3_ids_df, percentile_df], axis=1)
        else:
            feature_vectors_df = percentile_df

        OUTPUT_VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving feature vectors to: '{OUTPUT_VECTORS_PATH}'...")
        feature_vectors_df.to_csv(OUTPUT_VECTORS_PATH, index=False)
        print(f"Feature vectors saved successfully. Output shape: {feature_vectors_df.shape}")
        
        print("\n--- Sample Output (First 5 Rows) ---")
        print(feature_vectors_df.head().to_markdown(index=False, numalign="left", stralign="left"))

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
    except FileNotFoundError:
        print(f"Error: Input GeoPackage not found at '{INPUT_GPKG_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()