import sys
from pathlib import Path
import pandas as pd

# 1. Setup project root and import custom paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]  # Adjust as needed (e.g., parents[2] or parents[3])
sys.path.insert(0, str(project_root))

from Utils.paths import CAUSAL_RELATION_MODEL_PATH

# Define output/input directory using your global path
output_dir = CAUSAL_RELATION_MODEL_PATH / "output"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure base output folder exists

csv_file_path = output_dir / "0_importances" / "feature_vectors.csv"

# 2. Load your dataset
df = pd.read_csv(csv_file_path)

# 3. Automatically identify feature columns (exclude the H3 'index')
feature_cols = [col for col in df.columns if col != 'index']

# =====================================================================
# CONFIGURABLE VARIABLE LISTS FOR EXTREME THRESHOLDS
# =====================================================================
# Fill these lists manually with your specific variable names
min_extreme_vars = [ 
    'dist_to_waterway_m',
]

max_extreme_vars = [ 
    'total_intervention_cost_1ring', 'total_fountains_count_1ring',
    'landslide_point_count_1ring', 'max_peak_elevation_1ring',
    'seismic_event_count_1ring', 'avg_seismic_magnitude_1ring', 'm_per_hex',
    'rooting_depth_class', 'surface_stoniness_class',
    'landslide_surface_class', 'hydraulic_hazard_level', 
    'epr_NTAXP', 'epr_TAXABINC', 'epr_SUBEMPTR', 'epr_PENSINCR', 
    'epr_PENSINCF', 'epr_ENTROAIN', 'epr_ENTROAIN01', 'erd_E0_10000', 
    'erd_E10000_1', 'erd_E15000_2', 'erd_E26000_5', 'erd_E55000_7',
    'erd_E75000_1', 'erd_E_GE1200', 'edst_acq_imm', 'edst_acq_erog', 
    'eidx_LAND_CON', 'idw_rain_mm_max_monthly', 'idw_temp_thermal_range',
    'idw_wind_speed_max_max', 'dtmidcnt_max', 'census_pop', 'inflow_total'
]
# =====================================================================

# Threshold definitions
MAX_THRESHOLD = 0.99
MIN_THRESHOLD = 0.01

# 4. Iterate and capture conditional extreme flags per hexagon
extreme_records = []

for idx, row in df.iterrows():
    h3_index = row['index']
    triggered_vars = []
    
    for col in feature_cols:
        val = row[col]
        if pd.isna(val):
            continue
            
        if col in min_extreme_vars:
            # Filter by minimum non-zero values (e.g., 0 < val <= 0.01)
            if 0 < val <= MIN_THRESHOLD:
                triggered_vars.append(col)
        else:
            # Default or explicitly max-filtered variables (val >= 0.99)
            if val >= MAX_THRESHOLD:
                triggered_vars.append(col)
    
    # Keep only hexagons that contain at least one extreme flag
    if len(triggered_vars) > 0:
        extreme_records.append({
            'index': h3_index,
            'extreme_variables': triggered_vars,
            'extreme_count': len(triggered_vars)
        })

# 5. Convert summary results into a DataFrame
extreme_df = pd.DataFrame(extreme_records)

print(f"Total H3 hexagons with at least one extreme flag: {len(extreme_df)} out of {len(df)}")
if not extreme_df.empty:
    print(extreme_df.head())

# 6. Create a long-form 'Exploded' table for Ontology Mapping
if not extreme_df.empty:
    exploded_df = extreme_df.explode('extreme_variables').rename(columns={'extreme_variables': 'attribute_variable'})
    # Clean up column name suffix for ontology matching
    exploded_df['ontology_term'] = exploded_df['attribute_variable'].str.replace('_log1p_pctl', '', regex=False).str.replace('_yeojohnson', '', regex=False)
else:
    exploded_df = pd.DataFrame(columns=['index', 'attribute_variable', 'extreme_count', 'ontology_term'])

print("\n--- Exploded Mapping Sample (Ready for Ontology Lookup) ---")
print(exploded_df.head(10))

# 7. Create output subdirectory and save safely
importance_filter_dir = output_dir / "1_importance_filter"
importance_filter_dir.mkdir(parents=True, exist_ok=True)


extreme_df.to_json(importance_filter_dir / "hexagons_extreme_summary.json", orient='records', indent=2)
exploded_df.to_csv(importance_filter_dir / "hexagon_to_ontology_triggers.csv", index=False)

print(f"\nOutputs successfully saved to: {importance_filter_dir}")