import sys
from pathlib import Path
import pandas as pd
from itertools import combinations

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
        'idw_rain_mm_max_monthly',
        'idw_wind_speed_max_max',
        'dtmidcnt_max',
        'avg_seismic_magnitude_1ring',
        'idw_temp_thermal_range',
        'avg_descending_soil_speed',
        'avg_ascending_soil_speed',
        'landslide_point_count_1ring',
        'hydraulic_hazard_level'
]
# =====================================================================

# Threshold definitions
MAX_THRESHOLD = 0.99
MIN_THRESHOLD = 0.01

# 4. Iterate and capture conditional extreme flags per hexagon (filtered for >= 2)
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
    
    # Keep only hexagons that contain at least 2 extreme flags
    if len(triggered_vars) >= 2:
        sorted_triggered = sorted(triggered_vars)
        hex_pairs = [list(pair) for pair in combinations(sorted_triggered, 2)]
        
        extreme_records.append({
            'index': h3_index,
            'extreme_variables': sorted_triggered,
            'extreme_pairs': hex_pairs,
            'extreme_count': len(triggered_vars)
        })

# 5. Convert summary results into a DataFrame
extreme_df = pd.DataFrame(extreme_records)

print(f"Total H3 hexagons with at least 2 extreme flags: {len(extreme_df)} out of {len(df)}")
if not extreme_df.empty:
    print(extreme_df.head())

# 6. Create long-form 'Exploded' table mapped with pairs for Ontology Lookup
if not extreme_records:
    exploded_df = pd.DataFrame(columns=['index', 'attribute_variable', 'attribute_variable_2', 'extreme_count', 'ontology_term', 'ontology_term_2'])
else:
    pair_rows = []
    for rec in extreme_records:
        h3 = rec['index']
        count = rec['extreme_count']
        for v1, v2 in rec['extreme_pairs']:
            pair_rows.append({
                'index': h3,
                'attribute_variable': v1,
                'attribute_variable_2': v2,
                'extreme_count': count
            })
    
    exploded_df = pd.DataFrame(pair_rows)
    # Clean up ontology term suffixes for both variables in the pair
    exploded_df['ontology_term'] = exploded_df['attribute_variable'].str.replace('_log1p_pctl', '', regex=False).str.replace('_yeojohnson', '', regex=False)
    exploded_df['ontology_term_2'] = exploded_df['attribute_variable_2'].str.replace('_log1p_pctl', '', regex=False).str.replace('_yeojohnson', '', regex=False)

print("\n--- Updated Exploded Mapping Sample (With Pairs) ---")
print(exploded_df.head(10))

# 7. Create output subdirectory and save safely
importance_filter_dir = output_dir / "1_importance_filter"
importance_filter_dir.mkdir(parents=True, exist_ok=True)

extreme_df.to_json(importance_filter_dir / "hexagons_extreme_summary.json", orient='records', indent=2)
exploded_df.to_csv(importance_filter_dir / "hexagon_to_ontology_triggers.csv", index=False)

print(f"\nOutputs successfully saved to: {importance_filter_dir}")