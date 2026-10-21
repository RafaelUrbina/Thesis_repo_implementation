import sys
from pathlib import Path
import shutil
import sqlite3
import json
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

# 1. Project Root & Paths Setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]  # Adjust based on your script nesting
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import CAUSAL_RELATION_MODEL_PATH, GRID_PATH

# Define file paths
traceable_csv_path = CAUSAL_RELATION_MODEL_PATH / "output" / "3_h3_causal_bridge" / "h3_geospatial_patent_failure_chains_traceable.csv"
extreme_summary_json_path = CAUSAL_RELATION_MODEL_PATH / "output" / "1_importance_filter" / "hexagons_extreme_summary.json"
input_gpkg_path = GRID_PATH / 'h310_grid_final_datacube.gpkg'

# Define final self-contained output path (fixed duplicate folder name)
output_dir = CAUSAL_RELATION_MODEL_PATH / "output" / "4_join_causals_to_gpkg"
output_dir.mkdir(parents=True, exist_ok=True)
output_gpkg_path = output_dir / 'h310_grid_final_datacube_causal.gpkg'

def build_self_contained_causal_geopackage():
    print("=== Phase 4: Unified GeoPackage Causal Integration ===")
    
    # 2. Validate Inputs
    if not traceable_csv_path.exists():
        raise FileNotFoundError(f"Traceable CSV not found at {traceable_csv_path}. Run the bridge script first.")
    if not input_gpkg_path.exists():
        raise FileNotFoundError(f"Original GeoPackage not found at {input_gpkg_path}.")

    # 3. Copy original GeoPackage to preserve it
    print(f"Copying original GeoPackage to: {output_gpkg_path}")
    shutil.copy(input_gpkg_path, output_gpkg_path)

    # 4. Load Data Inputs
    print(f"Loading traceable chains from: {traceable_csv_path}")
    traceable_df = pd.read_csv(traceable_csv_path)
    print(f"Loaded {len(traceable_df)} traceable rows.")
    
    # Load and parse Layer 4 pairs from hexagons_extreme_summary.json
    pairs_rows = []
    if extreme_summary_json_path.exists():
        print(f"Loading extreme summary JSON from: {extreme_summary_json_path}")
        with open(extreme_summary_json_path, 'r', encoding='utf-8') as f:
            extreme_summary_data = json.load(f)
            
        for entry in extreme_summary_data:
            h3_idx = entry.get('index')
            pairs = entry.get('extreme_pairs', [])
            for pair in pairs:
                if len(pair) == 2:
                    pairs_rows.append({
                        'index': h3_idx,
                        'variable_1': pair[0],
                        'variable_2': pair[1]
                    })
    else:
        print("[Notice] hexagons_extreme_summary.json not found. Layer 4 table will be empty.")

    pairs_df = pd.DataFrame(pairs_rows)
    print(f"Extracted {len(pairs_df)} extreme variable pair records for Layer 4.")

    print(f"Loading original H3 GeoPackage layer 'final_data_cube' from: {input_gpkg_path}")
    gdf = gpd.read_file(input_gpkg_path, layer='final_data_cube')
    print(f"Loaded GeoPackage with {len(gdf)} hexagons.")

    # 5. Aggregate Traceable Rows to Hexagon-Level Summary Metrics (Layer 1 prep)
    print("\nAggregating failure chains by H3 index for macro-metrics...")
    summary_geo_df = traceable_df.groupby('index').agg(
        total_failure_chains=('failure_term', 'count'),
        unique_technical_triggers=('technical_term', 'nunique'),
        top_failure_mode=('failure_term', lambda x: x.mode()[0] if not x.empty else ''),
        patent_count=('patent_ids', lambda x: len(set(pid.strip() for val in x.dropna() for pid in str(val).split(',') if pid.strip()))),
        all_unique_failures=('failure_term', lambda x: ", ".join(sorted(set(str(v).strip() for v in x.dropna() if str(v).strip() and str(v).lower() != 'nan')))),
        all_unique_attributes=('attribute_term', lambda x: ", ".join(sorted(set(str(v).strip() for v in x.dropna() if str(v).strip() and str(v).lower() != 'nan'))))
    ).reset_index()

    # 5b. Aggregate Traceable Rows by H3 Index + Unique Triplet (Layer 3 prep)
    print("Aggregating failure chains by H3 index and unique triplets...")
    triplet_summary_df = traceable_df.groupby(['index', 'technical_term', 'attribute_term', 'failure_term']).agg(
        triplet_chain_count=('failure_term', 'count'),
        triplet_patent_count=('patent_ids', lambda x: len(set(pid.strip() for val in x.dropna() for pid in str(val).split(',') if pid.strip()))),
        triplet_patent_ids=('patent_ids', lambda x: ", ".join(sorted(set(pid.strip() for val in x.dropna() for pid in str(val).split(',') if pid.strip()))))
    ).reset_index()

    # 6. Merge Summary Metrics via Left Join on 'index'
    print("Merging aggregated causal metrics into the polygon GeoDataFrame...")
    merged_gdf = gdf.merge(summary_geo_df, on='index', how='left')

    # Fill NaN values for safe background hexagons with no active failure links
    merged_gdf['total_failure_chains'] = merged_gdf['total_failure_chains'].fillna(0).astype(int)
    merged_gdf['unique_technical_triggers'] = merged_gdf['unique_technical_triggers'].fillna(0).astype(int)
    merged_gdf['patent_count'] = merged_gdf['patent_count'].fillna(0).astype(int)
    merged_gdf['top_failure_mode'] = merged_gdf['top_failure_mode'].fillna('None')
    merged_gdf['all_unique_failures'] = merged_gdf['all_unique_failures'].fillna('None')
    merged_gdf['all_unique_attributes'] = merged_gdf['all_unique_attributes'].fillna('None')

    # 7. Calculate Weighted Composite Index
    print("Computing weighted socio-economic index...")
    index_columns = [
        'm_per_hex', 'census_pop', 'inflow_total'
    ]
    
    for col in index_columns:
        if col not in merged_gdf.columns:
            merged_gdf[col] = 0

    index_subset = merged_gdf[index_columns].fillna(0)

    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(index_subset)
    norm_df = pd.DataFrame(normalized_values, columns=index_columns, index=merged_gdf.index)

    priority_cols = {'census_pop', 'inflow_total'}
    weights = {col: (1.1 if col in priority_cols else 1.0) for col in index_columns}

    composite_score = sum(norm_df[col] * weights[col] for col in index_columns)
    total_weight = sum(weights.values())
    
    merged_gdf['causal_socioeconomic_index'] = (composite_score / total_weight) * 100

    # 8. Write Back Spatial Polygons Layer to the Copy
    print(f"Writing updated spatial layer to GeoPackage copy...")
    merged_gdf.to_file(output_gpkg_path, layer='final_data_cube', driver="GPKG")

    # 9. Inject All Tables (Layers 2, 3, and 4) into SQLite (.gpkg)
    print("Embedding tables into the GeoPackage database...")
    conn = sqlite3.connect(output_gpkg_path)
    
    # Layer 2: Raw multi-row table
    traceable_df.to_sql('traceable_causal_chains', conn, if_exists='replace', index=False)
    
    # Layer 3: Unique Triplet per Hexagon aggregated table
    triplet_summary_df.to_sql('h3_triplet_summary', conn, if_exists='replace', index=False)
    
    # Layer 4: H3 Extreme Variable Pairs table from JSON
    pairs_df.to_sql('h3_extreme_pairs', conn, if_exists='replace', index=False)
    
    conn.close()

    print("\nPipeline complete! Successfully created unified self-contained GeoPackage:")
    print(f"Saved to: {output_gpkg_path}")
    print(" - Layer 1 [Spatial]: 'final_data_cube' (Hexagon macro-metrics & socio-economic index)")
    print(" - Layer 2 [Tabular]: 'traceable_causal_chains' (Full raw multi-row causal links)")
    print(" - Layer 3 [Tabular]: 'h3_triplet_summary' (Unique index + triplet combinations)")
    print(" - Layer 4 [Tabular]: 'h3_extreme_pairs' (Hex index with variable_1 and variable_2 extracted from JSON)")

if __name__ == "__main__":
    build_self_contained_causal_geopackage()