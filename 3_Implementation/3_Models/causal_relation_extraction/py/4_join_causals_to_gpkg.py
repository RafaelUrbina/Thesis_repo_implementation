import sys
from pathlib import Path
import shutil
import sqlite3
import pandas as pd
import geopandas as gpd

# 1. Project Root & Paths Setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]  # Adjust based on your script nesting
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import CAUSAL_RELATION_MODEL_PATH, GRID_PATH

# Define file paths
traceable_csv_path = CAUSAL_RELATION_MODEL_PATH / "output" / "3_h3_causal_bridge" / "h3_geospatial_patent_failure_chains_traceable.csv"
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
    
    print(f"Loading original H3 GeoPackage layer 'final_data_cube' from: {input_gpkg_path}")
    gdf = gpd.read_file(input_gpkg_path, layer='final_data_cube')
    print(f"Loaded GeoPackage with {len(gdf)} hexagons.")

    # 5. Aggregate Traceable Rows to Hexagon-Level Summary Metrics (with Order-Independent Unique Strings)
    print("\nAggregating failure chains by H3 index for macro-metrics...")
    summary_geo_df = traceable_df.groupby('index').agg(
        total_failure_chains=('failure_term', 'count'),
        unique_technical_triggers=('technical_term', 'nunique'),
        top_failure_mode=('failure_term', lambda x: x.mode()[0] if not x.empty else ''),
        patent_count=('patent_ids', lambda x: len(set(pid.strip() for val in x.dropna() for pid in str(val).split(',') if pid.strip()))),
        all_unique_failures=('failure_term', lambda x: ", ".join(sorted(set(str(v).strip() for v in x.dropna() if str(v).strip() and str(v).lower() != 'nan')))),
        all_unique_attributes=('attribute_term', lambda x: ", ".join(sorted(set(str(v).strip() for v in x.dropna() if str(v).strip() and str(v).lower() != 'nan'))))
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

    # 7. Write Back Spatial Polygons Layer to the Copy
    print(f"Writing updated spatial layer to GeoPackage copy...")
    merged_gdf.to_file(output_gpkg_path, layer='final_data_cube', driver="GPKG")

    # 8. Inject Full Traceable CSV as a Secondary Table Layer into SQLite (.gpkg)
    print("Embedding full traceable causal chain table into the GeoPackage database...")
    conn = sqlite3.connect(output_gpkg_path)
    traceable_df.to_sql('traceable_causal_chains', conn, if_exists='replace', index=False)
    conn.close()

    print("\nPipeline complete! Successfully created unified self-contained GeoPackage:")
    print(f"Saved to: {output_gpkg_path}")
    print(" - Layer 1 [Spatial]: 'final_data_cube' (Updated with choropleth metrics & unique string columns)")
    print(" - Layer 2 [Tabular]: 'traceable_causal_chains' (Full multi-row causal links)")
    
    print(f"\n--- Preview of Updated Spatial Data (Non-Zero Hexagons) ---")
    print(merged_gdf[merged_gdf['total_failure_chains'] > 0][
        ['index', 'total_failure_chains', 'top_failure_mode', 'all_unique_failures']
    ].head(5).to_markdown(index=False))

if __name__ == "__main__":
    build_self_contained_causal_geopackage()