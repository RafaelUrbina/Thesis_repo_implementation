import sys
from pathlib import Path
import fiona

# Add the project root to the Python path to allow for absolute imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.insert(0, str(project_root))

from Utils.paths import GPKG_PATH
import geopandas as gpd

# Define the path to your GeoPackage
source_gpkg_path_str = GPKG_PATH + "static_spatial_dimensions.gpkg"
source_gpkg_path = Path(source_gpkg_path_str)

# Define the path for the new cleaned GeoPackage
cleaned_gpkg_path = source_gpkg_path.with_name(f"{source_gpkg_path.stem}_cleaned.gpkg")

def interactive_column_cleaner():
    """
    Loads a GeoPackage and interactively prompts the user to drop columns
    from each layer, then saves the result to a new file.
    """
    try:
        layers = fiona.listlayers(source_gpkg_path)
        print(f"Found GeoPackage at: {source_gpkg_path}")
        print(f"Cleaned file will be saved as: {cleaned_gpkg_path}\n")
    except fiona.errors.DriverError as e:
        print(f"Error: Could not open the GeoPackage file. Please check the path.")
        print(f"Details: {e}")
        return

    for layer_name in layers:
        print(f"\n--- Processing Layer: '{layer_name}' ---")
        gdf = gpd.read_file(source_gpkg_path, layer=layer_name)
        
        columns_to_drop = []
        
        for column in gdf.columns:
            # The 'geometry' column is essential and should not be dropped
            if column == 'geometry':
                continue

            print(f"\n-- Analyzing column: '{column}' in layer '{layer_name}' --")
            
            unique_values = gdf[column].unique()
            sample_values = unique_values[:10]
            
            print(f"Found {len(unique_values)} unique values. Here are up to 10 samples:")
            for val in sample_values:
                print(f"  - {val}")

            while True:
                decision = input(f"Drop column '{column}'? (y/n): ").lower().strip()
                if decision in ['y', 'n']:
                    break
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
            
            if decision == 'y':
                columns_to_drop.append(column)
        
        if columns_to_drop:
            gdf.drop(columns=columns_to_drop, inplace=True)
            print(f"\nDropped {len(columns_to_drop)} columns from layer '{layer_name}'.")
        
        # Save the (potentially modified) layer to the new GeoPackage
        gdf.to_file(cleaned_gpkg_path, layer=layer_name, driver="GPKG")
        print(f"Saved cleaned version of layer '{layer_name}' to '{cleaned_gpkg_path.name}'.")

    print("\n--- All layers processed. Cleaning complete! ---")

if __name__ == "__main__":
    interactive_column_cleaner()