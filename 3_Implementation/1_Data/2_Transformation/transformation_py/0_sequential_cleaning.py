import sys
from pathlib import Path
import fiona

import pyogrio
# Add the project root to the Python path to allow for absolute imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]
sys.path.insert(0, str(project_root))

from Utils.paths import GPKG_PATH
import geopandas as gpd

# Define the path to your GeoPackage
source_gpkg_path = GPKG_PATH / "merged_pedologgia_static.gpkg"

# Define the path for the new cleaned GeoPackage
cleaned_gpkg_path = source_gpkg_path.with_name(f"{source_gpkg_path.stem}_cleaned.gpkg")


def explode_multipart_geometry_layers_in_file(
    input_gpkg_path=source_gpkg_path,
    output_gpkg_path=cleaned_gpkg_path,
):
    """
    Explodes every layer that contains multipart geometry types (such as
    MultiPolygon or MultiLineString) in the input GeoPackage and writes the
    result to the cleaned output file.
    """
    try:
        layers = fiona.listlayers(input_gpkg_path)
    except fiona.errors.DriverError as e:
        print(
            f"Error: Could not open the GeoPackage file at '{input_gpkg_path}'."
        )
        print(f"Details: {e}")
        return

    print(f"Reading GeoPackage from: {input_gpkg_path}")
    print(f"Writing exploded output to: {output_gpkg_path}\n")

    for layer_name in layers:
        gdf = gpd.read_file(input_gpkg_path, layer=layer_name)

        if gdf.empty or "geometry" not in gdf.columns:
            print(f"Writing non-spatial table '{layer_name}' without changes.")
            # Use pyogrio to write non-spatial DataFrames
            pyogrio.write_dataframe(
                gdf, output_gpkg_path, layer=layer_name, driver="GPKG"
            )
            continue

        multipart_types = {"MultiPolygon", "MultiLineString"}
        multipart_mask = gdf.geometry.geom_type.isin(multipart_types)
        if not multipart_mask.any():
            print(
                f"Layer '{layer_name}' does not contain multipart geometries; writing unchanged."
            )
            gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
            continue

        print(
            f"Exploding multipart geometries in layer '{layer_name}'..."
        )
        exploded_gdf = gdf.explode(index_parts=False)
        exploded_gdf.to_file(
            output_gpkg_path,
            layer=layer_name,
            driver="GPKG",
        )
        print(
            f"Saved exploded layer '{layer_name}' to '{output_gpkg_path.name}'."
        )

    print("\n--- All multipart geometry layers exploded. ---")


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

        while True:
            layer_action = input(
                f"Process layer '{layer_name}'? (p = process, s = skip, c = copy as is): "
            ).lower().strip()
            if layer_action in ['p', 's', 'c']:
                break
            print(
                "Invalid input. Please enter 'p' to process, 's' to skip, or 'c' to copy."
            )

        if layer_action in ['s', 'c']:
            if layer_action == 's':
                print(f"Skipping layer '{layer_name}'. It will not be in the output.")
                continue
            # Write the original, unmodified layer to the new file
            gdf.to_file(cleaned_gpkg_path, layer=layer_name, driver="GPKG")
            print(f"Saved original layer '{layer_name}' to '{cleaned_gpkg_path.name}'.")
            continue
        
        columns_to_drop = []
        
        column_names = [col for col in gdf.columns if col != 'geometry']
        i = 0

        while i < len(column_names):
            column = column_names[i]

            print(f"\n-- Analyzing column: '{column}' in layer '{layer_name}' --")
            
            unique_values = gdf[column].unique()
            sample_values = unique_values[:10]
            
            print(f"Found {len(unique_values)} unique values. Here are up to 10 samples:")
            for val in sample_values:
                print(f"  - {val}")

            while True:
                decision = input(
                    f"Drop column '{column}'? (y = yes, n = no, u = undo previous decision): "
                ).lower().strip()
                if decision in ['y', 'n', 'u']:
                    break
                print(
                    "Invalid input. Please enter 'y' for yes, 'n' for no, or 'u' to go back to the previous column."
                )

            if decision == 'u':
                if i == 0:
                    print("There is no previous column to go back to.")
                    continue

                previous_column = column_names[i - 1]
                if previous_column in columns_to_drop:
                    columns_to_drop.remove(previous_column)
                    print(f"Removed '{previous_column}' from the drop list.")
                else:
                    print(f"Kept '{previous_column}' as not dropped.")

                i -= 1
                print(f"Going back to previous column '{previous_column}'.")
                continue
            
            if decision == 'y':
                columns_to_drop.append(column)

            i += 1
        
        if columns_to_drop:
            gdf.drop(columns=columns_to_drop, inplace=True)
            print(f"\nDropped {len(columns_to_drop)} columns from layer '{layer_name}'.")
        
        # Save the (potentially modified) layer to the new GeoPackage
        gdf.to_file(cleaned_gpkg_path, layer=layer_name, driver="GPKG")
        print(f"Saved cleaned version of layer '{layer_name}' to '{cleaned_gpkg_path.name}'.")

    print("\n--- All layers processed. Cleaning complete! ---")

if __name__ == "__main__":
    #interactive_column_cleaner()
    #explode_multipart_geometry_layers_in_file()
    print("Check the functionality and the input and output paths before running the script.")