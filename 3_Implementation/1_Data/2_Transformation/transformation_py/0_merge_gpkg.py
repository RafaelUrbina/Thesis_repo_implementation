from pathlib import Path
import sys

import fiona
import geopandas as gpd
import pyogrio

# Make the repository root importable when running this script directly.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.insert(0, str(project_root))

from Utils.paths import GPKG_PATH


INPUT_GPKGS = [
    Path(GPKG_PATH) / "pedologgia_cleaned.gpkg",
    Path(GPKG_PATH) / "static_spatial_dimensions_cleaned.gpkg",
]
OUTPUT_GPKG = Path(GPKG_PATH) / "merged_pedologgia_static.gpkg"


def has_duplicate_layer_name(layer_name, seen_layers, char_difference=0):
    """
    Return True if layer_name duplicates any seen layer.
    Two layer names are considered duplicates if they differ by char_difference or fewer characters.
    """
    if layer_name in seen_layers:
        return True

    for seen in seen_layers:
        # Calculate character-level difference
        diff_count = 0
        max_len = max(len(layer_name), len(seen))
        
        for i in range(max_len):
            char1 = layer_name[i] if i < len(layer_name) else None
            char2 = seen[i] if i < len(seen) else None
            if char1 != char2:
                diff_count += 1
        
        if diff_count <= char_difference:
            return True

    return False


def merge_geopackages(input_gpkgs, output_gpkg, overwrite=True):
    """
    Merge multiple GeoPackages into one output GeoPackage.

    Layers whose names already appeared in an earlier source GeoPackage are
    skipped so the output does not contain duplicate layer names.
    Attribute-only tables are also supported.
    """
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and output_gpkg.exists():
        output_gpkg.unlink()

    seen_layers = set()
    written_layers = []

    for input_gpkg in input_gpkgs:
        if not input_gpkg.exists():
            print(f"Skipping missing file: {input_gpkg}")
            continue

        try:
            layers = fiona.listlayers(input_gpkg)
        except Exception as exc:
            print(f"Could not read layers from {input_gpkg}: {exc}")
            continue

        print(f"Reading source GeoPackage: {input_gpkg.name}")

        for layer_name in layers:
            if has_duplicate_layer_name(layer_name, seen_layers):
                print(
                    f"Skipping duplicate layer '{layer_name}' from {input_gpkg.name}"
                )
                continue

            try:
                gdf = gpd.read_file(input_gpkg, layer=layer_name)
            except Exception as exc:
                print(
                    f"Could not read layer '{layer_name}' from {input_gpkg.name}: {exc}"
                )
                continue

            try:
                if isinstance(gdf, gpd.GeoDataFrame) and "geometry" in gdf.columns:
                    gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
                else:
                    pyogrio.write_dataframe(
                        gdf,
                        output_gpkg,
                        layer=layer_name,
                        driver="GPKG",
                    )
            except Exception as exc:
                print(
                    f"Could not write layer '{layer_name}' to {output_gpkg.name}: {exc}"
                )
                continue

            seen_layers.add(layer_name)
            written_layers.append(layer_name)
            print(f"Wrote layer '{layer_name}' to {output_gpkg.name}")

    print(f"\nMerged {len(written_layers)} unique layers into {output_gpkg.name}")
    return output_gpkg


if __name__ == "__main__":
    merge_geopackages(INPUT_GPKGS, OUTPUT_GPKG)
