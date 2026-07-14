"""
Performs zonal statistics to aggregate Digital Terrain Model (DTM) data into
a hexagonal grid.

This script loads an H3 hexagonal grid from a GeoPackage and two DTM raster
files in ASCII format. For each raster, it calculates the mean and maximum
pixel values that fall within each hexagonal cell.

The resulting statistics are added as new columns to the hexagonal grid's
attribute table. The final enriched GeoDataFrame is then saved to a new
GeoPackage file.
"""

import sys
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box, shape
from rasterio.warp import transform_geom

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import MASTER_DATA_PATH, MIDPOINTS_PATH, GPKG_PATH


def main() -> None:
    """
    Main function to execute the zonal statistics workflow.
    """
    # 1. Define file paths
    gpkg_path = GPKG_PATH / "merged_pedologgia_static.gpkg"
    hex_layer_name = "h310grid"

    # Define the paths to your DTM raster files
    # We only need dtmidcnt.asc for the continental part.
    dtm_folder = MASTER_DATA_PATH / "terrain_surface_models" / "DTM_Idrologico"
    raster_path = dtm_folder / "dtmidcnt.asc"

    # Define the output path
    output_dir = MIDPOINTS_PATH / "dtm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg_path = output_dir / "h3_grid_with_dtm_stats.gpkg"

    # 2. Load the hexagonal vector layer
    print(f"Loading hexagonal grid '{hex_layer_name}' from '{gpkg_path.name}'...")
    try:
        hex_grid = gpd.read_file(gpkg_path, layer=hex_layer_name)
    except Exception as e:
        print(f"Error: Could not read layer '{hex_layer_name}'. Please ensure it exists in the GeoPackage. Details: {e}")
        return

    # Ensure the CRS is set for reprojection if necessary
    # The project CRS is EPSG:32632
    if hex_grid.crs is None or hex_grid.crs != "EPSG:32632":
        print(f"Warning: Hex grid CRS is {hex_grid.crs}. Forcing to EPSG:32632.")
        hex_grid = hex_grid.set_crs("EPSG:32632", allow_override=True)

    # 3. Perform zonal statistics for the continental DTM
    if not raster_path.exists():
        print(f"Error: Raster file not found, aborting: {raster_path}")
        return

    print(f"\nProcessing raster: {raster_path.name}")
    raster_name = raster_path.stem.lower()

    print(f"Original number of hexagons: {len(hex_grid)}")

    # The DTM raster is in EPSG:3003. Reproject vector to match raster CRS for zonal_stats.
    with rasterio.open(raster_path) as src:
        # The .asc format may not have CRS info. We know it's EPSG:3003.
        if src.crs:
            dtm_crs = src.crs
        else:
            print("Warning: Raster has no CRS. Assigning EPSG:3003 as specified.")
            dtm_crs = "EPSG:3003"
        print(f"Using DTM CRS: {dtm_crs}")

        # --- Optimization: Filter hexagons by raster bounding box before reprojection ---
        # 1. Get raster bounds and create a geometry
        raster_bounds_geom = box(*src.bounds)
        # 2. Reproject the bounding box to the hexagon grid's CRS
        projected_bounds_dict = transform_geom(
            dtm_crs, hex_grid.crs, raster_bounds_geom
        )
        # 3. Convert the reprojected dictionary back to a Shapely geometry
        projected_bounds_geom = shape(projected_bounds_dict)
        # 4. Filter the grid to keep only intersecting hexagons
        hex_grid_filtered = hex_grid[hex_grid.intersects(projected_bounds_geom)]
        print(
            f"Number of hexagons after filtering to raster extent: {len(hex_grid_filtered)}"
        )
        # --- End of Optimization ---

        # --- Zonal Statistics Calculation ---
        # For robust results, explicitly reproject the filtered hexagons to match the raster's CRS.
        # This avoids potential on-the-fly reprojection issues within rasterstats.
        print(f"Reprojecting {len(hex_grid_filtered)} hexagons to raster CRS ({dtm_crs})...")
        hex_grid_reprojected = hex_grid_filtered.to_crs(dtm_crs)

        # Use the raster's own nodata value if available, otherwise fall back.
        nodata_val = src.nodata if src.nodata is not None else -999

        print("Calculating zonal statistics...")
        stats = zonal_stats(
            hex_grid_reprojected,
            src.read(1),
            affine=src.transform,
            stats=["mean", "max"],
            geojson_out=True,
            nodata=nodata_val,
        )

    # Create the new column names
    mean_col = f"{raster_name}_mean"
    max_col = f"{raster_name}_max"

    # Convert stats to a GeoDataFrame and merge back into the main grid
    stats_gdf = gpd.GeoDataFrame.from_features(stats)
    stats_gdf = stats_gdf.rename(columns={"mean": mean_col, "max": max_col})

    # Use the existing 'index' column as the unique key for the merge.
    hex_grid = hex_grid.merge(
        stats_gdf[["index", mean_col, max_col]], on="index", how="left"
    )
    print(f"Added '{raster_name}_mean' and '{raster_name}_max' columns.")

    # 4. Save the result to a new GeoPackage
    print(f"\nSaving enriched hexagonal grid to: {output_gpkg_path}")
    hex_grid.to_file(output_gpkg_path, driver="GPKG", layer="h3_grid_with_dtm_stats")

    print("\nDone. Zonal statistics calculation complete.")


if __name__ == "__main__":
    main()