from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MASTER_DATA_PATH, MIDPOINTS_PATH


def main() -> None:
    gpkg_path = GPKG_PATH / "merged_pedologgia_static.gpkg"
    excel_path = MASTER_DATA_PATH / "seismic_fenomena" / "CPTI15_v4.0.xlsx"
    output_path = MIDPOINTS_PATH / "seismic" / "seismic_points_utm32n.shp"

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    print(f"Loading boundary layer from GeoPackage: {gpkg_path}")
    boundary = gpd.read_file(gpkg_path, layer="provcm01012026_wgs84")
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:32632")
    else:
        boundary = boundary.to_crs("EPSG:32632")

    print(f"Loading seismic catalogue from: {excel_path}")
    catalogue = pd.read_excel(excel_path, sheet_name="catalogue")

    required_cols = {"LatDef", "LonDef"}
    missing = required_cols - set(catalogue.columns)
    if missing:
        raise ValueError(f"Missing required columns in Excel sheet 'catalogue': {sorted(missing)}")

    catalogue = catalogue.dropna(subset=["LatDef", "LonDef"]).copy()
    catalogue["LonDef"] = pd.to_numeric(catalogue["LonDef"], errors="coerce")
    catalogue["LatDef"] = pd.to_numeric(catalogue["LatDef"], errors="coerce")
    catalogue = catalogue.dropna(subset=["LonDef", "LatDef"])

    print(f"Creating point geometries for {len(catalogue)} records")
    points = gpd.GeoDataFrame(
        catalogue,
        geometry=gpd.points_from_xy(catalogue["LonDef"], catalogue["LatDef"]),
        crs="EPSG:4326",
    )

    points = points[points["geometry"].notna()].copy()
    print(f"Reprojecting point geometries to EPSG:32632")
    points = points.to_crs("EPSG:32632")

    print(f"Filtering points inside provcm01012026_wgs84")
    points_in_boundary = gpd.clip(points, boundary)

    if points_in_boundary.empty:
        raise ValueError("No points were found inside the boundary layer.")

    points_utm = points_in_boundary

    print(f"Saving point shapefile to: {output_path}")
    points_utm.to_file(output_path, driver="ESRI Shapefile")
    print("Done.")


if __name__ == "__main__":
    main()
