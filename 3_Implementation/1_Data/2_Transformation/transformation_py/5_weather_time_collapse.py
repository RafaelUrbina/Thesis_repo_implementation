"""
Performs temporal aggregation on station data to collapse the time dimension.

This script processes several point-based station layers from the main
GeoPackage. For each station identified by 'station_id', it calculates
a series of statistical metrics from its time-series data (e.g., daily
readings).

The script computes metrics like peak values, averages, standard deviations,
and percentiles for variables such as temperature, rainfall, wind speed, and
water levels. The date column is used to handle temporal calculations like
monthly or annual summaries.

The output is a new GeoPackage containing layers of the aggregated station
data, with each station now having a single entry with the computed metrics.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import GPKG_PATH, MIDPOINTS_PATH


def finalize_gdf_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Selects and reorders columns for the final output.

    Ensures the output contains only the station ID, its static metrics,
    and geometry, removing any intermediate or raw data columns.

    Args:
        gdf: The GeoDataFrame to clean.

    Returns:
        A GeoDataFrame with a clean and ordered set of columns.
    """
    # Define essential columns that should always be first
    core_cols = ["station_id", "quota"]
    # Find all other columns that are not the geometry column
    metric_cols = [col for col in gdf.columns if col not in core_cols + ["geometry"]]
    # Combine and set the final order
    final_cols = core_cols + sorted(metric_cols) + ["geometry"]
    return gdf[final_cols]


def aggregate_stations(gdf: gpd.GeoDataFrame, aggregations: dict, date_col: str = "date") -> gpd.GeoDataFrame:
    """
    Aggregates a GeoDataFrame of station data by station_id.

    Args:
        gdf: GeoDataFrame with time-series data for multiple stations.
        aggregations: A dictionary defining the aggregation rules.
        date_col: The name of the column containing date information.

    Returns:
        A GeoDataFrame with one row per station and aggregated metrics.
    """
    if gdf.empty:
        return gpd.GeoDataFrame()

    # Ensure date column is in datetime format
    gdf[date_col] = pd.to_datetime(gdf[date_col])

    # --- Step 1: Perform Primary Aggregations ---
    # This calculates the base metrics for each station.
    aggregated_data = gdf.groupby("station_id").agg(**aggregations)

    # --- Step 2: Perform Secondary, Post-Aggregation Calculations ---
    # These calculations depend on the results from Step 1.

    # Calculate thermal range from the newly created peak/nadir columns.
    if "temp_max_peak" in aggregated_data.columns and "temp_min_nadir" in aggregated_data.columns:
        aggregated_data["temp_thermal_range"] = (
            aggregated_data["temp_max_peak"] - aggregated_data["temp_min_nadir"]
        )

    # --- Step 3: Finalize the GeoDataFrame ---
    station_geometries = gdf.groupby("station_id")["geometry"].first()

    result_gdf = gpd.GeoDataFrame(aggregated_data, geometry=station_geometries, crs="EPSG:32632")
    result_gdf.reset_index(inplace=True)

    return result_gdf


def main() -> None:
    """
    Main function to run the aggregation for all specified layers.
    """
    # 1. Define file paths
    input_dir = MIDPOINTS_PATH / "weather_aggregates"
    gpkg_path = input_dir / "weather_station_monthly.gpkg"
    output_dir = MIDPOINTS_PATH / "aggregated_stations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg = output_dir / "aggregated_station_metrics.gpkg"

    # 2. Define layers and their specific aggregation rules
    processing_config = {
        "termometri_stations_firenze": {
            "layer_name": "temperature",
            "aggregations": [
                ("temp_max_peak", "temp_max_c", "max"),
                ("temp_min_nadir", "temp_min_c", "min"),
                # The 'temp_thermal_range' will be calculated after this initial aggregation.
                ("quota", "quota", "first"),
            ]
        },
        "idrometri_stations_firenze": {
            "layer_name": "idrometry",
            "aggregations": [
                ("river_stage_max_m", "water_level_m", "max"),
                ("river_stage_mean_m", "water_level_m", "mean"),
                ("river_stage_std_m", "water_level_m", "std"),
                ("quota", "quota", "first"),
            ]
        },
        "cf_pluviometri": {
            "layer_name": "rain",
            "aggregations": [
                ("rain_mm_std", "rain_mm", "std"),
                # All other rain metrics (min/max/avg monthly, annual sum) are calculated below.
                ("quota", "quota", "first"),
            ]
        },
        "anemometri_stations_firenze": {
            "layer_name": "wind",
            "aggregations": [
                ("wind_speed_max_max", "wind_speed_max_m_s", "max"),
                ("wind_speed_max_95p", "wind_speed_max_m_s", lambda x: x.quantile(0.95)),
                ("wind_speed_avg_mean", "wind_speed_mean_m_s", "mean"),
                ("quota", "quota", "first"),
            ]
        },
    }

    print(f"Reading data from: {gpkg_path}")
    print(f"Saving aggregated output to: {output_gpkg}\n")

    # 3. Loop through layers, process, and save
    for station_type, config in processing_config.items():
        layer_name = config["layer_name"]
        print(f"--- Processing layer: '{layer_name}' (from {station_type}) ---")
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
        except Exception as e:
            print(f"Warning: Could not read layer '{layer_name}'. Skipping. Error: {e}")
            continue

        # Reformat for new pandas >1.0 agg syntax if needed
        final_aggs = {
            new_name: pd.NamedAgg(column=col, aggfunc=op)
            for new_name, col, op in config["aggregations"]
        }

        # Perform aggregation
        aggregated_gdf = aggregate_stations(gdf, final_aggs)

        # --- Secondary Calculation for all Monthly Rainfall Metrics ---
        if config["layer_name"] == "rain" and not gdf.empty:
            print("  - Calculating monthly and annual rainfall metrics...")

            # --- Metrics calculated directly from the monthly average daily values ---
            monthly_rain_by_station = gdf.groupby("station_id")["rain_mm"]
            max_monthly = monthly_rain_by_station.max().rename("rain_mm_max_monthly")
            min_monthly = monthly_rain_by_station.min().rename("rain_mm_min_monthly")
            avg_monthly = monthly_rain_by_station.mean().rename("rain_mm_avg_monthly")

            # --- Corrected Annual Sum Calculation ---
            # To get the true annual sum, we must multiply the daily average by days in month.
            gdf["rain_mm_monthly_total"] = gdf["rain_mm"] * gdf["date"].dt.days_in_month
            total_rain = gdf.groupby("station_id")["rain_mm_monthly_total"].sum()

            # --- Finalization and Merging ---
            num_years = (gdf["date"].dt.year.max() - gdf["date"].dt.year.min()) + 1
            if num_years > 0:
                # Normalize total rain by years to get the average annual sum
                annual_sum = (total_rain / num_years).rename("rain_mm_sum_annual")

                # Merge all new metrics into the aggregated dataframe
                aggregated_gdf = aggregated_gdf.merge(max_monthly, on='station_id', how='left')
                aggregated_gdf = aggregated_gdf.merge(min_monthly, on='station_id', how='left')
                aggregated_gdf = aggregated_gdf.merge(avg_monthly, on='station_id', how='left')
                aggregated_gdf = aggregated_gdf.merge(annual_sum, on='station_id', how='left')
                print("  - Rainfall metrics calculated.")

        # Clean up and reorder columns for final output
        if not aggregated_gdf.empty:
            aggregated_gdf = finalize_gdf_columns(aggregated_gdf)

        # Save the processed layer
        output_layer_name = f"{station_type}_aggregated"
        print(f"  - Aggregation complete. Found {len(aggregated_gdf)} stations.")
        aggregated_gdf.to_file(output_gpkg, layer=output_layer_name, driver="GPKG")
        print(f"  - Saved to layer '{output_layer_name}' in '{output_gpkg.name}'.\n")

    print("--- All layers processed successfully! ---")


if __name__ == "__main__":
    main()