import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from Utils
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4] 
sys.path.insert(0, str(project_root))

import colorsys
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon
from Utils.paths import GRID_PATH, GRID_SUMMARY_STATISTICS_PATH


# -----------------------------------------------------------------------------
# Color Generator Helper
# -----------------------------------------------------------------------------
def get_unique_colors(n_colors: int):
    """Generates n distinct, high-contrast RGB colors for clear plotting."""
    if n_colors == 0:
        return []
    
    if n_colors <= 20:
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
        return [palette[i] for i in range(n_colors)]
    
    # Dynamically generate HLS colors for large feature counts
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.75 + (i % 2) * 0.2  # Alternates saturation
        lightness = 0.4 + (i % 3) * 0.15   # Alternates lightness
        # Python's colorsys expects (Hue, Lightness, Saturation)
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


# -----------------------------------------------------------------------------
# Spatial Calculation Helpers
# -----------------------------------------------------------------------------
def compute_standard_deviational_ellipse(centroids, weights=None):
    """Calculates parameters for a 1-Standard Deviational Ellipse safely."""
    x = centroids.x.to_numpy()
    y = centroids.y.to_numpy()

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.nan_to_num(weights, nan=0.0)

    w_sum = weights.sum()
    if w_sum <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)

    dx = x - x_mean
    dy = y - y_mean

    sum_dx2 = np.sum(weights * (dx**2))
    sum_dy2 = np.sum(weights * (dy**2))
    sum_dx_dy = np.sum(weights * dx * dy)

    num = (sum_dx2 - sum_dy2) + np.sqrt(
        np.maximum(0, (sum_dx2 - sum_dy2) ** 2 + 4 * (sum_dx_dy**2))
    )
    den = 2 * sum_dx_dy

    theta_rad = 0.0 if den == 0 else np.arctan(num / den)

    # Use np.maximum(0, ...) to prevent floating-point precision sqrt errors
    var_x = np.sum(
        weights * (dx * np.cos(theta_rad) - dy * np.sin(theta_rad)) ** 2
    ) / w_sum
    var_y = np.sum(
        weights * (dx * np.sin(theta_rad) + dy * np.cos(theta_rad)) ** 2
    ) / w_sum

    std_x = np.sqrt(np.maximum(0, var_x))
    std_y = np.sqrt(np.maximum(0, var_y))

    theta_deg = np.degrees(theta_rad)

    return x_mean, y_mean, std_x, std_y, theta_deg


def create_ellipse_geometry(
    center_x, center_y, semi_major, semi_minor, rotation_deg, num_points=100
):
    """Generates a Shapely Polygon representing a Standard Deviational Ellipse."""
    if np.isnan(center_x) or np.isnan(semi_major) or semi_major == 0 or semi_minor == 0:
        return None

    angles = np.linspace(0, 2 * np.pi, num_points)
    unit_x = np.cos(angles)
    unit_y = np.sin(angles)

    base_poly = Polygon(zip(unit_x, unit_y))
    scaled_poly = scale(base_poly, xfact=semi_major, yfact=semi_minor)
    rotated_poly = rotate(scaled_poly, rotation_deg, origin=(0, 0))
    ellipse_poly = translate(rotated_poly, xoff=center_x, yoff=center_y)

    return ellipse_poly


# -----------------------------------------------------------------------------
# Batch Mapping Function
# -----------------------------------------------------------------------------
def save_batched_spatial_maps(
    gdf: gpd.GeoDataFrame, spatial_df: pd.DataFrame, output_dir: Path
):
    """Generates separate static figures with unique color maps per variable."""
    
    batch_definitions = {
        "spatial_map_epr_erd_eidx.png": lambda v: any(k in v.lower() for k in ["epr", "erd", "eidx"]),
        "spatial_map_idw_nearest.png": lambda v: any(k in v.lower() for k in ["idw", "nearest"]),
    }

    unweighted_row = spatial_df[
        spatial_df["variable"] == "UNWEIGHTED_GEOMETRY"
    ].iloc[0]

    weighted_df = spatial_df[spatial_df["variable"] != "UNWEIGHTED_GEOMETRY"].copy()

    assigned_vars = set()
    batches = {}

    for file_name, condition in batch_definitions.items():
        matched = weighted_df[weighted_df["variable"].apply(condition)]
        if not matched.empty:
            batches[file_name] = matched
            assigned_vars.update(matched["variable"].tolist())

    rest_df = weighted_df[~weighted_df["variable"].isin(assigned_vars)]
    if not rest_df.empty:
        batches["spatial_map_other_variables.png"] = rest_df

    for file_name, batch_df in batches.items():
        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Base Grid
        gdf.plot(
            ax=ax,
            facecolor="#f5f5f5",
            edgecolor="#d9d9d9",
            linewidth=0.2,
            alpha=0.8,
        )

        # 2. Unweighted Baseline (Black Reference)
        ax.scatter(
            unweighted_row["mean_center_x"],
            unweighted_row["mean_center_y"],
            color="black",
            marker="*",
            s=250,
            zorder=6,
            label="Unweighted Mean Center",
        )
        ax.scatter(
            unweighted_row["median_center_x"],
            unweighted_row["median_center_y"],
            color="black",
            marker="D",
            s=120,
            zorder=6,
            label="Unweighted Median Center",
        )

        unweighted_ellipse = create_ellipse_geometry(
            unweighted_row["mean_center_x"],
            unweighted_row["mean_center_y"],
            unweighted_row["sde_semi_major_axis"],
            unweighted_row["sde_semi_minor_axis"],
            unweighted_row["sde_rotation_deg"],
        )
        if unweighted_ellipse:
            gpd.GeoSeries([unweighted_ellipse]).plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linestyle="--",
                linewidth=2,
                zorder=5,
                label="Unweighted SDE",
            )

        # 3. Unique Color Assignment for Batch Variables
        unique_colors = get_unique_colors(len(batch_df))

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            var_name = row["variable"]
            color = unique_colors[idx]

            # Weighted Mean Center
            if not np.isnan(row["mean_center_x"]):
                ax.scatter(
                    row["mean_center_x"],
                    row["mean_center_y"],
                    color=color,
                    marker="o",
                    s=120,
                    edgecolors="black",
                    linewidth=0.5,
                    zorder=7,
                    label=f"{var_name} (Mean)",
                )

            # SDE Ellipse
            weighted_ellipse = create_ellipse_geometry(
                row["mean_center_x"],
                row["mean_center_y"],
                row["sde_semi_major_axis"],
                row["sde_semi_minor_axis"],
                row["sde_rotation_deg"],
            )
            if weighted_ellipse:
                gpd.GeoSeries([weighted_ellipse]).plot(
                    ax=ax,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=1.5,
                    zorder=4,
                )
                gpd.GeoSeries([weighted_ellipse]).plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor=color,
                    linestyle="-",
                    linewidth=1.5,
                    zorder=4,
                )

        title_tag = file_name.replace("spatial_map_", "").replace(".png", "").replace("_", " ").upper()
        ax.set_title(
            f"Spatial Statistics: {title_tag}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_axis_off()
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            fontsize=9,
        )

        output_file = output_dir / file_name
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Batch map saved to: {output_file}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def generate_geopackage_summary(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GeoPackage from: {input_path}")
    gdf = gpd.read_file(input_path)

    ignore_cols = {"h3_index", "geometry"}
    cols_to_analyze = [c for c in gdf.columns if c.lower() not in ignore_cols]

    float_cols = []
    int_cols = []
    string_cols = []

    for col in cols_to_analyze:
        dtype = gdf[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            float_cols.append(col)
        elif pd.api.types.is_integer_dtype(dtype):
            int_cols.append(col)
        elif pd.api.types.is_string_dtype(
            dtype
        ) or pd.api.types.is_object_dtype(dtype):
            string_cols.append(col)

    # 1. Aspatial Summaries
    if float_cols:
        float_summary = gdf[float_cols].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        ).T
        float_summary["missing_count"] = gdf[float_cols].isna().sum()
        float_summary["missing_pct"] = (float_summary["missing_count"] / len(gdf)) * 100
        float_summary["variance"] = gdf[float_cols].var()
        float_summary["skewness"] = gdf[float_cols].skew()

        col_order = [
            "count", "missing_count", "missing_pct", "mean", "std", "variance",
            "min", "5%", "25%", "50%", "75%", "95%", "max", "skewness"
        ]
        float_summary[col_order].to_csv(output_dir / "summary_floats_continuous.csv")

    if int_cols:
        int_summary = gdf[int_cols].describe(percentiles=[0.25, 0.50, 0.75]).T
        int_summary["missing_count"] = gdf[int_cols].isna().sum()
        int_summary["unique_categories"] = gdf[int_cols].nunique()
        int_summary["mode"] = gdf[int_cols].mode().iloc[0]

        col_order = [
            "count", "missing_count", "unique_categories", "min", "25%", 
            "50%", "75%", "max", "mean", "mode", "std"
        ]
        int_summary[col_order].to_csv(output_dir / "summary_integers_ordinal.csv")

    if string_cols:
        string_summary = gdf[string_cols].describe().T
        string_summary["missing_count"] = gdf[string_cols].isna().sum()
        string_summary["missing_pct"] = (string_summary["missing_count"] / len(gdf)) * 100
        string_summary = string_summary.rename(
            columns={
                "unique": "unique_categories",
                "top": "most_frequent_value",
                "freq": "most_frequent_count",
            }
        )
        string_summary["most_frequent_pct"] = (
            string_summary["most_frequent_count"] / string_summary["count"]
        ) * 100

        col_order = [
            "count", "missing_count", "missing_pct", "unique_categories", 
            "most_frequent_value", "most_frequent_count", "most_frequent_pct"
        ]
        string_summary[col_order].to_csv(output_dir / "summary_strings_categorical.csv")

    # 2. Spatial Metrics Calculation
    print("Calculating spatial metrics...")
    centroids = gdf.geometry.centroid

    mc_x, mc_y = centroids.x.mean(), centroids.y.mean()
    med_x, med_y = centroids.x.median(), centroids.y.median()
    _, _, std_x, std_y, theta = compute_standard_deviational_ellipse(centroids)

    spatial_metrics = [{
        "variable": "UNWEIGHTED_GEOMETRY",
        "mean_center_x": mc_x,
        "mean_center_y": mc_y,
        "median_center_x": med_x,
        "median_center_y": med_y,
        "sde_semi_major_axis": max(std_x, std_y),
        "sde_semi_minor_axis": min(std_x, std_y),
        "sde_rotation_deg": theta,
    }]

    for num_col in float_cols + int_cols:
        weights = gdf[num_col].to_numpy()
        w_x, w_y, w_std_x, w_std_y, w_theta = (
            compute_standard_deviational_ellipse(centroids, weights=weights)
        )

        spatial_metrics.append({
            "variable": num_col,
            "mean_center_x": w_x,
            "mean_center_y": w_y,
            "median_center_x": np.nan,
            "median_center_y": np.nan,
            "sde_semi_major_axis": max(w_std_x, w_std_y)
            if not np.isnan(w_std_x)
            else np.nan,
            "sde_semi_minor_axis": min(w_std_x, w_std_y)
            if not np.isnan(w_std_x)
            else np.nan,
            "sde_rotation_deg": w_theta,
        })

    spatial_df = pd.DataFrame(spatial_metrics)
    spatial_df.to_csv(output_dir / "spatial_statistics_summary.csv", index=False)

    # 3. Render Batched Maps
    print("Generating batched spatial map figures...")
    save_batched_spatial_maps(gdf, spatial_df, output_dir)

    print(f"\nExecution complete! Output directory:\n{output_dir}")


if __name__ == "__main__":
    gpkg_input = GRID_PATH / "h310_grid_final_datacube.gpkg"
    output_dir = GRID_SUMMARY_STATISTICS_PATH / "output"

    generate_geopackage_summary(gpkg_input, output_dir)