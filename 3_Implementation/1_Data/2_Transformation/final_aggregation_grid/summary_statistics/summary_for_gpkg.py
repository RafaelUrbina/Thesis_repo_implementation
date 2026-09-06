from pathlib import Path
import geopandas as gpd
import pandas as pd

def generate_datacube_summary(
    gpkg_path: Path, 
    layer_name: str, 
    ordinal_cols: list[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates summary statistics for continuous, ordinal, and text-categorical
    variables from a GeoPackage data cube.
    """
    ordinal_cols = ordinal_cols or []
    
    # 1. Load GeoPackage layer & drop geometry for pure tabular analysis
    print(f"Loading layer '{layer_name}' from {gpkg_path}...")
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    df = gdf.drop(columns=['geometry'], errors='ignore')

    # 2. Separate variables into distinct schema lists
    text_cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns]
    
    # Separate numeric columns into continuous vs ordinal
    all_numeric = df.select_dtypes(include=['number']).columns.tolist()
    ordinal_numeric_cols = [c for c in all_numeric if c in ordinal_cols]
    continuous_cols = [c for c in all_numeric if c not in ordinal_cols]

    # --- A. Continuous Numerical Summary ---
    numeric_summary = pd.DataFrame()
    if continuous_cols:
        num_stats = df[continuous_cols].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
        num_stats['skewness'] = df[continuous_cols].skew()
        num_stats['zeros_count'] = (df[continuous_cols] == 0).sum()
        num_stats['zeros_pct'] = (num_stats['zeros_count'] / len(df)) * 100
        num_stats['null_count'] = df[continuous_cols].isnull().sum()
        
        numeric_summary = num_stats.round(4)

    # --- B. Ordinal Category Summary ---
    ordinal_summary_list = []
    for col in ordinal_numeric_cols:
        counts = df[col].value_counts(dropna=False, normalize=False)
        pcts = df[col].value_counts(dropna=False, normalize=True) * 100
        
        # Calculate ordinal-specific metrics (Median & IQR)
        med_val = df[col].median()
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        
        for cat_val in counts.index:
            ordinal_summary_list.append({
                "Variable": col,
                "Category_Code": cat_val,
                "Count": counts[cat_val],
                "Percentage (%)": round(pcts[cat_val], 2),
                "Median": med_val,
                "IQR": q75 - q25
            })
    ordinal_summary = pd.DataFrame(ordinal_summary_list)

    # --- C. Text Categorical Summary ---
    text_summary_list = []
    for col in text_cat_cols:
        counts = df[col].value_counts(dropna=False, normalize=False)
        pcts = df[col].value_counts(dropna=False, normalize=True) * 100
        for cat_val in counts.index:
            text_summary_list.append({
                "Variable": col,
                "Category": str(cat_val),
                "Count": counts[cat_val],
                "Percentage (%)": round(pcts[cat_val], 2)
            })
    text_summary = pd.DataFrame(text_summary_list)

    return numeric_summary, ordinal_summary, text_summary


def main():
    gpkg_file = Path("path/to/GRID_PATH") / "h310_grid_final_datacube.gpkg"
    layer = "final_data_cube"

    # Define any integer/numeric columns that represent ordered categories (e.g., land cover codes, risk ranks)
    ordinal_columns = ["slope_risk_class", "land_use_code", "station_density_rank"]

    num_sum, ord_sum, text_sum = generate_datacube_summary(
        gpkg_path=gpkg_file,
        layer_name=layer,
        ordinal_cols=ordinal_columns
    )

    # Output to console
    print("\n=== CONTINUOUS NUMERICAL SUMMARY ===")
    print(num_sum)

    print("\n=== ORDINAL NUMERICAL SUMMARY ===")
    print(ord_sum)

    print("\n=== TEXT CATEGORICAL SUMMARY ===")
    print(text_sum)

    # Save to Excel report with multiple tabs
    output_excel = gpkg_file.parent / "datacube_summary_statistics.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        num_sum.to_excel(writer, sheet_name="Continuous_Numeric")
        ord_sum.to_excel(writer, sheet_name="Ordinal_Categories", index=False)
        text_sum.to_excel(writer, sheet_name="Text_Categories", index=False)

    print(f"\nSummary statistics report exported to: {output_excel}")

if __name__ == "__main__":
    main()