from pathlib import Path
import geopandas as gpd
import pandas as pd
import sys

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

from Utils.paths import GRID_PATH

# Threshold for unique values: numeric columns with <= this many distinct values
# are treated as ordinal/categorical rather than continuous.
ORDINAL_UNIQUE_THRESHOLD = 10

# List of column names excluded from analysis (e.g., index)
EXCLUDED_COLUMNS = ["index"]


def generate_datacube_summary(
    gpkg_path: Path,
    layer_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Generates summary statistics for continuous, ordinal, and text-categorical
    variables from a GeoPackage data cube.
    """
    # 1. Load GeoPackage layer & drop geometry for pure tabular analysis
    print(f"Loading layer '{layer_name}' from {gpkg_path}...")
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    df = gdf.drop(columns=['geometry'], errors='ignore')

    # 2. Separate variables into distinct schema lists
    text_cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns]

    # Separate numeric columns into continuous vs ordinal
    # Numeric columns with few unique values are treated as ordinal/categorical
    all_numeric = df.select_dtypes(include=['number']).columns.tolist()
    ordinal_numeric_cols = [
        c for c in all_numeric
        if df[c].nunique(dropna=False) <= ORDINAL_UNIQUE_THRESHOLD
    ]
    continuous_cols = [c for c in all_numeric if c not in ordinal_numeric_cols]

    # Exclude columns listed in EXCLUDED_COLUMNS from analysis
    continuous_cols = [c for c in continuous_cols if c not in EXCLUDED_COLUMNS]
    ordinal_numeric_cols = [c for c in ordinal_numeric_cols if c not in EXCLUDED_COLUMNS]
    text_cat_cols = [c for c in text_cat_cols if c not in EXCLUDED_COLUMNS]

    print(f"Detected {len(continuous_cols)} continuous numeric column(s): {continuous_cols}")
    print(f"Detected {len(ordinal_numeric_cols)} ordinal numeric column(s): {ordinal_numeric_cols}")
    print(f"Detected {len(text_cat_cols)} text categorical column(s): {text_cat_cols}")

    # --- A. Continuous Numerical Summary ---
    numeric_summary = pd.DataFrame()
    if continuous_cols:
        num_stats = df[continuous_cols].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
        num_stats['skewness'] = df[continuous_cols].skew()
        num_stats['zeros_count'] = (df[continuous_cols] == 0).sum()
        num_stats['zeros_pct'] = (num_stats['zeros_count'] / len(df)) * 100
        num_stats['null_count'] = df[continuous_cols].isnull().sum()

        numeric_summary = num_stats.round(4)
        # Make the variable names appear as a column instead of just as the index
        numeric_summary = numeric_summary.reset_index()

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
    # Build a dict mapping each text-categorical variable to its own DataFrame
    text_summaries: dict[str, pd.DataFrame] = {}
    for col in text_cat_cols:
        counts = df[col].value_counts(dropna=False, normalize=False)
        pcts = df[col].value_counts(dropna=False, normalize=True) * 100
        text_summaries[col] = pd.DataFrame({
            "Category": [str(cat_val) for cat_val in counts.index],
            "Count": counts.values,
            "Percentage (%)": [round(pcts[cat_val], 2) for cat_val in counts.index]
        })

    return numeric_summary, ordinal_summary, text_summaries


def main():
    gpkg_file = GRID_PATH / "h310_grid_final_datacube.gpkg"
    layer = "final_data_cube"

    num_sum, ord_sum, text_sum = generate_datacube_summary(
        gpkg_path=gpkg_file,
        layer_name=layer,
    )

    # Output to console
    print("\n=== CONTINUOUS NUMERICAL SUMMARY ===")
    print(num_sum)

    print("\n=== ORDINAL NUMERICAL SUMMARY ===")
    print(ord_sum)

    print("\n=== TEXT CATEGORICAL SUMMARY ===")
    print(text_sum)

    # Save numeric summaries to one workbook.
    output_excel = gpkg_file.parent / "datacube_summary_statistics.xlsx"
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        num_sum.to_excel(writer, sheet_name="Continuous_Numeric", index=False)
        ord_sum.to_excel(writer, sheet_name="Ordinal_Categories", index=False)

    print(f"\nSummary statistics report exported to: {output_excel}")

    # Save each text-categorical variable as its own sheet in a separate workbook.
    text_output_excel = gpkg_file.parent / "summary_statistics" / "datacube_text_category_statistics.xlsx"
    if text_sum:
        with pd.ExcelWriter(text_output_excel, engine="xlsxwriter") as writer:
            used_sheet_names = set()
            for variable, summary in text_sum.items():
                sheet_name = "".join(
                    "_" if character in "[]:*?/\\" else character
                    for character in variable
                )[:31] or "Text_Category"
                base_sheet_name = sheet_name
                suffix = 1
                while sheet_name in used_sheet_names:
                    suffix_text = f"_{suffix}"
                    sheet_name = f"{base_sheet_name[:31 - len(suffix_text)]}{suffix_text}"
                    suffix += 1
                used_sheet_names.add(sheet_name)
                summary.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(text_output_excel, engine="xlsxwriter") as writer:
            pd.DataFrame({"Message": ["No text categorical variables found."]}).to_excel(
                writer,
                sheet_name="No_Text_Categories",
                index=False,
            )

    print(f"Text categorical report exported to: {text_output_excel}")

if __name__ == "__main__":
    main()