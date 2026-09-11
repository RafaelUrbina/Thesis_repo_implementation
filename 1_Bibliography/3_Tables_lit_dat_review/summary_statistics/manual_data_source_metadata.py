import json
import os
import pandas as pd
from difflib import SequenceMatcher


import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]
sys.path.insert(0, str(project_root))

from Utils.paths import SUMMARY_STATISTICS_PATH

# 1. Load the JSON metadata
json_file_path = SUMMARY_STATISTICS_PATH / "metadata_datasources.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
total_records = len(df)

# 2. String similarity clustering function (50% character similarity threshold)
def group_similar_terms(series, similarity_threshold=0.50):
    # Lowercase & strip whitespace for normalized comparison
    normalized = series.dropna().astype(str).str.strip().str.lower()
    mapping = {}
    canonical_labels = {}
    
    for original, norm in zip(series, normalized):
        if not norm or pd.isna(original):
            continue
        found_group = False
        
        # Compare against existing canonical cluster representatives
        for canon_norm in canonical_labels:
            if SequenceMatcher(None, norm, canon_norm).ratio() >= similarity_threshold:
                mapping[original] = canonical_labels[canon_norm]
                found_group = True
                break
                
        # Register new canonical representative if no match found
        if not found_group:
            canonical_labels[norm] = original
            mapping[original] = original
            
    return series.map(mapping)

# 3. Apply 50% similarity clustering to ALL text variables
target_columns = ['Data Type', 'Source', 'Access (Open or Not)', 'Possible Issues']

grouped_cols = {}
for col in target_columns:
    if col in df.columns:
        grouped_col_name = f"{col} Grouped"
        df[grouped_col_name] = group_similar_terms(df[col], similarity_threshold=0.30)
        grouped_cols[col] = grouped_col_name

# 4. Helper function to generate clean individual summary DataFrames
def get_breakdown_df(column_name):
    counts = df[column_name].value_counts(dropna=False)
    records = []
    for item, count in counts.items():
        pct = (count / total_records) * 100
        records.append({
            "Category": item if pd.notna(item) else "N/A",
            "Count": count,
            "Percentage (%)": f"{pct:.2f}%"
        })
    return pd.DataFrame(records)

# 5. Export results to separate CSV files (Robust and dependency-free)
output_dir = SUMMARY_STATISTICS_PATH / "outputdata"
os.makedirs(output_dir, exist_ok=True)

for original_col, grouped_col_name in grouped_cols.items():
    summary_df = get_breakdown_df(grouped_col_name)
    
    # Generate clean file name (e.g., summary_access_open_or_not.csv)
    clean_fname = f"summary_{original_col.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    csv_path = os.path.join(output_dir, clean_fname)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved: {csv_path}")

# 6. Optional: Export to multi-sheet Excel file if openpyxl is available
try:
    excel_path = os.path.join(output_dir, "dataset_summary_statistics_grouped.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for original_col, grouped_col_name in grouped_cols.items():
            sheet_title = original_col[:30].replace('/', '_')
            get_breakdown_df(grouped_col_name).to_excel(writer, sheet_name=sheet_title, index=False)
    print(f"Excel summary saved: {excel_path}")
except ImportError:
    print("openpyxl not installed — skipping Excel workbook export (CSVs generated successfully).")