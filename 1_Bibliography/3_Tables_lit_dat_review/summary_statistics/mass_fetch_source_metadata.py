import csv
import os
from collections import Counter
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]
sys.path.insert(0, str(project_root))

from Utils.paths import SUMMARY_STATISTICS_PATH

file_path = SUMMARY_STATISTICS_PATH / "mass_fetch_data_meta" / "df_for_mapping_FINAL_READY.csv"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total_rows = len(rows)
print(f"Total rows in dataset: {total_rows}\n")

columns_to_summarize = [
    'MacroArea',
    'Source_DOMAIN',
    'Time cadence (or last uploaded or time of data)_CLEANED',
    'Territory_REG',
    'Data Type_CLEANED'
]

output_dir = SUMMARY_STATISTICS_PATH / "outputmassdata"
os.makedirs(output_dir, exist_ok=True)

for col in columns_to_summarize:
    header = f"=== Summary Statistics for: {col} ==="
    print(header)
    
    if col == 'Data Type_CLEANED':
        tokens = []
        for row in rows:
            v = row.get(col, '')
            if v is not None:
                v = v.strip()
                if v.startswith('[') and v.endswith(']'):
                    v = v[1:-1].strip()
                if v:
                    row_tokens = v.split()
                    tokens.extend(row_tokens)
        counter = Counter(tokens)
    else:
        values = [row.get(col, '').strip() if row.get(col, '') is not None else '' for row in rows]
        values = [v if v != '' else '<NA>' for v in values]
        counter = Counter(values)
    
    sorted_counts = counter.most_common()
    
    # Generate clean CSV filename
    clean_col_name = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    csv_filename = f"summary_{clean_col_name}.csv"
    csv_path = output_dir / csv_filename
    
    with open(csv_path, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Value', 'Count', 'Percentage (%)'])
        for val, count in sorted_counts:
            pct = (count / total_rows) * 100
            writer.writerow([val, count, f"{pct:.2f}%"])
            
    print(f"Saved: {csv_path}\n" + "-"*50 + "\n")

print(f"All summary CSV reports successfully generated in {output_dir}")


