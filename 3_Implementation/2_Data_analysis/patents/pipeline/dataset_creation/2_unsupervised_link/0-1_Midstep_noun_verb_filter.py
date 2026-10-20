import ast
from pathlib import Path
import re
import sys
import pandas as pd
import json

# Setup project root imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]  # Adjust parent index if needed
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "dataset_creation"
    / "2_unsupervised_link"
    / "output"
    / "causal_filtered_unsupervised_sentiment_pos.csv"
)

# Output path in the exact same directory with a filtered suffix (or overwrite if desired)
OUTPUT_CSV_PATH = INPUT_CSV_PATH.parent / "causal_filtered_unsupervised_sentiment_pos_strict.csv"

# Part-of-Speech Filtering Thresholds (Adjustable)
MAX_VERBS_ALLOWED = 7
MAX_NOUNS_ALLOWED = 15

MAX_ROWS = None


# =====================================================================
# PARSING HELPER
# =====================================================================
def parse_field(field_val):
    """Safely parses stringified Python lists, JSON arrays, or raw list/string objects."""
    if isinstance(field_val, list):
        return [str(i).strip() for i in field_val if str(i).strip()]

    if pd.isna(field_val) or field_val is None:
        return []

    if not isinstance(field_val, str):
        val_str = str(field_val).strip()
        return [val_str] if val_str else []

    cleaned = field_val.strip()
    if not cleaned or cleaned.lower() == "nan":
        return []

    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple, set)):
                return [str(i).strip() for i in parsed if str(i).strip()]
        except (ValueError, SyntaxError):
            pass

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if str(i).strip()]
        except json.JSONDecodeError:
            pass

        tokens = re.findall(r"[^\s,\[\]'\"']+", cleaned)
        return [t.strip() for t in tokens if t.strip()]

    if "," in cleaned:
        return [
            item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()
        ]

    return [cleaned.strip("'\"")]


# =====================================================================
# MAIN FILTERING PIPELINE
# =====================================================================
def filter_dataset_by_pos(input_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Reading dataset from '{input_path}'...")
    df = pd.read_csv(input_path, nrows=MAX_ROWS)
    initial_row_count = len(df)

    filtered_rows = []

    print("[Pipeline] Applying strict POS filtering on verbs and nouns...")
    for _, row in df.iterrows():
        verbs = parse_field(row.get("verbs", []))
        nouns = parse_field(row.get("nouns", []))

        # Skip rows exceeding either threshold
        if len(verbs) > MAX_VERBS_ALLOWED or len(nouns) > MAX_NOUNS_ALLOWED:
            continue

        filtered_rows.append(row)

    df_filtered = pd.DataFrame(filtered_rows)
    df_filtered.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print(f"Initial Rows: {initial_row_count:,}")
    print(f"Filtered Rows Retained: {len(df_filtered):,}")
    print(f"Rows Dropped: {initial_row_count - len(df_filtered):,}")
    print(f"[Success] Filtered dataset saved to: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    print("\n=== Processing POS Dataset Filter ===")
    filter_dataset_by_pos(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    print("\n[Done] Filtering pipeline finished successfully.")