import ast
from pathlib import Path
import random
import re
import sys
import json

import pandas as pd

# Setup project root imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]  # Adjust parent index if needed
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH, CAUSAL_RELATION_MODEL_PATH

# =====================================================================
# CONFIGURATION & OUTPUT PATHS
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "dataset_creation"
    / "2_unsupervised_link"
    / "output"
    / "causal_filtered_unsupervised_sentiment_pos_strict.csv"
)

OUTPUT_DIR = CAUSAL_RELATION_MODEL_PATH / "precision_test"
SUMMARY_CSV_FILE = OUTPUT_DIR / "precision_test.csv"
TESTED_CSV_FILE = OUTPUT_DIR / "tested_precision_test.csv"
METRICS_TXT_FILE = OUTPUT_DIR / "precision_metrics.txt"

MAX_ROWS = None

# Part-of-Speech Filtering Thresholds (Adjustable)
MAX_VERBS_ALLOWED = 7
MAX_NOUNS_ALLOWED = 15

# Random Selector Parameters:
# - SAMPLE_SIZE: Set to an integer (e.g., 367) to randomly sample that many observations, or None to include all.
# - RANDOM_SEED: Set to an integer for reproducible random sampling, or None for non-deterministic behavior.
SAMPLE_SIZE = 381
RANDOM_SEED = 42


# =====================================================================
# 1. PARSING HELPER
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
# 2. PRECISION METRICS EVALUATOR & TXT EXPORTER
# =====================================================================
def evaluate_precision(tested_path=TESTED_CSV_FILE, metrics_txt_path=METRICS_TXT_FILE):
    """Reads tested_precision_test.csv if it exists, calculates precision metrics, and saves them to a text file."""
    tested_path = Path(tested_path)
    metrics_txt_path = Path(metrics_txt_path)
    
    if not tested_path.exists():
        print(f"\n[Metrics] Tested file not found at '{tested_path}'. Skipping precision evaluation.")
        return

    print(f"\n=== Precision Metrics Evaluation ===")
    print(f"[Metrics] Reading evaluated dataset from '{tested_path}'...")
    try:
        df_tested = pd.read_csv(tested_path)
    except Exception as e:
        print(f"[Error] Failed to read {tested_path}: {e}")
        return

    if "eval" not in df_tested.columns:
        print("[Warning] The 'eval' column was not found in the tested CSV file.")
        return

    # Clean and ensure binary format for eval column
    eval_series = pd.to_numeric(df_tested["eval"], errors="coerce")
    valid_mask = eval_series.isin([0, 1])
    
    total_rows = len(df_tested)
    evaluated_rows = valid_mask.sum()
    
    if evaluated_rows == 0:
        print("[Metrics] No valid binary (0 or 1) values found in the 'eval' column.")
        return

    clean_evals = eval_series[valid_mask].astype(int)
    
    total_positives = (clean_evals == 1).sum()
    total_negatives = (clean_evals == 0).sum()
    precision = total_positives / evaluated_rows

    # Format output text block
    metrics_output = (
        "=== PRECISION TEST METRICS REPORT ===\n"
        f"Tested File Source: {tested_path.name}\n"
        f"--------------------------------------------------\n"
        f" Total Rows in Tested Dataset : {total_rows}\n"
        f" Valid Evaluated Rows         : {evaluated_rows}\n"
        f" Positive Predictions (1)     : {total_positives}\n"
        f" Negative Predictions (0)     : {total_negatives}\n"
        f" Overall Precision            : {precision:.4f} ({precision * 100:.2f}%)\n"
        f"--------------------------------------------------\n"
    )

    # Print to console
    print(metrics_output)

    # Save to text file
    metrics_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(metrics_output)
    print(f"[Metrics] Saved precision report to text file: {metrics_txt_path}")


# =====================================================================
# 3. STREAMLINED SINGLE-OBSERVATION CSV EXPORTER
# =====================================================================
def generate_single_observation_csv(input_path, output_filename=SUMMARY_CSV_FILE, max_rows=MAX_ROWS):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Reading dataset from '{input_path}'...")
    df = pd.read_csv(input_path, nrows=max_rows)

    records = []

    print("[Pipeline] Extracting and filtering single observations...")
    for _, row in df.iterrows():
        pat_id = str(row.get("pat_id", "UNKNOWN")).strip()
        para_id = str(row.get("paragraph_id", "UNKNOWN")).strip()

        text_val = ""
        for col in ["text", "paragraph_text", "content", "sentence", "abstract"]:
            if col in row and pd.notna(row[col]):
                text_val = str(row[col]).strip()
                break

        techs = parse_field(row.get("occurrence_technical", []))
        attrs = parse_field(row.get("occurrence_variable", []))
        failures = parse_field(row.get("occurrence_failures", []))
        causals = parse_field(row.get("causal_ocurrence_words", []))

        nouns = parse_field(row.get("nouns", []))
        verbs = parse_field(row.get("verbs", []))

        # 1. Apply POS Threshold Filters directly to the observation row
        if len(verbs) > MAX_VERBS_ALLOWED or len(nouns) > MAX_NOUNS_ALLOWED:
            continue

        # 2. Ensure row contains all required causal components
        if not (causals and techs and attrs and failures):
            continue

        # Build a single observation record representing this specific row
        rec = {
            "pat_id": pat_id,
            "paragraph_id": para_id,
            "technical": ", ".join(techs),
            "attribute": ", ".join(attrs),
            "failure": ", ".join(failures),
            "trigger": causals[0].lower() if causals else "",
            "link_text": text_val,
            "nouns": ", ".join(sorted(nouns)),
            "verbs": ", ".join(sorted(verbs)),
            "noun_count": len(nouns),
            "verb_count": len(verbs),
        }

        records.append(rec)

    print(f"Total Valid Observations Found (after filtering): {len(records)}")

    # Random Selector for Rows if SAMPLE_SIZE is set
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(records):
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        records = random.sample(records, SAMPLE_SIZE)
        print(f"Randomly sampled {SAMPLE_SIZE} observations for export (Seed: {RANDOM_SEED}).")

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(output_path, index=False)
    print(f"[CSV Exporter] Saved summary report directly to: {output_path}")
    return df_summary


if __name__ == "__main__":
    print("\n=== Processing Single Observation CSV Exporter ===")
    generate_single_observation_csv(INPUT_CSV_PATH, output_filename=SUMMARY_CSV_FILE)
    
    # Check, evaluate precision metrics, and export to TXT if tested file exists
    evaluate_precision(TESTED_CSV_FILE, METRICS_TXT_FILE)
    
    print("\n[Done] CSV export pipeline finished successfully.")