# =====================================================================
# NOTEE FOR CONTEXT, the current script does the causal links generation but since 
# it uses dictionaries it collapses all occurrences that repeat to a single
# dict key: [Technical, Attribute, Failure] hence the lower number of rows
# =====================================================================

import ast
from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys

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
    / "causal_filtered_unsupervised_sentiment_pos.csv"
)

OUTPUT_DIR = CAUSAL_RELATION_MODEL_PATH / "precision_test"
SUMMARY_CSV_FILE = OUTPUT_DIR / "precision_test.csv"

MAX_ROWS = None

# Always include text column in CSV output
TEXT_SENTENCES = True

# Random Selector Parameters:
# - SAMPLE_SIZE: Set to an integer (e.g., 500) to randomly sample that many chains, or None to include all.
# - RANDOM_SEED: Set to an integer for reproducible random sampling, or None for non-deterministic behavior.
SAMPLE_SIZE = 367
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
# 2. STREAMLINED DIRECT CSV EXPORTER PIPELINE
# =====================================================================
def generate_causal_chains_csv(input_path, output_filename=SUMMARY_CSV_FILE, max_rows=MAX_ROWS):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Reading dataset from '{input_path}'...")
    df = pd.read_csv(input_path, nrows=max_rows)

    # Dictionary to aggregate chain data directly by (Technical, Attribute, Failure) key
    chain_aggregates = defaultdict(lambda: {
        "patents": set(),
        "triggers": set(),
        "nouns": set(),
        "verbs": set(),
        "occurrences": []
    })

    print("[Pipeline] Extracting causal chains and occurrences...")
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

        if not (causals and techs and attrs and failures):
            continue

        trigger = str(causals[0]).lower()
        occ_entry = {"pat_id": pat_id, "para_id": para_id, "text": text_val}

        # Extract strict [Technical] -> [Attribute] -> [Failure] combinations per row
        for t, a, f in itertools.product(techs, attrs, failures):
            if t != a and a != f and t != f:
                key = (t, a, f)
                agg = chain_aggregates[key]
                agg["patents"].add(pat_id)
                agg["triggers"].add(trigger)
                agg["nouns"].update(nouns)
                agg["verbs"].update(verbs)
                agg["occurrences"].append(occ_entry)

    records = []
    for (t_node, a_node, f_node), data in chain_aggregates.items():
        all_patents = sorted(list(data["patents"]))
        all_triggers = sorted(list(data["triggers"]))
        chain_nouns = sorted(list(data["nouns"]))
        chain_verbs = sorted(list(data["verbs"]))
        occurrences = data["occurrences"]

        has_complete = len(occurrences) > 0
        flag = "Complete" if has_complete else "Partial"

        # Grab a single representative text sentence instead of keeping all of them combined
        rep_text = ""
        for occ in occurrences:
            if occ["text"]:
                rep_text = occ["text"]
                break

        rec = {
            "technical": t_node,
            "attribute": a_node,
            "failure": f_node,
            "flag": flag,
        }

        if TEXT_SENTENCES:
            rec["link_text"] = rep_text

        rec.update({
            "patent_ids": ", ".join(all_patents),
            "patent_count": len(all_patents),
            "triggers": ", ".join(all_triggers),
            "nouns": ", ".join(chain_nouns),
            "verbs": ", ".join(chain_verbs),
            "noun_count": len(chain_nouns),
            "verb_count": len(chain_verbs),
        })

        records.append(rec)

    print(f"Total Unique Causal Chains Found: {len(records)}")

    # Random Selector for Rows if SAMPLE_SIZE is set
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(records):
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        records = random.sample(records, SAMPLE_SIZE)
        print(f"Randomly sampled {SAMPLE_SIZE} chains for export (Seed: {RANDOM_SEED}).")

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(output_path, index=False)
    print(f"[CSV Exporter] Saved summary report directly to: {output_path}")
    return df_summary


if __name__ == "__main__":
    print("\n=== Processing Streamlined Causal Chain CSV Exporter ===")
    generate_causal_chains_csv(INPUT_CSV_PATH, output_filename=SUMMARY_CSV_FILE)
    print("\n[Done] CSV export pipeline finished successfully.")