import ast
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

# Setup project root imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]  # Adjust parent index if needed
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

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

OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "summary_table"


def parse_list_column(value) -> list:
    """Safely parse string representations of list literals (e.g. "['noun1', 'noun2']")."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    value = value.strip()
    if not value or value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def compute_combinations(subset_df: pd.DataFrame) -> dict:
    """Helper function to compute counts and proportions for the 7 combination keys."""
    total = len(subset_df)
    if total == 0:
        keys = [
            "all_three",
            "tech_and_attr_only",
            "attr_and_fail_only",
            "tech_and_fail_only",
            "singular_tech_only",
            "singular_attr_only",
            "singular_fail_only",
        ]
        return {k: {"count": 0, "proportion": 0.0} for k in keys}

    comb_all_three = (
        subset_df["has_technical"]
        & subset_df["has_attribute"]
        & subset_df["has_failures"]
    )
    comb_tech_attr_only = (
        subset_df["has_technical"]
        & subset_df["has_attribute"]
        & (~subset_df["has_failures"])
    )
    comb_attr_fail_only = (
        subset_df["has_attribute"]
        & subset_df["has_failures"]
        & (~subset_df["has_technical"])
    )
    comb_tech_fail_only = (
        subset_df["has_technical"]
        & subset_df["has_failures"]
        & (~subset_df["has_attribute"])
    )
    comb_tech_only = (
        subset_df["has_technical"]
        & (~subset_df["has_attribute"])
        & (~subset_df["has_failures"])
    )
    comb_attr_only = (
        subset_df["has_attribute"]
        & (~subset_df["has_technical"])
        & (~subset_df["has_failures"])
    )
    comb_fail_only = (
        subset_df["has_failures"]
        & (~subset_df["has_technical"])
        & (~subset_df["has_attribute"])
    )

    mapping = [
        ("all_three", comb_all_three),
        ("tech_and_attr_only", comb_tech_attr_only),
        ("attr_and_fail_only", comb_attr_fail_only),
        ("tech_and_fail_only", comb_tech_fail_only),
        ("singular_tech_only", comb_tech_only),
        ("singular_attr_only", comb_attr_only),
        ("singular_fail_only", comb_fail_only),
    ]

    occurrences = {}
    for key, series in mapping:
        cnt = int(series.sum())
        prop = cnt / total if total > 0 else 0.0
        occurrences[key] = {"count": cnt, "proportion": prop}

    return occurrences


def generate_summary_stats(df: pd.DataFrame) -> dict:
    # -----------------------------------------------------------------
    # 1. Parse List Columns
    # -----------------------------------------------------------------
    list_columns = [
        "occurrence_technical",
        "occurrence_failures",
        "occurrence_causal",
        "occurrence_variable",
        "nouns",
        "verbs",
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_column)

    # Determine row-level (sentence-level) presence flags
    df["has_technical"] = df["occurrence_technical"].apply(lambda x: len(x) > 0)
    df["has_failures"] = df["occurrence_failures"].apply(lambda x: len(x) > 0)
    df["has_attribute"] = df["occurrence_variable"].apply(lambda x: len(x) > 0)

    # -----------------------------------------------------------------
    # 2. Sentence-Level Statistics (Row-by-Row)
    # -----------------------------------------------------------------
    total_sentences = len(df)
    sentence_occurrences = compute_combinations(df)

    # -----------------------------------------------------------------
    # 3. Paragraph-Level Aggregation (Uses pat_id + paragraph_id)
    # -----------------------------------------------------------------
    paragraph_grouped = (
        df.groupby(["pat_id", "paragraph_id"])
        .agg(
            has_technical=("has_technical", "any"),
            has_failures=("has_failures", "any"),
            has_attribute=("has_attribute", "any"),
        )
        .reset_index()
    )
    total_paragraphs = len(paragraph_grouped)
    paragraph_occurrences = compute_combinations(paragraph_grouped)

    # -----------------------------------------------------------------
    # 4. Patent-Level Aggregation
    # -----------------------------------------------------------------
    patent_grouped = (
        df.groupby("pat_id")
        .agg(
            appearances=("paragraph_id", "count"),  # count of sentence rows per patent
            has_technical=("has_technical", "any"),
            has_failures=("has_failures", "any"),
            has_attribute=("has_attribute", "any"),
            mean_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    total_patents = len(patent_grouped)
    avg_appearances_per_patent = patent_grouped["appearances"].mean()
    patent_occurrences = compute_combinations(patent_grouped)

    # -----------------------------------------------------------------
    # 5. Sentiment Score Statistics
    # -----------------------------------------------------------------
    sentiment_mean = df["sentiment_score"].mean()
    sentiment_std = df["sentiment_score"].std()

    # -----------------------------------------------------------------
    # 6. Top 10 Nouns and Top 10 Verbs
    # -----------------------------------------------------------------
    all_nouns = [
        noun for sublist in df["nouns"] for noun in sublist if isinstance(sublist, list)
    ]
    all_verbs = [
        verb for sublist in df["verbs"] for verb in sublist if isinstance(sublist, list)
    ]

    top_10_nouns = Counter(all_nouns).most_common(10)
    top_10_verbs = Counter(all_verbs).most_common(10)

    # Build Summary Result Object
    return {
        "total_rows": total_sentences,
        "total_unique_patents": total_patents,
        "total_paragraphs": total_paragraphs,
        "avg_appearances_per_patent": avg_appearances_per_patent,
        "sentence_occurrences": sentence_occurrences,
        "paragraph_occurrences": paragraph_occurrences,
        "patent_occurrences": patent_occurrences,
        "sentiment_mean": sentiment_mean,
        "sentiment_std": sentiment_std,
        "top_10_nouns": top_10_nouns,
        "top_10_verbs": top_10_verbs,
    }


def main():
    print(f"Loading data from: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    stats = generate_summary_stats(df)

    # Print Results to Console
    print("\n" + "=" * 60)
    print("PATENT DATASET SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total Sentence Rows: {stats['total_rows']}")
    print(f"Total Unique Paragraphs: {stats['total_paragraphs']}")
    print(f"Total Unique Patents: {stats['total_unique_patents']}")
    print(
        f"Mean Patent Appearances (Sentences/Patent): {stats['avg_appearances_per_patent']:.2f}"
    )

    labels = [
        ("Contains Technical + Attribute + Failure", "all_three"),
        ("Contains Technical + Attribute ONLY", "tech_and_attr_only"),
        ("Contains Attribute + Failure ONLY", "attr_and_fail_only"),
        ("Contains Technical + Failure ONLY", "tech_and_fail_only"),
        ("Singular - Technical ONLY", "singular_tech_only"),
        ("Singular - Attributes ONLY", "singular_attr_only"),
        ("Singular - Failures ONLY", "singular_fail_only"),
    ]

    # --- Patent Level ---
    print("\n--- OCCURRENCE PRESENCE & PROPORTIONS (PATENT LEVEL) ---")
    occ_pat = stats["patent_occurrences"]
    for label, key in labels:
        cnt = occ_pat[key]["count"]
        prop = occ_pat[key]["proportion"]
        print(f"{label:<42}: {cnt:<5} ({prop:.2%})")

    # --- Paragraph Level ---
    print("\n--- OCCURRENCE PRESENCE & PROPORTIONS (PARAGRAPH LEVEL) ---")
    occ_para = stats["paragraph_occurrences"]
    for label, key in labels:
        cnt = occ_para[key]["count"]
        prop = occ_para[key]["proportion"]
        print(f"{label:<42}: {cnt:<5} ({prop:.2%})")

    # --- Sentence Level ---
    print("\n--- OCCURRENCE PRESENCE & PROPORTIONS (SENTENCE LEVEL) ---")
    occ_sent = stats["sentence_occurrences"]
    for label, key in labels:
        cnt = occ_sent[key]["count"]
        prop = occ_sent[key]["proportion"]
        print(f"{label:<42}: {cnt:<5} ({prop:.2%})")

    print("\n--- SENTIMENT SCORE ---")
    print(f"Mean Sentiment Score: {stats['sentiment_mean']:.4f}")
    print(f"Std Sentiment Score:  {stats['sentiment_std']:.4f}")

    print("\n--- TOP 10 NOUNS ---")
    for noun, count in stats["top_10_nouns"]:
        print(f"  {noun:<20}: {count}")

    print("\n--- TOP 10 VERBS ---")
    for verb, count in stats["top_10_verbs"]:
        print(f"  {verb:<20}: {count}")

    # Export Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_DIR / "summary_metrics.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("PATENT DATASET SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Sentence Rows: {stats['total_rows']}\n")
        f.write(f"Total Unique Paragraphs: {stats['total_paragraphs']}\n")
        f.write(f"Total Unique Patents: {stats['total_unique_patents']}\n")
        f.write(
            f"Mean Patent Appearances: {stats['avg_appearances_per_patent']:.2f}\n\n"
        )

        for tier_name, occ_dict in [
            ("PATENT LEVEL", stats["patent_occurrences"]),
            ("PARAGRAPH LEVEL", stats["paragraph_occurrences"]),
            ("SENTENCE LEVEL", stats["sentence_occurrences"]),
        ]:
            f.write(f"--- OCCURRENCE PRESENCE & PROPORTIONS ({tier_name}) ---\n")
            for key, item in occ_dict.items():
                f.write(
                    f"  {key:<20}: Count = {item['count']}, Proportion = {item['proportion']:.4f} ({item['proportion']:.2%})\n"
                )
            f.write("\n")

        f.write(f"SENTIMENT:\n")
        f.write(f"  Mean: {stats['sentiment_mean']:.4f}\n")
        f.write(f"  Std: {stats['sentiment_std']:.4f}\n\n")
        
        f.write("TOP 10 NOUNS:\n")
        for word, count in stats["top_10_nouns"]:
            f.write(f"  {word}: {count}\n")
            
        f.write("\nTOP 10 VERBS:\n")
        for word, count in stats["top_10_verbs"]:
            f.write(f"  {word}: {count}\n")

    print(f"\nSummary stats exported to: {summary_file}")


if __name__ == "__main__":
    main()