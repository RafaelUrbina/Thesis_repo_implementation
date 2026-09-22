import sys
from pathlib import Path
import ast
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter

# =====================================================================
# Setup project root imports & Path Logic
# =====================================================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]  # Adjust parent index if needed
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "supervised_link"
    / "output"
    / "causal_filtered_supervised_exploded.csv"
)

OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "summary_statistics"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output File Destinations
TOP_ASSOC_TECH_ATTR_CSV = OUTPUT_DIR / "top_associations_tech_vs_attr.csv"
TOP_ASSOC_ATTR_FAIL_CSV = OUTPUT_DIR / "top_associations_attr_vs_fail.csv"

HEATMAP_TECH_ATTR_PNG = OUTPUT_DIR / "heatmap_top20_tech_vs_attr.png"
HEATMAP_ATTR_FAIL_PNG = OUTPUT_DIR / "heatmap_top20_attr_vs_fail.png"

SUMMARY_CARDS_CSV = OUTPUT_DIR / "patent_summary_cards.csv"
SUMMARY_CARDS_JSON = OUTPUT_DIR / "patent_summary_cards.json"


# ==========================================
# 1. DATA LOADING & PARSING FUNCTION
# ==========================================

def load_and_preprocess_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads dataset and parses string-encoded list columns 
    (e.g., "['reservoir', 'pump']") back into Python lists.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    list_columns = [
        "occurrence_technical",
        "occurrence_failures",
        "occurrence_causal",
        "occurrence_variable",
        "causal_ocurrence_words"
    ]

    def safe_parse_list(val):
        if isinstance(val, list):
            return val
        if pd.isna(val) or not val:
            return []
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return []
            elif val:
                return [x.strip() for x in val.split(",") if x.strip()]
        return []

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)

    return df


# ==========================================
# TASK 1: TOP ASSOCIATION TABLES (NPMI BASED, NO P-VALUE)
# ==========================================

def compute_npmi_associations(
    df: pd.DataFrame, 
    col_a: str, 
    col_b: str, 
    min_freq: int = 2, 
    npmi_thresh: float = 0.3
) -> pd.DataFrame:
    """
    Computes pair associations between two occurrence columns using NPMI.
    Excludes p-values entirely to focus on actual association strength.
    """
    N = len(df)
    
    contexts_a = df[col_a].apply(lambda x: list(set(x)) if isinstance(x, list) else [])
    contexts_b = df[col_b].apply(lambda x: list(set(x)) if isinstance(x, list) else [])

    freq_a = Counter()
    freq_b = Counter()
    pair_counts = Counter()

    for terms_a, terms_b in zip(contexts_a, contexts_b):
        for t_a in set(terms_a):
            freq_a[t_a] += 1
        for t_b in set(terms_b):
            freq_b[t_b] += 1
        for t_a, t_b in product(set(terms_a), set(terms_b)):
            pair_counts[(t_a, t_b)] += 1

    records = []
    for (t1, t2), observed_co in pair_counts.items():
        if observed_co < min_freq:
            continue
            
        f1 = freq_a[t1]
        f2 = freq_b[t2]

        p_xy = observed_co / N
        p_x = f1 / N
        p_y = f2 / N
        
        pmi = np.log2(p_xy / (p_x * p_y))
        npmi = pmi / (-np.log2(p_xy))
        
        if npmi >= npmi_thresh:
            records.append({
                "term_1": t1,
                "term_2": t2,
                "co_occurrences": observed_co,
                "freq_term_1": f1,
                "freq_term_2": f2,
                "npmi": round(npmi, 4)
            })

    result_df = pd.DataFrame(records)
    
    if result_df.empty:
        return pd.DataFrame()

    return result_df.sort_values(by="npmi", ascending=False).reset_index(drop=True)


# ==========================================
# TASK 2: CO-OCCURRENCE HEATMAP (LIMITED TO TOP 20 MOST FREQUENT)
# ==========================================

def save_top20_category_heatmap(
    associations_df: pd.DataFrame, 
    category_a_name: str, 
    category_b_name: str, 
    output_path: Path, 
    top_k: int = 20, 
    metric: str = "npmi"
):
    """
    Filters the association dataframe to the Top 20 most frequent terms on each axis
    and saves an annotated heatmap as a high-resolution PNG image.
    """
    if associations_df.empty:
        print(f"No associations to plot for {category_a_name} vs {category_b_name}.")
        return

    # Extract Top 20 terms based on total frequency in the dataset
    top_terms_1 = associations_df.groupby("term_1")["freq_term_1"].max().nlargest(top_k).index
    top_terms_2 = associations_df.groupby("term_2")["freq_term_2"].max().nlargest(top_k).index

    filtered_df = associations_df[
        associations_df["term_1"].isin(top_terms_1) & 
        associations_df["term_2"].isin(top_terms_2)
    ]

    if filtered_df.empty:
        print(f"No overlapping pairs found within top {top_k} terms for {category_a_name} vs {category_b_name}.")
        return

    # Pivot into Category Matrix
    matrix = filtered_df.pivot(index="term_1", columns="term_2", values=metric).fillna(0)

    # Dynamic sizing based on grid dimension
    fig_width = max(8, len(matrix.columns) * 0.55)
    fig_height = max(6, len(matrix.index) * 0.45)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd", 
        cbar_kws={'label': metric.upper()},
        linewidths=0.5,
        square=True
    )
    
    plt.title(f"Top {top_k} Co-occurrence Matrix: {category_a_name} vs {category_b_name} ({metric.upper()})", fontsize=11, fontweight="bold")
    plt.xlabel(category_b_name, fontsize=10)
    plt.ylabel(category_a_name, fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap successfully saved to: {output_path}")


# ==========================================
# TASK 3: PATENT SUMMARY CARDS
# ==========================================

def generate_and_save_patent_summary_cards(df: pd.DataFrame, csv_output_path: Path, json_output_path: Path) -> pd.DataFrame:
    """
    Aggregates data at the pat_id level showing primary failure pathways and saves to CSV and JSON formats.
    """
    cards = []
    grouped = df.groupby("pat_id")
    
    for pat_id, group in grouped:
        title = group["title"].iloc[0] if "title" in group.columns else "N/A"
        
        tech_terms = list(set([t for sub in group["occurrence_technical"] for t in sub]))
        causal_words = list(set([c for sub in group["causal_ocurrence_words"] for c in sub]))
        causal_terms = list(set([c for sub in group["occurrence_causal"] for c in sub]))
        failure_terms = list(set([f for sub in group["occurrence_failures"] for f in sub]))
        variable_terms = list(set([v for sub in group["occurrence_variable"] for v in sub]))
        
        # Sentence-level triplet matches
        pathways = []
        for _, row in group.iterrows():
            row_techs = row["occurrence_technical"]
            row_causes = row["causal_ocurrence_words"] + row["occurrence_causal"]
            row_failures = row["occurrence_failures"]
            
            for t, c, f in product(row_techs, row_causes, row_failures):
                pathways.append(f"{t} ➔ [{c}] ➔ {f}")
                
        pathway_counts = Counter(pathways).most_common(3)
        formatted_pathways = [f"{path} (x{count})" for path, count in pathway_counts]
        
        cards.append({
            "pat_id": pat_id,
            "title": title,
            "total_sentences": len(group),
            "technical_terms": tech_terms,
            "causal_words_and_terms": list(set(causal_words + causal_terms)),
            "failure_terms": failure_terms,
            "variable_terms": variable_terms,
            "primary_failure_pathways": formatted_pathways if formatted_pathways else ["No sentence-level triplet found"]
        })
        
    summary_cards_df = pd.DataFrame(cards)
    
    # Save as JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=4)
    print(f"Patent summary cards saved as JSON to: {json_output_path}")

    # Save as CSV
    csv_df = summary_cards_df.copy()
    for list_col in ["technical_terms", "causal_words_and_terms", "failure_terms", "variable_terms", "primary_failure_pathways"]:
        csv_df[list_col] = csv_df[list_col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
    csv_df.to_csv(csv_output_path, index=False)
    print(f"Patent summary cards saved as CSV to: {csv_output_path}")

    return summary_cards_df


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def main():
    print("=" * 60)
    print("STARTING REVISED PATENT SUMMARY STATISTICS PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print(f"\nLoading data from: {INPUT_CSV_PATH}")
    df = load_and_preprocess_dataset(INPUT_CSV_PATH)
    print(f"Successfully loaded {len(df)} rows.")

    # 2. TASK 1: Compute Associations (Without P-Value)
    print("\n[Task 1a] Computing NPMI Associations (Technical Terms vs Attributes)...")
    tech_attr_assoc = compute_npmi_associations(
        df, col_a="occurrence_technical", col_b="occurrence_variable", min_freq=2, npmi_thresh=0.3
    )
    if not tech_attr_assoc.empty:
        tech_attr_assoc.to_csv(TOP_ASSOC_TECH_ATTR_CSV, index=False)
        print(f"  -> Saved table to: {TOP_ASSOC_TECH_ATTR_CSV}")

    print("\n[Task 1b] Computing NPMI Associations (Attributes vs Failures)...")
    attr_fail_assoc = compute_npmi_associations(
        df, col_a="occurrence_variable", col_b="occurrence_failures", min_freq=2, npmi_thresh=0.3
    )
    if not attr_fail_assoc.empty:
        attr_fail_assoc.to_csv(TOP_ASSOC_ATTR_FAIL_CSV, index=False)
        print(f"  -> Saved table to: {TOP_ASSOC_ATTR_FAIL_CSV}")

    # 3. TASK 2: Heatmaps (Top 20 Terms)
    print("\n[Task 2a] Generating Top 20 Heatmap (Technical Terms vs Attributes)...")
    save_top20_category_heatmap(
        tech_attr_assoc, 
        category_a_name="Technical Terms", 
        category_b_name="Attribute Terms", 
        output_path=HEATMAP_TECH_ATTR_PNG,
        top_k=20,
        metric="npmi"
    )

    print("\n[Task 2b] Generating Top 20 Heatmap (Attributes vs Failure Terms)...")
    save_top20_category_heatmap(
        attr_fail_assoc, 
        category_a_name="Attribute Terms", 
        category_b_name="Failure Terms", 
        output_path=HEATMAP_ATTR_FAIL_PNG,
        top_k=20,
        metric="npmi"
    )

    # 4. TASK 3: Patent Summary Cards
    print("\n[Task 3] Generating and saving Patent Summary Cards...")
    generate_and_save_patent_summary_cards(
        df, 
        csv_output_path=SUMMARY_CARDS_CSV, 
        json_output_path=SUMMARY_CARDS_JSON
    )

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"All revised deliverables exported to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()