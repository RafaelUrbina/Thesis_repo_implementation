import ast
import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from scipy.stats import chi2_contingency

# ==========================================
# 1. DATA LOADING & PARSING FUNCTION
# ==========================================

def load_and_preprocess_dataset(data_input):
    """
    Loads dataset and parses string-encoded list columns 
    (e.g., "['reservoir', 'pump']") back into Python lists.
    """
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    else:
        df = data_input.copy()

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
# DELIVERABLE 1: TOP ASSOCIATION TABLES 
# (Chi-Square p < 0.05 AND NPMI > 0.3)
# ==========================================

def compute_statistically_significant_associations(
    df, col_a, col_b, min_freq=1, p_value_thresh=0.05, npmi_thresh=0.3
):
    """
    Computes pair associations between two occurrence columns.
    Applies Chi-Square contingency test for statistical significance ($p < 0.05$)
    and filters by NPMI threshold ($NPMI > 0.3$).
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
        
        # Contingency table: [[Both present, T1 present/T2 absent], [T1 absent/T2 present, Both absent]]
        a = observed_co
        b = f1 - a
        c = f2 - a
        d = N - (a + b + c)
        
        contingency_table = [[a, b], [c, d]]
        
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
        except ValueError:
            p_val = 1.0

        p_xy = a / N
        p_x = f1 / N
        p_y = f2 / N
        
        pmi = np.log2(p_xy / (p_x * p_y))
        npmi = pmi / (-np.log2(p_xy))
        
        records.append({
            "term_1": t1,
            "term_2": t2,
            "co_occurrences": observed_co,
            "freq_term_1": f1,
            "freq_term_2": f2,
            "npmi": round(npmi, 4),
            "p_value": round(p_val, 5),
            "significant": (p_val < p_value_thresh) and (npmi > npmi_thresh)
        })

    result_df = pd.DataFrame(records)
    
    if result_df.empty:
        return pd.DataFrame()

    filtered_df = result_df[result_df["significant"] == True].sort_values(
        by="npmi", ascending=False
    ).reset_index(drop=True)
    
    return filtered_df


# ==========================================
# DELIVERABLE 2: CO-OCCURRENCE MATRIX (HEATMAP)
# ==========================================

def plot_and_save_category_heatmap(associations_df, category_a_name, category_b_name, output_dir="output", metric="npmi"):
    """
    Transforms an association dataframe into a Category vs Category matrix, 
    plots the heatmap, and saves it to a PNG image file.
    """
    if associations_df.empty:
        print(f"No significant associations to plot for {category_a_name} vs {category_b_name}.")
        return

    matrix = associations_df.pivot(index="term_1", columns="term_2", values=metric).fillna(0)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd", 
        cbar_kws={'label': metric.upper()},
        linewidths=0.5
    )
    plt.title(f"Co-occurrence Profile: {category_a_name} vs {category_b_name} ({metric.upper()})")
    plt.xlabel(category_b_name)
    plt.ylabel(category_a_name)
    plt.tight_layout()

    # Export figure
    filename = f"heatmap_{category_a_name.lower().replace(' ', '_')}_vs_{category_b_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"✓ Saved Heatmap Image: {filepath}")


# ==========================================
# DELIVERABLE 3: PATENT SUMMARY CARDS / DASHBOARD
# ==========================================

def generate_patent_summary_cards(df):
    """
    Aggregates data at the pat_id level showing primary failure pathways.
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
            "causal_expressions": list(set(causal_words + causal_terms)),
            "failure_terms": failure_terms,
            "variable_terms": variable_terms,
            "primary_failure_pathways": formatted_pathways if formatted_pathways else ["No sentence-level triplet found"]
        })
        
    return pd.DataFrame(cards)


# ==========================================
# EXPORT ENGINE MODULE
# ==========================================

def export_results(associations_dict, patent_cards_df, output_dir="output_results"):
    """
    Saves all pipeline outputs to structured files (CSV, JSON, PNG).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to directory: ./{output_dir}/")

    # 1. Export Association Tables (CSV & Excel)
    for name, assoc_df in associations_dict.items():
        if not assoc_df.empty:
            csv_path = os.path.join(output_dir, f"top_associations_{name}.csv")
            assoc_df.to_csv(csv_path, index=False)
            print(f"✓ Saved Association Table (CSV): {csv_path}")

    # 2. Export Patent Summary Cards (CSV & JSON)
    if not patent_cards_df.empty:
        # Save CSV version (lists flattened to string)
        cards_csv_df = patent_cards_df.copy()
        for col in ["technical_terms", "causal_expressions", "failure_terms", "variable_terms", "primary_failure_pathways"]:
            cards_csv_df[col] = cards_csv_df[col].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)
        
        cards_csv_path = os.path.join(output_dir, "patent_summary_cards.csv")
        cards_csv_df.to_csv(cards_csv_path, index=False)
        print(f"✓ Saved Patent Cards (CSV): {cards_csv_path}")

        # Save JSON version (keeps structured list hierarchy for web/dashboards)
        cards_json_path = os.path.join(output_dir, "patent_summary_cards.json")
        patent_cards_df.to_json(cards_json_path, orient="records", indent=4)
        print(f"✓ Saved Patent Cards (JSON): {cards_json_path}")


# ==========================================
# PIPELINE EXECUTION DEMO
# ==========================================

if __name__ == "__main__":
    # Sample dataset matching your schema
    raw_sample = [
        {
            "pat_id": "EP-0003327-B1",
            "title": "PROCESS FOR CHEMICAL-MECHANICAL TREATMENT...",
            "text_type": "description",
            "paragraph_id": "para_15",
            "text": "due to the shorter dwell time...",
            "occurrence_technical": [], "occurrence_failures": [], "occurrence_causal": [], 
            "occurrence_variable": ["nature"], "causal_ocurrence_words": ["due to"]
        },
        {
            "pat_id": "EP-0003327-B1",
            "title": "PROCESS FOR CHEMICAL-MECHANICAL TREATMENT...",
            "text_type": "description",
            "paragraph_id": "para_24",
            "text": "this results in a significantly increased...",
            "occurrence_technical": ["sedimentation basin"], "occurrence_failures": ["clogging"], 
            "occurrence_causal": [], "occurrence_variable": ["load capacity"], "causal_ocurrence_words": ["results in"]
        },
        {
            "pat_id": "EP-0004056-B1",
            "title": "PRESSURE-REGULATED WATER SUPPLY INSTALLATION",
            "text_type": "claim",
            "paragraph_id": "claim_1",
            "text": "a pressure regulated water supply system...",
            "occurrence_technical": "['reservoir', 'discharge line', 'valve', 'pump']", 
            "occurrence_failures": "['pressure drop']", 
            "occurrence_causal": [], 
            "occurrence_variable": [], 
            "causal_ocurrence_words": "['as a result of', 'gives rise to', 'leads to']"
        }
    ]

    OUTPUT_DIR = "patent_pipeline_outputs"

    # 1. Load & Preprocess Data
    df = load_and_preprocess_dataset(pd.DataFrame(raw_sample))

    # 2. Deliverable 1: Calculate Top Associations
    tech_vs_causal = compute_statistically_significant_associations(
        df, 
        col_a="occurrence_technical", 
        col_b="causal_ocurrence_words", 
        min_freq=1, 
        p_value_thresh=0.05, 
        npmi_thresh=-1.0  # Threshold lowered for sample demo size
    )

    associations = {
        "technical_vs_causal": tech_vs_causal
    }

    # 3. Deliverable 2: Plot and Save Heatmaps
    plot_and_save_category_heatmap(
        tech_vs_causal, "Technical Terms", "Causal Words", output_dir=OUTPUT_DIR, metric="npmi"
    )

    # 4. Deliverable 3: Generate Patent Summary Cards
    patent_cards_df = generate_patent_summary_cards(df)

    # 5. Export All Results (CSVs, JSON)
    export_results(associations, patent_cards_df, output_dir=OUTPUT_DIR)