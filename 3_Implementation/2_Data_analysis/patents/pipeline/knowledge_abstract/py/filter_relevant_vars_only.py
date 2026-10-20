import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys

try:
    import networkx as nx
except ImportError:
    pass

import pandas as pd
import plotly.graph_objects as go

# Setup project root imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]  # Adjust parent index if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH, CAUSAL_RELATION_MODEL_PATH

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "causal_graph"
SEMANTIC_MAPPINGS_DIR = CAUSAL_RELATION_MODEL_PATH / "output" / "2_semantic_mappings"

# Path to the category mapped children JSON
CATEGORY_MAPPED_CHILDREN_JSON = SEMANTIC_MAPPINGS_DIR / "filtered_category_mapped_children.json"

# Explicit subset of variables used to find matching categories in the JSON
EXPLICIT_FILTER_VARIABLES = {
    'total_intervention_cost_1ring',
    'm_per_hex',
    'idw_rain_mm_max_monthly',
    'idw_wind_speed_max_max',
    'dtmidcnt_max',
    'avg_seismic_magnitude_1ring',
    'idw_temp_thermal_range',
    'dist_to_waterway_m',
    'avg_descending_soil_speed',
    'avg_ascending_soil_speed',
    'landslide_point_count_1ring',
    'hydraulic_hazard_level'
}

# Input Unfiltered Deaggregated CSV
INPUT_DEAGGREGATED_CSV = OUTPUT_DIR / "complete_patent_causal_chains_pos_deaggregated.csv"

# Filtered Output Files (Deaggregated, Graph Cache, and Sankey Only - No Summary CSVs)
FILTERED_DEAGGREGATED_CSV = OUTPUT_DIR / "filtered_complete_patent_causal_chains_pos_deaggregated.csv"
FILTERED_GRAPH_CACHE = OUTPUT_DIR / "filtered_patent_causal_pos_graph.graphml"
FILTERED_SANKEY_OUTPUT = OUTPUT_DIR / "filtered_patent_edge_cases_pos_sankey.html"

# Filtering Parameters
MIN_TOTAL_DEGREE = 3
TOP_PERCENT_TRIPLETS = 0.10  # Retains only the top 10% most frequent [Technical, Attribute, Failure] triplets

NODE_COLOR_MAP = {
    "Technical": "rgba(31, 119, 180, 0.8)",  # Blue
    "Attribute": "rgba(44, 160, 44, 0.8)",   # Green
    "Failure": "rgba(214, 39, 40, 0.8)",     # Red
}

TYPE_SHORT_MAP = {
    "Technical": "T",
    "Attribute": "A",
    "Failure": "F",
}


# =====================================================================
# 1. HELPER FUNCTIONS: PARSING & FILTER COMPILATION
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


def get_allowed_filter_terms() -> set:
    """
    Reads filtered_category_mapped_children.json, identifies categories whose 
    'mapped_spatial_variables' intersect with EXPLICIT_FILTER_VARIABLES, 
    and collects all mapped variables and available child terms from those categories.
    """
    allowed = set()
    explicit_set = {v.lower().strip() for v in EXPLICIT_FILTER_VARIABLES}

    if CATEGORY_MAPPED_CHILDREN_JSON.exists():
        try:
            print(f"[Filter] Reading categories from: {CATEGORY_MAPPED_CHILDREN_JSON}")
            with open(CATEGORY_MAPPED_CHILDREN_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cat_key, cat_val in data.items():
                    cat_vars = {v.lower().strip() for v in cat_val.get("mapped_spatial_variables", [])}
                    
                    if explicit_set & cat_vars:
                        print(f"  -> Matched Category: '{cat_key}'")
                        for var in cat_val.get("mapped_spatial_variables", []):
                            allowed.add(var.strip().lower())
                        for child in cat_val.get("available_child_terms", []):
                            if isinstance(child, dict) and "child_term" in child:
                                allowed.add(child["child_term"].strip().lower())
                            elif isinstance(child, str):
                                allowed.add(child.strip().lower())
        except Exception as e:
            print(f"[Filter] Warning: Could not parse category JSON ({e}). Falling back to explicit set.")

    if not allowed:
        allowed = explicit_set

    print(f"[Filter] Total compiled allowed filter terms: {len(allowed)}")
    return allowed


def filter_low_degree_nodes(G, min_degree):
    """Iteratively removes nodes whose total edge count <= min_degree."""
    while True:
        nodes_to_remove = [
            n for n in G.nodes() if max(G.in_degree(n), G.out_degree(n)) < min_degree
        ]
        if not nodes_to_remove:
            break
        G.remove_nodes_from(nodes_to_remove)


def is_passive_voice(text, verbs_list):
    """Heuristic check for passive voice based on auxiliary 'be' verbs + past participle or verb patterns."""
    passive_auxiliaries = {"is", "was", "are", "were", "be", "been", "being", "gets", "got"}
    text_lower = str(text).lower()
    words = text_lower.split()
    
    # Check if text contains auxiliary 'be' near past participle verbs (-ed / irregular)
    for i, word in enumerate(words):
        if word in passive_auxiliaries and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word.endswith("ed") or next_word.endswith("en") or next_word in {"caused", "affected", "damaged", "impaired", "restricted"}:
                return True
                
    # Check verbs list for explicit passive markers
    for v in verbs_list:
        v_clean = str(v).lower().strip()
        if any(v_clean.startswith(aux) for aux in ["is ", "was ", "are ", "were "]):
            return True
            
    return False


def select_informative_triplet_row(group_df):
    """
    Among rows sharing the exact same [technical, attribute, failure] triplet, 
    selects the most informative row (preferring forward/active over backward/static/passive).
    """
    if len(group_df) == 1:
        return group_df.iloc[0]

    best_row = None
    best_score = -float('inf')

    for _, row in group_df.iterrows():
        score = 0
        cat = str(row.get("category", "")).strip().lower()
        text = str(row.get("link_text", ""))
        verbs = parse_field(row.get("verbs", []))

        # Priority 1: Informative mode (forward category) gets highest preference
        if cat == "forward":
            score += 100
        elif cat == "mixed":
            score += 40
        else:  # backward or static
            score += 10

        # Priority 2: Active voice vs Passive voice (prefer active / non-passive)
        if not is_passive_voice(text, verbs):
            score += 50
        else:
            score -= 30

        # Priority 3: Richness / detail of sentence description (length & noun/verb counts)
        score += min(len(text), 150) / 50
        score += int(row.get("noun_count", 0)) * 2
        score += int(row.get("verb_count", 0)) * 2

        if score > best_score:
            best_score = score
            best_row = row

    return best_row


# =====================================================================
# 2. FILTER DEAGGREGATED TABLE & GENERATE OUTPUTS
# =====================================================================
def filter_deaggregated_table_and_export(allowed_terms):
    print("\n[Pipeline] Loading existing unfiltered deaggregated CSV table...")
    
    if not INPUT_DEAGGREGATED_CSV.exists():
        print(f"[Error] Required unfiltered input file not found: {INPUT_DEAGGREGATED_CSV}. Please run the full pipeline first.")
        return

    # 1. Load Deaggregated CSV
    df_deagg = pd.read_csv(INPUT_DEAGGREGATED_CSV)
    print(f"  - Loaded deaggregated rows: {len(df_deagg)}")

    def matches_filter(attr_val):
        if not allowed_terms:
            return True
        a_clean = str(attr_val).strip().lower()
        return any(term in a_clean or a_clean in term for term in allowed_terms)

    # Filter by Attribute ontology match
    df_deagg_filtered = df_deagg[df_deagg["attribute"].apply(matches_filter)].copy()
    print(f"  - Rows after attribute category filter: {len(df_deagg_filtered)}")

    if df_deagg_filtered.empty:
        print("[Warning] Filtered deaggregated dataset is empty after attribute match! Skipping remaining steps.")
        return

    # 2. Apply Top 10% Most Frequent Triplet Filter [Technical, Attribute, Failure]
    if TOP_PERCENT_TRIPLETS and 0 < TOP_PERCENT_TRIPLETS < 1.0:
        triplet_counts = df_deagg_filtered.groupby(["technical", "attribute", "failure"]).size().reset_index(name="triplet_freq")
        triplet_counts = triplet_counts.sort_values(by="triplet_freq", ascending=False)
        
        cutoff_idx = max(1, int(len(triplet_counts) * TOP_PERCENT_TRIPLETS))
        top_triplets_df = triplet_counts.head(cutoff_idx)
        
        top_triplet_set = set(
            zip(
                top_triplets_df["technical"].str.strip(),
                top_triplets_df["attribute"].str.strip(),
                top_triplets_df["failure"].str.strip()
            )
        )

        df_deagg_filtered["t_clean"] = df_deagg_filtered["technical"].str.strip()
        df_deagg_filtered["a_clean"] = df_deagg_filtered["attribute"].str.strip()
        df_deagg_filtered["f_clean"] = df_deagg_filtered["failure"].str.strip()

        df_deagg_filtered = df_deagg_filtered[
            df_deagg_filtered.apply(lambda row: (row["t_clean"], row["a_clean"], row["f_clean"]) in top_triplet_set, axis=1)
        ].drop(columns=["t_clean", "a_clean", "f_clean"])
        
        print(f"  - Rows after keeping top {int(TOP_PERCENT_TRIPLETS * 100)}% most frequent triplets: {len(df_deagg_filtered)}")

    if df_deagg_filtered.empty:
        print("[Warning] Filtered deaggregated dataset is empty after triplet frequency filter! Skipping graph generation.")
        return

    # 3. Disambiguate Identical Triplets: Keep only the informative/forward/active mode row per triplet
    print("\n[Pipeline] Disambiguating duplicate triplets (prioritizing informative/forward/active mode)...")
    initial_row_count = len(df_deagg_filtered)
    
    df_deagg_filtered["t_clean"] = df_deagg_filtered["technical"].str.strip()
    df_deagg_filtered["a_clean"] = df_deagg_filtered["attribute"].str.strip()
    df_deagg_filtered["f_clean"] = df_deagg_filtered["failure"].str.strip()

    optimized_rows = []
    for _, group in df_deagg_filtered.groupby(["t_clean", "a_clean", "f_clean"]):
        best_row = select_informative_triplet_row(group)
        optimized_rows.append(best_row)

    df_deagg_filtered = pd.DataFrame(optimized_rows).drop(columns=["t_clean", "a_clean", "f_clean"], errors="ignore")
    print(f"  - Rows after informative mode disambiguation: {len(df_deagg_filtered)} (Pruned {initial_row_count - len(df_deagg_filtered)} conflicting/passive duplicates)")

    if df_deagg_filtered.empty:
        print("[Warning] Filtered deaggregated dataset is empty after disambiguation! Skipping graph generation.")
        return

    # 4. Build Filtered Graph Cache (.graphml) and apply MIN_TOTAL_DEGREE pruning
    print("\n[Pipeline] Building filtered NetworkX graph...")
    G_filtered = nx.DiGraph()

    for _, row in df_deagg_filtered.iterrows():
        t = str(row["technical"]).strip()
        a = str(row["attribute"]).strip()
        f = str(row["failure"]).strip()
        pat_id = str(row["pat_id"]).strip()
        trigger = str(row.get("trigger1", row.get("trigger", ""))).strip()
        text_val = str(row.get("link_text", "")).strip()

        if not G_filtered.has_node(t):
            G_filtered.add_node(t, node_type="Technical")
        if not G_filtered.has_node(a):
            G_filtered.add_node(a, node_type="Attribute")
        if not G_filtered.has_node(f):
            G_filtered.add_node(f, node_type="Failure")

        occ_entry = {"pat_id": pat_id, "para_id": str(row.get("paragraph_id", "")), "text": text_val}

        for u, v in [(t, a), (a, f)]:
            if G_filtered.has_edge(u, v):
                G_filtered[u][v]["weight"] += 1
                G_filtered[u][v]["patents"].add(pat_id)
                G_filtered[u][v]["triggers"].add(trigger)
                for n in parse_field(row.get("nouns", "")):
                    G_filtered[u][v]["nouns"].add(n)
                for vb in parse_field(row.get("verbs", "")):
                    G_filtered[u][v]["verbs"].add(vb)
                G_filtered[u][v]["occurrences"].append(occ_entry)
            else:
                edge_attr = {
                    "weight": 1,
                    "trigger": trigger,
                    "patents": {pat_id},
                    "triggers": {trigger},
                    "nouns": set(parse_field(row.get("nouns", ""))),
                    "verbs": set(parse_field(row.get("verbs", ""))),
                    "occurrences": [occ_entry],
                }
                G_filtered.add_edge(u, v, **edge_attr)

    # Apply minimum total degree pruning
    filter_low_degree_nodes(G_filtered, MIN_TOTAL_DEGREE)

    # Retain only deaggregated rows whose nodes survived graph filtering
    surviving_nodes = set(G_filtered.nodes())
    df_deagg_filtered = df_deagg_filtered[
        df_deagg_filtered["technical"].isin(surviving_nodes) &
        df_deagg_filtered["attribute"].isin(surviving_nodes) &
        df_deagg_filtered["failure"].isin(surviving_nodes)
    ].copy()

    FILTERED_DEAGGREGATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_deagg_filtered.to_csv(FILTERED_DEAGGREGATED_CSV, index=False)
    print(f"  -> Saved filtered deaggregated CSV: {FILTERED_DEAGGREGATED_CSV} ({len(df_deagg_filtered)} rows remaining)")

    # Save GraphML copy
    G_save = G_filtered.copy()
    for u, v, d in G_save.edges(data=True):
        d["patents"] = json.dumps(list(d["patents"]))
        d["triggers"] = json.dumps(list(d["triggers"]))
        d["nouns"] = json.dumps(list(d.get("nouns", set())))
        d["verbs"] = json.dumps(list(d.get("verbs", set())))
        d["occurrences"] = json.dumps(d.get("occurrences", []))

    nx.write_graphml(G_save, FILTERED_GRAPH_CACHE)
    print(f"  -> Saved filtered graph cache: {FILTERED_GRAPH_CACHE}")

    # 5. Generate Filtered Sankey Diagram (.html) from filtered deaggregated rows
    print("\n[Pipeline] Generating filtered Sankey diagram...")
    stage_nodes = {}
    edge_counts = defaultdict(int)
    edge_meta = defaultdict(lambda: {"patents": set(), "triggers": set(), "nouns": set(), "verbs": set()})

    for _, row in df_deagg_filtered.iterrows():
        t_node = str(row["technical"]).strip()
        a_node = str(row["attribute"]).strip()
        f_node = str(row["failure"]).strip()
        pat_id = str(row["pat_id"]).strip()
        trigger = str(row.get("trigger1", row.get("trigger", ""))).strip().lower()

        stage_nodes[(t_node, 0)] = "Technical"
        stage_nodes[(a_node, 1)] = "Attribute"
        stage_nodes[(f_node, 2)] = "Failure"

        e1_key = ((t_node, 0), (a_node, 1), t_node, a_node)
        e2_key = ((a_node, 1), (f_node, 2), a_node, f_node)

        edge_counts[e1_key] += 1
        edge_counts[e2_key] += 1

        edge_meta[e1_key]["patents"].add(pat_id)
        if trigger:
            edge_meta[e1_key]["triggers"].add(trigger)
        edge_meta[e2_key]["patents"].add(pat_id)
        if trigger:
            edge_meta[e2_key]["triggers"].add(trigger)

        for n in parse_field(row.get("nouns", "")):
            edge_meta[e1_key]["nouns"].add(n)
            edge_meta[e2_key]["nouns"].add(n)
        for v in parse_field(row.get("verbs", "")):
            edge_meta[e1_key]["verbs"].add(v)
            edge_meta[e2_key]["verbs"].add(v)

    indexed_nodes = list(stage_nodes.keys())
    node_to_idx = {node_key: idx for idx, node_key in enumerate(indexed_nodes)}

    node_labels = []
    node_colors = []
    for node_name, stage in indexed_nodes:
        n_type = stage_nodes[(node_name, stage)]
        short_type = TYPE_SHORT_MAP.get(n_type, n_type)
        node_labels.append(f"{node_name} ({short_type})")
        node_colors.append(NODE_COLOR_MAP.get(n_type, "rgba(100,100,100,0.8)"))

    sources, targets, values, link_labels = [], [], [], []

    for key, flow_value in edge_counts.items():
        src_key, tgt_key, orig_u, orig_v = key
        sources.append(node_to_idx[src_key])
        targets.append(node_to_idx[tgt_key])
        values.append(flow_value)

        meta = edge_meta[key]
        patents_list = sorted(list(meta["patents"]))
        patents_str = ", ".join(patents_list[:8]) + ("..." if len(patents_list) > 8 else "")
        triggers = ", ".join(sorted(list(meta["triggers"])))

        hover_lines = [
            f"<b>From:</b> {orig_u}",
            f"<b>To:</b> {orig_v}",
            f"<b>Patent IDs:</b> {patents_str or 'None'}",
            f"<b>Triggers:</b> {triggers}",
            f"<b>Path Flow Count:</b> {flow_value}",
        ]

        link_labels.append("<br>".join(hover_lines))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=25,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    customdata=link_labels,
                    hovertemplate="%{customdata}<extra></extra>",
                    color="rgba(200, 200, 200, 0.4)",
                    hovercolor="#e6e200",
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Filtered Patent Causal Chain [Top 10% + Informative Mode Disambiguation + Degree Filter]: [Technical] ➔ [Attribute] ➔ [Failure]",
        font_size=12,
        height=900,
        width=900,
        margin=dict(l=50, r=200, t=60, b=50),
    )

    FILTERED_SANKEY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(FILTERED_SANKEY_OUTPUT)
    print(f"  -> Saved filtered Sankey diagram: {FILTERED_SANKEY_OUTPUT}")


# =====================================================================
# 3. MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("\n=== Starting Post-Processing Filter Pipeline ===")
    
    # Compile the filter terms using JSON category matching
    allowed_terms = get_allowed_filter_terms()
    
    # Process deaggregated table, build graph cache, and generate filtered Sankey
    filter_deaggregated_table_and_export(allowed_terms)

    print("\nFiltered post-processing completed successfully!")