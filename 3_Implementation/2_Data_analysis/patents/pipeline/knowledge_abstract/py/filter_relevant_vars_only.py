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

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Ensure necessary NLTK data is downloaded
for pkg in ('punkt', 'averaged_perceptron_tagger', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger_eng'):
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'taggers/{pkg}' if 'tagger' in pkg else f'corpora/{pkg}')
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

# Setup project root imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]  # Adjust parent index if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH, CAUSAL_RELATION_MODEL_PATH, ONTOLOGY_LISTS_PATH

# =====================================================================
# CONFIGURATION & PATHS
# =====================================================================
OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "causal_graph"
SEMANTIC_MAPPINGS_DIR = CAUSAL_RELATION_MODEL_PATH / "output" / "2_semantic_mappings"

# Dedicated subfolder for all filtered final outputs
FILTERED_OUTPUT_DIR = OUTPUT_DIR / "filtered_final"

# Path to the category mapped children JSON
CATEGORY_MAPPED_CHILDREN_JSON = SEMANTIC_MAPPINGS_DIR / "filtered_category_mapped_children.json"

FORWARD_TXT_PATH = ONTOLOGY_LISTS_PATH / "categorized_causal" / "Forward.txt"
BACKWARD_TXT_PATH = ONTOLOGY_LISTS_PATH / "categorized_causal" / "Backward.txt"
STATIC_TXT_PATH = ONTOLOGY_LISTS_PATH / "categorized_causal" / "Static.txt"

# Explicit subset of variables used to find matching categories in the JSON
EXPLICIT_FILTER_VARIABLES = {
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

# Filtered Output Files (Placed inside filtered_final)
FILTERED_DEAGGREGATED_CSV = FILTERED_OUTPUT_DIR / "filtered_complete_patent_causal_chains_pos_deaggregated.csv"
FILTERED_GRAPH_CACHE = FILTERED_OUTPUT_DIR / "filtered_patent_causal_pos_graph.graphml"
FILTERED_SUMMARY_TXT = FILTERED_OUTPUT_DIR / "filtered_pipeline_metrics_summary.txt"

# Sankey Outputs (Combined + Separated Categorized, placed inside filtered_final)
FILTERED_SANKEY_OUTPUT = FILTERED_OUTPUT_DIR / "filtered_patent_edge_cases_pos_sankey.html"
FORWARD_SANKEY_OUTPUT = FILTERED_OUTPUT_DIR / "filtered_forward_patent_edge_cases_pos_sankey.html"
BACKWARD_SANKEY_OUTPUT = FILTERED_OUTPUT_DIR / "filtered_backward_patent_edge_cases_pos_sankey.html"
STATIC_SANKEY_OUTPUT = FILTERED_OUTPUT_DIR / "filtered_static_patent_edge_cases_pos_sankey.html"

# Filtering Parameters (Applied strictly to Sankey Visualizations)
MIN_TOTAL_DEGREE = 2
TOP_PERCENT_TRIPLETS = 0.10  # Top 10% most frequent triplets applied to Sankey views only

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
# 1. HELPER FUNCTIONS: PARSING, CATEGORY FILES & LEMMATIZATION
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


def load_category_words(file_path):
    """Loads a set of lowercased category trigger phrases from a text file."""
    path = Path(file_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    return set()


def get_allowed_filter_terms() -> set:
    """Reads filtered_category_mapped_children.json and compiles ONLY natural language child terms."""
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


def get_row_category(row, forward_words, backward_words, static_words):
    """Determines category tone based on trigger word match with Forward.txt, Backward.txt, Static.txt."""
    tr = str(row.get("trigger1", row.get("trigger", ""))).strip().lower()
    is_fwd = tr in forward_words
    is_bwd = tr in backward_words
    is_stc = tr in static_words

    if is_fwd and not is_bwd and not is_stc:
        return "forward"
    elif is_bwd and not is_fwd and not is_stc:
        return "backward"
    elif is_stc and not is_fwd and not is_bwd:
        return "static"
    else:
        cat = str(row.get("category", "")).strip().lower()
        return cat if cat in {"forward", "backward", "static"} else "mixed"


def get_trigger_lemma(trigger_str):
    """Extracts the main verb/action root of a causal term using POS tagging and lemmatization."""
    if not trigger_str or not isinstance(trigger_str, str):
        return ""
    text = trigger_str.strip().lower()
    if not text:
        return ""
    
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        
        content_words = []
        for word, pos in tagged:
            if word in {'because', 'by', 'from', 'as', 'to'}:
                continue
            
            wn_pos = 'n'
            if pos.startswith('V'):
                wn_pos = 'v'
            elif pos.startswith('J'):
                wn_pos = 'a'
            elif pos.startswith('R'):
                wn_pos = 'r'
            elif pos.startswith('N'):
                wn_pos = 'n'
            else:
                continue
            
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            content_words.append(lemma)
            
        if content_words:
            return content_words[0]
    except Exception:
        pass
    
    stop_words = {'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'a', 'an', 'the', 'and', 'or', 'but', 'because'}
    words = [w.strip() for w in text.split() if w.strip() and w.strip() not in stop_words]
    return words[0] if words else text


def are_causally_equivalent(lemma1, lemma2):
    """Checks if two verb lemmas are semantically equivalent or derivationally related using WordNet."""
    if not lemma1 or not lemma2:
        return False
    
    if lemma1 == lemma2 or lemma1 in lemma2 or lemma2 in lemma1:
        return True
        
    try:
        synsets1 = wordnet.synsets(lemma1, pos=wordnet.VERB)
        synsets2 = wordnet.synsets(lemma2, pos=wordnet.VERB)
        
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim and sim > 0.4:
                    return True
                for lemma_obj in s1.lemmas():
                    for deriv in lemma_obj.derivationally_related_forms():
                        if deriv.name().lower() == lemma2:
                            return True
    except Exception:
        pass
        
    # Common patent causal equivalencies fallback map
    causal_equivalents = {
        ('cause', 'induce'), ('cause', 'trigger'), ('cause', 'produce'), ('cause', 'generate'),
        ('lead', 'result'), ('derive', 'produce'), ('create', 'produce'), ('stimulate', 'induce'),
        ('provoke', 'trigger'), ('promote', 'trigger')
    }
    if (lemma1, lemma2) in causal_equivalents or (lemma2, lemma1) in causal_equivalents:
        return True
        
    return False


def resolve_triplet_conflicts(group_df, forward_words, backward_words, static_words):
    """Resolves tone conflicts between forward and backward triplets using WordNet semantic/derivational matching."""
    rows_with_cat = []
    for _, row in group_df.iterrows():
        cat = get_row_category(row, forward_words, backward_words, static_words)
        row_dict = row.to_dict()
        row_dict["_computed_cat"] = cat
        rows_with_cat.append(row_dict)

    non_static_rows = [r for r in rows_with_cat if r["_computed_cat"] != "static"]
    static_rows = [r for r in rows_with_cat if r["_computed_cat"] == "static"]

    unique_cats = {r["_computed_cat"] for r in non_static_rows}
    has_forward = "forward" in unique_cats
    has_backward = "backward" in unique_cats

    examples = []
    if has_forward and has_backward:
        forward_candidates = [r for r in non_static_rows if r["_computed_cat"] == "forward"]
        backward_candidates = [r for r in non_static_rows if r["_computed_cat"] in {"backward", "mixed"}]
        
        for b_row in backward_candidates:
            b_tr = str(b_row.get("trigger1", b_row.get("trigger", ""))).strip()
            b_lemma = get_trigger_lemma(b_tr)
            
            matched_f_row = None
            for f_row in forward_candidates:
                f_tr = str(f_row.get("trigger1", f_row.get("trigger", ""))).strip()
                f_lemma = get_trigger_lemma(f_tr)
                
                if b_lemma and f_lemma:
                    is_semantic_match = are_causally_equivalent(b_lemma, f_lemma)
                    
                    if is_semantic_match and not (b_lemma == 'because' and f_lemma != 'cause'):
                        matched_f_row = f_row
                        break
            
            if matched_f_row is not None:
                kept_row = matched_f_row
                kept_trigger = str(kept_row.get("trigger1", kept_row.get("trigger", ""))).strip()
                kept_cat = kept_row["_computed_cat"]
                
                orig_row_copy = b_row.copy()
                orig_tr = b_tr
                orig_link = str(b_row.get("link_text", ""))
                
                if "trigger1" in b_row:
                    b_row["trigger1"] = kept_trigger
                if "trigger" in b_row:
                    b_row["trigger"] = kept_trigger
                
                b_row["_computed_cat"] = kept_cat
                if "category" in b_row:
                    b_row["category"] = kept_cat
                
                if orig_tr and orig_tr.lower() != kept_trigger.lower():
                    pattern = re.compile(re.escape(orig_tr), re.IGNORECASE)
                    b_row["link_text"] = pattern.sub(f"{orig_tr} ({kept_trigger})", orig_link, count=1)
                
                examples.append((kept_row, orig_row_copy, b_row))
                
        return non_static_rows + static_rows, examples
    else:
        return non_static_rows + static_rows, []


# =====================================================================
# 2. SANKEY RENDERER
# =====================================================================
def export_sankey_diagram(df_subset, output_filename, title_text, min_degree=MIN_TOTAL_DEGREE, top_percent=TOP_PERCENT_TRIPLETS, omit_stage1_triggers=True):
    if df_subset.empty:
        print(f"[Visualizer] No rows to render for {output_filename}.")
        return

    df_filtered = df_subset.copy()

    if top_percent and 0 < top_percent < 1.0:
        triplet_counts = df_filtered.groupby(["technical", "attribute", "failure"]).size().reset_index(name="triplet_freq")
        triplet_counts = triplet_counts.sort_values(by="triplet_freq", ascending=False)
        cutoff_idx = max(1, int(len(triplet_counts) * top_percent))
        top_triplets_df = triplet_counts.head(cutoff_idx)
        
        top_triplet_set = set(
            zip(
                top_triplets_df["technical"].str.strip(),
                top_triplets_df["attribute"].str.strip(),
                top_triplets_df["failure"].str.strip()
            )
        )
        df_filtered["t_c"] = df_filtered["technical"].str.strip()
        df_filtered["a_c"] = df_filtered["attribute"].str.strip()
        df_filtered["f_c"] = df_filtered["failure"].str.strip()
        df_filtered = df_filtered[
            df_filtered.apply(lambda r: (r["t_c"], r["a_c"], r["f_c"]) in top_triplet_set, axis=1)
        ].drop(columns=["t_c", "a_c", "f_c"])

    if df_filtered.empty:
        print(f"[Visualizer] Subset empty after Sankey top-percent filter for {output_filename}.")
        return

    G_temp = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        t, a, f = str(row["technical"]).strip(), str(row["attribute"]).strip(), str(row["failure"]).strip()
        for u, v, n_type_u, n_type_v in [(t, a, "Technical", "Attribute"), (a, f, "Attribute", "Failure")]:
            if not G_temp.has_node(u):
                G_temp.add_node(u, node_type=n_type_u)
            if not G_temp.has_node(v):
                G_temp.add_node(v, node_type=n_type_v)
            if G_temp.has_edge(u, v):
                G_temp[u][v]["weight"] += 1
            else:
                G_temp.add_edge(u, v, weight=1)

    filter_low_degree_nodes(G_temp, min_degree)
    surviving_nodes = set(G_temp.nodes())

    df_filtered = df_filtered[
        df_filtered["technical"].isin(surviving_nodes) &
        df_filtered["attribute"].isin(surviving_nodes) &
        df_filtered["failure"].isin(surviving_nodes)
    ].copy()

    if df_filtered.empty:
        print(f"[Visualizer] Subset empty after degree filtering for {output_filename}.")
        return

    stage_nodes = {}
    edge_counts = defaultdict(int)
    edge_meta = defaultdict(lambda: {"patents": set(), "triggers": set(), "nouns": set(), "verbs": set()})

    for _, row in df_filtered.iterrows():
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
        if trigger and not omit_stage1_triggers:
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
        ]
        if triggers:
            hover_lines.append(f"<b>Triggers:</b> {triggers}")
        hover_lines.append(f"<b>Path Flow Count:</b> {flow_value}")

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
        title_text=title_text,
        font_size=12,
        height=900,
        width=900,
        margin=dict(l=50, r=200, t=60, b=50),
    )

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[Visualizer] Saved Sankey diagram: {output_path}")


# =====================================================================
# 3. MAIN FILTER PIPELINE
# =====================================================================
def filter_deaggregated_table_and_export(allowed_terms, remove_static_terms=True):
    print("\n[Pipeline] Loading existing unfiltered deaggregated CSV table...")
    
    if not INPUT_DEAGGREGATED_CSV.exists():
        print(f"[Error] Required unfiltered input file not found: {INPUT_DEAGGREGATED_CSV}. Please run the full pipeline first.")
        return

    df_deagg = pd.read_csv(INPUT_DEAGGREGATED_CSV)
    total_raw_rows = len(df_deagg)
    print(f"  - Loaded deaggregated rows: {total_raw_rows}")

    def matches_filter(attr_val):
        if not allowed_terms:
            return True
        a_clean = str(attr_val).strip().lower()
        return any(term == a_clean or term in a_clean for term in allowed_terms)

    # 1. Attribute Category Filter
    df_deagg_filtered = df_deagg[df_deagg["attribute"].apply(matches_filter)].copy()
    print(f"  - Rows after attribute category filter: {len(df_deagg_filtered)}")

    if df_deagg_filtered.empty:
        print("[Warning] Filtered dataset is empty after attribute match!")
        return

    # Load Category Word Sets from txt files
    forward_words = load_category_words(FORWARD_TXT_PATH)
    backward_words = load_category_words(BACKWARD_TXT_PATH)
    static_words = load_category_words(STATIC_TXT_PATH)

    # Apply remove_static_terms keyword handling
    if remove_static_terms:
        df_deagg_filtered = df_deagg_filtered[
            df_deagg_filtered.apply(lambda r: get_row_category(r, forward_words, backward_words, static_words) != "static", axis=1)
        ].copy()

    # 2. Tone Conflict Disambiguation & Trigger Alignment (Retaining all rows)
    print("\n[Pipeline] Aligning conflicting triplet tones and updating triggers instead of pruning...")
    initial_row_count = len(df_deagg_filtered)
    
    df_deagg_filtered["t_clean"] = df_deagg_filtered["technical"].str.strip()
    df_deagg_filtered["a_clean"] = df_deagg_filtered["attribute"].str.strip()
    df_deagg_filtered["f_clean"] = df_deagg_filtered["failure"].str.strip()

    optimized_rows = []
    sample_metrics = []
    total_modified_rows = 0

    for _, group in df_deagg_filtered.groupby(["t_clean", "a_clean", "f_clean"]):
        kept_rows, conflict_examples = resolve_triplet_conflicts(group, forward_words, backward_words, static_words)
        optimized_rows.extend(kept_rows)
        total_modified_rows += len(conflict_examples)
        if conflict_examples and len(sample_metrics) < 20:
            sample_metrics.extend(conflict_examples[:20 - len(sample_metrics)])

    df_deagg_filtered = pd.DataFrame(optimized_rows).drop(columns=["t_clean", "a_clean", "f_clean", "_computed_cat"], errors="ignore")
    
    total_remaining_rows = len(df_deagg_filtered)
    total_unmodified_rows = total_remaining_rows - total_modified_rows

    print(f"  [Alignment Metrics]")
    print(f"    - Original Candidate Rows: {initial_row_count}")
    print(f"    - Total Retained Rows: {total_remaining_rows}")
    print(f"    - Unmodified Chains: {total_unmodified_rows}")
    print(f"    - Conflicting/Modified Chains: {total_modified_rows}")
    
    if sample_metrics:
        print(f"\n  [Trigger & Category Alignment Examples (Showing up to 20)]")
        for idx, (kept, orig_modified, updated) in enumerate(sample_metrics[:20], 1):
            kept_trigger = str(kept.get('trigger1', kept.get('trigger', 'N/A')))
            orig_trigger = str(orig_modified.get('trigger1', orig_modified.get('trigger', 'N/A')))
            updated_trigger = str(updated.get('trigger1', updated.get('trigger', 'N/A')))
            print(f"    Example {idx}:")
            print(f"      Triplet: [{updated.get('technical')} ➔ {updated.get('attribute')} ➔ {updated.get('failure')}]")
            print(f"      [KEPT/REFERENCE - Forward]: Trigger = '{kept_trigger}' | Category = {kept.get('_computed_cat','N/A')}")
            print(f"      [ORIGINAL CONFLICTING/DUPLICATE]: Trigger = '{orig_trigger}' | Category = {orig_modified.get('_computed_cat','N/A')}")
            print(f"        Original Sentence: \"{orig_modified.get('link_text','')}\"")
            print(f"      [ALIGNED/UPDATED]: Trigger = '{updated_trigger}' | Category = {updated.get('_computed_cat','N/A')}")
            print(f"        Updated Sentence:  \"{updated.get('link_text','')}\"\n")

    # Ensure filtered_final subfolder exists
    FILTERED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save Clean Unpruned Filtered Table to CSV inside filtered_final
    df_deagg_filtered.to_csv(FILTERED_DEAGGREGATED_CSV, index=False)
    print(f"  -> Saved filtered deaggregated CSV: {FILTERED_DEAGGREGATED_CSV} ({total_remaining_rows} rows)")

    # Output metrics summary text file inside filtered_final
    summary_text = (
        "====================================================\n"
        "           FILTERED PIPELINE METRICS SUMMARY        \n"
        "====================================================\n"
        f"Total Raw Rows in Input Dataset: {total_raw_rows}\n"
        f"Rows Remaining After Filtering: {total_remaining_rows}\n"
        f"Unmodified Chains: {total_unmodified_rows}\n"
        f"Conflicting/Modified Chains: {total_modified_rows}\n"
        "====================================================\n"
    )
    FILTERED_SUMMARY_TXT.write_text(summary_text, encoding="utf-8")
    print(f"  -> Saved metrics summary text file: {FILTERED_SUMMARY_TXT}")

    # 3. Build Full Unpruned Filtered Graph Cache (.graphml) inside filtered_final
    print("\n[Pipeline] Building filtered NetworkX graph cache...")
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

    G_save = G_filtered.copy()
    for u, v, d in G_save.edges(data=True):
        d["patents"] = json.dumps(list(d["patents"]))
        d["triggers"] = json.dumps(list(d["triggers"]))
        d["nouns"] = json.dumps(list(d.get("nouns", set())))
        d["verbs"] = json.dumps(list(d.get("verbs", set())))
        d["occurrences"] = json.dumps(d.get("occurrences", []))

    nx.write_graphml(G_save, FILTERED_GRAPH_CACHE)
    print(f"  -> Saved filtered graph cache: {FILTERED_GRAPH_CACHE}")

    # 4. Generate Sankey Diagrams inside filtered_final
    print("\n[Pipeline] Generating Sankey diagrams...")
    
    export_sankey_diagram(
        df_deagg_filtered,
        FILTERED_SANKEY_OUTPUT,
        title_text="Filtered Patent Causal Chain [Combined]: [Technical] ➔ [Attribute] ➔ [Failure]"
    )

    df_deagg_filtered["_cat"] = df_deagg_filtered.apply(lambda r: get_row_category(r, forward_words, backward_words, static_words), axis=1)

    df_forward = df_deagg_filtered[df_deagg_filtered["_cat"] == "forward"]
    df_backward = df_deagg_filtered[df_deagg_filtered["_cat"] == "backward"]
    df_static = df_deagg_filtered[df_deagg_filtered["_cat"] == "static"]

    export_sankey_diagram(
        df_forward,
        FORWARD_SANKEY_OUTPUT,
        title_text="Filtered Patent Causal Chain [Forward / Informative]: [Technical] ➔ [Attribute] ➔ [Failure]"
    )
    export_sankey_diagram(
        df_backward,
        BACKWARD_SANKEY_OUTPUT,
        title_text="Filtered Patent Causal Chain [Backward]: [Technical] ➔ [Attribute] ➔ [Failure]"
    )
    export_sankey_diagram(
        df_static,
        STATIC_SANKEY_OUTPUT,
        title_text="Filtered Patent Causal Chain [Static]: [Technical] ➔ [Attribute] ➔ [Failure]"
    )


# =====================================================================
# 4. MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("\n=== Starting Post-Processing Filter Pipeline ===")
    
    allowed_terms = get_allowed_filter_terms()
    
    # Keyword remove_static_terms can be set to True or False here
    filter_deaggregated_table_and_export(allowed_terms, remove_static_terms=True)

    print("\nFiltered post-processing completed successfully!")