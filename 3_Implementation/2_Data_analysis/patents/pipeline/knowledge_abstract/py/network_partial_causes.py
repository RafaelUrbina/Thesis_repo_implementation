import ast
from collections import defaultdict
import itertools
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
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

# =====================================================================
# CONFIGURATION & OUTPUT PATHS (POS-Enriched Only)
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "dataset_creation"
    / "2_unsupervised_link"
    / "output"
    / "causal_filtered_unsupervised_sentiment_pos.csv"
)

OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "causal_graph"

# POS-Enriched Outputs
GRAPH_CACHE_FILE = OUTPUT_DIR / "patent_causal_pos_graph.graphml"
SANKEY_OUTPUT_FILE = OUTPUT_DIR / "patent_edge_cases_pos_sankey.html"
SUMMARY_CSV_FILE = OUTPUT_DIR / "patent_causal_chains_pos_summary.csv"

MAX_ROWS = None

# Toggle: If True, include complete_link_text & partial_link_text in CSV output. If False, omit them.
TEXT_SENTENCES = False

# Parameter: Nodes with total degree (in_degree + out_degree) <= this value will be pruned.
# Set to 1 so that terms must appear more than once (degree > 1) to remain in the graph.
MIN_TOTAL_DEGREE = 20

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
# 1. PARSING & CAUSAL CHAIN EXTRACTION
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


def extract_causal_triplets(row):
    """Parses a row from the dataset and extracts directed edges strictly:

    [Technical] -> [Attribute] -> [Failure]
    """
    edges = []

    pat_id = str(row.get("pat_id", "UNKNOWN")).strip()
    para_id = str(row.get("paragraph_id", "UNKNOWN")).strip()

    # Attempt to retrieve text column from standard dataset names
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
        return edges

    trigger = str(causals[0]).lower()

    # Step 1: Technical -> Attribute
    for t, a in itertools.product(techs, attrs):
        if t != a:
            edges.append(
                {
                    "source": t,
                    "source_type": "Technical",
                    "target": a,
                    "target_type": "Attribute",
                    "trigger": trigger,
                    "pat_id": pat_id,
                    "para_id": para_id,
                    "text": text_val,
                    "nouns": nouns,
                    "verbs": verbs,
                }
            )

    # Step 2: Attribute -> Failure
    for a, f in itertools.product(attrs, failures):
        if a != f:
            edges.append(
                {
                    "source": a,
                    "source_type": "Attribute",
                    "target": f,
                    "target_type": "Failure",
                    "trigger": trigger,
                    "pat_id": pat_id,
                    "para_id": para_id,
                    "text": text_val,
                    "nouns": nouns,
                    "verbs": verbs,
                }
            )

    return edges


# =====================================================================
# 2. NETWORKX GRAPH BUILDER WITH DISK CACHING & DEGREE FILTERING
# =====================================================================
def build_or_load_graph(df_or_path, cache_file=GRAPH_CACHE_FILE, max_rows=None):
    """Checks if a pre-computed POS GraphML file exists.

    If present, loads it directly. Otherwise, processes the dataframe and saves it.
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[Cache] Found existing POS graph at '{cache_path}'. Loading from disk...")
        G = nx.read_graphml(cache_path)

        for u, v, d in G.edges(data=True):
            for key in ("patents", "triggers", "nouns", "verbs"):
                if key in d and isinstance(d[key], str):
                    try:
                        d[key] = set(json.loads(d[key]))
                    except json.JSONDecodeError:
                        d[key] = set()
            if "occurrences" in d and isinstance(d["occurrences"], str):
                try:
                    d["occurrences"] = json.loads(d["occurrences"])
                except json.JSONDecodeError:
                    d["occurrences"] = []
        
        # Apply frequency filter even when loading from cache
        filter_low_degree_nodes(G, MIN_TOTAL_DEGREE)
        return G

    print("[Pipeline] Computing POS Causal Graph from dataset...")

    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path, nrows=max_rows)
    else:
        df = df_or_path.head(max_rows) if max_rows is not None else df_or_path

    G = nx.DiGraph()

    for _, row in df.iterrows():
        edges = extract_causal_triplets(row)
        for e in edges:
            if not G.has_node(e["source"]):
                G.add_node(e["source"], node_type=e["source_type"])

            if not G.has_node(e["target"]):
                G.add_node(e["target"], node_type=e["target_type"])

            occ_entry = {
                "pat_id": e["pat_id"],
                "para_id": e["para_id"],
                "text": e["text"],
            }

            if G.has_edge(e["source"], e["target"]):
                G[e["source"]][e["target"]]["weight"] += 1
                G[e["source"]][e["target"]]["patents"].add(e["pat_id"])
                G[e["source"]][e["target"]]["triggers"].add(e["trigger"])
                G[e["source"]][e["target"]]["nouns"].update(e.get("nouns", []))
                G[e["source"]][e["target"]]["verbs"].update(e.get("verbs", []))
                G[e["source"]][e["target"]]["occurrences"].append(occ_entry)
            else:
                edge_attr = {
                    "weight": 1,
                    "trigger": e["trigger"],
                    "patents": {e["pat_id"]},
                    "triggers": {e["trigger"]},
                    "nouns": set(e.get("nouns", [])),
                    "verbs": set(e.get("verbs", [])),
                    "occurrences": [occ_entry],
                }
                G.add_edge(e["source"], e["target"], **edge_attr)

    # Filter out nodes based on MIN_TOTAL_DEGREE threshold
    filter_low_degree_nodes(G, MIN_TOTAL_DEGREE)

    # Save graph for future runs
    G_save = G.copy()
    for u, v, d in G_save.edges(data=True):
        d["patents"] = json.dumps(list(d["patents"]))
        d["triggers"] = json.dumps(list(d["triggers"]))
        d["nouns"] = json.dumps(list(d.get("nouns", set())))
        d["verbs"] = json.dumps(list(d.get("verbs", set())))
        d["occurrences"] = json.dumps(d.get("occurrences", []))

    nx.write_graphml(G_save, cache_path)
    print(f"[Cache] Saved POS graph to '{cache_path}'.")

    return G


def filter_low_degree_nodes(G, min_degree):
    """Iteratively removes nodes whose total edge count (in_degree + out_degree) <= min_degree."""
    while True:
        nodes_to_remove = [
            n for n in G.nodes() if max(G.in_degree(n), G.out_degree(n)) < min_degree
        ]
        if not nodes_to_remove:
            break
        G.remove_nodes_from(nodes_to_remove)


# =====================================================================
# 3. PATHFINDER
# =====================================================================
def find_strict_cross_patent_chains_fast(G):
    """Fast path finder enforcing zero node-type repetition along any path."""
    valid_paths = []
    node_type_map = {
        n: d.get("node_type", "Unknown") for n, d in G.nodes(data=True)
    }

    for middle in G.nodes():
        type_middle = node_type_map.get(middle, "Unknown")
        if type_middle != "Attribute":
            continue

        predecessors = [u for u in G.predecessors(middle) if u != middle]
        successors = [v for v in G.successors(middle) if v != middle]

        if not predecessors or not successors:
            continue

        for t in predecessors:
            if node_type_map.get(t) != "Technical":
                continue

            for f in successors:
                if node_type_map.get(f) != "Failure":
                    continue

                e1 = G[t][middle]
                e2 = G[middle][f]

                shared = e1["patents"].intersection(e2["patents"])
                if not shared:
                    has_causal_1 = any(
                        tr not in ("context_pair", "") for tr in e1["triggers"]
                    )
                    has_causal_2 = any(
                        tr not in ("context_pair", "") for tr in e2["triggers"]
                    )
                    if not (has_causal_1 and has_causal_2):
                        continue

                valid_paths.append([t, middle, f])

    return valid_paths


# =====================================================================
# 4. SANKEY DIAGRAM GENERATOR
# =====================================================================
def export_sankey_html(G, valid_paths, output_filename=SANKEY_OUTPUT_FILE):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not valid_paths:
        print(f"\n[Visualizer] No valid paths to render for {output_filename}.")
        return

    stage_nodes = {}
    edge_counts = defaultdict(int)

    for path in valid_paths:
        t_node, a_node, f_node = path[0], path[1], path[2]

        stage_nodes[(t_node, 0)] = G.nodes[t_node].get("node_type", "Technical")
        stage_nodes[(a_node, 1)] = G.nodes[a_node].get("node_type", "Attribute")
        stage_nodes[(f_node, 2)] = G.nodes[f_node].get("node_type", "Failure")

        e1_key = ((t_node, 0), (a_node, 1), t_node, a_node)
        e2_key = ((a_node, 1), (f_node, 2), a_node, f_node)

        edge_counts[e1_key] += 1
        edge_counts[e2_key] += 1

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

    for (src_key, tgt_key, orig_u, orig_v), flow_value in edge_counts.items():
        sources.append(node_to_idx[src_key])
        targets.append(node_to_idx[tgt_key])
        values.append(flow_value)

        data = G[orig_u][orig_v] if G.has_edge(orig_u, orig_v) else {}

        patents_list = sorted(list(data.get("patents", [])))
        patents_str = ", ".join(patents_list[:8]) + (
            "..." if len(patents_list) > 8 else ""
        )
        triggers = ", ".join(sorted(list(data.get("triggers", []))))

        hover_lines = [
            f"<b>From:</b> {orig_u}",
            f"<b>To:</b> {orig_v}",
            f"<b>Patent IDs:</b> {patents_str or 'None'}",
            f"<b>Triggers:</b> {triggers}",
            f"<b>Path Flow Count:</b> {flow_value}",
        ]

        nouns_list = sorted(list(data.get("nouns", [])))
        verbs_list = sorted(list(data.get("verbs", [])))
        nouns_str = ", ".join(nouns_list[:8]) + (
            "..." if len(nouns_list) > 8 else ""
        )
        verbs_str = ", ".join(verbs_list[:8]) + (
            "..." if len(verbs_list) > 8 else ""
        )

        hover_lines.append(f"<b>Nouns:</b> {nouns_str or 'None'}")
        hover_lines.append(f"<b>Verbs:</b> {verbs_str or 'None'}")

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
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Patent Causal Chain (With POS Meta): [Technical] ➔ [Attribute] ➔ [Failure]",
        font_size=12,
        height=900,
        width=900,
        margin=dict(l=50, r=200, t=60, b=50),
    )

    fig.write_html(output_path)
    print(f"[Visualizer] Sankey diagram exported to: {output_path}")


# =====================================================================
# 5. CSV EXPORTER (With Completeness Flag & Conditional Text Columns)
# =====================================================================
def export_causal_chains_summary(G, valid_paths, output_filename=SUMMARY_CSV_FILE):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for path in valid_paths:
        t_node, a_node, f_node = path[0], path[1], path[2]

        e1_data = G[t_node][a_node] if G.has_edge(t_node, a_node) else {}
        e2_data = G[a_node][f_node] if G.has_edge(a_node, f_node) else {}

        all_patents = sorted(
            list(
                set(e1_data.get("patents", set())).union(
                    e2_data.get("patents", set())
                )
            )
        )
        all_triggers = sorted(
            list(
                set(e1_data.get("triggers", set())).union(
                    e2_data.get("triggers", set())
                )
            )
        )

        chain_nouns = sorted(
            list(
                set(e1_data.get("nouns", set())).union(
                    e2_data.get("nouns", set())
                )
            )
        )
        chain_verbs = sorted(
            list(
                set(e1_data.get("verbs", set())).union(
                    e2_data.get("verbs", set())
                )
            )
        )

        # Determine if the chain is Complete or Partial based on row/paragraph match
        e1_occ = e1_data.get("occurrences", [])
        e2_occ = e2_data.get("occurrences", [])

        e2_lookup = {(occ["pat_id"], occ["para_id"]): occ["text"] for occ in e2_occ}

        has_complete = False
        complete_texts = set()
        partial_text_pairs = set()

        for occ1 in e1_occ:
            key = (occ1["pat_id"], occ1["para_id"])
            if key in e2_lookup:
                has_complete = True
                if occ1["text"]:
                    complete_texts.add(occ1["text"])
            else:
                for occ2 in e2_occ:
                    if occ1["text"] or occ2["text"]:
                        pair_str = f"[T->A]: {occ1['text']} | [A->F]: {occ2['text']}"
                        partial_text_pairs.add(pair_str)

        flag = "Complete" if has_complete else "Partial"

        rec = {
            "technical": t_node,
            "attribute": a_node,
            "failure": f_node,
            "flag": flag,
        }

        # Conditionally add text columns based on TEXT_SENTENCES toggle
        if TEXT_SENTENCES:
            complete_link_text = " | ".join(sorted(list(complete_texts))) if has_complete else ""
            partial_link_text = " || ".join(sorted(list(partial_text_pairs))) if not has_complete else ""
            rec["complete_link_text"] = complete_link_text
            rec["partial_link_text"] = partial_link_text

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

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(output_path, index=False)
    print(f"[CSV Exporter] Saved summary report to: {output_path}")
    return df_summary


# =====================================================================
# 6. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    if os.path.exists(GRAPH_CACHE_FILE):
        os.remove(GRAPH_CACHE_FILE)

    print("\n=== Processing POS-Enriched Causal Graph Pipeline ===")
    G_pos = build_or_load_graph(
        INPUT_CSV_PATH,
        cache_file=GRAPH_CACHE_FILE,
        max_rows=MAX_ROWS,
    )

    paths_pos = find_strict_cross_patent_chains_fast(G_pos)
    print(f"Strict Cross-Patent Chains (POS) Found: {len(paths_pos)}")

    if paths_pos:
        export_sankey_html(
            G_pos,
            paths_pos,
            output_filename=SANKEY_OUTPUT_FILE,
        )
        export_causal_chains_summary(
            G_pos,
            paths_pos,
            output_filename=SUMMARY_CSV_FILE,
        )

    print("\n[Done] POS pipeline outputs generated successfully.")