import ast
import itertools
import json
import os
from pathlib import Path
import sys
from collections import defaultdict

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
# CONFIGURATION & OUTPUT PATHS
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "unsupervised_link"
    / "output"
    / "causal_filtered_unsupervised_sentiment_pos.csv"
)

OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "causal_graph"

# Primary Outputs
GRAPH_CACHE_FILE = OUTPUT_DIR / "patent_causal_graph.graphml"
SANKEY_OUTPUT_FILE = OUTPUT_DIR / "patent_edge_cases_sankey.html"
SUMMARY_CSV_FILE = OUTPUT_DIR / "patent_causal_chains_summary.csv"

# Secondary POS-Enriched Outputs
POS_GRAPH_CACHE_FILE = OUTPUT_DIR / "patent_causal_pos_graph.graphml"
POS_SANKEY_OUTPUT_FILE = OUTPUT_DIR / "patent_edge_cases_pos_sankey.html"
POS_SUMMARY_CSV_FILE = OUTPUT_DIR / "patent_causal_chains_pos_summary.csv"

MAX_ROWS = None

NODE_COLOR_MAP = {
    "Technical": "rgba(31, 119, 180, 0.8)",  # Blue
    "Attribute": "rgba(44, 160, 44, 0.8)",  # Green
    "Failure": "rgba(214, 39, 40, 0.8)",    # Red
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
    """Safely parses stringified Python lists, JSON arrays, or raw list objects."""
    if isinstance(field_val, list):
        return [str(i).strip() for i in field_val if str(i).strip()]
    
    if pd.isna(field_val):
        return []
        
    if isinstance(field_val, str) and field_val.strip():
        cleaned = field_val.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
                if isinstance(parsed, list):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except (ValueError, SyntaxError):
                pass
            
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except json.JSONDecodeError:
                pass
                
            items = cleaned.strip("[]").split(",")
            return [i.strip().strip("'\"") for i in items if i.strip()]
        return [cleaned]
    return []


def extract_causal_triplets(row):
    """Parses a row from the real dataset schema and extracts directed edges strictly:
    [Technical] -> [Attribute] -> [Failure]
    """
    edges = []

    pat_id = str(row.get("pat_id", "UNKNOWN"))
    para_id = str(row.get("paragraph_id", "UNKNOWN"))

    techs = parse_field(row.get("occurrence_technical", []))
    attrs = parse_field(row.get("occurrence_variable", []))
    failures = parse_field(row.get("occurrence_failures", []))
    causals = parse_field(row.get("causal_ocurrence_words", []))

    # CORRECTED: Read 'nouns' and 'verbs' directly matching your CSV column headers
    nouns = parse_field(row.get("nouns", []))
    verbs = parse_field(row.get("verbs", []))

    if not (causals and techs and attrs and failures):
        return edges

    trigger = str(causals[0]).lower()

    # Step 1: Technical -> Attribute
    for t, a in itertools.product(techs, attrs):
        if t != a:
            edges.append({
                "source": t,
                "source_type": "Technical",
                "target": a,
                "target_type": "Attribute",
                "trigger": "context_pair",
                "pat_id": pat_id,
                "para_id": para_id,
                "nouns": nouns,
                "verbs": verbs,
            })

    # Step 2: Attribute -> Failure
    for a, f in itertools.product(attrs, failures):
        if a != f:
            edges.append({
                "source": a,
                "source_type": "Attribute",
                "target": f,
                "target_type": "Failure",
                "trigger": trigger,
                "pat_id": pat_id,
                "para_id": para_id,
                "nouns": nouns,
                "verbs": verbs,
            })

    return edges


# =====================================================================
# 2. NETWORKX GRAPH BUILDER WITH DISK CACHING
# =====================================================================
def build_or_load_graph(df_or_path, cache_file=GRAPH_CACHE_FILE, max_rows=None, include_pos=False):
    """Checks if a pre-computed GraphML file exists.
    If present, loads it directly. Otherwise, processes the dataframe and saves it.
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    target_name = "POS Causal Graph" if include_pos else "Primary Causal Graph"

    if cache_path.exists():
        print(f"[Cache] Found existing graph ({target_name}) at '{cache_path}'. Loading from disk...")
        G = nx.read_graphml(cache_path)

        for u, v, d in G.edges(data=True):
            for key in ("patents", "triggers", "nouns", "verbs"):
                if key in d and isinstance(d[key], str):
                    try:
                        d[key] = set(json.loads(d[key]))
                    except json.JSONDecodeError:
                        d[key] = set()
        return G

    print(f"[Pipeline] Computing {target_name} from dataset...")

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

            if G.has_edge(e["source"], e["target"]):
                G[e["source"]][e["target"]]["weight"] += 1
                G[e["source"]][e["target"]]["patents"].add(e["pat_id"])
                G[e["source"]][e["target"]]["triggers"].add(e["trigger"])
                if include_pos:
                    # CORRECTED: Key matching 'nouns' and 'verbs' from dictionary
                    G[e["source"]][e["target"]]["nouns"].update(e.get("nouns", []))
                    G[e["source"]][e["target"]]["verbs"].update(e.get("verbs", []))
            else:
                edge_attr = {
                    "weight": 1,
                    "trigger": e["trigger"],
                    "patents": {e["pat_id"]},
                    "triggers": {e["trigger"]},
                }
                if include_pos:
                    edge_attr["nouns"] = set(e.get("nouns", []))
                    edge_attr["verbs"] = set(e.get("verbs", []))

                G.add_edge(e["source"], e["target"], **edge_attr)

    # Save graph for future runs
    G_save = G.copy()
    for u, v, d in G_save.edges(data=True):
        d["patents"] = json.dumps(list(d["patents"]))
        d["triggers"] = json.dumps(list(d["triggers"]))
        if include_pos:
            d["nouns"] = json.dumps(list(d.get("nouns", set())))
            d["verbs"] = json.dumps(list(d.get("verbs", set())))

    nx.write_graphml(G_save, cache_path)
    print(f"[Cache] Saved {target_name} to '{cache_path}'.")

    return G


# =====================================================================
# 3. PATHFINDER
# =====================================================================
def find_strict_cross_patent_chains_fast(G):
    """Fast path finder enforcing zero node-type repetition along any path."""
    valid_paths = []
    node_type_map = {n: d.get("node_type", "Unknown") for n, d in G.nodes(data=True)}

    for middle in G.nodes():
        type_middle = node_type_map[middle]
        if type_middle != "Attribute":
            continue

        predecessors = [u for u in G.predecessors(middle) if u != middle]
        successors = [v for v in G.successors(middle) if v != middle]

        if not predecessors or not successors:
            continue

        for t in predecessors:
            if node_type_map[t] != "Technical":
                continue

            for f in successors:
                if node_type_map[f] != "Failure":
                    continue

                e1 = G[t][middle]
                e2 = G[middle][f]

                shared = e1["patents"].intersection(e2["patents"])
                if not shared:
                    has_causal_1 = any(tr != "context_pair" for tr in e1["triggers"])
                    has_causal_2 = any(tr != "context_pair" for tr in e2["triggers"])
                    if not (has_causal_1 and has_causal_2):
                        continue

                valid_paths.append([t, middle, f])

    return valid_paths


# =====================================================================
# 4. SANKEY DIAGRAM GENERATOR
# =====================================================================
def export_sankey_html(G, valid_paths, output_filename=SANKEY_OUTPUT_FILE, include_pos=False):
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
        patents_list = list(data.get("patents", []))
        patents = ", ".join(patents_list[:5]) + ("..." if len(patents_list) > 5 else "")
        triggers = ", ".join(data.get("triggers", []))

        hover_lines = [
            f"<b>From:</b> {orig_u}",
            f"<b>To:</b> {orig_v}",
            f"<b>Triggers:</b> {triggers}",
            f"<b>Patents:</b> {patents}",
            f"<b>Path Flow Count:</b> {flow_value}",
        ]

        if include_pos:
            nouns_list = sorted(list(data.get("nouns", [])))
            verbs_list = sorted(list(data.get("verbs", [])))
            nouns_str = ", ".join(nouns_list[:8]) + ("..." if len(nouns_list) > 8 else "")
            verbs_str = ", ".join(verbs_list[:8]) + ("..." if len(verbs_list) > 8 else "")

            hover_lines.append(f"<b>Nouns:</b> {nouns_str or 'None'}")
            hover_lines.append(f"<b>Verbs:</b> {verbs_str or 'None'}")

        link_labels.append("<br>".join(hover_lines))

    title_suffix = " (With POS Meta)" if include_pos else ""

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
        title_text=f"Patent Causal Chain: [Technical] ➔ [Attribute] ➔ [Failure]{title_suffix}",
        font_size=12,
        height=750,
        margin=dict(l=50, r=200, t=60, b=50),
    )

    fig.write_html(output_path)
    print(f"[Visualizer] Sankey diagram exported to: {output_path}")


# =====================================================================
# 5. CSV EXPORTER
# =====================================================================
def export_causal_chains_summary(G, valid_paths, output_filename=SUMMARY_CSV_FILE, include_pos=False):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for path in valid_paths:
        t_node, a_node, f_node = path[0], path[1], path[2]

        e1_data = G[t_node][a_node] if G.has_edge(t_node, a_node) else {}
        e2_data = G[a_node][f_node] if G.has_edge(a_node, f_node) else {}

        all_patents = sorted(list(set(e1_data.get("patents", set())).union(e2_data.get("patents", set()))))
        all_triggers = sorted(list(set(e1_data.get("triggers", set())).union(e2_data.get("triggers", set()))))

        rec = {
            "technical": t_node,
            "attribute": a_node,
            "failure": f_node,
            "patents": ", ".join(all_patents),
            "triggers": ", ".join(all_triggers),
        }

        if include_pos:
            chain_nouns = sorted(list(set(e1_data.get("nouns", set())).union(e2_data.get("nouns", set()))))
            chain_verbs = sorted(list(set(e1_data.get("verbs", set())).union(e2_data.get("verbs", set()))))

            rec["nouns"] = ", ".join(chain_nouns)
            rec["verbs"] = ", ".join(chain_verbs)
            rec["noun_count"] = len(chain_nouns)
            rec["verb_count"] = len(chain_verbs)

        records.append(rec)

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(output_path, index=False)
    print(f"[CSV Exporter] Saved summary report to: {output_path}")
    return df_summary


# =====================================================================
# 6. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Remove cached graphs to force re-computation with new POS fields
    for f in (GRAPH_CACHE_FILE, POS_GRAPH_CACHE_FILE):
        if os.path.exists(f):
            os.remove(f)

    print("\n=== Pipeline 1: Processing Primary Causal Graph ===")
    G_primary = build_or_load_graph(
        INPUT_CSV_PATH, cache_file=GRAPH_CACHE_FILE, max_rows=1000, include_pos=False
    )

    paths_primary = find_strict_cross_patent_chains_fast(G_primary)
    print(f"Strict Cross-Patent Chains Found: {len(paths_primary)}")

    if paths_primary:
        export_sankey_html(G_primary, paths_primary, output_filename=SANKEY_OUTPUT_FILE, include_pos=False)
        export_causal_chains_summary(G_primary, paths_primary, output_filename=SUMMARY_CSV_FILE, include_pos=False)

    print("\n=== Pipeline 2: Processing Secondary POS-Enriched Graph ===")
    G_pos = build_or_load_graph(
        INPUT_CSV_PATH, cache_file=POS_GRAPH_CACHE_FILE, max_rows=1000, include_pos=True
    )

    paths_pos = find_strict_cross_patent_chains_fast(G_pos)
    print(f"Strict Cross-Patent Chains (POS) Found: {len(paths_pos)}")

    if paths_pos:
        export_sankey_html(G_pos, paths_pos, output_filename=POS_SANKEY_OUTPUT_FILE, include_pos=True)
        export_causal_chains_summary(G_pos, paths_pos, output_filename=POS_SUMMARY_CSV_FILE, include_pos=True)

    print("\n[Done] All primary and secondary outputs generated successfully.")