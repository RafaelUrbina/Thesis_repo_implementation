import itertools
import json
import os
from pathlib import Path
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
# CONFIGURATION & CONFIGURABLE ROW LIMIT
# =====================================================================
INPUT_CSV_PATH = (
    PATENT_PIPELINE_PATH
    / "supervised_link"
    / "output"
    / "causal_filtered_supervised_exploded.csv"
)

OUTPUT_DIR = PATENT_PIPELINE_PATH / "knowledge_abstract" / "output" / "causal_graph"
GRAPH_CACHE_FILE = OUTPUT_DIR / "patent_causal_graph.graphml"
SANKEY_OUTPUT_FILE = OUTPUT_DIR / "patent_edge_cases_sankey.html"

# Set integer (e.g., 500) to inspect subset, or None to process entire file
MAX_ROWS = None  

NODE_COLOR_MAP = {
    "Technical": "rgba(31, 119, 180, 0.8)",  # Blue
    "Attribute": "rgba(44, 160, 44, 0.8)",  # Green
    "Failure": "rgba(214, 39, 40, 0.8)",  # Red
}

TYPE_SHORT_MAP = {
    "Technical": "T",
    "Attribute": "A",
    "Failure": "F",
}


# =====================================================================
# 1. STRICT CAUSAL CHAIN EXTRACTION
# =====================================================================
def parse_field(field_val):
    """Safely parses stringified lists or raw list objects."""
    if isinstance(field_val, list):
        return field_val
    if isinstance(field_val, str) and field_val.strip():
        # Handle stringified Python lists or JSON arrays
        cleaned = field_val.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                # Replace single quotes for valid JSON parsing
                return json.loads(cleaned.replace("'", '"'))
            except Exception:
                # Fallback simple string cleaning if json parser fails
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

    # Map dataset column names to graph taxonomy
    techs = parse_field(row.get("occurrence_technical", []))
    attrs = parse_field(row.get("occurrence_variable", []))
    failures = parse_field(row.get("occurrence_failures", []))
    causals = parse_field(row.get("causal_ocurrence_words", []))

    # Discard if any required component is missing
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
                    "trigger": "context_pair",
                    "pat_id": pat_id,
                    "para_id": para_id,
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
                }
            )

    return edges


# =====================================================================
# 2. NETWORKX GRAPH BUILDER WITH DISK CACHING
# =====================================================================
def build_or_load_graph(df_or_path, cache_file=GRAPH_CACHE_FILE, max_rows=None):
    """Checks if a pre-computed GraphML file exists.
    If present, loads it directly. Otherwise, processes the data frame and saves it.
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[Cache] Found existing graph at '{cache_path}'. Loading from disk...")
        G = nx.read_graphml(cache_path)

        # Deserialize sets stored as stringified JSON in GraphML
        for u, v, d in G.edges(data=True):
            if "patents" in d and isinstance(d["patents"], str):
                d["patents"] = set(json.loads(d["patents"]))
            if "triggers" in d and isinstance(d["triggers"], str):
                d["triggers"] = set(json.loads(d["triggers"]))
        return G

    print("[Pipeline] Computing knowledge graph from dataset...")

    if isinstance(df_or_path, (str, Path)):
        print(f"[Data] Loading CSV: {df_or_path} (nrows={max_rows})")
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
            else:
                G.add_edge(
                    e["source"],
                    e["target"],
                    weight=1,
                    trigger=e["trigger"],
                    patents={e["pat_id"]},
                    triggers={e["trigger"]},
                )

    # Save graph for future runs
    G_save = G.copy()
    for u, v, d in G_save.edges(data=True):
        d["patents"] = json.dumps(list(d["patents"]))
        d["triggers"] = json.dumps(list(d["triggers"]))

    nx.write_graphml(G_save, cache_path)
    print(f"[Cache] Knowledge graph saved to '{cache_path}'.")

    return G


# =====================================================================
# 3. HIGH-PERFORMANCE GENERALIZED PATHFINDER (NO TYPE REPETITION)
# =====================================================================
def find_strict_cross_patent_chains_fast(G):
    """Fast path finder enforcing zero node-type repetition along any path.
    Enforces strict categorical progression without hardcoded term checks.
    """
    valid_paths = []

    node_type_map = {n: d.get("node_type", "Unknown") for n, d in G.nodes(data=True)}

    for middle in G.nodes():
        type_middle = node_type_map[middle]

        predecessors = [u for u in G.predecessors(middle) if u != middle]
        successors = [v for v in G.successors(middle) if v != middle]

        if not predecessors or not successors:
            continue

        for t in predecessors:
            type_t = node_type_map[t]

            # Rule 1: Node type of Source cannot match Node type of Middle
            if type_t == type_middle:
                continue

            for f in successors:
                type_f = node_type_map[f]

                # Rule 2: No repeated node types along the chain (T != M, M != F, T != F)
                if type_middle == type_f or type_t == type_f:
                    continue

                # Rule 3: Enforce standard forward progression order
                if (type_t, type_middle, type_f) != ("Technical", "Attribute", "Failure"):
                    continue

                e1 = G[t][middle]
                e2 = G[middle][f]

                # Check Cross-Patent Transition Validation
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
def export_sankey_html(G, valid_paths, output_filename=SANKEY_OUTPUT_FILE):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not valid_paths:
        print("\n[Visualizer] No valid paths to render.")
        return

    stage_nodes = {}
    # Use a dictionary to ACCUMULATE flow weights across paths
    edge_weights = {} 
    edge_metadata = {}

    for path in valid_paths:
        t_node, a_node, f_node = path[0], path[1], path[2]

        stage_nodes[(t_node, 0)] = G.nodes[t_node].get("node_type", "Technical")
        stage_nodes[(a_node, 1)] = G.nodes[a_node].get("node_type", "Attribute")
        stage_nodes[(f_node, 2)] = G.nodes[f_node].get("node_type", "Failure")

        # Define stage-aware edge keys
        e1_key = ((t_node, 0), (a_node, 1), t_node, a_node)
        e2_key = ((a_node, 1), (f_node, 2), a_node, f_node)

        # Accumulate weight for e1
        w1 = G[t_node][a_node].get("weight", 1) if G.has_edge(t_node, a_node) else 1
        edge_weights[e1_key] = edge_weights.get(e1_key, 0) + w1
        edge_metadata[e1_key] = (t_node, a_node)

        # Accumulate weight for e2
        w2 = G[a_node][f_node].get("weight", 1) if G.has_edge(a_node, f_node) else 1
        edge_weights[e2_key] = edge_weights.get(e2_key, 0) + w2
        edge_metadata[e2_key] = (a_node, f_node)

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

    for (src_key, tgt_key, orig_u, orig_v), total_weight in edge_weights.items():
        sources.append(node_to_idx[src_key])
        targets.append(node_to_idx[tgt_key])
        values.append(total_weight)

        data = G[orig_u][orig_v] if G.has_edge(orig_u, orig_v) else {}
        patents_list = list(data.get("patents", []))
        patents = ", ".join(patents_list[:5]) + ("..." if len(patents_list) > 5 else "")
        triggers = ", ".join(data.get("triggers", []))

        hover_info = (
            f"<b>From:</b> {orig_u}<br>"
            f"<b>To:</b> {orig_v}<br>"
            f"<b>Triggers:</b> {triggers}<br>"
            f"<b>Patents:</b> {patents}<br>"
            f"<b>Total Flow Weight:</b> {total_weight}"
        )
        link_labels.append(hover_info)

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
        title_text="Patent Causal Chain: [Technical] ➔ [Attribute] ➔ [Failure]",
        font_size=12,
        height=750,
        margin=dict(l=50, r=200, t=60, b=50),
    )

    fig.write_html(output_path)
    print(f"\n[Visualizer] Interactive Sankey diagram exported to: {output_path}")

# =====================================================================
# 5. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    if os.path.exists(GRAPH_CACHE_FILE):
        os.remove(GRAPH_CACHE_FILE)

    G = build_or_load_graph(
        INPUT_CSV_PATH, cache_file=GRAPH_CACHE_FILE, max_rows=1000
    )

    print("\n=== Knowledge Extraction Pipeline ===")
    print(f"Total Nodes Processed: {G.number_of_nodes()}")
    print(f"Total Edges Extracted: {G.number_of_edges()}")

    paths = find_strict_cross_patent_chains_fast(G)
    print(f"\nStrict Cross-Patent Chains Found: {len(paths)}")
    for p in paths[:20]:
        print(" ➔ ".join(p))
    if len(paths) > 20:
        print(f"... and {len(paths) - 20} more chains.")

    if paths:
        export_sankey_html(G, paths, output_filename=SANKEY_OUTPUT_FILE)
    else:
        print("\n[Visualizer] No valid 3-step causal chains found to visualize.")