import itertools
import json
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

FORWARD_TRIGGERS = {
    "results in",
    "gives rise to",
    "leads to",
    "causes",
    "produces",
    "controls",
}
BACKWARD_TRIGGERS = {"due to", "as a result of", "caused by", "owing to"}

NODE_COLOR_MAP = {
    "Technical": "rgba(31, 119, 180, 0.8)",  # Blue
    "Attribute": "rgba(44, 160, 44, 0.8)",  # Green
    "Failure": "rgba(214, 39, 40, 0.8)",  # Red
}


# =====================================================================
# 1. STRICT 4-ELEMENT CAUSAL CHAIN EXTRACTION
# =====================================================================
def extract_causal_triplets(row):
    """Parses a sentence row and extracts directed edges strictly adhering to:
    [Technical] -> [Attribute] -> [Failure]

    Strict Constraints:
    - Requires complete sequence (Technical, Attribute, Failure, and Causal terms).
    - Incomplete or partial sequences are completely discarded.
    """
    edges = []

    pat_id = row.get("pat_id", "UNKNOWN")
    para_id = row.get("paragraph_id", "UNKNOWN")

    techs = row.get("technical_terms", [])
    attrs = row.get("attribute_terms", [])
    failures = row.get("failure_terms", [])
    causals = row.get("causal_ocurrence_words", [])

    # Strict Requirement: Discard if any required component is missing
    if not (causals and techs and attrs and failures):
        return edges

    trigger = causals[0].lower()

    # -----------------------------------------------------------------
    # STEP 1: Context Pair: [Technical] -> [Attribute]
    # -----------------------------------------------------------------
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
                    "is_complete": True,
                }
            )

    # -----------------------------------------------------------------
    # STEP 2: Causal Link: [Attribute] -> [Failure]
    # -----------------------------------------------------------------
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
                    "is_complete": True,
                }
            )

    return edges


# =====================================================================
# 2. NETWORKX GRAPH BUILDER & STRICT TOPOLOGY PATHFINDER
# =====================================================================
def build_knowledge_graph(df):
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
                    pat_id=e["pat_id"],
                    para_id=e["para_id"],
                    patents={e["pat_id"]},
                    triggers={e["trigger"]},
                    is_complete=e["is_complete"],
                )

    return G


def find_strict_cross_patent_chains(G):
    """Finds valid causal paths strictly following:
    [Technical] -> [Attribute] -> [Failure] (or longer valid cross-patent extensions).

    Strict Topology Rules:
    - Path must consist of at least 3 nodes (2 edges).
    - Prevents type shortcuts/stacking.
    - Ensures valid node-type progression: Technical -> Attribute -> Failure.
    """
    valid_paths = []

    tech_nodes = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "Technical"
    ]
    failure_nodes = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "Failure"
    ]

    for start in tech_nodes:
        for end in failure_nodes:
            if nx.has_path(G, start, end):
                for path in nx.all_simple_paths(G, start, end):
                    # Rule 0: Minimum length enforcement (Must have at least 3 nodes)
                    if len(path) < 3:
                        continue

                    is_valid_chain = True

                    # Rule 1: Strict Type Progression & No Type Stacking
                    node_types = [
                        G.nodes[node].get("node_type") for node in path
                    ]

                    # Enforce strict initial sequence: Technical -> Attribute -> Failure
                    if node_types[0] != "Technical" or node_types[1] != "Attribute":
                        continue

                    for i in range(len(node_types) - 1):
                        if node_types[i] == node_types[i + 1]:
                            is_valid_chain = False
                            break

                    if not is_valid_chain:
                        continue

                    # Rule 2: Strict Cross-Patent Hop Validation
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_data = G[u][v]

                        if i > 0:
                            prev_u, prev_v = path[i - 1], path[i]
                            prev_edge_data = G[prev_u][prev_v]

                            shared_patents = prev_edge_data[
                                "patents"
                            ].intersection(edge_data["patents"])

                            if not shared_patents:
                                has_causal_in_prev = any(
                                    t != "context_pair"
                                    for t in prev_edge_data["triggers"]
                                )
                                has_causal_in_next = any(
                                    t != "context_pair"
                                    for t in edge_data["triggers"]
                                )

                                if not (
                                    has_causal_in_prev and has_causal_in_next
                                ):
                                    is_valid_chain = False
                                    break

                    if is_valid_chain:
                        valid_paths.append(path)

    return valid_paths


# =====================================================================
# 3. SANKEY DIAGRAM GENERATOR (PLOTLY)
# =====================================================================
def export_sankey_html(G, valid_paths, output_filename="causal_sankey_diagram.html"):
    """Transforms valid NetworkX DiGraph paths into an interactive Plotly Sankey diagram HTML file.

    Only edges that participate in valid paths are included.
    """
    # 1. Collect only nodes and edges that participate in valid paths
    valid_nodes = set()
    valid_edges = set()

    for path in valid_paths:
        for node in path:
            valid_nodes.add(node)
        for i in range(len(path) - 1):
            valid_edges.add((path[i], path[i + 1]))

    nodes = list(valid_nodes)
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    node_labels = []
    node_colors = []
    for node in nodes:
        n_type = G.nodes[node].get("node_type", "Technical")
        node_labels.append(f"{node} ({n_type})")
        node_colors.append(NODE_COLOR_MAP.get(n_type, "rgba(100,100,100,0.8)"))

    sources, targets, values, link_labels = [], [], [], []

    for u, v in valid_edges:
        data = G[u][v]
        sources.append(node_indices[u])
        targets.append(node_indices[v])
        values.append(data.get("weight", 1))

        patents = ", ".join(data.get("patents", [data.get("pat_id", "N/A")]))
        triggers = ", ".join(data.get("triggers", [data.get("trigger", "link")]))
        hover_info = (
            f"<b>From:</b> {u}<br>"
            f"<b>To:</b> {v}<br>"
            f"<b>Triggers:</b> {triggers}<br>"
            f"<b>Patents:</b> {patents}<br>"
            f"<b>Flow Count:</b> {data.get('weight', 1)}"
        )
        link_labels.append(hover_info)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
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
        title_text="4-Element Patent Causal Chain: [Technical] ➔ [Attribute] ➔ [Causal Term] ➔ [Failure]",
        font_size=12,
        height=750,
    )

    fig.write_html(output_filename)
    print(f"\n[Visualizer] Interactive Sankey diagram exported to: {output_filename}")


# =====================================================================
# 4. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    sample_rows = [
        # =====================================================================
        # 1. NORMAL / COMMON EXPECTED CASES
        # =====================================================================
        {
            # Normal Case 1: Standard 4-element forward chain
            "pat_id": "US-1001-A",
            "paragraph_id": "para_01",
            "text": "excess heat exchanger temperature leads to overheating of the core",
            "technical_terms": ["heat exchanger"],
            "attribute_terms": ["temperature"],
            "failure_terms": ["overheating"],
            "causal_ocurrence_words": ["leads to"],
        },
        {
            # Normal Case 2: Standard 4-element backward chain
            "pat_id": "US-1002-A",
            "paragraph_id": "para_05",
            "text": "cavitation occurs in the fuel pump owing to a sudden pressure drop",
            "technical_terms": ["fuel pump"],
            "attribute_terms": ["pressure drop"],
            "failure_terms": ["cavitation"],
            "causal_ocurrence_words": ["owing to"],
        },
        {
            # Normal Case 3: Valid Cross-Patent Bridge
            # Patent A: turbine blade -> vibration -> fatigue
            # Patent B: turbine blade -> fatigue -> cracking
            # Expected Full Chain: turbine blade -> vibration -> fatigue -> cracking
            "pat_id": "US-1003-A",
            "paragraph_id": "para_12",
            "text": "turbine blade vibration results in structural fatigue",
            "technical_terms": ["turbine blade"],
            "attribute_terms": ["vibration"],
            "failure_terms": ["fatigue"],
            "causal_ocurrence_words": ["results in"],
        },
        {
            "pat_id": "EP-2003-B",
            "paragraph_id": "para_08",
            "text": "severe fatigue causes surface cracking",
            "technical_terms": ["turbine blade"],
            "attribute_terms": ["fatigue"],
            "failure_terms": ["cracking"],
            "causal_ocurrence_words": ["causes"],
        },

        # =====================================================================
        # 2. NOISY & EDGE CASES (Pipeline Stress-Tests)
        # =====================================================================
        {
            # Edge Case 1: Type-Stacking Attempt / Missing Attribute
            "pat_id": "US-9001-X",
            "paragraph_id": "para_30",
            "text": "corrosion results in severe leakage in the pipe",
            "technical_terms": ["pipe"],
            "attribute_terms": [],
            "failure_terms": ["corrosion", "leakage"],
            "causal_ocurrence_words": ["results in"],
        },
        {
            # Edge Case 2: Passive Mention without Causal Trigger
            "pat_id": "US-9002-X",
            "paragraph_id": "para_03",
            "text": "the overall vibration of the housing is monitored continuously",
            "technical_terms": ["housing"],
            "attribute_terms": ["vibration"],
            "failure_terms": [],
            "causal_ocurrence_words": [],
        },
        {
            # Edge Case 3: Cartesian Product Combinatorial Noise
            "pat_id": "US-9003-X",
            "paragraph_id": "para_45",
            "text": "high rotor speed and valve voltage cause thermal deformation and stalling",
            "technical_terms": ["rotor", "valve"],
            "attribute_terms": ["speed", "voltage"],
            "failure_terms": ["thermal deformation", "stalling"],
            "causal_ocurrence_words": ["cause"],
        },
        {
            # Edge Case 4: Incomplete Tuple (Missing Technical Term)
            "pat_id": "EP-9004-Y",
            "paragraph_id": "para_11",
            "text": "excessive friction causes overheating",
            "technical_terms": [],
            "attribute_terms": ["friction"],
            "failure_terms": ["overheating"],
            "causal_ocurrence_words": ["causes"],
        },
    ]

    df = pd.DataFrame(sample_rows)
    G = build_knowledge_graph(df)

    print("=== Knowledge Extraction Pipeline ===")
    print(f"Total Nodes Processed: {G.number_of_nodes()}")
    print(f"Total Edges Extracted: {G.number_of_edges()}")

    paths = find_strict_cross_patent_chains(G)
    print(f"\nStrict Cross-Patent Chains Found: {len(paths)}")
    for p in paths:
        print(" ➔ ".join(p))

    # Export visualization passing both graph and validated paths
    export_sankey_html(G, paths, output_filename="patent_edge_cases_sankey.html")