import ast
import csv
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# RELATIVE IMPORTS & PATH RESOLUTION
# -------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

# -------------------------------------------------------------------------
# CONFIGURATION & COLUMN MAPPING
# -------------------------------------------------------------------------
INPUT_CSV = (
    PATENT_PIPELINE_PATH
    / "unsupervised_link/output/causal_filtered_unsupervised_sentiment_pos.csv"
)
OUTPUT_LINKS_CSV = (
    PATENT_PIPELINE_PATH / "knowledge_abstract/output/incomplete_causal_graph/incomplete_pairwise_links.csv"
)
OUTPUT_HTML = (
    PATENT_PIPELINE_PATH / "knowledge_abstract/output/incomplete_causal_graph/incomplete_sankey_diagram.html"
)

# Set the percentile threshold for link occurrence (e.g., 0.75 = top 25% links, 0.50 = above median)
# Set to 0.0 to keep all links.
PERCENTILE_THRESHOLD = 0.90

# Ordered stages matching your exact CSV column names
STAGES = [
    "occurrence_technical",
    "occurrence_variable",
    "causal_ocurrence_words",
    "occurrence_failures",
]

# Prefixes for clear stage differentiation (T, A, C, F)
STAGE_PREFIXES = {
    "occurrence_technical": "(T)",
    "occurrence_variable": "(A)",
    "causal_ocurrence_words": "(C)",
    "occurrence_failures": "(F)",
}

# Color palette matching the pipeline's visual design system
STAGE_COLORS = {
    "occurrence_technical": "#1f77b4",  # Muted Blue
    "occurrence_variable": "#2ca02c",   # Cooked Green
    "causal_ocurrence_words": "#ff7f0e",  # Safety Orange
    "occurrence_failures": "#d62728",   # Brick Red
}
DEFAULT_NODE_COLOR = "#7f7f7f"


def parse_list_cell(cell_value):
    """Safely converts stringified lists like \"['valve', 'pump']\" into Python lists."""
    if pd.isna(cell_value):
        return []
    cell_str = str(cell_value).strip()
    if not cell_str or cell_str in ["[]", "None", "nan", "NaN"]:
        return []

    try:
        parsed = ast.literal_eval(cell_str)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        cleaned = cell_str.strip("[]'\"")
        return [item.strip() for item in cleaned.split(",") if item.strip()]


def get_node_color(node_name: str) -> str:
    """Assigns node colors based on stage prefix."""
    for stage_col, prefix in STAGE_PREFIXES.items():
        if node_name.startswith(f"{prefix} "):
            return STAGE_COLORS.get(stage_col, DEFAULT_NODE_COLOR)
    return DEFAULT_NODE_COLOR


def build_pairwise_sankey(
    input_path: Path,
    output_csv_path: Path,
    output_html_path: Path,
    only_incomplete: bool = True,
    percentile_threshold: float = 0.0,
):
    # 1. Load Data
    df = pd.read_csv(input_path)

    links = []

    # 2. Extract Stage Values per Row
    for idx, row in df.iterrows():
        stage_items = {}
        for stage in STAGES:
            if stage in df.columns:
                stage_items[stage] = parse_list_cell(row[stage])
            else:
                stage_items[stage] = []

        # Identify active non-empty stages in this row
        active_stages = [stage for stage in STAGES if stage_items[stage]]

        # Isolate 2-node dyads (incomplete links)
        if only_incomplete and len(active_stages) != 2:
            continue

        # Extract pairwise connections across consecutive active stages
        for i in range(len(active_stages) - 1):
            src_stage = active_stages[i]
            tgt_stage = active_stages[i + 1]

            src_prefix = STAGE_PREFIXES.get(src_stage, src_stage)
            tgt_prefix = STAGE_PREFIXES.get(tgt_stage, tgt_stage)

            for src_item in stage_items[src_stage]:
                for tgt_item in stage_items[tgt_stage]:
                    links.append(
                        {
                            "source": f"{src_prefix} {src_item}",
                            "target": f"{tgt_prefix} {tgt_item}",
                            "source_stage": src_stage,
                            "target_stage": tgt_stage,
                        }
                    )

    if not links:
        print("[WARNING] No valid pairwise links found in dataset matching criteria.")
        return

    # 3. Aggregate Links
    links_df = pd.DataFrame(links)
    aggregated_links = (
        links_df.groupby(["source", "target", "source_stage", "target_stage"])
        .size()
        .reset_index(name="value")
    )

    # -------------------------------------------------------------------------
    # PERCENTILE FILTERING LOGIC
    # -------------------------------------------------------------------------
    if percentile_threshold > 0.0:
        cutoff_value = aggregated_links["value"].quantile(percentile_threshold)
        print(
            f"[INFO] Filtering links above {percentile_threshold * 100:.0f}th percentile "
            f"(Minimum occurrences required: > {cutoff_value:.2f})"
        )
        # Keep links strictly greater than the percentile cutoff (or >= if you want boundary included)
        aggregated_links = aggregated_links[aggregated_links["value"] > cutoff_value].reset_index(drop=True)

        if aggregated_links.empty:
            print("[WARNING] No links remained after applying the percentile filter. Try lowering PERCENTILE_THRESHOLD.")
            return

    # Save filtered output CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated_links.to_csv(output_csv_path, index=False)
    print(f"[INFO] Pairwise links CSV saved to: {output_csv_path} ({len(aggregated_links)} links)")

    # 4. Map Node Labels to Plotly Indices & Build Color Lists
    unique_nodes = list(
        pd.unique(aggregated_links[["source", "target"]].values.ravel())
    )
    node_indices = {node_name: i for i, node_name in enumerate(unique_nodes)}

    node_colors = [get_node_color(node) for node in unique_nodes]

    plotly_sources = aggregated_links["source"].map(node_indices).tolist()
    plotly_targets = aggregated_links["target"].map(node_indices).tolist()
    plotly_values = aggregated_links["value"].tolist()

    # Link colors with transparency matching Plotly's pipeline style
    link_colors = [
        get_node_color(src).replace("rgb", "rgba").replace(")", ", 0.35)")
        if "rgb" in get_node_color(src)
        else "rgba(200, 200, 200, 0.35)"
        for src in aggregated_links["source"]
    ]

    # 5. Render Plotly Sankey Diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                domain=dict(x=[0, 1], y=[0, 1]),
                orientation="h",
                valueformat=".0f",
                valuesuffix=" occurrences",
                node=dict(
                    pad=20,
                    thickness=18,
                    line=dict(color="#222222", width=0.6),
                    label=unique_nodes,
                    color=node_colors,
                    hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Arial"),
                ),
                link=dict(
                    source=plotly_sources,
                    target=plotly_targets,
                    value=plotly_values,
                    color=link_colors,
                ),
            )
        ]
    )

    p_label = f" (Top {(1 - percentile_threshold) * 100:.0f}% Links)" if percentile_threshold > 0 else ""
    fig.update_layout(
        title=dict(
            text=f"<b>Incomplete 2-Node Association Pathways{p_label}</b><br><sup>(T) Technical, (A) Attribute, (C) Causal Term, and (F) Failure Dyads</sup>",
            x=0.02,
            y=0.96,
            font=dict(size=18, family="Arial, sans-serif", color="#111111"),
        ),
        font=dict(size=12, family="Arial, sans-serif", color="#333333"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=80, b=40),
        autosize=True,
    )

    # Save interactive HTML
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html_path))
    print(f"[INFO] Interactive Sankey saved to: {output_html_path}")


if __name__ == "__main__":
    build_pairwise_sankey(
        input_path=INPUT_CSV,
        output_csv_path=OUTPUT_LINKS_CSV,
        output_html_path=OUTPUT_HTML,
        only_incomplete=True,
        percentile_threshold=PERCENTILE_THRESHOLD,
    )