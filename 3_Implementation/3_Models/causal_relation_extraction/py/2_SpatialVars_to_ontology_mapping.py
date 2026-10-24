"""
Category-Level Semantic Variable Mapper & Patent Causal Linker

This script:
1. Filters and loads metadata strictly from the provided manual variable list.
2. Uses descriptions.csv as a semantic reference dictionary for embedding calculations.
3. Performs strict inner-join matching so unlinked items are completely dropped.
4. Exports filtered linked CSV and NetworkX Bipartite Graph (.graphml / .json).
"""

import json
import os
import re
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Configuration Toggles & Global Filter Set
# -------------------------------------------------------------------------
# Set to "ON" to filter final outputs to only the specified dataset variables.
# Set to "OFF" to keep all variables.
CAUSAL_LINK_FILTER = "ON"

# Explicit global filter set containing your chosen dataset variables.
EXPLICIT_CAUSAL_FILTER_SET = {
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

# -------------------------------------------------------------------------
# Dynamic Path Resolution & Relative Imports
# -------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent

project_root = next(
    (p for p in [current_dir] + list(current_dir.parents) if (p / "Utils").exists()), 
    current_dir.parents[3]
)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import (
    REPO_ROOT,
    PATENT_PIPELINE_PATH,
    ONTOLOGY_OUTPUT_PATH,
    CAUSAL_RELATION_MODEL_PATH,
    KNOWLEDGE_ABSTRACT_PATH,
)

import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

# Embedding & NLP dependencies
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import spacy
from rdflib import Graph, SKOS, RDFS, Namespace, URIRef

# -------------------------------------------------------------------------
# 1. Project Paths & Explicit Variable List Setup
# -------------------------------------------------------------------------
METADATA_CSV_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "descriptions.csv"
CAUSAL_LINKS_CSV_PATH = KNOWLEDGE_ABSTRACT_PATH / "output" / "causal_graph" / "patent_causal_chains_pos_summary.csv"
ONTOLOGY_RDF_PATH = ONTOLOGY_OUTPUT_PATH / "pipeline_knowledge_graph.ttl"

# Dynamic output prefix based on filter status
output_prefix = "filtered_" if CAUSAL_LINK_FILTER.upper() == "ON" else ""

OUTPUT_LINKED_RESULTS_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "2_semantic_mappings" / f"{output_prefix}linked_feature_causal_chain.csv"
OUTPUT_GRAPHML_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "2_semantic_mappings" / f"{output_prefix}bipartite_mapping_graph.graphml"
OUTPUT_GRAPH_JSON_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "2_semantic_mappings" / f"{output_prefix}bipartite_mapping_graph.json"
OUTPUT_CATEGORY_CHILDREN_JSON_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "2_semantic_mappings"/ f"{output_prefix}category_mapped_children.json"

# Explicit target analysis variables list
TARGET_VARIABLES_LIST = [
    'total_intervention_cost_1ring', 'total_fountains_count_1ring',
    'landslide_point_count_1ring', 'max_peak_elevation_1ring',
    'seismic_event_count_1ring', 'avg_seismic_magnitude_1ring', 
    'm_per_hex', 'dist_to_waterway_m', 'rooting_depth_class', 
    'surface_stoniness_class', 'landslide_surface_class',
    'hydraulic_hazard_level', 'epr_NTAXP', 'epr_TAXABINC',
    'epr_SUBEMPTR', 'epr_PENSINCR', 'epr_PENSINCF', 'epr_ENTROAIN', 
    'epr_ENTROAIN01', 'erd_E0_10000', 'erd_E10000_1', 'erd_E15000_2', 
    'erd_E26000_5', 'erd_E55000_7', 'erd_E75000_1', 'erd_E_GE1200', 
    'edst_acq_imm', 'edst_acq_erog', 'eidx_LAND_CON',
    'idw_rain_mm_max_monthly', 'idw_temp_thermal_range',
    'idw_wind_speed_max_max', 'dtmidcnt_max', 'census_pop', 
    'inflow_total', 'avg_descending_soil_speed', 'avg_ascending_soil_speed', 
    'nearest_hydro_river_stage_max_m', 'descending_soil_presence', 'ascending_soil_presence'
]

# Category Definitions Dictionary
CATEGORY_DEFINITIONS = {
    "Class_MeteorologicalMonitoring": (
        "Meteorological and Hydrological Monitoring: weather station, "
        "sensors, gauge, real-time discharge, telemetry, pluviometer."
    ),
    "Class_HydrologicalSettingsAndFloodRisk": (
        "Hydrological Settings and Flood Risk: river inundation, "
        "waterways, culvert, catchment, runoff, stormwater, hydraulic risk."
    ),
    "Class_GeotechnicalDynamics": (
        "Geotechnical Dynamics and Landslide Hazards: slope movement, "
        "mass wasting, seismic activity, earthquake, ground deformation, "
        "subsidence, liquefaction, soil failure."
    ),
    "Class_PedologicalProperties": (
        "Pedological Soil Properties and Agricultural Capacity: soil depth, "
        "rooting depth, ksat, hydraulic conductivity, salinity, erodibility, "
        "soil texture, organic matter, pedology, agricultural land."
    ),
    "Class_TopographicAndClimaticIndicators": (
        "Topographic and Climatic Indicators: elevation, slope aspect, "
        "altitudinal profile, drought index, moisture deficit, precipitation, "
        "temperature variations."
    ),
    "Class_BuiltEnvironmentAndLandUse": (
        "Built Environment, Land Use, and Human Interventions: building footprint, "
        "architectural density, remediation cost, repair costs, roads, "
        "infrastructure, urban land use."
    ),
    "Class_LandscapeRegionalization": (
        "Landscape Regionalization: landscape unit, unita di paesaggio, "
        "ecoregion, geomorphological unit, landscape system classification."
    ),
    "Class_SocioEconomicIndicators": (
        "Socio-Economic Indicators, Vulnerability, and Municipal Metabolism: "
        "population density, average income, wealth, daytime population, "
        "pension, taxable income, municipal water metabolism, social vulnerability."
    ),
}

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------------------------
# 2. Extract Category Terms & Rich Definitions from RDF Graph
# -------------------------------------------------------------------------
def get_node_labels(rdf_graph: Graph, node_uri: URIRef) -> list[str]:
    labels = []
    for _, _, l in rdf_graph.triples((node_uri, SKOS.prefLabel, None)):
        labels.append(str(l))
    for _, _, l in rdf_graph.triples((node_uri, SKOS.altLabel, None)):
        labels.append(str(l))
    for _, _, l in rdf_graph.triples((node_uri, RDFS.label, None)):
        labels.append(str(l))
    return list(set(labels))


def extract_risk_category_terms(rdf_graph_path: Path) -> tuple[pd.DataFrame, Graph]:
    g = Graph()
    EX = Namespace("http://www.semanticweb.org/thesis/pipeline-risk#")

    if rdf_graph_path.exists():
        print(f"Parsing RDF Knowledge Graph from: {rdf_graph_path}...")
        g.parse(str(rdf_graph_path), format="turtle")

    category_concepts = set()
    for s, p, o in g.triples((EX.EnvironmentalSocialRiskScheme, SKOS.hasTopConcept, None)):
        category_concepts.add(o)
    for s, p, o in g.triples((None, SKOS.inScheme, EX.EnvironmentalSocialRiskScheme)):
        if (s, None, EX.VariableClassification) in g:
            category_concepts.add(s)

    categories = []
    if category_concepts:
        for concept in category_concepts:
            uri_str = str(concept)
            concept_key = uri_str.split("#")[-1] if "#" in uri_str else uri_str.split("/")[-1]
            
            labels = get_node_labels(g, concept)
            primary_label = labels[0] if labels else concept_key

            graph_def = ""
            for _, _, d in g.triples((concept, SKOS.definition, None)):
                graph_def = str(d)
            if not graph_def:
                for _, _, c in g.triples((concept, RDFS.comment, None)):
                    graph_def = str(c)

            rich_definition = graph_def if graph_def else CATEGORY_DEFINITIONS.get(concept_key, primary_label)
            
            categories.append({
                "uri": uri_str,
                "category_key": concept_key,
                "category_term": primary_label.lower().strip(),
                "rich_definition": rich_definition,
                "semantic_search_text": f"{primary_label}. {rich_definition}".lower().strip()
            })
    else:
        print("Using fallback category definitions dictionary...")
        for cat_key, cat_def in CATEGORY_DEFINITIONS.items():
            categories.append({
                "uri": f"http://www.semanticweb.org/thesis/pipeline-risk#{cat_key}",
                "category_key": cat_key,
                "category_term": cat_key.replace("Class_", "").lower().strip(),
                "rich_definition": cat_def,
                "semantic_search_text": cat_def.lower().strip()
            })

    df_categories = pd.DataFrame(categories).drop_duplicates(subset=["category_key"]).reset_index(drop=True)
    print(f"Loaded {len(df_categories)} Categories with rich contextual definitions.")
    return df_categories, g


# -------------------------------------------------------------------------
# 3. Load & Filter Metadata Strictly Against Target Variables List
# -------------------------------------------------------------------------
def load_variable_metadata(csv_path: Path, target_vars: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file '{csv_path}' not found! Please check '{csv_path}'.")

    print(f"Loading variable metadata dictionary from: {csv_path}...")
    df_meta = pd.read_csv(csv_path)

    col_mapping = {}
    for col in df_meta.columns:
        clean_col = col.strip().lower()
        if clean_col in ["variable", "variable_name", "var"]:
            col_mapping[col] = "variable_name"
        elif clean_col in ["description", "desc"]:
            col_mapping[col] = "description"
        elif clean_col in ["clean_label", "label"]:
            col_mapping[col] = "clean_label"
        elif clean_col in ["variable_category", "category"]:
            col_mapping[col] = "variable_category"

    df_meta.rename(columns=col_mapping, inplace=True)

    if "variable_name" not in df_meta.columns or "description" not in df_meta.columns:
        raise ValueError(f"Missing required columns in {csv_path.name}. Found: {list(df_meta.columns)}")

    df_meta["variable_name"] = df_meta["variable_name"].astype(str).str.strip()
    
    # STRICT FILTER & RESET INDEX to guarantee contiguous row indices for similarity matrices
    df_meta = df_meta[df_meta["variable_name"].isin(target_vars)].copy()
    df_meta = df_meta.reset_index(drop=True)
    print(f"Filtered metadata to {len(df_meta)} target variables from the explicit variable list.")

    df_meta["description"] = df_meta["description"].fillna("").astype(str).str.strip()

    if "clean_label" not in df_meta.columns or df_meta["clean_label"].str.strip().eq("").all():
        df_meta["clean_label"] = df_meta["variable_name"].apply(
            lambda x: re.sub(r"[_\-]+", " ", x).title()
        )
    else:
        df_meta["clean_label"] = df_meta["clean_label"].fillna("")

    if "variable_category" not in df_meta.columns:
        df_meta["variable_category"] = ""
    else:
        df_meta["variable_category"] = df_meta["variable_category"].fillna("")

    df_meta["semantic_text"] = (
        df_meta["clean_label"] + ". " + 
        df_meta["description"] + " " + 
        df_meta["variable_category"]
    ).str.strip().str.lower()

    return df_meta


# -------------------------------------------------------------------------
# 4. Semantic Embedding Mapping
# -------------------------------------------------------------------------
class SemanticMapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Initializing Transformer Embedding Model ('{model_name}')...")
            self.model = SentenceTransformer(model_name)
            self.use_transformers = True
        else:
            print("SentenceTransformers not installed. Falling back to spaCy similarity...")
            self.use_transformers = False

    def compute_similarity_matrix(self, variable_texts: list, category_search_texts: list) -> np.ndarray:
        if self.use_transformers:
            var_embeddings = self.model.encode(variable_texts, convert_to_tensor=True)
            cat_embeddings = self.model.encode(category_search_texts, convert_to_tensor=True)
            return util.cos_sim(var_embeddings, cat_embeddings).cpu().numpy()
        else:
            sim_matrix = np.zeros((len(variable_texts), len(category_search_texts)))
            var_docs = [nlp(text) for text in variable_texts]
            cat_docs = [nlp(term) for term in category_search_texts]

            for i, v_doc in enumerate(var_docs):
                v_tokens = set([t.lemma_ for t in v_doc if not t.is_stop and t.is_alpha])
                for j, c_doc in enumerate(cat_docs):
                    if c_doc.text in v_doc.text:
                        sim_matrix[i, j] = 0.95
                        continue
                    c_tokens = set([t.lemma_ for t in c_doc if not t.is_stop and t.is_alpha])
                    if v_tokens and c_tokens:
                        intersection = v_tokens.intersection(c_tokens)
                        union = v_tokens.union(c_tokens)
                        sim_matrix[i, j] = len(intersection) / len(union) if union else 0.0
            return sim_matrix


def map_variables_to_categories(df_meta: pd.DataFrame, df_categories: pd.DataFrame) -> pd.DataFrame:
    mapper = SemanticMapper()
    
    variable_texts = df_meta["semantic_text"].tolist()
    category_search_texts = df_categories["semantic_search_text"].tolist()
    
    print("\nComputing embedding similarity matrix for target variables...")
    sim_matrix = mapper.compute_similarity_matrix(variable_texts, category_search_texts)
    
    mapping_results = []
    for idx, row in df_meta.iterrows():
        similarities = sim_matrix[idx]
        best_match_idx = np.argmax(similarities)
        best_score = float(similarities[best_match_idx])
        best_cat_row = df_categories.iloc[best_match_idx]
        
        if best_score >= 0.40:
            quality = "High"
            explanation = "Strong semantic vector alignment between variable description and category concept definition."
        elif best_score >= 0.25:
            quality = "Medium"
            explanation = "Moderate semantic match based on contextual keywords in variable metadata."
        else:
            quality = "Low/Fallback"
            explanation = "Assigned as best-fit candidate category (highest cosine similarity score among available risk categories)."

        mapping_results.append({
            "variable_name": row["variable_name"],
            "clean_label": row["clean_label"],
            "description": row["description"],
            "mapped_category": best_cat_row["category_term"],
            "category_key": best_cat_row["category_key"],
            "category_uri": best_cat_row["uri"],
            "match_confidence": round(best_score, 4),
            "mapping_quality": quality,
            "mapping_explanation": explanation
        })
            
    return pd.DataFrame(mapping_results)


# -------------------------------------------------------------------------
# 5. Extract Category Children Terms & Build Explained Category JSON
# -------------------------------------------------------------------------
def export_category_children_hierarchy_json(
    df_mappings: pd.DataFrame, 
    rdf_graph: Graph, 
    output_json_path: Path
) -> dict:
    print("\nExtracting all child terms & writing mapping explanations for JSON output...")
    category_json = {}

    for _, row in df_mappings.iterrows():
        cat_name = row["mapped_category"]
        cat_uri_str = row["category_uri"]
        var_name = row["variable_name"]
        
        var_detail = {
            "variable_name": var_name,
            "clean_label": row["clean_label"],
            "description": row["description"],
            "match_confidence": row["match_confidence"],
            "mapping_quality": row["mapping_quality"],
            "explanation": row["mapping_explanation"]
        }

        if cat_name in category_json:
            category_json[cat_name]["mapped_spatial_variables"].append(var_name)
            category_json[cat_name]["mapped_variables_detail"].append(var_detail)
            continue

        cat_uri = URIRef(cat_uri_str)
        child_nodes = set()

        for _, _, child in rdf_graph.triples((cat_uri, SKOS.narrower, None)):
            child_nodes.add(child)
        for child, _, _ in rdf_graph.triples((None, SKOS.broader, cat_uri)):
            child_nodes.add(child)

        children_list = []
        for child_uri in child_nodes:
            c_labels = get_node_labels(rdf_graph, child_uri)
            primary_c_label = c_labels[0] if c_labels else str(child_uri).split("#")[-1]
            children_list.append({
                "child_uri": str(child_uri),
                "child_term": primary_c_label.lower().strip(),
                "all_labels": c_labels
            })

        category_json[cat_name] = {
            "category_name": cat_name,
            "category_uri": cat_uri_str,
            "mapped_spatial_variables": [var_name],
            "mapped_variables_detail": [var_detail],
            "total_child_terms": len(children_list),
            "available_child_terms": children_list
        }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(category_json, f, indent=2)

    print(f"Exported Explained Category Hierarchy JSON to: {output_json_path}")
    return category_json


# -------------------------------------------------------------------------
# 6. Link Variables & Patent Causal Chains (Standard Full Link + Final Filter)
# -------------------------------------------------------------------------
def link_variables_to_causal_chain(
    df_mappings: pd.DataFrame,
    category_children_dict: dict,
    causal_links_path: Path,
    filter_mode: str = CAUSAL_LINK_FILTER,
    filter_set: set = EXPLICIT_CAUSAL_FILTER_SET
) -> pd.DataFrame:
    if not causal_links_path.exists():
        raise FileNotFoundError(f"Patent causal links file mandatory but not found at: {causal_links_path}")

    print(f"Loading Patent Causal Links from: {causal_links_path}")
    df_causal = pd.read_csv(causal_links_path)

    exploded_rows = []
    for _, row in df_mappings.iterrows():
        cat_name = row["mapped_category"]
        if cat_name in category_children_dict:
            child_terms = category_children_dict[cat_name]["available_child_terms"]
            for child in child_terms:
                row_dict = row.to_dict()
                row_dict["child_attribute_term"] = child["child_term"]
                row_dict["child_uri"] = child["child_uri"]
                exploded_rows.append(row_dict)

    df_exploded = pd.DataFrame(exploded_rows)

    if df_exploded.empty:
        print("Warning: Exploded DataFrame is empty!")
        return pd.DataFrame()

    # STRICT INNER JOIN: Drops any variable/child term combination that has no matching patent causal link
    df_linked = pd.merge(
        df_exploded,
        df_causal,
        left_on="child_attribute_term",
        right_on="attribute",
        how="inner"
    )

    # FINAL FILTER APPLIED HERE: If ON, keep only rows matching the specified dataset variables list
    if filter_mode.upper() == "ON" and filter_set:
        print(f"Applying final filter: keeping only {len(filter_set)} specified variables in causal link results...")
        df_linked = df_linked[df_linked["variable_name"].isin(filter_set)].copy()
        print(f"Retained {len(df_linked)} rows after filtering.")

    primary_cols = [
        "variable_name", "clean_label", "mapped_category", 
        "child_attribute_term", "match_confidence", "mapping_quality", "mapping_explanation",
        "technical", "failure", "patent_ids", "patent_count", "triggers", "nouns", "verbs", "description"
    ]
    existing_primary = [c for c in primary_cols if c in df_linked.columns]
    other_cols = [c for c in df_linked.columns if c not in existing_primary]
    
    return df_linked[existing_primary + other_cols]


# -------------------------------------------------------------------------
# 7. Build & Export Bipartite NetworkX Graph
# -------------------------------------------------------------------------
def export_bipartite_graph(df_linked_results: pd.DataFrame, graphml_path: Path, json_path: Path):
    print("\nBuilding Undirected Bipartite NetworkX Graph (Variables -> Category Parents)...")
    G = nx.Graph()

    for idx, row in df_linked_results.iterrows():
        var_name = str(row["variable_name"])
        var_node = f"VAR:{var_name}"
        
        G.add_node(
            var_node,
            bipartite=0,
            type="spatial_variable",
            label=str(row.get("clean_label", var_name)),
            description=str(row.get("description", ""))
        )

        cat_term = str(row.get("mapped_category", ""))
        if cat_term:
            cat_node = f"CAT:{cat_term}"
            
            G.add_node(
                cat_node,
                bipartite=1,
                type="risk_category",
                label=cat_term
            )

            G.add_edge(
                var_node,
                cat_node,
                weight=float(row.get("match_confidence", 1.0)),
                confidence=float(row.get("match_confidence", 1.0))
            )

    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(graphml_path))
    print(f"Exported Bipartite GraphML to: {graphml_path}")

    graph_json_data = json_graph.node_link_data(G)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_json_data, f, indent=2)
    print(f"Exported Bipartite JSON Graph to: {json_path}")


# -------------------------------------------------------------------------
# 8. Main Pipeline Execution
# -------------------------------------------------------------------------
def main():
    print(f"=== Pipeline Configuration: CAUSAL_LINK_FILTER = {CAUSAL_LINK_FILTER} ===")
    
    print("\n=== Step 1: Load & Filter Variable Metadata ===")
    df_meta = load_variable_metadata(METADATA_CSV_PATH, TARGET_VARIABLES_LIST)
    print(f"Loaded {len(df_meta)} target variables with rich descriptions.")

    print("\n=== Step 2: Extract Risk Categories & Rich Definitions ===")
    df_categories, rdf_graph = extract_risk_category_terms(ONTOLOGY_RDF_PATH)

    print("\n=== Step 3: Run High-Precision Mapping ===")
    df_mappings = map_variables_to_categories(df_meta, df_categories)

    print("\n--- Variable -> Category Mapping Preview ---")
    print(df_mappings[["variable_name", "mapped_category", "match_confidence", "mapping_quality"]].to_string(index=False))

    print("\n=== Step 4: Extract All Children & Export Explained JSON ===")
    category_children_dict = export_category_children_hierarchy_json(
        df_mappings, rdf_graph, OUTPUT_CATEGORY_CHILDREN_JSON_PATH
    )

    print("\n=== Step 5: Link with Patent Causal Chains (Strict Inner Join) ===")
    df_linked_results = link_variables_to_causal_chain(
        df_mappings,
        category_children_dict,
        CAUSAL_LINKS_CSV_PATH,
        filter_mode=CAUSAL_LINK_FILTER,
        filter_set=EXPLICIT_CAUSAL_FILTER_SET
    )

    print(f"\nSaving strictly linked output CSV to: {OUTPUT_LINKED_RESULTS_PATH}")
    OUTPUT_LINKED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_linked_results.to_csv(OUTPUT_LINKED_RESULTS_PATH, index=False)

    print("\n=== Step 6: Export Bipartite Graph ===")
    export_bipartite_graph(df_linked_results, OUTPUT_GRAPHML_PATH, OUTPUT_GRAPH_JSON_PATH)

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()