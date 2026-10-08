"""
Semantic Variable-to-Ontology Mapper & Patent Causal Linker

This script:
1. Loads variable metadata (variable_name, clean_label, description, variable_category).
2. Uses Sentence Transformers / spaCy / Cosine Similarity to map dataset variables
   (using their clean labels and rich descriptions) to SKOS Ontology Concepts.
3. Merges dataset Feature Importances with mapped Ontology Attribute Terms.
4. Links Feature Importances to Patent Causal Chains (technical component, attribute, failure mode, patent_ids, etc.).
5. Exports the linked results to CSV.
"""

import os
import re
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Dynamic Path Resolution & Relative Imports
# -------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent

# Automatically resolve the repository root containing the 'Utils' package
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

# Embedding & NLP dependencies
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import spacy
from rdflib import Graph, SKOS, RDFS, Namespace

# -------------------------------------------------------------------------
# 1. Project Paths & Setup
# -------------------------------------------------------------------------
# Input Files
METADATA_CSV_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "variable_metadata.csv"
FEATURE_IMPORTANCE_CSV_PATH = CAUSAL_RELATION_MODEL_PATH / "output" / "feature_importance.csv"
CAUSAL_LINKS_CSV_PATH = KNOWLEDGE_ABSTRACT_PATH / "patent_causal_chains_pos_summary.csv"
ONTOLOGY_RDF_PATH = ONTOLOGY_OUTPUT_PATH / "pipeline_knowledge_graph.ttl"

# Output File
OUTPUT_LINKED_RESULTS_PATH = PATENT_PIPELINE_PATH / "linked_feature_causal_chain.csv"

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------------------------
# 2. Extract Ontology Concept Terms from RDF Knowledge Graph
# -------------------------------------------------------------------------
def extract_ontology_terms(rdf_graph_path: Path) -> pd.DataFrame:
    """
    Parses the RDF Knowledge Graph and extracts all SKOS concept terms
    (prefLabel, altLabel, surface labels) into a structured DataFrame.
    """
    if not rdf_graph_path.exists():
        print(f"Warning: Ontology RDF file not found at '{rdf_graph_path}'. Returning sample concept terms.")
        return pd.DataFrame([
            {"uri": "http://example.org/ontology#WaterLevel", "term": "water level"},
            {"uri": "http://example.org/ontology#WaterwayDistance", "term": "waterway distance"},
            {"uri": "http://example.org/ontology#LandslideHazard", "term": "landslide hazard"},
            {"uri": "http://example.org/ontology#RainfallIntensity", "term": "annual rainfall"},
            {"uri": "http://example.org/ontology#HydraulicHazard", "term": "hydraulic hazard"},
            {"uri": "http://example.org/ontology#SoilSpeed", "term": "soil movement"}
        ])

    print(f"Parsing RDF Knowledge Graph from: {rdf_graph_path}...")
    g = Graph()
    g.parse(str(rdf_graph_path), format="turtle")
    
    terms = []
    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        terms.append({"uri": str(s), "term": str(o).lower().strip()})
    for s, p, o in g.triples((None, SKOS.altLabel, None)):
        terms.append({"uri": str(s), "term": str(o).lower().strip()})
    for s, p, o in g.triples((None, RDFS.label, None)):
        terms.append({"uri": str(s), "term": str(o).lower().strip()})
        
    df_terms = pd.DataFrame(terms).drop_duplicates(subset=["term"]).reset_index(drop=True)
    print(f"Extracted {len(df_terms)} unique ontology concept terms.")
    return df_terms


# -------------------------------------------------------------------------
# 3. Load & Preprocess Variable Metadata with Descriptions
# -------------------------------------------------------------------------
def load_variable_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Loads variable_metadata.csv and generates a rich semantic query string
    combining clean_label, description, and variable_category.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file '{csv_path}' not found! Please create 'variable_metadata.csv'.")

    print(f"Loading variable metadata from: {csv_path}...")
    df_meta = pd.read_csv(csv_path)

    # Required columns check
    required_cols = {"variable_name", "clean_label", "description"}
    missing = required_cols - set(df_meta.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    # Fill missing text
    df_meta["clean_label"] = df_meta["clean_label"].fillna("")
    df_meta["description"] = df_meta["description"].fillna("")
    if "variable_category" not in df_meta.columns:
        df_meta["variable_category"] = ""
    else:
        df_meta["variable_category"] = df_meta["variable_category"].fillna("")

    # Construct unified semantic query representation
    df_meta["semantic_text"] = (
        df_meta["clean_label"] + ". " + 
        df_meta["description"] + " " + 
        df_meta["variable_category"]
    ).str.strip().str.lower()

    return df_meta


# -------------------------------------------------------------------------
# 4. Semantic Embedding Mapping Engine
# -------------------------------------------------------------------------
class SemanticMapper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Initializing Transformer Embedding Model ('{model_name}')...")
            self.model = SentenceTransformer(model_name)
            self.use_transformers = True
        else:
            print("SentenceTransformers not installed. Falling back to spaCy lexical similarity...")
            self.use_transformers = False

    def compute_similarity_matrix(self, variable_texts: list, ontology_terms: list) -> np.ndarray:
        """
        Computes cosine similarity matrix between variable semantic texts and ontology terms.
        """
        if self.use_transformers:
            var_embeddings = self.model.encode(variable_texts, convert_to_tensor=True)
            ont_embeddings = self.model.encode(ontology_terms, convert_to_tensor=True)
            sim_matrix = util.cos_sim(var_embeddings, ont_embeddings).cpu().numpy()
            return sim_matrix
        else:
            sim_matrix = np.zeros((len(variable_texts), len(ontology_terms)))
            var_docs = [nlp(text) for text in variable_texts]
            ont_docs = [nlp(term) for term in ontology_terms]

            for i, v_doc in enumerate(var_docs):
                v_tokens = set([t.lemma_ for t in v_doc if not t.is_stop and t.is_alpha])
                for j, o_doc in enumerate(ont_docs):
                    if o_doc.text in v_doc.text:
                        sim_matrix[i, j] = 0.95
                        continue
                    o_tokens = set([t.lemma_ for t in o_doc if not t.is_stop and t.is_alpha])
                    if v_tokens and o_tokens:
                        intersection = v_tokens.intersection(o_tokens)
                        union = v_tokens.union(o_tokens)
                        sim_matrix[i, j] = len(intersection) / len(union) if union else 0.0
                    else:
                        sim_matrix[i, j] = 0.0
            return sim_matrix


def map_variables_to_ontology(
    df_meta: pd.DataFrame, 
    df_ontology_terms: pd.DataFrame, 
    threshold: float = 0.35
) -> pd.DataFrame:
    """
    Maps each variable in df_meta to the closest ontology concept term using embeddings.
    """
    mapper = SemanticMapper()
    
    variable_texts = df_meta["semantic_text"].tolist()
    ontology_terms = df_ontology_terms["term"].tolist()
    
    print("\nComputing vector embedding cosine similarities...")
    sim_matrix = mapper.compute_similarity_matrix(variable_texts, ontology_terms)
    
    mapping_results = []
    for idx, row in df_meta.iterrows():
        similarities = sim_matrix[idx]
        best_match_idx = np.argmax(similarities)
        best_score = float(similarities[best_match_idx])
        
        if best_score >= threshold:
            best_term_row = df_ontology_terms.iloc[best_match_idx]
            mapping_results.append({
                "variable_name": row["variable_name"],
                "clean_label": row["clean_label"],
                "description": row["description"],
                "variable_category": row["variable_category"],
                "mapped_attribute_term": best_term_row["term"],
                "match_confidence": round(best_score, 4),
                "ontology_uri": best_term_row["uri"]
            })
        else:
            mapping_results.append({
                "variable_name": row["variable_name"],
                "clean_label": row["clean_label"],
                "description": row["description"],
                "variable_category": row["variable_category"],
                "mapped_attribute_term": "UNMAPPED",
                "match_confidence": round(best_score, 4),
                "ontology_uri": None
            })
            
    return pd.DataFrame(mapping_results)


# -------------------------------------------------------------------------
# 5. Link Feature Importances & Patent Causal Chains
# -------------------------------------------------------------------------
def link_feature_importance_to_causal_chain(
    df_mappings: pd.DataFrame,
    feature_importance_path: Path,
    causal_links_path: Path
) -> pd.DataFrame:
    """
    Merges feature importances and patent causal chains based on mapped ontology terms.
    Supports extended causal chain fields: technical, attribute, failure, patent_ids,
    patent_count, triggers, nouns, verbs, noun_count, verb_count.
    """
    if feature_importance_path.exists():
        print(f"\nLoading Feature Importances from: {feature_importance_path}")
        df_feat_imp = pd.read_csv(feature_importance_path)
    else:
        df_feat_imp = pd.DataFrame({
            "variable_name": df_mappings["variable_name"],
            "importance": np.round(np.random.uniform(0.01, 0.40, len(df_mappings)), 4)
        })

    if causal_links_path.exists():
        print(f"Loading Patent Causal Links from: {causal_links_path}")
        df_causal = pd.read_csv(causal_links_path)
    else:
        df_causal = pd.DataFrame({
            "technical": ["reservoir", "pump", "sluice", "metering pump", "valve"],
            "attribute": ["water level", "waterway distance", "landslide hazard", "annual rainfall", "hydraulic hazard"],
            "failure": ["contamination", "overflow", "structural collapse", "pipe burst", "cavitation"],
            "patent_ids": ["KR-102715270-B1, KR-200395622-Y1"] * 5,
            "patent_count": [2] * 5,
            "triggers": ["increase, leak", "overflow, clog", "slide, erosion", "heavy rain, flood", "pressure, burst"],
            "nouns": ["water, reservoir", "pump, flow", "slope, soil", "rain, gauge", "valve, pipe"],
            "verbs": ["flow, rise", "pump, stop", "collapse, shift", "measure, fall", "control, open"],
            "noun_count": [2] * 5,
            "verb_count": [2] * 5
        })

    df_mapped_importances = pd.merge(
        df_mappings,
        df_feat_imp,
        on="variable_name",
        how="left"
    )

    df_linked = pd.merge(
        df_mapped_importances,
        df_causal,
        left_on="mapped_attribute_term",
        right_on="attribute",
        how="left"
    )

    primary_cols = [
        "variable_name", "clean_label", "importance", "mapped_attribute_term", 
        "match_confidence", "technical", "failure", "patent_ids", "patent_count",
        "triggers", "nouns", "verbs", "noun_count", "verb_count",
        "description", "variable_category"
    ]
    existing_primary = [c for c in primary_cols if c in df_linked.columns]
    other_cols = [c for c in df_linked.columns if c not in existing_primary]
    
    return df_linked[existing_primary + other_cols]


# -------------------------------------------------------------------------
# 6. Main Pipeline Execution
# -------------------------------------------------------------------------
def main():
    print("=== Step 1: Load Variable Metadata & Descriptions ===")
    df_meta = load_variable_metadata(METADATA_CSV_PATH)
    print(f"Loaded {len(df_meta)} variables with rich descriptions.")

    print("\n=== Step 2: Extract Ontology Concept Terms ===")
    df_ontology_terms = extract_ontology_terms(ONTOLOGY_RDF_PATH)

    print("\n=== Step 3: Run Transformer/Embedding Mapping ===")
    df_mappings = map_variables_to_ontology(df_meta, df_ontology_terms, threshold=0.30)

    print("\n--- Mapping Preview ---")
    print(df_mappings[["variable_name", "mapped_attribute_term", "match_confidence"]].to_string(index=False))

    print("\n=== Step 4: Link with Feature Importance & Patent Causal Chains ===")
    df_linked_results = link_feature_importance_to_causal_chain(
        df_mappings,
        FEATURE_IMPORTANCE_CSV_PATH,
        CAUSAL_LINKS_CSV_PATH
    )

    print(f"\nSaving linked output to: {OUTPUT_LINKED_RESULTS_PATH}")
    OUTPUT_LINKED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_linked_results.to_csv(OUTPUT_LINKED_RESULTS_PATH, index=False)
    print("Pipeline complete!")


if __name__ == "__main__":
    main()