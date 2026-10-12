import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# 1. Project Root & Paths Setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]  # Adjust based on your script nesting
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import CAUSAL_RELATION_MODEL_PATH, KNOWLEDGE_ABSTRACT_PATH

# Define file paths
output_dir = CAUSAL_RELATION_MODEL_PATH / "output"
output_dir.mkdir(parents=True, exist_ok=True)

extreme_csv_path = output_dir / "1_importance_filter" / "hexagon_to_ontology_triggers.csv" 
causal_links_csv_path = KNOWLEDGE_ABSTRACT_PATH / "output" / "causal_graph" / "patent_causal_chains_pos_summary.csv" 
variable_mapping_json_path = output_dir / "2_semantic_mappings" / "category_mapped_children.json" 

# Path to your dataset variable descriptions file
variable_descriptions_csv_path = output_dir / "descriptions.csv" 

def run_spatial_causal_bridge():
    print("=== Phase 3: Spatial-Causal Graph Bridge Execution (Highest-Similarity Semantic Pairing) ===")
    
    # 2. Load Inputs
    print(f"Loading extreme hexagon triggers from: {extreme_csv_path}")
    extreme_df = pd.read_csv(extreme_csv_path)
    
    print(f"Loading patent causal links from: {causal_links_csv_path}")
    causal_df = pd.read_csv(causal_links_csv_path)
    
    print(f"Loading ontology mapping from: {variable_mapping_json_path}")
    with open(variable_mapping_json_path, 'r', encoding='utf-8') as f:
        ontology_mapping = json.load(f)

    # 3. Load Dataset Variable Descriptions with Automatic Header Mapping
    if variable_descriptions_csv_path.exists():
        print(f"Loading dataset variable descriptions from: {variable_descriptions_csv_path}")
        var_desc_df = pd.read_csv(variable_descriptions_csv_path)
        
        # Standardize column headers if using 'Variable' / 'Description'
        column_rename_map = {}
        if 'Variable' in var_desc_df.columns:
            column_rename_map['Variable'] = 'variable_name'
        if 'Description' in var_desc_df.columns:
            column_rename_map['Description'] = 'description'
            
        if column_rename_map:
            var_desc_df = var_desc_df.rename(columns=column_rename_map)
            
        dataset_vars = var_desc_df['variable_name'].astype(str).tolist()
        dataset_texts = var_desc_df['description'].fillna(var_desc_df['variable_name']).astype(str).tolist()
    else:
        print("[Notice] descriptions.csv not found. Auto-extracting unique variables from extreme triggers data...")
        cleaned_terms = extreme_df['ontology_term'].astype(str).str.replace('_log1p_pctl', '', regex=False).str.replace('_yeojohnson', '', regex=False)
        dataset_vars = cleaned_terms.unique().tolist()
        dataset_texts = dataset_vars

    # 4. Transformer-Based Highest-Similarity Mapping: Patent Attribute -> Specific Dataset Variable
    print("\nComputing highest-similarity semantic pairings between patent attributes and dataset variables...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get unique patent attributes from causal links
    unique_patent_attrs = causal_df['attribute'].dropna().unique().tolist()
    
    # Encode embeddings
    attr_embeddings = model.encode(unique_patent_attrs, convert_to_tensor=True)
    var_embeddings = model.encode(dataset_texts, convert_to_tensor=True)
    
    # Compute cosine similarity matrix (Patent Attributes x Dataset Variables)
    cos_sim = util.cos_sim(attr_embeddings, var_embeddings)
    
    # Assign each patent attribute to its single highest-similarity dataset variable
    patent_attr_to_best_var = {}
    for i, attr in enumerate(unique_patent_attrs):
        best_match_idx = torch.argmax(cos_sim[i]).item()
        patent_attr_to_best_var[attr] = dataset_vars[best_match_idx]

    print(f"Successfully generated highest-similarity mappings for {len(patent_attr_to_best_var)} unique patent attributes.")

    # 5. Filter Hexagons: Must contain at least 1 extreme variable
    print("\nFiltering H3 hexagons for multi-hazard criteria (>= 1 extreme variables)...")
    hex_counts = extreme_df.groupby('index')['attribute_variable'].nunique().reset_index()
    valid_hex_indexes = hex_counts[hex_counts['attribute_variable'] >= 1]['index']
    
    filtered_extreme_df = extreme_df[extreme_df['index'].isin(valid_hex_indexes)]
    print(f"Retained {len(valid_hex_indexes)} multi-hazard hexagons out of {extreme_df['index'].nunique()} total flagged cells.")

    # 6. Build Lookups: Spatial Variable -> Category
    var_to_category = {}

    for cat_key, cat_data in ontology_mapping.items():
        category_name = cat_data.get('category_name')
        for var in cat_data.get('mapped_spatial_variables', []):
            clean_var = var.replace('_log1p_pctl', '').replace('_yeojohnson', '')
            var_to_category[clean_var] = category_name

    # 7. Connect Hexagons to Patent Causal Links via Highest-Similarity Assignment
    print("\nBridging individual extreme spatial variables to patent causal chains using highest-similarity mapping...")
    
    final_results = []
    
    # Pre-map causal df attributes to their best assigned variable
    causal_df['assigned_variable'] = causal_df['attribute'].map(patent_attr_to_best_var)
    
    for _, row in filtered_extreme_df.iterrows():
        h3_idx = row['index']
        raw_attr_var = row['attribute_variable']
        ontology_term = row['ontology_term'].replace('_log1p_pctl', '').replace('_yeojohnson', '')
        
        category = var_to_category.get(ontology_term, '')
        
        if 'attribute' not in causal_df.columns:
            continue
            
        # Find causal links where the assigned variable matches this hexagon's ontology term
        matching_causal = causal_df[causal_df['assigned_variable'] == ontology_term]
        
        for _, causal_row in matching_causal.iterrows():
            pat_attr = causal_row.get('attribute', '')
            
            final_results.append({
                'index': h3_idx,
                'extreme_spatial_variable': raw_attr_var,
                'variable_name': ontology_term,
                'ontology_category': category,
                'technical_term': causal_row.get('technical', ''),
                'attribute_term': pat_attr,
                'causal_term_triggers': causal_row.get('triggers', ''),
                'failure_term': causal_row.get('failure', ''),
                'patent_ids': causal_row.get('patent_ids', '')
            })

    # 8. Save Final Traceable Thesis Output Table
    final_output_df = pd.DataFrame(final_results)
    output_file_path = output_dir / "3_h3_causal_bridge" / "h3_geospatial_patent_failure_chains_traceable.csv"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_df.to_csv(output_file_path, index=False)
    
    print(f"\nPipeline complete! Successfully generated {len(final_output_df)} highest-similarity paired traceable rows.")
    print(f"Saved final results to: {output_file_path}")
    print("\n--- Sample Output Preview ---")
    print(final_output_df.head(5).to_markdown(index=False))

if __name__ == "__main__":
    run_spatial_causal_bridge()