import sys
from pathlib import Path
import json
import pandas as pd

# 1. Project Root & Paths Setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]  # Adjust based on your script nesting
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.paths import CAUSAL_RELATION_MODEL_PATH, KNOWLEDGE_ABSTRACT_PATH

# Define file paths
output_dir = CAUSAL_RELATION_MODEL_PATH / "output"
output_dir.mkdir(parents=True, exist_ok=True)

extreme_csv_path = output_dir / "hexagon_to_ontology_triggers.csv" 
causal_links_csv_path = KNOWLEDGE_ABSTRACT_PATH / "output" / "causal_graph" / "patent_causal_chains_pos_summary.csv" 
variable_mapping_json_path = output_dir / "category_mapped_children.json" 

def run_spatial_causal_bridge():
    print("=== Phase 3: Spatial-Causal Graph Bridge Execution (Traceable Row-Level) ===")
    
    # 2. Load Inputs
    print(f"Loading extreme hexagon triggers from: {extreme_csv_path}")
    extreme_df = pd.read_csv(extreme_csv_path)
    
    print(f"Loading patent causal links from: {causal_links_csv_path}")
    causal_df = pd.read_csv(causal_links_csv_path)
    
    print(f"Loading ontology mapping from: {variable_mapping_json_path}")
    with open(variable_mapping_json_path, 'r', encoding='utf-8') as f:
        ontology_mapping = json.load(f)

    # 3. Filter Hexagons: Must contain at least 2 extreme variables
    print("\nFiltering H3 hexagons for multi-hazard criteria (>= 2 extreme variables)...")
    hex_counts = extreme_df.groupby('index')['attribute_variable'].nunique().reset_index()
    valid_hex_indexes = hex_counts[hex_counts['attribute_variable'] >= 2]['index']
    
    filtered_extreme_df = extreme_df[extreme_df['index'].isin(valid_hex_indexes)]
    print(f"Retained {len(valid_hex_indexes)} multi-hazard hexagons out of {extreme_df['index'].nunique()} total flagged cells.")

    # 4. Build Lookups: Spatial Variable -> Category & Category -> Child Terms
    var_to_category = {}
    category_to_child_terms = {}

    for cat_key, cat_data in ontology_mapping.items():
        category_name = cat_data.get('category_name')
        
        # Extract child terms for matching against patent attributes
        child_terms = set(
            child['child_term'].lower().strip() 
            for child in cat_data.get('available_child_terms', [])
        )
        category_to_child_terms[category_name] = child_terms
        
        # Map each spatial variable listed in this category back to the category name
        for var in cat_data.get('mapped_spatial_variables', []):
            clean_var = var.replace('_log1p_pctl', '').replace('_yeojohnson', '')
            var_to_category[clean_var] = category_name

    # 5. Connect Hexagons to Patent Causal Links at the Variable Level
    print("\nBridging individual extreme spatial variables to patent causal chains...")
    
    final_results = []
    
    # Iterate row-by-row through every extreme variable flag inside valid multi-hazard cells
    for _, row in filtered_extreme_df.iterrows():
        h3_idx = row['index']
        raw_attr_var = row['attribute_variable']
        ontology_term = row['ontology_term'].replace('_log1p_pctl', '').replace('_yeojohnson', '')
        
        # Determine which category this specific variable belongs to
        category = var_to_category.get(ontology_term)
        if not category:
            continue
            
        # Get allowed patent attribute child terms for this category
        allowed_patent_attributes = category_to_child_terms.get(category, set())
        
        if not allowed_patent_attributes or 'attribute' not in causal_df.columns:
            continue
            
        # Filter patent causal links matching this category's child terms
        matched_chains = causal_df[
            causal_df['attribute'].str.lower().str.strip().isin(allowed_patent_attributes)
        ]
        
        # Append each matched link alongside the specific spatial variable that triggered it
        for _, causal_row in matched_chains.iterrows():
            final_results.append({
                'index': h3_idx,
                'extreme_spatial_variable': raw_attr_var,
                'variable_name': ontology_term,  # Updated column name here
                'ontology_category': category,
                'technical_term': causal_row.get('technical', ''),
                'attribute_term': causal_row.get('attribute', ''),
                'causal_term_triggers': causal_row.get('triggers', ''),
                'failure_term': causal_row.get('failure', ''),
                'patent_ids': causal_row.get('patent_ids', '')
            })

    # 6. Save Final Traceable Thesis Output Table
    final_output_df = pd.DataFrame(final_results)
    output_file_path = output_dir / "h3_geospatial_patent_failure_chains_traceable.csv"
    final_output_df.to_csv(output_file_path, index=False)
    
    print(f"\nPipeline complete! Successfully generated {len(final_output_df)} traceable rows.")
    print(f"Saved final results to: {output_file_path}")
    print("\n--- Sample Output Preview ---")
    print(final_output_df.head(5).to_markdown(index=False))

if __name__ == "__main__":
    run_spatial_causal_bridge()