import os
import re
import time
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from Utils.paths import PATENT_PIPELINE_PATH, ONTOLOGY_OUTPUT_PATH, REPO_ROOT

import pandas as pd
from rdflib import Graph, RDF, SKOS, RDFS

def load_ontology(rdf_path):
    print(f'Loading ontology from {rdf_path}...')
    g = Graph()
    g.parse(str(rdf_path), format='xml')
    print(f'Loaded ontology with {len(g)} triples.')
    return g

def extract_ontology_data(g):
    term_to_uri = {}
    uri_to_hierarchy = {}
    term_to_category = {}

    concepts = list(g.subjects(RDF.type, SKOS.Concept))
    
    for concept in concepts:
        pref_labels = list(g.objects(concept, SKOS.prefLabel))
        pref_label = str(pref_labels[0]).lower() if pref_labels else None
        
        alt_labels = [str(obj).lower() for obj in g.objects(concept, SKOS.altLabel)]
        schemes = [str(obj) for obj in g.objects(concept, SKOS.inScheme)]
        
        category = 'other'
        for scheme_uri in schemes:
            if 'TechnicalComponentScheme' in scheme_uri:
                category = 'technical'
            elif 'FailureModeScheme' in scheme_uri:
                category = 'failures'
            elif 'EnvironmentalSocialRiskScheme' in scheme_uri:
                category = 'variable'
            elif 'CausalRelationalScheme' in scheme_uri:
                category = 'causal'
        
        broader_terms = []
        for b in g.objects(concept, SKOS.broader):
            b_labels = list(g.objects(b, SKOS.prefLabel))
            if b_labels:
                broader_terms.append(str(b_labels[0]).lower())
                
        narrower_terms = []
        for n in g.objects(concept, SKOS.narrower):
            n_labels = list(g.objects(n, SKOS.prefLabel))
            if n_labels:
                narrower_terms.append(str(n_labels[0]).lower())

        sub_classes = []
        for sc in g.objects(concept, RDFS.subClassOf):
            sc_labels = list(g.objects(sc, SKOS.prefLabel))
            if sc_labels:
                sub_classes.append(str(sc_labels[0]).lower())

        hierarchy_info = {
            'uri': str(concept),
            'category': category,
            'pref_label': pref_label,
            'alt_labels': alt_labels,
            'schemes': schemes,
            'broader': broader_terms,
            'narrower': narrower_terms,
            'sub_class_of': sub_classes
        }

        uri_to_hierarchy[str(concept)] = hierarchy_info

        if pref_label:
            term_to_uri[pref_label] = str(concept)
            term_to_category[pref_label] = category
        for alt in alt_labels:
            term_to_uri[alt] = str(concept)
            term_to_category[alt] = category

    print(f'Extracted {len(concepts)} concepts and {len(term_to_uri)} unique term strings.')
    return term_to_uri, uri_to_hierarchy, term_to_category

def process_dataset(
    csv_path, 
    rdf_path, 
    output_csv_path=None, 
    chunksize=100000,
    include_technical=True,
    include_failures=True,
    include_causal=True,
    include_variable=True
):
    g = load_ontology(rdf_path)
    term_to_uri, uri_to_hierarchy, term_to_category = extract_ontology_data(g)

    active_categories = []
    if include_technical: active_categories.append('technical')
    if include_failures: active_categories.append('failures')
    if include_causal: active_categories.append('causal')
    if include_variable: active_categories.append('variable')

    print(f'Active matching categories (topics): {active_categories}')

    filtered_term_to_uri = {
        term: uri for term, uri in term_to_uri.items() 
        if term_to_category.get(term) in active_categories
    }

    all_terms = sorted(list(filtered_term_to_uri.keys()), key=lambda x: len(x.split()), reverse=True)
    
    print(f'Compiling regex pattern for {len(all_terms)} active ontology terms...')
    escaped_terms = [re.escape(t) for t in all_terms if len(t) > 1]
    if escaped_terms:
        pattern_str = r'\b(' + '|'.join(escaped_terms) + r')\b'
        compiled_regex = re.compile(pattern_str)
    else:
        compiled_regex = None
    print('Regex compiled successfully.')

    print(f'Processing patent dataset from {csv_path} in chunks of {chunksize}...')
    
    filtered_chunks = []
    total_rows = 0
    total_matched = 0
    start_time = time.time()

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        total_rows += len(chunk)
        texts = chunk['text'].fillna('').str.lower()
        
        ocurrence_words_col = []
        hierarchy_col = []
        matched_mask = []
        
        for text in texts:
            matches = list(set(compiled_regex.findall(text))) if compiled_regex else []
            if matches:
                matched_mask.append(True)
                ocurrence_words_col.append(matches)
                term_hierarchy_map = {term: uri_to_hierarchy[filtered_term_to_uri[term]] for term in matches if term in filtered_term_to_uri}
                hierarchy_col.append(term_hierarchy_map)
            else:
                matched_mask.append(False)
                ocurrence_words_col.append([])
                hierarchy_col.append({})
                
        filtered_chunk = chunk[matched_mask].copy()
        if len(filtered_chunk) > 0:
            filtered_chunk['ocurrence_words'] = [ocurrence_words_col[i] for i, m in enumerate(matched_mask) if m]
            filtered_chunk['ontology_hierarchy'] = [hierarchy_col[i] for i, m in enumerate(matched_mask) if m]
            filtered_chunks.append(filtered_chunk)
            total_matched += len(filtered_chunk)
            
        print(f'Processed chunk {chunk_idx + 1} (Total rows scanned: {total_rows}, Matched so far: {total_matched})')

    final_df = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame(columns=list(pd.read_csv(csv_path, nrows=1).columns) + ['ocurrence_words', 'ontology_hierarchy'])

    print(f'Processing complete in {time.time() - start_time:.2f} seconds.')
    print(f'Total rows: {total_rows}, Remaining after filtering: {len(final_df)}')

    if output_csv_path:
        print(f'Saving filtered dataset to {output_csv_path}...')
        final_df.to_csv(output_csv_path, index=False)
        print('Saved successfully.')

    return final_df

if __name__ == '__main__':
    csv_file = PATENT_PIPELINE_PATH / 'data_creation/output/patent_dataset.csv'
    rdf_file = ONTOLOGY_OUTPUT_PATH / 'pipeline_knowledge_graph.rdf'
    output_file = PATENT_PIPELINE_PATH / 'data_creation/output/patent_dataset_filtered_ontology.csv'
    
    df_filtered = process_dataset(
        csv_file, 
        rdf_file, 
        output_file,
        include_technical=True,
        include_failures=True,
        include_causal=False,
        include_variable=True
    )
    print(df_filtered[['pat_id', 'ocurrence_words']].head(10))
