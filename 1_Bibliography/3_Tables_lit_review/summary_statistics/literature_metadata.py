import json
import os
import re
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher

# 1. Load the literature review JSON dataset
json_file_path = "1_Bibliography/3_Tables_lit_review/metadata_literature.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
total_papers = len(df)

# 2. Strict String Similarity Matcher for Phrase Chains (80% threshold)
def group_similar_phrases(phrase_list, similarity_threshold=0.80):
    mapping = {}
    canonical_labels = {}
    
    for phrase in phrase_list:
        norm = phrase.strip().lower()
        if not norm:
            continue
        found_group = False
        
        for canon_norm in canonical_labels:
            if SequenceMatcher(None, norm, canon_norm).ratio() >= similarity_threshold:
                mapping[phrase] = canonical_labels[canon_norm]
                found_group = True
                break
                
        if not found_group:
            canonical_labels[norm] = phrase
            mapping[phrase] = phrase
            
    return mapping

# 3. Lemmatizer & Stopword Definitions
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'an', 'and', 'any', 'are', 
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'his', 'how', 'i', 'if', 'in', 'into', 'is',
    'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'also', 'etc', 'study',
    'paper', 'article', 'author', 'authors', 'future', 'work', 'section', 'present', 'plan', 'intend',
    'aim', 'focus', 'will', 'explicitly', 'mentioned'
}

def lemmatize_word(word):
    word = word.lower()
    irregulars = {'analyses': 'analysis', 'matrices': 'matrix', 'data': 'data'}
    if word in irregulars:
        return irregulars[word]
    if word.endswith('abilities') or word.endswith('ibility'):
        return word[:-7] + 'able'
    if word.endswith('ties'):
        return word[:-3] + 'ty'
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('ical'):
        return word[:-2]
    if word.endswith('ing'):
        base = word[:-3]
        return base[:-1] if len(base) > 3 and base[-1] == base[-2] else base
    if word.endswith('ed'):
        base = word[:-2]
        return base[:-1] + 'y' if base.endswith('i') else base
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word

# 4. Extract Multi-Word Phrase Chains (Bigrams and Trigrams)
def extract_phrase_chains(text):
    if not isinstance(text, str) or "not explicitly mentioned" in text.lower():
        return []
    
    # Split text into logical clauses
    clauses = re.split(r'[.;!?]+', text)
    chains = []
    
    for clause in clauses:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', clause)
        lemmas = []
        for t in tokens:
            low = t.lower()
            if low not in STOPWORDS:
                lem = lemmatize_word(low)
                if lem not in STOPWORDS and len(lem) >= 3:
                    lemmas.append(lem)
                    
        # Construct multi-word phrase chains (2-word and 3-word combinations)
        if len(lemmas) >= 2:
            for i in range(len(lemmas) - 1):
                chains.append(f"{lemmas[i]} {lemmas[i+1]}")
        if len(lemmas) >= 3:
            for i in range(len(lemmas) - 2):
                chains.append(f"{lemmas[i]} {lemmas[i+1]} {lemmas[i+2]}")
                
    return list(set(chains))

# 5. Extract phrase chains per paper
df['FW_Phrase_Chains'] = df['Future Work or Improvements'].apply(extract_phrase_chains)

# Collect all phrase chains across articles
all_chains = [chain for chains in df['FW_Phrase_Chains'] for chain in chains]

# Group similar phrase chains using strict 80% threshold
phrase_mapping = group_similar_phrases(all_chains, similarity_threshold=0.70)

# Calculate article count per phrase chain group (deduplicated per article)
grouped_paper_counts = {canon_phrase: 0 for canon_phrase in set(phrase_mapping.values())}

for chains in df['FW_Phrase_Chains']:
    article_phrases = {phrase_mapping[c] for c in chains if c in phrase_mapping}
    for canon_phrase in article_phrases:
        grouped_paper_counts[canon_phrase] += 1

# Build the aggregated phrase chain dataframe
df_future_work_aggregated = pd.DataFrame([
    {
        "Distilled Future Work Concept Chain": concept,
        "Paper Count": count,
        "Percentage of Total Articles (%)": f"{(count / total_papers) * 100:.2f}%"
    }
    for concept, count in grouped_paper_counts.items()
]).sort_values(by="Paper Count", ascending=False).reset_index(drop=True)

# 6. Export outputs
output_dir = "./1_Bibliography/3_Tables_lit_review/summary_statistics/outputliterature"
os.makedirs(output_dir, exist_ok=True)

df_future_work_aggregated.to_csv(os.path.join(output_dir, "summary_future_work_aggregated.csv"), index=False)

print("Aggregated future work phrase chains exported successfully.")