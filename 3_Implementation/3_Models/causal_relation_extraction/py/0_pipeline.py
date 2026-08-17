import re
import json
import pandas as pd
import networkx as nx

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import spacy


# ============================================================
# CONFIGURATION
# ============================================================

PATENT_CSV = "patents.csv"

FAILURE_TERMS = "failure_terms.txt"
CAUSAL_TERMS = "causal_terms.txt"
VARIABLE_TERMS = "variable_terms.txt"
TECHNICAL_TERMS = "technical_terms.txt"

OUTPUT_RETRIEVAL = "retrieved_patents.csv"
OUTPUT_CONTEXT = "causal_contexts.csv"
OUTPUT_RELATIONS = "causal_relations.csv"
OUTPUT_GRAPHS = "causal_graphs.json"

TEXT_COLUMN = "full_text"
PATENT_ID_COLUMN = "patent_id"

# Number of sentences around a relevant sentence
CONTEXT_WINDOW = 2

# Maximum sentence distance allowed between entities
# and causal cue
MAX_CAUSAL_DISTANCE = 2

nlp = spacy.load("en_core_web_sm")


# ============================================================
# 1. LOAD TERM LISTS
# ============================================================

def load_terms(path):

    text = Path(path).read_text(encoding="utf-8")

    text = text.replace("\n", ",")

    terms = []

    for term in text.split(","):

        term = term.strip().strip('"').strip("'")

        if term:
            terms.append(term)

    # Remove exact duplicates while preserving order
    seen = set()
    unique_terms = []

    for term in terms:

        key = term.lower()

        if key not in seen:

            seen.add(key)
            unique_terms.append(term)

    return unique_terms


failure_terms = load_terms(FAILURE_TERMS)
causal_terms = load_terms(CAUSAL_TERMS)
variable_terms = load_terms(VARIABLE_TERMS)
technical_terms = load_terms(TECHNICAL_TERMS)


print("Failure terms:", len(failure_terms))
print("Causal terms:", len(causal_terms))
print("Variable terms:", len(variable_terms))
print("Technical terms:", len(technical_terms))


# ============================================================
# 2. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove PDF line breaks
    text = text.replace("\n", " ")

    # Join words broken by PDF hyphenation
    text = re.sub(
        r"(\w)-\s+(\w)",
        r"\1\2",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 3. TERM MATCHING
# ============================================================

def compile_term_pattern(terms):

    escaped = [
        re.escape(term)
        for term in sorted(
            terms,
            key=len,
            reverse=True
        )
    ]

    if not escaped:
        return None

    return re.compile(
        r"(?<!\w)(?:" +
        "|".join(escaped) +
        r")(?!\w)",
        flags=re.IGNORECASE
    )


failure_pattern = compile_term_pattern(failure_terms)
causal_pattern = compile_term_pattern(causal_terms)
variable_pattern = compile_term_pattern(variable_terms)
technical_pattern = compile_term_pattern(technical_terms)


# ============================================================
# 4. TERM EXTRACTION
# ============================================================

def find_terms(text, pattern):

    if pattern is None:
        return []

    return [
        m.group(0)
        for m in pattern.finditer(text)
    ]


def extract_term_categories(text):

    return {

        "failure": find_terms(
            text,
            failure_pattern
        ),

        "causal": find_terms(
            text,
            causal_pattern
        ),

        "variable": find_terms(
            text,
            variable_pattern
        ),

        "technical": find_terms(
            text,
            technical_pattern
        )
    }


# ============================================================
# 5. PATENT RETRIEVAL
# ============================================================

def retrieve_patent(text):

    """
    IMPORTANT CHANGE:

    We no longer require a patent to contain all four
    categories.

    The goal is to retrieve patents that contain useful
    PARTIAL evidence.

    Examples:

        variable + causal
        variable + failure
        failure + causal
        failure + technical
        variable + technical
        variable + failure + technical
    """

    categories = extract_term_categories(text)

    counts = {
        category: len(
            set(
                x.lower()
                for x in values
            )
        )
        for category, values
        in categories.items()
    }

    has_failure = counts["failure"] > 0
    has_variable = counts["variable"] > 0
    has_technical = counts["technical"] > 0
    has_causal = counts["causal"] > 0

    score = 0

    # --------------------------------------------------------
    # Basic evidence
    # --------------------------------------------------------

    if has_variable:
        score += 1

    if has_failure:
        score += 1

    if has_technical:
        score += 1

    if has_causal:
        score += 1

    # --------------------------------------------------------
    # Partial relationships
    # --------------------------------------------------------

    if has_variable and has_causal:
        score += 3

    if has_variable and has_failure:
        score += 4

    if has_failure and has_causal:
        score += 3

    if has_failure and has_technical:
        score += 3

    if has_variable and has_technical:
        score += 2

    # --------------------------------------------------------
    # Stronger evidence
    # --------------------------------------------------------

    if (
        has_variable
        and has_failure
        and has_causal
    ):
        score += 4

    if (
        has_failure
        and has_causal
        and has_technical
    ):
        score += 4

    if (
        has_variable
        and has_failure
        and has_technical
    ):
        score += 4

    # Full combination is useful,
    # but NOT required.
    if (
        has_variable
        and has_failure
        and has_causal
        and has_technical
    ):
        score += 5

    return score, categories


# ============================================================
# 6. SENTENCE SPLITTING
# ============================================================

def get_sentences(text):

    doc = nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]


# ============================================================
# 7. SENTENCE CATEGORY PROFILE
# ============================================================

def sentence_categories(sentence):

    return extract_term_categories(sentence)


def sentence_has_relevant_terms(sentence):

    cats = sentence_categories(sentence)

    return any(
        len(cats[category]) > 0
        for category in [
            "failure",
            "causal",
            "variable",
            "technical"
        ]
    )


# ============================================================
# 8. EXTRACT RELEVANT SENTENCES
# ============================================================

def extract_contexts(
    text,
    window=CONTEXT_WINDOW
):

    sentences = get_sentences(text)

    relevant_indices = []

    for i, sentence in enumerate(sentences):

        if sentence_has_relevant_terms(sentence):

            relevant_indices.append(i)

    # --------------------------------------------------------
    # Merge overlapping windows
    # --------------------------------------------------------

    selected = set()

    for i in relevant_indices:

        start = max(
            0,
            i - window
        )

        end = min(
            len(sentences),
            i + window + 1
        )

        for j in range(start, end):

            selected.add(j)

    # --------------------------------------------------------
    # Create contexts
    # --------------------------------------------------------

    contexts = []

    for i in sorted(selected):

        start = max(
            0,
            i - window
        )

        end = min(
            len(sentences),
            i + window + 1
        )

        context = " ".join(
            sentences[start:end]
        )

        contexts.append({

            "sentence_index": i,

            "trigger_sentence":
                sentences[i],

            "context":
                context,

            "sentence_start":
                start,

            "sentence_end":
                end - 1
        })

    return contexts


# ============================================================
# 9. ENTITY EXTRACTION
# ============================================================

def extract_entities(sentence):

    entities = []

    patterns = [

        (
            "failure",
            failure_pattern
        ),

        (
            "variable",
            variable_pattern
        ),

        (
            "technical",
            technical_pattern
        )
    ]

    for category, pattern in patterns:

        if pattern is None:
            continue

        for match in pattern.finditer(sentence):

            entities.append({

                "text":
                    match.group(0),

                "category":
                    category,

                "start":
                    match.start(),

                "end":
                    match.end()
            })

    # --------------------------------------------------------
    # Remove overlapping entities
    # --------------------------------------------------------

    entities.sort(
        key=lambda x: (
            x["start"],
            -(x["end"] - x["start"])
        )
    )

    selected = []

    for entity in entities:

        overlap = False

        for existing in selected:

            if (
                entity["start"]
                < existing["end"]
                and
                entity["end"]
                > existing["start"]
            ):

                overlap = True
                break

        if not overlap:

            selected.append(entity)

    return selected


# ============================================================
# 10. CAUSAL CUE EXTRACTION
# ============================================================

def extract_causal_cues(sentence):

    if causal_pattern is None:
        return []

    return [

        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end()
        }

        for match
        in causal_pattern.finditer(sentence)
    ]


# ============================================================
# 11. CAUSAL DIRECTION
# ============================================================

def infer_direction(
    left_entity,
    right_entity,
    causal_term
):

    cue = causal_term.lower().strip()

    # --------------------------------------------------------
    # Explicit reverse-causality constructions
    # --------------------------------------------------------

    reverse_patterns = [

        "due to",
        "because of",
        "resulting from",
        "caused by",
        "originated as",
        "derived from",
        "introduced by"
    ]

    for pattern in reverse_patterns:

        if pattern in cue:

            return (
                right_entity,
                left_entity
            )

    # --------------------------------------------------------
    # Normal forward construction
    #
    # rainfall causes flooding
    # sediment causes blockage
    #
    # left -> right
    # --------------------------------------------------------

    return (
        left_entity,
        right_entity
    )


# ============================================================
# 12. ENTITY PAIR GENERATION
# ============================================================

def generate_entity_pairs(
    entities,
    cue_start,
    cue_end
):

    before = [
        e for e in entities
        if e["end"] <= cue_start
    ]

    after = [
        e for e in entities
        if e["start"] >= cue_end
    ]

    if not before or not after:

        return []

    # --------------------------------------------------------
    # Rather than only taking the closest entities,
    # consider several nearby candidates.
    # --------------------------------------------------------

    before = before[-3:]
    after = after[:3]

    pairs = []

    for left in before:

        for right in after:

            distance = (
                abs(
                    left["start"]
                    -
                    right["start"]
                )
            )

            pairs.append(
                (
                    distance,
                    left,
                    right
                )
            )

    pairs.sort(
        key=lambda x: x[0]
    )

    return pairs[:5]


# ============================================================
# 13. CAUSAL RELATION EXTRACTION
# ============================================================

def extract_causal_relations(
    sentence,
    patent_id=None,
    sentence_index=None
):

    entities = extract_entities(
        sentence
    )

    cues = extract_causal_cues(
        sentence
    )

    relations = []

    if not entities or not cues:

        return relations

    for cue in cues:

        pairs = generate_entity_pairs(
            entities,
            cue["start"],
            cue["end"]
        )

        for _, left, right in pairs:

            source, target = infer_direction(
                left,
                right,
                cue["text"]
            )

            # ------------------------------------------------
            # Relationship type
            # ------------------------------------------------

            relation_type = (
                source["category"]
                + "_to_"
                + target["category"]
            )

            relations.append({

                "patent_id":
                    patent_id,

                "sentence_index":
                    sentence_index,

                "source":
                    source["text"],

                "source_type":
                    source["category"],

                "causal_term":
                    cue["text"],

                "target":
                    target["text"],

                "target_type":
                    target["category"],

                "relation_type":
                    relation_type,

                "sentence":
                    sentence
            })

    return relations


# ============================================================
# 14. CROSS-SENTENCE CAUSAL EXTRACTION
# ============================================================

def extract_cross_sentence_relations(
    context,
    patent_id,
    sentence_start
):

    """
    Important addition.

    Example:

        Heavy rainfall occurred in the region.
        This resulted in increased runoff.
        The runoff caused sediment accumulation.

    No single sentence necessarily contains all entities.

    We therefore inspect the entire context.
    """

    sentences = get_sentences(
        context
    )

    relations = []

    for i, sentence in enumerate(sentences):

        # ----------------------------------------------------
        # Direct relation
        # ----------------------------------------------------

        direct = extract_causal_relations(
            sentence,
            patent_id,
            sentence_start + i
        )

        relations.extend(direct)

        # ----------------------------------------------------
        # Cross-sentence relation
        #
        # Look at previous sentence + current sentence
        # ----------------------------------------------------

        if i == 0:
            continue

        previous = sentences[i - 1]

        previous_entities = extract_entities(
            previous
        )

        current_entities = extract_entities(
            sentence
        )

        current_cues = extract_causal_cues(
            sentence
        )

        if (
            not previous_entities
            or not current_entities
            or not current_cues
        ):
            continue

        for cue in current_cues:

            # Current sentence entities after cue
            after = [

                e for e in current_entities

                if e["start"] >= cue["end"]
            ]

            if not after:
                continue

            target = after[0]

            # Nearest entity from previous sentence
            source = previous_entities[-1]

            relations.append({

                "patent_id":
                    patent_id,

                "sentence_index":
                    sentence_start + i,

                "source":
                    source["text"],

                "source_type":
                    source["category"],

                "causal_term":
                    cue["text"],

                "target":
                    target["text"],

                "target_type":
                    target["category"],

                "relation_type":
                    source["category"]
                    + "_to_"
                    + target["category"],

                "sentence":
                    previous
                    + " "
                    + sentence
            })

    return relations


# ============================================================
# 15. NORMALIZE ENTITY NAMES
# ============================================================

def normalize_entity(text):

    """
    Used for graph merging.

    Keeps the original wording in the evidence,
    but creates a normalized graph node.
    """

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# 16. BUILD CROSS-PATENT GRAPH
# ============================================================

def build_corpus_graph(
    relation_df
):

    G = nx.DiGraph()

    for _, row in relation_df.iterrows():

        source = normalize_entity(
            row["source"]
        )

        target = normalize_entity(
            row["target"]
        )

        source_type = row[
            "source_type"
        ]

        target_type = row[
            "target_type"
        ]

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        if not G.has_node(source):

            G.add_node(
                source,
                category=source_type
            )

        if not G.has_node(target):

            G.add_node(
                target,
                category=target_type
            )

        # ----------------------------------------------------
        # Edge
        # ----------------------------------------------------

        if G.has_edge(
            source,
            target
        ):

            edge = G[
                source
            ][
                target
            ]

            edge["weight"] += 1

            edge["patents"].add(
                str(row["patent_id"])
            )

            edge["evidence"].append({

                "patent_id":
                    str(row["patent_id"]),

                "sentence_index":
                    int(row["sentence_index"]),

                "causal_term":
                    row["causal_term"],

                "sentence":
                    row["sentence"],

                "relation_type":
                    row["relation_type"]
            })

        else:

            G.add_edge(

                source,
                target,

                weight=1,

                patents={
                    str(row["patent_id"])
                },

                evidence=[{

                    "patent_id":
                        str(row["patent_id"]),

                    "sentence_index":
                        int(row["sentence_index"]),

                    "causal_term":
                        row["causal_term"],

                    "sentence":
                        row["sentence"],

                    "relation_type":
                        row["relation_type"]
                }]
            )

    return G


# ============================================================
# 17. SERIALIZE GRAPH
# ============================================================

def graph_to_json(G):

    data = nx.node_link_data(
        G
    )

    # Convert sets to lists
    for node in data["nodes"]:

        if isinstance(
            node.get("patents"),
            set
        ):

            node["patents"] = list(
                node["patents"]
            )

    for edge in data["links"]:

        if isinstance(
            edge.get("patents"),
            set
        ):

            edge["patents"] = list(
                edge["patents"]
            )

    return data


# ============================================================
# 18. PROCESS PATENTS
# ============================================================

df = pd.read_csv(
    PATENT_CSV
)

retrieval_rows = []
context_rows = []
relation_rows = []


for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processing patents"
):

    # --------------------------------------------------------
    # Patent ID
    # --------------------------------------------------------

    if PATENT_ID_COLUMN in df.columns:

        patent_id = row[
            PATENT_ID_COLUMN
        ]

    else:

        patent_id = idx

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = normalize_text(
        row[TEXT_COLUMN]
    )

    if not text:
        continue

    # ========================================================
    # RETRIEVAL
    # ========================================================

    score, categories = retrieve_patent(
        text
    )

    # Do not retrieve patents containing only
    # generic causal language.
    #
    # We want patents with at least two
    # meaningful categories.
    #
    meaningful_categories = sum(
        bool(categories[x])
        for x in [
            "failure",
            "variable",
            "technical"
        ]
    )

    if meaningful_categories < 2:
        continue

    # --------------------------------------------------------
    # Retrieval output
    # --------------------------------------------------------

    retrieval_rows.append({

        "patent_id":
            patent_id,

        "retrieval_score":
            score,

        "failure_terms":
            json.dumps(
                sorted(
                    set(
                        categories["failure"]
                    )
                )
            ),

        "variable_terms":
            json.dumps(
                sorted(
                    set(
                        categories["variable"]
                    )
                )
            ),

        "technical_terms":
            json.dumps(
                sorted(
                    set(
                        categories["technical"]
                    )
                )
            ),

        "causal_terms":
            json.dumps(
                sorted(
                    set(
                        categories["causal"]
                    )
                )
            )
    })

    # ========================================================
    # CONTEXT EXTRACTION
    # ========================================================

    contexts = extract_contexts(
        text,
        window=CONTEXT_WINDOW
    )

    for context in contexts:

        context_rows.append({

            "patent_id":
                patent_id,

            "sentence_index":
                context["sentence_index"],

            "trigger_sentence":
                context["trigger_sentence"],

            "context":
                context["context"],

            "sentence_start":
                context["sentence_start"],

            "sentence_end":
                context["sentence_end"]
        })

        # ====================================================
        # CAUSAL RELATIONS
        # ====================================================

        relations = extract_cross_sentence_relations(

            context["context"],

            patent_id,

            context["sentence_start"]
        )

        for relation in relations:

            relation_rows.append(
                relation
            )


# ============================================================
# 19. CREATE DATAFRAMES
# ============================================================

retrieval_df = pd.DataFrame(
    retrieval_rows
)

context_df = pd.DataFrame(
    context_rows
)

relation_df = pd.DataFrame(
    relation_rows
)


# ============================================================
# 20. REMOVE DUPLICATE RELATIONS
# ============================================================

if not relation_df.empty:

    relation_df = relation_df.drop_duplicates(

        subset=[

            "patent_id",

            "source",

            "source_type",

            "causal_term",

            "target",

            "target_type",

            "sentence"
        ]
    )


# ============================================================
# 21. SAVE RESULTS
# ============================================================

retrieval_df.to_csv(

    OUTPUT_RETRIEVAL,

    index=False,

    encoding="utf-8"
)

context_df.to_csv(

    OUTPUT_CONTEXT,

    index=False,

    encoding="utf-8"
)

relation_df.to_csv(

    OUTPUT_RELATIONS,

    index=False,

    encoding="utf-8"
)


print(
    "Retrieved patents:",
    len(retrieval_df)
)

print(
    "Contexts:",
    len(context_df)
)

print(
    "Causal relations:",
    len(relation_df)
)


# ============================================================
# 22. BUILD CORPUS-LEVEL CAUSAL GRAPH
# ============================================================

if not relation_df.empty:

    G = build_corpus_graph(
        relation_df
    )

    graph_json = graph_to_json(
        G
    )

else:

    graph_json = {
        "nodes": [],
        "links": []
    }


# ============================================================
# 23. SAVE GRAPH
# ============================================================

with open(

    OUTPUT_GRAPHS,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        graph_json,

        f,

        ensure_ascii=False,

        indent=2
    )


print(
    "Corpus graph nodes:",
    len(graph_json["nodes"])
)

print(
    "Corpus graph edges:",
    len(graph_json["links"])
)

print(
    "Graph saved:",
    OUTPUT_GRAPHS
)