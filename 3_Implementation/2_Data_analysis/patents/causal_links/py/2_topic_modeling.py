"""
Performs guided topic modeling on cleaned patent text files to align with
geospatial data concepts.
This script takes a directory of cleaned text files and uses machine learning
to discover latent topics within the corpus.

The pipeline is as follows:
1.  Load all cleaned text documents into memory.
2.  Use TfidfVectorizer to convert the text into a matrix of TF-IDF features,
    which weighs words by their importance in the corpus.
    Crucially, the vectorizer is constrained to a custom vocabulary derived
    from the project's geospatial data layers to ensure topics are relevant.
3.  Apply Non-Negative Matrix Factorization (NMF), an algorithm well-suited
    for this task, to decompose the TF-IDF matrix into a set of topics.
4.  For each discovered topic, identify and display the top N most
    representative words.
5.  Determine the most dominant topic for each patent document.
6.  Save the topic definitions and the document-topic mapping to separate files.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import CAUSAL_LINKS_PATH

# --- Custom Vocabulary for Guided Topic Modeling ---
# This vocabulary is derived from the conceptual descriptions of the geospatial data layers.
# It forces the model to build topics around these specific, relevant concepts.
DOMAIN_VOCABULARY = [
    # Hydrology & Water Management
    "water", "hydrology", "hydraulic", "hydrogeological", "precipitation", "rain",
    "storm", "flood", "inundation", "runoff", "drainage", "sewage", "effluent",
    "river", "stream", "waterway", "watercourse", "aquifer", "groundwater",
    "reservoir", "dam", "basin", "catchment", "level", "discharge", "flow",
    "pressure", "quality", "treatment", "filtration", "purification", "desalination",
    "irrigation", "drought", "scarcity",

    # Geotechnics & Soil Science
    "soil", "geotechnical", "pedological", "landslide", "erosion", "seismic",
    "earthquake", "ground", "terrain", "geology", "lithology", "bedrock",
    "sediment", "clay", "sand", "gravel", "porosity", "permeability",
    "infiltration", "conductivity", "saturation", "moisture", "stability",
    "deformation", "subsidence", "slope", "embankment", "foundation",
    "excavation", "drilling", "tunnelling",

    # Infrastructure & Systems
    "system", "network", "grid", "infrastructure", "pipeline", "pipe", "conduit",
    "culvert", "channel", "main", "duct", "tube", "pump", "valve", "manifold",
    "tank", "sump", "cistern", "well", "sensor", "meter", "gauge", "monitor",
    "control", "actuator", "automation", "scada", "telemetry", "data", "model",
    "simulation", "optimization", "management", "maintenance", "repair", "leak",
    "detection", "inspection", "construction", "material", "concrete", "steel",
    "polymer", "coating", "lining",

    # Environmental & Economic
    "environment", "contaminant", "pollutant", "remediation", "sustainability",
    "risk", "hazard", "assessment", "cost", "economic", "efficiency", "energy",
    "urban", "municipal", "utility", "regulation", "standard", "policy"
]

def get_document_paths(input_dir: Path) -> list[Path]:
    """Returns a sorted list of .txt files from the input directory."""
    return sorted(list(input_dir.glob("*.txt")))

def display_topics(model: NMF, feature_names: list, num_top_words: int) -> str:
    """Formats the topics found by the model into a readable string."""
    output_lines = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        message = f"Topic #{topic_idx + 1}: {' | '.join(top_words)}"
        print(message)
        output_lines.append(message)
    return "\n".join(output_lines)


def main():
    """
    Main function to run the topic modeling pipeline.
    """
    # --- Configuration ---
    input_dir = CAUSAL_LINKS_PATH / "causal_texts" / "cleaned_txt"
    analysis_output_dir = CAUSAL_LINKS_PATH / "analysis_results"
    topic_definitions_file = analysis_output_dir / "topic_definitions.txt"
    doc_topic_mapping_file = analysis_output_dir / "patent_topic_mapping.csv"

    # --- Model Hyperparameters ---
    # Adjust NUM_TOPICS to control the granularity of the discovered themes.
    NUM_TOPICS = 5  # The number of topics you want to discover.
    NUM_TOP_WORDS = 15  # The number of words to display for each topic.

    # --- Script Execution ---
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading cleaned text files from: {input_dir}")

    # 1. Load the corpus and filenames
    doc_paths = get_document_paths(input_dir)
    corpus = [path.read_text(encoding="utf-8") for path in doc_paths]
    doc_filenames = [path.name for path in doc_paths]

    if not corpus:
        print("Error: No cleaned text files found. Please run the cleaning script first.")
        return

    print(f"Loaded {len(corpus)} documents.")

    # 2. Vectorize the text with TF-IDF using our custom domain vocabulary
    print("Vectorizing text with TF-IDF using the custom domain vocabulary...")
    vectorizer = TfidfVectorizer(vocabulary=DOMAIN_VOCABULARY, max_df=0.95, min_df=2)
    tfidf = vectorizer.fit_transform(corpus)

    # 3. Apply NMF to find topics from the constrained feature set
    print(f"Running NMF to find {NUM_TOPICS} topics...")
    nmf = NMF(n_components=NUM_TOPICS, random_state=42, alpha_W=0.0, alpha_H=0.0, l1_ratio=0)
    nmf.fit(tfidf)

    # 4. Display and save the topics
    print(f"\n--- Top {NUM_TOP_WORDS} words per topic ---")
    feature_names = vectorizer.get_feature_names_out()
    topic_results = display_topics(nmf, feature_names, NUM_TOP_WORDS)

    topic_definitions_file.write_text(topic_results, encoding="utf-8")
    print(f"\n--- Topic definitions saved to: {topic_definitions_file} ---")

    # 5. Map each document to its most dominant topic
    print("\nMapping each patent to its dominant topic...")
    doc_topic_matrix = nmf.transform(tfidf)
    dominant_topic = doc_topic_matrix.argmax(axis=1)

    # Create a DataFrame for the results
    mapping_df = pd.DataFrame({
        "patent_filename": doc_filenames,
        "dominant_topic_id": dominant_topic,
        "dominant_topic_name": [f"Topic #{t + 1}" for t in dominant_topic]
    })
    mapping_df.to_csv(doc_topic_mapping_file, index=False)
    print(f"--- Patent-to-topic mapping saved to: {doc_topic_mapping_file} ---")


if __name__ == "__main__":
    main()