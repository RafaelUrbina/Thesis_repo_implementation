import os

# =====================================================================
# HARDWARE THROTTLING: Must be configured BEFORE importing torch / spacy
# Limit PyTorch / OpenMP / BLAS thread pools (e.g., to 2 threads)
# =====================================================================
MAX_CPU_THREADS = "4"  # Adjust: "1" (coolest), "2" (balanced), "4" (faster)
os.environ["OMP_NUM_THREADS"] = MAX_CPU_THREADS
os.environ["MKL_NUM_THREADS"] = MAX_CPU_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = MAX_CPU_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = MAX_CPU_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = MAX_CPU_THREADS

import csv
from pathlib import Path
import sys
import time  # For cooling delays
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import torch
from tqdm import tqdm

csv.field_size_limit(2147483647)

# Also limit PyTorch's internal thread pool explicitly
torch.set_num_threads(int(MAX_CPU_THREADS))

# Ensure NLTK VADER lexicon is available locally
nltk.download("vader_lexicon", quiet=True)

# Add project root to path using relative import formula
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

def extract_nouns_and_verbs(doc):
    """Extracts lemmas of Nouns (NOUN, PROPN) and Verbs/Auxiliaries (VERB, AUX) from a SpaCy Doc

    as separate sorted unique lists. Filters out standalone modal verbs like 'can'/'will'
    while capturing true action verbs and phrasal particles.
    """
    nouns = set()
    verbs = set()

    for token in doc:
        clean_term = token.lemma_.lower().strip()

        # Skip punctuation, non-alphabetic, or single-character noise
        if len(clean_term) <= 1 or not clean_term.isalpha():
            continue

        if token.pos_ in ("NOUN", "PROPN"):
            nouns.add(clean_term)

        elif token.pos_ in ("VERB", "AUX"):
            # Exclude standalone modal/auxiliary verbs unless part of a phrasal verb
            if token.dep_ == "aux" and token.pos_ == "AUX":
                continue

            # Capture phrasal particles (e.g., 'carry out' -> 'carry out')
            particle = ""
            for child in token.children:
                if child.dep_ == "prt":
                    particle = f" {child.lemma_.lower().strip()}"
                    break

            verbs.add(f"{clean_term}{particle}")

    return sorted(list(nouns)), sorted(list(verbs))


def main():
    input_path = (
        PATENT_PIPELINE_PATH
        / "dataset_creation"
        / "1_supervised_link"
        / "output"
        / "causal_filtered_supervised_exploded.csv"
    )
    output_path = (
        PATENT_PIPELINE_PATH
        / "dataset_creation"
        / "2_unsupervised_link"
        / "output"
        / "causal_filtered_unsupervised_sentiment_pos.csv"
    )

    print(f"Loading CSV from: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Determine how many logical rows have already been processed
    already_processed = 0
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, mode="r", encoding="utf-8", newline="") as f:
            reader_check = csv.reader(f)
            header_check = next(reader_check, None)
            if header_check:
                already_processed = sum(1 for _ in reader_check)
        print(f"Found existing output file. Resuming from row {already_processed:,}...")

    # Load SpaCy model
    MODEL_NAME = "en_core_web_lg"  # Switch Between en_core_web_lg and en_core_web_trf if maximum accuracy is needed
    print(f"Loading SpaCy model ('{MODEL_NAME}')...")
    print(f"CPU threads limited to: {MAX_CPU_THREADS}")

    nlp = spacy.load(MODEL_NAME)

    # Safely determine active pipeline components depending on whether model is TRF or Standard
    available_pipes = nlp.pipe_names
    disable_pipes = [
        pipe for pipe in ["ner", "parser", "senter"] if pipe in available_pipes
    ]

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # BATCH & COOLING CONFIGURATION
    BATCH_SIZE = 1
    COOLING_SLEEP_SEC = 0.5

    batch_rows = []
    processed_count = 0

    print("Processing CSV rows with throttled NLP processing...")
    try:
        # Open output in 'a' (append) mode so existing rows are preserved
        file_mode = "a" if already_processed > 0 else "w"

        with (
            open(input_path, mode="r", encoding="utf-8", newline="") as infile,
            open(output_path, mode=file_mode, encoding="utf-8", newline="") as outfile,
        ):
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader, None)
            if not header:
                print("Error: Empty CSV file.")
                return

            if "text" not in header:
                print("Error: 'text' column not found in header.")
                return

            text_idx = header.index("text")

            # Only write header if starting a fresh output file
            if already_processed == 0:
                new_header = header + ["sentiment_score", "nouns", "verbs"]
                writer.writerow(new_header)
                outfile.flush()

            # Fast-forward input reader past already processed rows
            for _ in range(already_processed):
                next(reader, None)

            unprocessed_batch_rows = []
            unprocessed_texts = []

            pbar = tqdm(desc="Processing rows", unit="row", initial=already_processed)
            batch_rows = []
            processed_count = 0

            for row in reader:
                if not row or len(row) <= text_idx:
                    continue

                text_val = row[text_idx].strip()
                
                # 1. Compute Sentiment via VADER (fast and safe)
                sentiment_score = (
                    analyzer.polarity_scores(text_val)["compound"]
                    if text_val
                    else 0.0
                )

                # 2. Process via SpaCy safely with a fallback
                nouns_list, verbs_list = [], []
                if text_val:
                    try:
                        doc = nlp(text_val)
                        nouns_list, verbs_list = extract_nouns_and_verbs(doc)
                    except Exception as nlp_err:
                        # If spaCy chokes on a weird text string, log it and keep moving
                        print(f"\n[Warning] SpaCy error on row: {nlp_err}")
                        nouns_list, verbs_list = [], []

                new_row = row + [
                    sentiment_score,
                    str(nouns_list),
                    str(verbs_list),
                ]
                batch_rows.append(new_row)
                processed_count += 1

                # Write in chunks of 100 to save memory and keep disk updated
                if len(batch_rows) >= 100:
                    writer.writerows(batch_rows)
                    outfile.flush()
                    pbar.update(len(batch_rows))
                    batch_rows.clear()

            # Write any remaining rows
            if batch_rows:
                writer.writerows(batch_rows)
                outfile.flush()
                pbar.update(len(batch_rows))
                batch_rows.clear()

            pbar.close()

        print(
            f"\nSuccessfully processed {processed_count:,} new rows with Sentiment and POS analysis."
        )
        print(f"Saved output CSV to: {output_path}")

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()