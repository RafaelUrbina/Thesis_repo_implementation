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

# Also limit PyTorch's internal thread pool explicitly
torch.set_num_threads(int(MAX_CPU_THREADS))

# Ensure NLTK VADER lexicon is available locally
nltk.download("vader_lexicon", quiet=True)

# Add project root to path using relative import formula
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH


def extract_nouns_and_verbs(doc):
    """Extracts lemmas of Nouns (NOUN, PROPN) and Verbs (VERB) from a SpaCy Doc

    as separate sorted unique lists.
    """
    nouns = set()
    verbs = set()
    for token in doc:
        clean_term = token.lemma_.lower().strip()
        if len(clean_term) > 1 and clean_term.isalpha():
            if token.pos_ in ("NOUN", "PROPN"):
                nouns.add(clean_term)
            elif token.pos_ == "VERB":
                verbs.add(clean_term)

    return sorted(list(nouns)), sorted(list(verbs))


def main():
    input_path = (
        PATENT_PIPELINE_PATH
        / "supervised_link"
        / "output"
        / "causal_filtered_supervised_exploded.csv"
    )
    output_path = (
        PATENT_PIPELINE_PATH
        / "unsupervised_link"
        / "output"
        / "causal_filtered_supervised_sentiment_pos.csv"
    )

    print(f"Loading CSV from: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Determine how many rows have already been processed
    already_processed = 0
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, mode="r", encoding="utf-8") as f:
            # Subtract 1 for header row
            already_processed = max(0, sum(1 for _ in f) - 1)
        print(f"Found existing output file. Resuming from row {already_processed:,}...")

    # Initialize SpaCy Transformer model
    print(f"Loading SpaCy transformer model ('en_core_web_trf')...")
    print(f"CPU threads limited to: {MAX_CPU_THREADS}")
    nlp = spacy.load("en_core_web_trf")

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # BATCH & COOLING CONFIGURATION
    BATCH_SIZE = 500
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

            # 2. Fast-forward input reader past already processed rows
            for _ in range(already_processed):
                next(reader, None)

            unprocessed_batch_rows = []
            unprocessed_texts = []

            pbar = tqdm(desc="Processing rows", unit="row", initial=already_processed)

            for row in reader:
                if not row or len(row) <= text_idx:
                    continue

                text_val = row[text_idx].strip()
                unprocessed_batch_rows.append(row)
                unprocessed_texts.append(text_val)

                # Process batch when buffer fills
                if len(unprocessed_texts) >= BATCH_SIZE:
                    with nlp.select_pipes(
                        enable=[
                            "transformer",
                            "tagger",
                            "attribute_ruler",
                            "lemmatizer",
                        ]
                    ):
                        docs = list(nlp.pipe(unprocessed_texts))

                    for orig_row, text, doc in zip(
                        unprocessed_batch_rows, unprocessed_texts, docs
                    ):
                        sentiment_score = (
                            analyzer.polarity_scores(text)["compound"]
                            if text
                            else 0.0
                        )

                        nouns_list, verbs_list = (
                            extract_nouns_and_verbs(doc) if text else ([], [])
                        )

                        new_row = orig_row + [
                            sentiment_score,
                            str(nouns_list),
                            str(verbs_list),
                        ]
                        batch_rows.append(new_row)
                        processed_count += 1

                    # Write out processed batch to disk & flush
                    writer.writerows(batch_rows)
                    outfile.flush()  # Forces immediate write to disk
                    pbar.update(len(batch_rows))

                    # Reset buffers
                    batch_rows.clear()
                    unprocessed_batch_rows.clear()
                    unprocessed_texts.clear()

                    if COOLING_SLEEP_SEC > 0:
                        time.sleep(COOLING_SLEEP_SEC)

            # Process remaining rows in final buffer
            if unprocessed_texts:
                with nlp.select_pipes(
                    enable=[
                        "transformer",
                        "tagger",
                        "attribute_ruler",
                        "lemmatizer",
                    ]
                ):
                    docs = list(nlp.pipe(unprocessed_texts))

                for orig_row, text, doc in zip(
                    unprocessed_batch_rows, unprocessed_texts, docs
                ):
                    sentiment_score = (
                        analyzer.polarity_scores(text)["compound"]
                        if text
                        else 0.0
                    )
                    nouns_list, verbs_list = (
                        extract_nouns_and_verbs(doc) if text else ([], [])
                    )
                    new_row = orig_row + [
                        sentiment_score,
                        str(nouns_list),
                        str(verbs_list),
                    ]
                    batch_rows.append(new_row)
                    processed_count += 1

                writer.writerows(batch_rows)
                outfile.flush()
                pbar.update(len(batch_rows))

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