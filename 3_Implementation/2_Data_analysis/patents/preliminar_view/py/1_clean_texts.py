"""
Cleans raw text files by removing stopwords and patent-specific jargon.

This script processes a directory of text files, performing several cleaning steps:
1.  Converts all text to lowercase.
2.  Removes a comprehensive list of generic English stopwords (e.g., 'the', 'a', 'is').
3.  Removes a custom list of patent-specific boilerplate and legal terms
    (e.g., 'United States Patent', 'claim', 'FIG.').
4.  Removes punctuation, numbers, and extra whitespace to leave only meaningful words.
5.  Saves the cleaned text to a new directory for further analysis, such as
    topic modeling.

This pre-processing step is crucial for improving the quality of any subsequent
NLP analysis.
"""

import re
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords

# --- Download NLTK stopwords if not already present ---
try:
    stopwords.words("english")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords")

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import CAUSAL_LINKS_PATH

# --- Custom Stopwords ---
# Generic English stopwords from NLTK
ENGLISH_STOPWORDS = set(stopwords.words("english"))

# Custom list of words and phrases commonly found in patent documents
PATENT_JARGON = {
    "patent", "united", "states", "claim", "claims", "fig", "us", "b1", "al",
    "date", "applicant", "inventor", "appl", "filed", "int", "cl", "cpc",
    "field", "classification", "search", "references", "cited", "documents",
    "examiner", "attorney", "agent", "firm", "abstract", "sheet", "drawing",
    "drawings", "description", "background", "summary", "detailed", "brief",
    "disclosure", "present", "invention", "system", "method", "device",
    "comprising", "comprises", "thereof", "herein", "said", "plurality",
    "therein", "wherein", "thereby", "thereon", "therefrom", "hereto",
    "therewith", "therein", "thereunder", "e g", "i e", "etc", "embodiment",
    "embodiments", "therefor", "thereof", "therein", "thereon", "thereby",
    "heretofore", "hereinafter", "hitherto", "however", "thereabout",
    "thereabouts", "thereafter", "thereat", "therebefore", "henceforth",
    "henceforward", "hereafter", "hereby", "herein", "hereof", "hereon",
    "hereto", "hereunder", "herewith", "hitherto", "however", "thereabout",
    "thereabouts", "thereafter", "thereat", "therebefore", "therefor",
    "therefrom", "therein", "thereof", "thereon", "thereto", "thereunder",
    "therewith", "thus", "whence", "whencesoever", "whenever", "whensoever",
    "where", "whereafter", "whereas", "whereat", "whereby", "wherefore",
    "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wheresoever",
    "whereto", "whereunto", "whereupon", "wherever", "wherewith", "wherewithal",
    "whether", "whither", "whithersoever", "least", "can", "may", "one", "first",
    "second", "third", "a", "b", "c", "d"
}

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(PATENT_JARGON)


def clean_text_file(raw_text: str) -> str:
    """
    Applies a series of cleaning operations to a string of text.
    """
    # 1. Convert to lowercase
    text = raw_text.lower()

    # 2. Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Tokenize (split into words) and remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in ALL_STOPWORDS and len(word) > 2]

    # 4. Join words back into a single string
    return " ".join(cleaned_words)


def main():
    """
    Main function to find, clean, and save all text files.
    """
    # --- Configuration ---
    input_dir = CAUSAL_LINKS_PATH / "causal_texts" / "txt"
    output_dir = CAUSAL_LINKS_PATH / "causal_texts" / "cleaned_txt"

    # --- Script Execution ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Searching for .txt files in: {input_dir}")

    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print("No .txt files found to process.")
        return

    print(f"Found {len(txt_files)} files. Starting cleaning process...")

    for txt_path in txt_files:
        print(f"  - Cleaning '{txt_path.name}'...")
        raw_text = txt_path.read_text(encoding="utf-8")
        cleaned_text = clean_text_file(raw_text)

        output_path = output_dir / txt_path.name
        output_path.write_text(cleaned_text, encoding="utf-8")

    print(f"\n--- Cleaning complete. Cleaned files saved in: {output_dir} ---")


if __name__ == "__main__":
    main()