import re
from pathlib import Path
import sys

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere in the project
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def clean_term_file(file_path: Path):
    """
    Reads a text file, removes lines containing numbers or specific keywords,
    and overwrites the file with the cleaned content.

    Args:
        file_path (Path): The path to the text file to clean.
    """
    if not file_path.exists():
        print(f"Error: File not found at '{file_path}'")
        return

    # Keywords to check for removal (all in lowercase for case-insensitive matching)
    keywords_to_remove = {
        "accident", "negotiation", "incident", "aero", "travel",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "business", "film", "international", "list", "sea"
    }

    try:
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        kept_lines = []
        # Iterate over each line to decide if it should be kept
        for line in original_lines:
            line_lower = line.lower()
            
            # Rule 1: Remove if the line contains any number
            if re.search(r'\d', line):
                continue  # Skip to the next line

            # Rule 2: Remove if the line contains any of the keywords
            if any(keyword in line_lower for keyword in keywords_to_remove):
                continue  # Skip to the next line

            # If no rules were met, keep the line
            kept_lines.append(line)

        # Write the filtered lines back to the original file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(kept_lines)

        removed_count = len(original_lines) - len(kept_lines)
        print(f"Processing complete for '{file_path.name}'.")
        print(f"Removed {removed_count} lines. Kept {len(kept_lines)} lines.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Import the path from the centralized paths.py
    from Utils.paths import PATENTS_OUTPUT_TERMS_PATH

    # Ensure the directory exists
    if not PATENTS_OUTPUT_TERMS_PATH.exists():
        print(f"Error: Directory not found at '{PATENTS_OUTPUT_TERMS_PATH}'")
    else:
        # Iterate over all .txt files in the target directory and clean them
        print(f"--- Starting cleaning process for files in '{PATENTS_OUTPUT_TERMS_PATH}' ---")
        for txt_file in PATENTS_OUTPUT_TERMS_PATH.glob("*.txt"):
            clean_term_file(txt_file)
        print("\n--- All files processed. ---")
