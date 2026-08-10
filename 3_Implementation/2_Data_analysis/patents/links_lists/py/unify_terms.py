import sys
from pathlib import Path
import re

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere in the project
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import PATENTS_OUTPUT_TERMS_PATH

def unify_and_clean_terms(
    input_dir: Path,
    files_to_process: list[str],
    output_filename: str
):
    """
    Reads multiple term files, cleans and normalizes the terms,
    removes duplicates, and saves the result to a single unified file.

    Args:
        input_dir (Path): The directory containing the term files.
        files_to_process (list[str]): A list of filenames to process.
        output_filename (str): The name for the unified output file.
    """
    unified_terms = set()
    
    print("--- Starting Term Unification and Cleaning Process ---")

    for filename in files_to_process:
        file_path = input_dir / filename
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path.name}")
            continue

        print(f"Processing file: {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the entire file content at once
                content = f.read()
                # Split the content by commas or newlines, but not spaces within terms.
                # The regex '[,\n]+' splits by one or more of the delimiters.
                terms = re.split(r'[,\n]+', content)
                for term in terms:
                    # Normalize: lowercase, remove quotes, strip whitespace
                    cleaned_term = term.lower().replace('"', '').strip()
                    # Add to set if the term is not empty
                    if cleaned_term: unified_terms.add(cleaned_term)
        except Exception as e:
            print(f"  - Error processing file {file_path.name}: {e}")

    print("\n--- Processing Complete ---")

    if not unified_terms:
        print("No terms were extracted. The output file will not be created.")
        return

    # Sort the terms for consistent output
    sorted_terms = sorted(list(unified_terms))

    # Define output path and save the unified list
    output_path = input_dir / output_filename
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for term in sorted_terms:
                f.write(term + "\n")
        print(f"Successfully unified {len(sorted_terms)} unique terms into '{output_path.name}'.")
    except Exception as e:
        print(f"Error writing to output file {output_path.name}: {e}")

if __name__ == "__main__":
    # List of the specific text files you want to merge
    selected_files = [
        "[variable_terms]_unified_terms.txt",
        "variable_list.txt",
    ]

    # Run the unification process
    unify_and_clean_terms(
        input_dir=PATENTS_OUTPUT_TERMS_PATH,
        files_to_process=selected_files,
        output_filename="variable_list.txt"
    )