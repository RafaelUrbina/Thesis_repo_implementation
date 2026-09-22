import ast
import csv
import sys
from pathlib import Path

# Add project root to path using relative import formula
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

# Occurrence columns to filter
OCCURRENCE_COLS = [
    "occurrence_technical",
    "occurrence_failures",
    "occurrence_causal",
    "occurrence_variable",
    "causal_ocurrence_words",
]


def parse_occurrence_list(raw_val: str) -> list[str]:
    """Safely parse a string representation of a list into a Python list."""
    if not raw_val or raw_val.strip() in ("", "[]", "None"):
        return []
    try:
        parsed = ast.literal_eval(raw_val)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []
    except (ValueError, SyntaxError):
        return []


def filter_occurrences(text: str, terms: list[str]) -> list[str]:
    """Simple substring check to retain terms present in the text."""
    if not text or not terms:
        return []
    text_lower = text.lower()
    return [term for term in terms if term.lower() in text_lower]


def main():
    input_path = (
        PATENT_PIPELINE_PATH
        / "supervised_link"
        / "output"
        / "causal_filtered_supervised.csv"
    )
    output_path = (
        PATENT_PIPELINE_PATH
        / "supervised_link"
        / "output"
        / "causal_filtered_supervised_exploded.csv"
    )

    print(f"Loading CSV from: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_count = 0
    exploded_count = 0
    batch_rows = []
    BATCH_SIZE = 10000

    print("Processing CSV rows with batch writing...")
    try:
        with (
            open(input_path, mode="r", encoding="utf-8", newline="") as infile,
            open(
                output_path, mode="w", encoding="utf-8", newline=""
            ) as outfile,
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

            # Get column indices for occurrence fields present in header
            occ_col_indices = {
                header.index(col): col
                for col in OCCURRENCE_COLS
                if col in header
            }

            writer.writerow(header)

            for row in reader:
                if not row or len(row) <= text_idx:
                    continue
                original_count += 1

                text_val = row[text_idx]

                if "..." in text_val:
                    parts = text_val.split("...")

                    # Parse original paragraph-level occurrences
                    parsed_occurrences = {
                        idx: parse_occurrence_list(
                            row[idx] if idx < len(row) else ""
                        )
                        for idx in occ_col_indices
                    }

                    for part in parts:
                        cleaned_part = part.strip()
                        if cleaned_part:
                            new_row = list(row)
                            new_row[text_idx] = cleaned_part

                            # Update occurrence columns for this sentence segment
                            for idx, terms in parsed_occurrences.items():
                                matched_terms = filter_occurrences(
                                    cleaned_part, terms
                                )
                                new_row[idx] = str(matched_terms)

                            batch_rows.append(new_row)
                            exploded_count += 1
                else:
                    batch_rows.append(row)
                    exploded_count += 1

                if len(batch_rows) >= BATCH_SIZE:
                    writer.writerows(batch_rows)
                    batch_rows.clear()

            if batch_rows:
                writer.writerows(batch_rows)

        print(f"Total original rows processed: {original_count:,}")
        print(f"Total exploded rows written: {exploded_count:,}")
        print(f"Saved exploded CSV to: {output_path}")
        print("Done!")
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()