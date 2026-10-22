#SCRIPT FOR SEPARATING THE SENTENCES WITH CONNECTORS AND OTHER STRUCTURAL WORDS.
import ast
import csv
import re
import sys
from pathlib import Path

# Prevent CSV field size limit errors on massive patent paragraphs
csv.field_size_limit(2147483647)

# Add project root to path using relative import formula
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]
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
    """Simple substring check to retain terms present in the text clause."""
    if not text or not terms:
        return []
    text_lower = text.lower()
    return [term for term in terms if term.lower() in text_lower]


def split_patent_claims_into_clauses(text: str, max_chars: int = 250) -> list[str]:
    """
    Splits massive, unpunctuated patent claims/descriptions into readable clause-level 
    sentences based on patent connectors, semicolons, and structural boundaries.
    """
    if not text or not text.strip():
        return []

    # 1. Primary Split: Semicolons, colons followed by space, or standard periods
    initial_splits = re.split(r"[;]|\b:\s+|[.!?]+(?:\s+|$)", text)
    
    clause_candidates = []
    
    # 2. Secondary Split: Major Patent Transition Connectors
    connector_pattern = re.compile(
        r",\s+(?=(?:wherein|in which|comprising|furthermore|finally|by means of|including|whereby)\b)", 
        flags=re.IGNORECASE
    )

    for chunk in initial_splits:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        if len(chunk) > max_chars:
            sub_clauses = connector_pattern.split(chunk)
            clause_candidates.extend(sub_clauses)
        else:
            clause_candidates.append(chunk)

    # 3. Tertiary Split: Enumerated Reference Fallback
    final_sentences = []
    for clause in clause_candidates:
        clause = clause.strip()
        if not clause:
            continue
            
        # Clean up leading commas or stray connectors left from regex splitting
        clause = re.sub(r"^,\s*", "", clause)
        
        if len(clause) > 300:
            ref_split = re.split(r"(?<=\)\s)(?=[a-z0-9\s]{3,}\s+\(\d+\))", clause)
            final_sentences.extend([s.strip() for s in ref_split if s.strip()])
        else:
            final_sentences.append(clause)

    return [s for s in final_sentences if len(s) > 10]


def main():
    input_path = (
        PATENT_PIPELINE_PATH
        / "dataset_creation"
        / "1_supervised_link"
        / "output"
        / "causal_filtered_supervised.csv"
    )
    output_path = (
        PATENT_PIPELINE_PATH
        / "dataset_creation"
        / "1_supervised_link"
        / "output"
        / "midstep_causal_filtered_supervised_exploded.csv"
    )
    stats_path = (
        PATENT_PIPELINE_PATH
        / "dataset_creation"
        / "1_supervised_link"
        / "output"
        / "midstep_causal_filtered_supervised_exploded_stats.txt"
    )

    print(f"Loading input CSV from: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    remaining_rows = 0
    batch_rows = []
    BATCH_SIZE = 10000

    print("Processing CSV rows with advanced patent clause splitting...")
    try:
        with (
            open(input_path, mode="r", encoding="utf-8", newline="") as infile,
            open(output_path, mode="w", encoding="utf-8", newline="") as outfile,
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

            # Pre-find header indices for fast lookup
            tech_idx = header.index("occurrence_technical") if "occurrence_technical" in header else None
            fail_idx = header.index("occurrence_failures") if "occurrence_failures" in header else None
            var_idx = header.index("occurrence_variable") if "occurrence_variable" in header else None
            causal_idx = header.index("occurrence_causal") if "occurrence_causal" in header else None
            causal_words_idx = header.index("causal_ocurrence_words") if "causal_ocurrence_words" in header else None

            # Get column indices for occurrence fields present in header
            occ_col_indices = {
                header.index(col): col
                for col in OCCURRENCE_COLS
                if col in header
            }

            writer.writerow(header)

            def satisfies_rule(occ_dict):
                """
                Rule: At least one occurrence in technical/failures/variable
                AND at least one occurrence in causal columns.
                """
                has_tech_fail_var = (
                    (tech_idx is not None and len(occ_dict.get(tech_idx, [])) > 0) or
                    (fail_idx is not None and len(occ_dict.get(fail_idx, [])) > 0) or
                    (var_idx is not None and len(occ_dict.get(var_idx, [])) > 0)
                )

                has_causal = (
                    (causal_idx is not None and len(occ_dict.get(causal_idx, [])) > 0) or
                    (causal_words_idx is not None and len(occ_dict.get(causal_words_idx, [])) > 0)
                )

                return has_tech_fail_var and has_causal

            for row in reader:
                total_rows += 1

                if not row or len(row) <= text_idx:
                    continue

                text_val = row[text_idx]
                clauses = split_patent_claims_into_clauses(text_val)

                # Parse original paragraph-level occurrences
                parsed_occurrences = {
                    idx: parse_occurrence_list(
                        row[idx] if idx < len(row) else ""
                    )
                    for idx in occ_col_indices
                }

                if clauses:
                    for clause in clauses:
                        new_row = list(row)
                        new_row[text_idx] = clause

                        # Update occurrence columns for this specific clause segment
                        updated_occ = {}
                        for idx, terms in parsed_occurrences.items():
                            matched_terms = filter_occurrences(clause, terms)
                            new_row[idx] = str(matched_terms)
                            updated_occ[idx] = matched_terms

                        # Apply filter rule to each exploded clause
                        if satisfies_rule(updated_occ):
                            batch_rows.append(new_row)
                            remaining_rows += 1
                else:
                    # Fallback if no clauses extracted
                    if satisfies_rule(parsed_occurrences):
                        batch_rows.append(row)
                        remaining_rows += 1

                if len(batch_rows) >= BATCH_SIZE:
                    writer.writerows(batch_rows)
                    batch_rows.clear()

            if batch_rows:
                writer.writerows(batch_rows)

        summary_text = (
            f"Total input rows: {total_rows}, Remaining exploded rows after filtering: {remaining_rows}"
        )

        with open(stats_path, mode="w", encoding="utf-8") as stats_file:
            stats_file.write(summary_text + "\n")

        print("\n" + "=" * 50)
        print(summary_text)
        print(f"Saved stats summary to: {stats_path}")
        print(f"Saved midstep exploded CSV to: {output_path}")
        print("=" * 50)

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()