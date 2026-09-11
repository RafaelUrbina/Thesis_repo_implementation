import json
import re
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[5]
sys.path.insert(0, str(project_root))

from Utils.paths import MASTER_DATA_PATH, PATENT_PIPELINE_PATH


def split_by_newlines(text: str) -> list[str]:
    """Splits text strictly by newline characters (\n)."""
    if not text:
        return []

    # Splits on unix (\n) or windows (\r\n) newlines
    lines = re.split(r"\r?\n+", text)

    # Clean leading/trailing spaces, discarding blank lines
    return [line.strip() for line in lines if line.strip()]


def process_patent_files(directory_path: Path | str) -> pd.DataFrame:
    directory = Path(directory_path)
    records = []

    print(f"Scanning directory: {directory.resolve()}")
    json_files = list(directory.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("Warning: No .json files found in the specified directory.")
        return pd.DataFrame()

    print(f"Found {total_files:,} JSON file(s) to process.\n")

    skipped_files = 0
    total_claims = 0
    total_description_lines = 0

    for filepath in tqdm(json_files, desc="Processing Patents", unit="file"):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                skipped_files += 1
                continue

        pat_id = data.get("pat_id", "")
        title = data.get("title", "")

        # 1. Process Claims (Each claim split by newlines if internal \n exist)
        claims = data.get("claims") or []
        for claim in claims:
            if not claim:
                continue
            claim_lines = split_by_newlines(claim)
            for line in claim_lines:
                records.append(
                    {
                        "pat_id": pat_id,
                        "title": title,
                        "text_type": "claim",
                        "text": line,
                    }
                )
                total_claims += 1

        # 2. Process Description (Split strictly by newlines)
        description = data.get("description") or ""
        desc_lines = split_by_newlines(description)
        for line in desc_lines:
            records.append(
                {
                    "pat_id": pat_id,
                    "title": title,
                    "text_type": "description",
                    "text": line,
                }
            )
            total_description_lines += 1

    # Processing summary
    processed_count = total_files - skipped_files
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Files Processed successfully : {processed_count:,} / {total_files:,}")
    if skipped_files > 0:
        print(f"Files Skipped (Invalid JSON) : {skipped_files:,}")
    print(f"Total Claim Rows Extracted   : {total_claims:,}")
    print(f"Total Description Rows       : {total_description_lines:,}")
    print(f"Total Dataset Rows Created   : {len(records):,}")
    print("=" * 50 + "\n")

    return pd.DataFrame(records)


if __name__ == "__main__":
    folder_path = MASTER_DATA_PATH / "patents/full_text"

    df = process_patent_files(folder_path)

    if not df.empty:
        output_dir = PATENT_PIPELINE_PATH / "data_creation/output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "patent_dataset.csv"
        print(f"Saving dataset to: {output_file.resolve()}...")
        df.to_csv(output_file, index=False, encoding="utf-8")
        print("Export completed successfully!\n")

        print("Dataset Preview:")
        print(df.head())
    else:
        print("No data extracted. Dataset export skipped.")