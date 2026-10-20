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


def clean_text(text: str) -> str:
    """Normalizes text to lowercase while preserving important patent punctuation 
    (such as hyphens, commas, colons, semicolons, parentheses, and brackets) 
    along with alphanumeric characters and basic sentence terminals.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()

    # Keep alphanumeric characters, whitespace, and valuable patent punctuation:
    # periods, commas, semicolons, colons, hyphens, parentheses, brackets, slashes, plus/equals
    text = re.sub(r"[^\w\s.,;:()\-\[%\]/+=]", " ", text)

    # Collapse multiple whitespaces into a single space and strip leading/trailing spaces
    return re.sub(r"\s+", " ", text).strip()


def split_by_paragraphs(text: str) -> list[str]:
    """Splits text by double newlines (blank lines) to capture actual paragraphs."""
    if not text:
        return []

    # Splits on one or more blank lines (with optional carriage returns)
    paragraphs = re.split(r"\r?\n\s*\r?\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def process_patent_files(directory_path: Path | str) -> pd.DataFrame:
    directory = Path(directory_path)
    records = []

    print(f"Scanning directory: {directory.resolve()}")
    json_files = list(directory.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("Warning: No .json files found in the specified directory.")
        return pd.DataFrame(), ""

    print(f"Found {total_files:,} JSON file(s) to process.\n")

    skipped_files = 0
    total_claims = 0
    total_description_paragraphs = 0

    for filepath in tqdm(json_files, desc="Processing Patents", unit="file"):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                skipped_files += 1
                continue

        pat_id = data.get("pat_id", "")
        title = data.get("title", "")

        # 1. Process Claims (Treating each claim element as a complete claim entity)
        claims = data.get("claims") or []
        for claim_idx, claim in enumerate(claims, start=1):
            if not claim:
                continue
            # Normalize internal whitespaces/newlines within the claim text
            normalized_claim = re.sub(r"\s+", " ", claim).strip()
            cleaned_claim = clean_text(normalized_claim)
            
            if cleaned_claim:
                records.append(
                    {
                        "pat_id": pat_id,
                        "title": title,
                        "text_type": "claim",
                        "paragraph_id": f"claim_{claim_idx}",
                        "text": cleaned_claim,
                    }
                )
                total_claims += 1

        # 2. Process Description (Split strictly by double newlines into true paragraphs)
        description = data.get("description") or ""
        desc_paragraphs = split_by_paragraphs(description)
        
        for para_idx, para in enumerate(desc_paragraphs, start=1):
            cleaned_para = clean_text(para)
            if cleaned_para:
                records.append(
                    {
                        "pat_id": pat_id,
                        "title": title,
                        "text_type": "description",
                        "paragraph_id": f"para_{para_idx}",
                        "text": cleaned_para,
                    }
                )
                total_description_paragraphs += 1
        
    # Processing summary
    processed_count = total_files - skipped_files
    summary_lines = [
        "=" * 50,
        "PROCESSING SUMMARY",
        "=" * 50,
        f"Files Processed successfully : {processed_count:,} / {total_files:,}",
    ]
    if skipped_files > 0:
        summary_lines.append(f"Files Skipped (Invalid JSON) : {skipped_files:,}")
    summary_lines.extend([
        f"Total Claim Rows Extracted   : {total_claims:,}",
        f"Total Description Paragraphs : {total_description_paragraphs:,}",
        f"Total Dataset Rows Created   : {len(records):,}",
        "=" * 50
    ])
    
    summary_text = "\n" + "\n".join(summary_lines) + "\n\n"
    print(summary_text)

    return pd.DataFrame(records), summary_text


if __name__ == "__main__":
    folder_path = MASTER_DATA_PATH / "patents/full_text"

    df, summary_text = process_patent_files(folder_path)

    if not df.empty:
        output_dir = PATENT_PIPELINE_PATH / "dataset_creation/0_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_file = output_dir / "processing_summary.txt"
        summary_file.write_text(summary_text, encoding="utf-8")
        print(f"Saved processing summary to: {summary_file.resolve()}")

        output_file = output_dir / "patent_dataset.csv"
        print(f"Saving dataset to: {output_file.resolve()}...")
        df.to_csv(output_file, index=False, encoding="utf-8")
        print("Export completed successfully!\n")

        print("Dataset Preview:")
        print(df.head())
    else:
        print("No data extracted. Dataset export skipped.")