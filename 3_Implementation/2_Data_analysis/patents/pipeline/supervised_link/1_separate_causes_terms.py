import sys
import csv
from pathlib import Path

# Add project root to path using relative import formula
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

from Utils.paths import PATENT_PIPELINE_PATH

def main():
    input_path = PATENT_PIPELINE_PATH / "supervised_link" / "output" / "causal_filtered_supervised.csv"
    output_path = PATENT_PIPELINE_PATH / "supervised_link" / "output" / "causal_filtered_supervised_exploded.csv"
    
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
        with open(input_path, mode="r", encoding="utf-8", newline="") as infile, \
             open(output_path, mode="w", encoding="utf-8", newline="") as outfile:
            
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
            writer.writerow(header)
            
            for row in reader:
                if not row or len(row) <= text_idx:
                    continue
                original_count += 1
                
                text_val = row[text_idx]
                if "..." in text_val:
                    parts = text_val.split("...")
                    for part in parts:
                        cleaned_part = part.strip()
                        if cleaned_part:
                            new_row = list(row)
                            new_row[text_idx] = cleaned_part
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










