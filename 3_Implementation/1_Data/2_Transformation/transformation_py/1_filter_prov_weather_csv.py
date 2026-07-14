import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import SIR_TOSCANA_PATH


def normalize_field(value: str) -> str:
    return value.strip().strip('"').strip()


def inspect_csv_files(folder: Path, dry_run: bool = False) -> None:
    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        print(f"No .csv files found in {folder}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {folder}")

    for file_path in csv_files:
        try:
            with file_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except Exception as exc:
            print(f"Could not read {file_path}: {exc}")
            continue

        provincia_line = None
        for line in lines:
            parts = [normalize_field(part) for part in line.split(";") if part.strip()]
            if parts and normalize_field(parts[0]).lower() == "provincia":
                provincia_line = line
                break

        if provincia_line is None:
            print(f"Skipping {file_path.name}: no Provincia line found")
            continue

        parts = [normalize_field(part) for part in provincia_line.split(";") if part.strip()]
        if len(parts) >= 2 and parts[1].upper() == "PI":
            print(f"Detected Provincia;PI in {file_path.name}: {provincia_line}")
            if dry_run:
                print("Dry run: would prompt to remove this file.")
                continue

            answer = input(f"Remove {file_path.name}? [y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                file_path.unlink()
                print(f"Removed {file_path.name}")
            else:
                print(f"Kept {file_path.name}")

        elif len(parts) >= 2 and parts[1].upper() == "FI":
            print(f"Skipping {file_path.name}: {provincia_line}")
        else:
            print(f"Skipping {file_path.name}: {provincia_line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect CSV headers and optionally remove files with Provincia;PI")
    parser.add_argument(
        "--folder",
        default=str(SIR_TOSCANA_PATH / "sir_rain_toscana_datasets"),
        help="Folder to scan for CSV files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything",
    )
    args = parser.parse_args()

    inspect_csv_files(Path(args.folder), dry_run=args.dry_run)
