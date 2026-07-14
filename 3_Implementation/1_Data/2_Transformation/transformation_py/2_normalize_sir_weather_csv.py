from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]
sys.path.insert(0, str(project_root))

from Utils.paths import SIR_WEATHER_TASKS


def parse_month_year(value: str) -> Tuple[int, int]:
    clean = value.strip().strip('"')
    dt = datetime.strptime(clean, "%d/%m/%Y")
    return dt.year, dt.month


def parse_date(value: str) -> datetime:
    clean = value.strip().strip('"')
    return datetime.strptime(clean, "%d/%m/%Y")


def parse_file_dates(path: Path) -> Tuple[List[Tuple[int, int]], List[str]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    data_header_index = None
    for idx, line in enumerate(lines):
        if "gg/mm/aaaa" in line:
            data_header_index = idx
            break

    if data_header_index is None:
        raise ValueError(f"No data header found in {path}")

    date_lines: List[Tuple[Tuple[int, int], str]] = []
    for line in lines[data_header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";;"):
            continue
        parts = [part.strip().strip('"') for part in stripped.split(";") if part.strip()]

        if len(parts) >= 2 and parts[0]:
            try:
                date_value = parts[0]
                month_year = parse_month_year(date_value)
                date_lines.append((month_year, line))
            except ValueError:
                continue

    sorted_dates = [d for d, _ in sorted(date_lines, key=lambda item: item[0])]
    date_strings = [f"{year:04d}-{month:02d}" for year, month in sorted_dates]
    lines_by_date: Dict[Tuple[int, int], str] = {d: line for d, line in date_lines}

    return sorted_dates, [lines_by_date[d] for d in sorted(lines_by_date)]


def normalize_folder(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        return

    print(f"Processing {source_dir.name} with {len(csv_files)} files")

    parsed_files: List[Tuple[Path, List[Tuple[int, int]], List[str]]] = []
    for csv_file in csv_files:
        try:
            month_years, _ = parse_file_dates(csv_file)
        except Exception as exc:
            print(f"Skipping {csv_file.name}: {exc}")
            continue

        parsed_files.append((csv_file, month_years, [f"{year:04d}-{month:02d}" for year, month in month_years]))

    if not parsed_files:
        print(f"No valid CSV files in {source_dir}")
        return

    all_dates = [dates for _, dates, _ in parsed_files]
    common_dates = set(all_dates[0])
    for dates in all_dates[1:]:
        common_dates &= set(dates)

    if not common_dates:
        print(f"No common dates found in {source_dir}")
        return

    common_date_strings = sorted([f"{year:04d}-{month:02d}" for year, month in common_dates])
    print(f"Common date range: {common_date_strings[0]} -> {common_date_strings[-1]} ({len(common_date_strings)} dates)")

    for csv_file, _, _ in parsed_files:
        try:
            lines = csv_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except Exception as exc:
            print(f"Could not read {csv_file.name}: {exc}")
            continue

        data_header_index = None
        for idx, line in enumerate(lines):
            if "gg/mm/aaaa" in line:
                data_header_index = idx
                break

        if data_header_index is None:
            print(f"Skipping {csv_file.name}: no data header found")
            continue

        metadata_lines = lines[:data_header_index]
        data_header_line = lines[data_header_index]
        selected_rows: List[str] = []

        for line in lines[data_header_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(";;"):
                continue
            if ";" in stripped:
                parts = [part.strip().strip('"') for part in stripped.split(";") if part.strip()]
                if len(parts) >= 2 and parts[0]:
                    try:
                        date_value = parts[0]
                        month_year = parse_month_year(date_value)
                        if f"{month_year[0]:04d}-{month_year[1]:02d}" in common_date_strings:
                            selected_rows.append(line)
                    except ValueError:
                        continue

        output_path = output_dir / csv_file.name
        output_lines = metadata_lines + [data_header_line] + selected_rows
        output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(f"Wrote {output_path.name} with {len(selected_rows)} rows")


if __name__ == "__main__":
    for source_dir, output_dir in SIR_WEATHER_TASKS:
        if source_dir.exists():
            normalize_folder(source_dir, output_dir)
        else:
            print(f"Missing folder: {source_dir}")
