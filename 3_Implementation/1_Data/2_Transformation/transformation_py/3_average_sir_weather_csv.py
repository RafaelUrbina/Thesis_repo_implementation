from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import SIR_WEATHER_TASKS


def parse_date(value: str) -> datetime:
    clean = value.strip().strip('"')
    return datetime.strptime(clean, "%d/%m/%Y")


def normalize_text(value: str) -> str:
    return str(value).strip().strip('"').strip()


def parse_numeric(value: str) -> float | None:
    clean = normalize_text(value)
    if not clean or clean.lower() in {"nan", "none", "@"}:
        return None

    cleaned = re.sub(r"\s+", "", clean)
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    cleaned = re.sub(r"[^0-9\-\.eE]", "", cleaned)
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def majority_value(values: Sequence[str]) -> str:
    cleaned = [normalize_text(v) for v in values if normalize_text(v)]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


def aggregate_monthly_rows(rows: List[Tuple[datetime, List[str]]]) -> List[Tuple[str, List[str]]]:
    grouped: dict[Tuple[int, int], List[List[str]]] = {}
    for date_value, values in rows:
        key = (date_value.year, date_value.month)
        grouped.setdefault(key, []).append(values)

    monthly_rows: List[Tuple[str, List[str]]] = []
    for (year, month), month_rows in sorted(grouped.items()):
        column_count = max(len(values) for values in month_rows) if month_rows else 0
        aggregated_values: List[str] = []
        for col_idx in range(column_count):
            column_values = [row[col_idx] if col_idx < len(row) else "" for row in month_rows]
            numeric_values = [parse_numeric(value) for value in column_values]
            numeric_candidates = [value for value in numeric_values if value is not None]
            text_candidates = [normalize_text(value) for value in column_values if parse_numeric(value) is None and normalize_text(value)]

            if numeric_candidates and not text_candidates:
                aggregated_values.append(str(round(sum(numeric_candidates) / len(numeric_candidates), 6)))
            elif text_candidates:
                aggregated_values.append(majority_value(text_candidates))
            else:
                aggregated_values.append("")

        monthly_rows.append((f"01/{month:02d}/{year}", aggregated_values))

    return monthly_rows


def aggregate_csv_file(source_path: Path, output_path: Path) -> None:
    lines = source_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    data_header_index = None
    for idx, line in enumerate(lines):
        if "gg/mm/aaaa" in line:
            data_header_index = idx
            break

    if data_header_index is None:
        raise ValueError(f"No data header found in {source_path}")

    metadata_lines = lines[:data_header_index]
    header_line = lines[data_header_index]
    data_rows: List[Tuple[datetime, List[str]]] = []

    for line in lines[data_header_index + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith(";;"):
            continue

        parts = [normalize_text(part) for part in stripped.split(";")]
        if not parts or not parts[0]:
            continue
        try:
            date_value = parse_date(parts[0])
        except ValueError:
            continue

        data_rows.append((date_value, parts[1:]))

    if not data_rows:
        raise ValueError(f"No data rows found in {source_path}")

    monthly_rows = aggregate_monthly_rows(data_rows)
    output_lines = metadata_lines + [header_line]
    for date_value, values in monthly_rows:
        output_lines.append(";".join([date_value, *values]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def aggregate_weather_folders() -> None:
    for _, midpoint_dir in SIR_WEATHER_TASKS:
        if not midpoint_dir.exists():
            print(f"Missing folder: {midpoint_dir}")
            continue

        midpoint_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(midpoint_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {midpoint_dir}")
            continue

        print(f"Processing {midpoint_dir.name} with {len(csv_files)} files")
        for csv_file in csv_files:
            try:
                aggregate_csv_file(csv_file, midpoint_dir / csv_file.name)
                print(f"Wrote monthly summary for {csv_file.name}")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Skipping {csv_file.name}: {exc}")


if __name__ == "__main__":
    aggregate_weather_folders()
