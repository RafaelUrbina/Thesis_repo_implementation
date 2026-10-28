"""Discovery and safe rendering helpers for insight assets."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".html", ".htm"}


def discover_plots(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.glob("*") if path.is_file() and path.suffix.lower() in SUPPORTED),
        key=lambda path: path.name.lower(),
    ) if folder.exists() else []


def slots_for_layout(layout: str) -> int:
    return {"1 Plot": 1, "2 Plots (Side-by-Side)": 2, "3 Plots (Row)": 3, "2x2 Grid (4 Plots)": 4}[layout]


def render_plot(path: Path, *, height: int = 480) -> None:
    if path.suffix.lower() in {".html", ".htm"}:
        try:
            st.html(path.read_text(encoding="utf-8"), unsafe_allow_javascript=True)
        except (OSError, UnicodeDecodeError) as error:
            st.error(f"Could not render {path.name}: {error}", icon=":material/error:")
    else:
        st.image(str(path), width="stretch")
