"""LaTeX source discovery and lightweight metadata extraction."""

from __future__ import annotations

import re
from pathlib import Path


def discover_tex_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.tex")) if folder.exists() else []


def read_tex_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
    except OSError as error:
        return f"% Unable to read {path.name}: {error}"


def latex_to_plain_text(source: str) -> str:
    """Make a readable overview without pretending to compile the thesis."""
    text = re.sub(r"%.*", "", source)
    text = re.sub(r"\\(textbf|textit|emph|section|subsection|chapter)\*?\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()
