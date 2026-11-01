import sys
from pathlib import Path


def add_project_root(levels_up: int = 1) -> Path:
    """Add the repository root to ``sys.path`` and return it.

    ``relative_import.py`` lives directly below the repository root in
    ``Utils``.  ``levels_up=1`` therefore resolves its parent directory.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[levels_up - 1]
    root_string = str(project_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    return project_root