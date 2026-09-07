# Add the project root to the Python path to allow for absolute imports

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents["# of parents to go up"]
sys.path.insert(0, str(project_root))

from Utils.paths import (variable)