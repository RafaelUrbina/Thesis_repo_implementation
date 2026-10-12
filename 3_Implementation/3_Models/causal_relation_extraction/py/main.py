import subprocess
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]
sys.path.insert(0, str(project_root))
from Utils.paths import CAUSAL_RELATION_MODEL_PATH 

Py_PIPELINE_DIR = CAUSAL_RELATION_MODEL_PATH / "py"
# List of script paths in the pipeline
scripts = [
    f"{Py_PIPELINE_DIR}/0_transform_normalization.py",
    f"{Py_PIPELINE_DIR}/1_threshold_filtering.py",
    f"{Py_PIPELINE_DIR}/2_SpatialVars_to_ontology_mapping.py",
    f"{Py_PIPELINE_DIR}/3_spatial_patent_join.py",
    f"{Py_PIPELINE_DIR}/4_join_causals_to_gpkg.py"
]

# Execute each script and capture output
for script in scripts:
    try:
        print(f"Executing {script}...")
        result = subprocess.run(['python', script], capture_output=True, text=True)
        
        # Print stdout from the executed script
        print(result.stdout.strip())
        
        # Check if there was an error during execution
        if result.stderr:
            print(f"Error in {script}:")
            print(result.stderr.strip())
            sys.exit(1)
    
    except Exception as e:
        print(f"Exception occurred while executing {script}:")
        print(str(e))
        sys.exit(1)

print("Pipeline completed successfully.")
