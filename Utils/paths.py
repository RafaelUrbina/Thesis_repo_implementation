from pathlib import Path

# Project paths and constants
REPO_ROOT = Path(__file__).resolve().parent.parent

MASTER_DATA_PATH = REPO_ROOT / "3_Implementation/1_Data/1_Master"
MIDPOINTS_PATH = REPO_ROOT / "3_Implementation/1_Data/2_Transformation/midpoints"
GRID_PATH = REPO_ROOT / "3_Implementation/1_Data/2_Transformation/final_aggregation_grid"

SIR_TOSCANA_PATH = MASTER_DATA_PATH / "weather/sir_toscana"
GPKG_PATH = REPO_ROOT / "3_Implementation/1_Data/2_Transformation/static_geospatial"
RASTER_PATH = REPO_ROOT / "3_Implementation/1_Data/2_Transformation/rasters"

PLOTS_SPATIAL_AUTOCORRELATION_PATH = REPO_ROOT / "3_Implementation/2_Data_analysis/spatial_correlation/plots"
TABLE_SPATIAL_AUTOCORRELATION_PATH = REPO_ROOT / "3_Implementation/2_Data_analysis/spatial_correlation/tables"

SIR_WEATHER_TASKS = (
    (
        SIR_TOSCANA_PATH / "sir_idrometry_toscana_datasets",
        MIDPOINTS_PATH / "sir_idrometry_toscana_datasets",
    ),
    (
        SIR_TOSCANA_PATH / "sir_rain_toscana_datasets",
        MIDPOINTS_PATH / "sir_rain_toscana_datasets",
    ),
    (
        SIR_TOSCANA_PATH / "sir_temperature_toscana_datasets",
        MIDPOINTS_PATH / "sir_temperature_toscana_datasets",
    ),
    (
        SIR_TOSCANA_PATH / "sir_wind_toscana_datasets",
        MIDPOINTS_PATH / "sir_wind_toscana_datasets",
    ),
)

PENDOLARISMO_PATH = MASTER_DATA_PATH / "pendolarismo"

CAUSAL_LINKS_PATH = REPO_ROOT / "3_Implementation/2_Data_analysis/patents/causal_links"

PATENTS_OUTPUT_TERMS_PATH = REPO_ROOT / "3_Implementation/2_Data_analysis/patents/links_lists/py/outputterms"
ONTOLOGY_LISTS_PATH = REPO_ROOT / "3_Implementation/2_Data_analysis/patents/ontology/py/lists"
ONTOLOGY_OUTPUT_PATH = REPO_ROOT / "3_Implementation/2_Data_analysis/patents/ontology/output"
