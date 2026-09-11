***

### Developer README & Documentation Guide
**Thesis Project: A Multimodal Data-Driven Framework for Water Distribution Network Resilience**

---

### 1. Project Overview

This repository contains the implementation for a master's thesis project focused on developing a **Decision Support System (DSS)** for the Italian water transportation infrastructure. The project integrates heterogeneous geospatial, environmental, economic, and patent data to analyze risks and inefficiencies in long-distance liquid transportation networks, with a specific case study in the Tuscany region of Italy.

The core objective is to demonstrate how multimodal open data and patent intelligence can be combined to provide actionable insights for policy makers and industry stakeholders, moving from a data-rich but siloed environment to a unified, risk-aware analytical framework.

**Keywords:** `Industry 4.0`, `Industry 5.0`, `Decision Support Systems`, `Machine Learning`, `Natural Language Processing`, `Data Integration`, `Heterogeneous Data`, `Geospatial Analysis`, `Patent Intelligence`.

**Status:** 🚧 *Work in Progress* - This repository is in its middle phase of implementation.

---

### 2. Methodology: The "Tri-Layer" Framework

The project's methodology is structured around three distinct layers of knowledge, which are integrated into a unified data cube for analysis.

1.  **The Detection Layer (Open Data & Remote Sensing):**
    *   **Goal:** To identify buried infrastructure and potential anomalies (like leaks) using remote sensing.
    *   **Data:** Utilizes satellite imagery (e.g., Sentinel-2), Digital Terrain Models (DTMs), and LiDAR-derived indices.
    *   **Implementation:** Focuses on analyzing "soil marks" and "vegetation indices" (e.g., NDWI, MSI) to detect subterranean features.

2.  **The Vulnerability Layer (Environmental & Socio-Economic Risk):**
    *   **Goal:** To assess the risk to the infrastructure from external factors.
    *   **Data:** Cross-references detected locations with hydrogeological risk maps (e.g., from ISPRA's IFFI and IdroGEO), seismic data (INGV), soil type data (Pedologgia), and economic/social data (ISTAT).
    *   **Implementation:** Uses geospatial overlays and joins (e.g., with H3 hex grids) to create a multidimensional risk score.

3.  **The Mitigation Layer (Patent Intelligence):**
    *   **Goal:** To identify technological solutions for the detected risks.
    *   **Data:** Analyzes patents (from Espacenet/Lens.org) related to pipeline integrity, leak detection, and repair.
    *   **Implementation:** Uses Natural Language Processing (NLP) on patent texts to categorize them by the problem they solve (e.g., "causal," "failure," "technical" solutions), creating a bridge between identified risks and potential mitigation technologies.

---

### 3. Repository Structure and Key Workflows

The project is organized into a logical pipeline, mirroring the data science lifecycle.

```
rafaelurbina-thesis_repo_implementation/
├── 0_Deliverables/          # Progress documentation for supervision meetings.
├── 1_Bibliography/          # Literature review, data source metadata,and notes.
├── 2_Pre_implementation_slides/ # Pitch deck and problem-framing presentations.
├── 2_Problem_framing/       # Core research questions and secondary objectives.
├── 3_Implementation/        # Main implementation of the solution.
│   ├── 1_Data/              # Data acquisition, cleaning, and transformation.
│   │   ├── 1_Master/        # Source catalogs and final curated datasets.
│   │   └── 2_Transformation/# Cleaning, standardization, and grid aggregation.
│   ├── 2_Data_analysis/     # Exploratory data analysis and spatial statistics.
│   │   ├── hex_mapping_*/   # Core logic for H3 grid integration.
│   │   ├── patents/         # Patent text processing, NLP, and ontology creation.
│   │   └── spatial_correlation/# Autocorrelation and multivariate analysis.
│   └── 3_Models/            # Machine Learning and causal extraction models.
│       └── causal_relation_extraction/ # Pipeline for linking risks to patents.
└── Utils/                   # Project utilities (e.g., path management).
```

---

### 4. Core Implementation Workflows

#### 4.1. Data Ingestion and Preprocessing (`3_Implementation/1_Data/`)
This is the most crucial step. The workflow ingests a wide variety of structured (tabular) and unstructured (geospatial, raster) data.

1.  **Cataloging Sources:** Metadata for all potential data sources is stored in `1_Bibliography/3_Tables_lit_review/metadata_datasources.json`. This includes geospatial data from regional geoportals (e.g., Regione Toscana), national datasets (ISTAT, ISPRA), and satellite data (Copernicus).
2.  **Acquisition & Cleaning:** Data is downloaded and then cleaned using QGIS and Python scripts located in `3_Implementation/1_Data/2_Transformation/transformation_py/`.
    *   **Key Scripts:**
        *   `0_sequential_cleaning.py`: Drops redundant columns, filters by region (e.g., Firenze province).
        *   `1_ispra_pericolo_join.py`: Joins ISPRA risk layers.
        *   `1_zonal_stats_dtm_to_hex.py`: Calculates statistics (e.g., average elevation, slope) for each H3 hexagon from DTM rasters.
        *   `2_normalize_sir_weather_csv.py`, `3_average_sir_weather_csv.py`: Normalizes and aggregates weather data from the SIR Toscana network.
3.  **Spatial Aggregation:** Data is transformed into a common spatial unit—an **H3 hex grid** (Resolution 10).
    *   `final_sources_categorized.ipynb` outlines the logic for a parsimonious "snowflake" schema:
        *   **Tier A (Static Base):** Soil type, elevation, slope.
        *   **Tier B (Coarse Aggregated Temporal Base):** InSAR subsidence, land use.
        *   **Tier C (Fine Temporal Base):** Aggregates Monthly rain, hydrological stress index.
    *   The transformation scripts in `2_Transformation/` handle the necessary spatial joins and overlays.

#### 4.2. Data Analysis and Feature Engineering (`3_Implementation/2_Data_analysis/`)

1.  **Hex Mapping (`hex_mapping_context/`, `hex_mapping_py/`):**
    *   The core of the feature engineering process.
    *   Scripts like `1_integrate_points_sjoin.py`, `2_integrate_lines.py`, and `3_integrate_polygons_overlay.py` handle different geometry types.
    *   `6_finalize_datacube.py` likely assembles the final data cube.

2.  **Patent Analysis (`patents/`):**
    *   **Term Extraction:** Uses scripts like `ampp_extractor.py` and `unify_terms.py` to categorize patent terms into causal (problems), failure (conditions), and variable (solutions) lists.
    *   **Ontology Creation:** `ontology/ontology.py` drafts an ontology to structure the knowledge, linking problems to technologies.
    *   **Topic Modeling:** `preliminar_view/py/0_pdf_to_text.py`, `1_clean_texts.py`, `2_topic_modeling.py` process patent PDFs to identify latent themes.

3.  **Spatial Correlation (`spatial_correlation/`):**
    *   The goal is to understand spatial dependencies.
    *   `py/spatial_autocorrelation_analysis.py`: Calculates global and local Moran's I to detect clustering of variables like water loss.
    *   `py/multivariate_spatial_analysis.py`: Explores relationships between multiple factors.
    *   **Output:** A set of CSV tables in `tables/` summarizing join counts and autocorrelation results.

#### 4.3. Modeling (`3_Implementation/3_Models/`)

1.  **Causal Relation Extraction (`causal_relation_extraction/`):**
    *   This is the "Mitigation Layer" in action.
    *   `py/0_pipeline.py`: A pipeline designed to take the risk factors and conditions (from the data cube) and link them to the technological terms extracted from patents. This creates a "causal chain" from risk to solution.

---

### 5. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd rafaelurbina-thesis_repo_implementation
    ```

2.  **Environment Setup:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    The project uses a standard `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `geopandas` and `rasterio` entries in the provided `requirements.txt` are empty. You may need to install them manually using `conda` or `pip`.*
    ```bash
    # Example for conda
    conda install geopandas rasterio rasterstats
    ```

4.  **Development Mode Installation (Optional):**
    You can install the project as a package to manage imports more easily. Check the `Utils/` folder for potential setup scripts.
    ```bash
    pip install -e .
    ```

5.  **Configuration:**
    *   Set the working directory to the root of the project.
    *   The `Utils/paths.py` file is likely used to manage relative paths for data and output directories.

---

### 6. Key Technologies

*   **Core:** Python 3.10+
*   **Geospatial:** `geopandas`, `rasterio`, `rasterstats`, `pyproj`, `h3`, `shapely`.
*   **Data Manipulation:** `pandas`, `numpy`.
*   **Machine Learning/NLP:** `scikit-learn`, `sentence-transformers`, `umap-learn`, `hdbscan`.
*   **Visualization:** `matplotlib`, `seaborn`, `plotly`, `wordcloud`.

### 7. License & Authors

*   **Author:** Rafael Ignacio Urbina Hincapie
*   **Affiliation:** Università di Pisa
*   **Supervisors:** Professors Irene Spada and Gualtiero Fantoni
*   **Start Date:** October 23, 2025