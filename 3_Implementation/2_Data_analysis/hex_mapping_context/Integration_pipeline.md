# Data Cube Integration Pipeline: Layer-to-Script Mapping

This document outlines the step-by-step process for integrating all source data layers into the master `h310grid`. The pipeline is organized into a series of modular Python scripts, each responsible for a specific type of spatial operation.

---

### **Step 1**
**Objective:** Prepare the foundational analysis grid.
- **Action:** Loads the `h310grid` layer.

| Layer Name             | Role                               |
| :--------------------- | :--------------------------------- |
| `h310grid`               | Base grid for the entire analysis. |

---

### **Step 2: `1_integrate_points_sjoin.py`**
**Objective:** Aggregate point data to calculate density, presence, or summary statistics within each hexagon.
- **Action:** Loads the `master_grid.gpkg` and performs spatial joins (`gpd.sjoin`) for each point layer.
- **Operation:** Groups the joined data by `h3_index` and applies the specified aggregation rule (count, sum, mean, max, mode).
- **Output:** `master_grid_with_points.gpkg`

| Layer Name                             | Spatial Operation | Statistical Aggregation Rule(s)        |
| :------------------------------------- | :---------------- | :------------------------------------- |
| `info_lotti_multipmpoint`              | `gpd.sjoin()`     | `sum` (importo_lo), `first` (tipo_disse), (from the current hexagon or the 1-ring scale, 1-level neighbors) |
| `pubacq_acq_fontanello_aq_attivipoint` | `gpd.sjoin()`     | `count`                               , (from the current hexagon or the 1-ring scale, 1-level neighbors) |
| `fontanellipoint`                      | `gpd.sjoin()`     | `count`                               ,(from the current hexagon or the 1-ring scale, 1-level neighbors) |
| `frane_piff_toscana_opendata`          | `gpd.sjoin()`     | `count`, `mode` (tipo_movimento)    , (from the current hexagon or the 1-ring scale, 1-level neighbors)   |
| `_peaks`                               | `gpd.sjoin()`     | `max` (ele)                         ,(from the current hexagon or the 1-ring scale, 1-level neighbors)   |
| `_places`                              | `gpd.sjoin()`     | `first` (name)                         |
| `seismic_points_utm32n`                | `gpd.sjoin()`     | `count`, `max` (MwDef), `mean` (MwDef) , (from the current hexagon or the 1-ring scale, 1-level neighbors) |

---

### **Step 3: `2_integrate_lines.py`**
**Objective:** Calculate metrics from linear features, such as density and distance.
- **Action:** Loads the grid and line layers. Uses `gpd.overlay` for density and `gpd.GeoSeries.distance` for proximity.
- **Operation:** Calculates total line length per hexagon for density. Calculates distance from each hexagon centroid to the nearest line.
- **Output:** `master_grid_with_lines.gpkg`

| Layer Name     | Spatial Operation                 | Statistical Aggregation Rule                               |
| :------------- | :-------------------------------- | :--------------------------------------------------------- |
| `_road`        | `gpd.overlay(how='intersection')` | `sum` of intersected line lengths / hexagon area.          |
| `waterwaysL`   | `gpd.GeoSeries.distance()`        | Distance from hexagon centroid to nearest line. `mode` of waterway type. |

---

### **Step 4: `3_integrate_polygons_overlay.py`**
**Objective:** Transfer attributes from complex polygon layers using area-based or priority rules.
- **Action:** Loads the grid and polygon layers. Uses `gpd.overlay(how='intersection')` to fragment the polygons against the grid.
- **Operation:** For each hexagon, calculates the area of intersecting fragments to determine the dominant category (**Majority Area Rule**) or highest priority value (**Ordinal Max Priority Rule**).
- **Output:** `master_grid_with_polygons.gpkg`

| Layer Name                                            | Statistical Aggregation Rule                                     |
| :---------------------------------------------------- | :--------------------------------------------------------------- |
| `profondita_utile_per_le_radici_cm`                   | Majority Area Rule on `profond` class.                           |
| `pietrosita_superficiale_`                            | Majority Area Rule on `ciottoli` class.                          |
| `natural_`                                            | Majority Area Rule on `natural` class (and top 2).               |
| `aggr_mosaicatura_ispra_2020_pericolosita_idraulica_firenze` | Ordinal Max Priority Rule on `pericolo` value.                   |
| `ksat_30__conducibilita_idraulica_satura_sezione_030_cm` | Majority Area Rule on `ksat_30` class.                           |
| `ksat_150__conducibilita_idraulica_satura_sezione_0150_cm` | Majority Area Rule on `ksat_150` class.                          |
| `interferenza_climatica_per_quota`                    | Majority Area Rule on `interf_cli` class.                        |
| `interferenza_climatica_per_deficit_idrico`           | Majority Area Rule on `deficit` class.                           |
| `gruppo_idrologico_usda`                              | Majority Area Rule on `gi` class.                                |
| `franosita__di_superficie_interessata_da_frane`       | Majority Area Rule on `franosita` class.                         |
| `frane_poly_toscana_opendata`                         | `sum` of intersected landslide area.                             |
| `velocita_movimento_nel_piano`                        | Majority Area Rule on `v_eozn` label.                            |
| `forest_`                                             | Majority Area Rule on `landuse` class.                           |
| `unita_di_paesaggio`                                  | Majority Area Rule on `udp` (and top 3).                         |
| `sottosistemi_di_paesaggio`                           | Majority Area Rule on `sst` (and top 3).                         |
| `sistemi_di_paesaggio`                                | Majority Area Rule on `sg` (and top 3).                          |
| `soil_region`                                         | Majority Area Rule on `srg` (and top 3).                         |
| `fertilita_chimica_dellorizzonte_superficiale`        | Majority Area Rule on `fertilita` class.                         |
| `erosione_potenziale_tha`                             | Majority Area Rule on `erosione` class.                          |
| `drenaggio_interno`                                   | Majority Area Rule on `drenag` class.                            |
| `capacita_duso_e_fertilita_dei_suoli`                 | Majority Area Rule on `lcc_classe`.                              |
| `building_`                                           | `sum` of intersected building area; `list` of building types.    |
| `salinita_dellorizzonte_superficiale_mscm__125`       | Majority Area Rule on `salinita` class.                          |
| `salinita_dellorizzonte_sottosuperficiale_1m_mscm__125` | Majority Area Rule on `sal_prof` class.                          |
| `rocciosita_`                                         | Majority Area Rule on `rocciosita` class.                        |
| `rischio_di_inondazione_con_tempo_di_ritorno_inferiore_a_30_anni` | Majority Area Rule on `inondaz` class.                           |
| `awc__available_water_capacity`                       | Majority Area Rule on `awc` class.                               |
| `mosaicatura_ispra_2024_pericolosita_frana_pai`       | Ordinal Max Priority Rule on `per_fr_ita` value.                 |
| `celle_soli_PS_descendenti`            | `max` (binary flag), `mean` (ave_vdesc)  |
| `celle_soli_PS_ascendenti`             | `max` (binary flag), `mean` (ave_vasc)   |

---

### **Step 5: `4_integrate_polygons_centroid.py`**
**Objective:** Assign attributes from large administrative polygons (like municipalities) to the hexagons they contain.
- **Action:** Loads the grid and communal-level data. Uses `gpd.sjoin(predicate='within')` on the hexagon centroids.
- **Operation:** Transfers attributes from the encompassing polygon to each hexagon. This is a direct 1-to-many assignment.
- **Output:** `master_grid_with_communal.gpkg`

| Layer Name                               | Spatial Operation                               | Statistical Aggregation Rule         |
| :--------------------------------------- | :---------------------------------------------- | :----------------------------------- |
| `tabular_comunal_census`                 | `gpd.sjoin(predicate='within')` on H3 Centroids | Centroid Structural Assignment.      |
| `tabular_comunal_economical_princ`       | `gpd.sjoin(predicate='within')` on H3 Centroids | Centroid Structural Assignment.      |
| `tabular_comunal_economical_reddito`     | `gpd.sjoin(predicate='within')` on H3 Centroids | Centroid Structural Assignment.      |
| `tabular_comunal_economical_distrib`     | `gpd.sjoin(predicate='within')` on H3 Centroids | Centroid Structural Assignment.      |
| `tabular_comunal_economical_indice_comp` | `gpd.sjoin(predicate='within')` on H3 Centroids | Centroid Structural Assignment.      |
| `pendolarismo_inflow_comunal`            | `gpd.sjoin(predicate='within')` on H3 Centroids | Centroid Structural Assignment.      |

---

### **Step 6: `5_integrate_stations.py`**
**Objective:** Transfer data from sparse monitoring stations to the continuous grid.
- **Action:** Loads the grid and the `_aggregated` station layers.
- **Operation:**
    - **Interpolation (IDW/Kriging):** For continuous variables like temperature and rain, create a continuous surface and sample the value at each hexagon's centroid.
    - **Nearest Neighbor:** For proximity metrics, calculate the distance from each hexagon to the nearest station (`gpd.sjoin_nearest`).
- **Output:** `master_grid_with_stations.gpkg`

| Layer Name                                 | Spatial Operation / Approach                          |
| :----------------------------------------- | :---------------------------------------------------- |
| `termometri_stations_firenze_aggregated`   | Spatial Interpolation or Nearest-Neighbor Join.       |
| `idrometri_stations_firenze_aggregated`    | Nearest-neighbor distance calculation (`sjoin_nearest`). |
| `cf_pluviometri_aggregated`                | Spatial Interpolation.                                |
| `anemometri_stations_firenze_aggregated`   | Spatial Interpolation or Nearest-Neighbor Join.       |

---

### **Step 7: `6_finalize_datacube.py`**
**Objective:** Perform the final merge and cleanup.
- **Action:** Loads the output from all previous steps.
- **Operation:**
    - **Direct Join:** Merges the `h3_grid_with_dtm_stats` data using the `h3_index`.
    - **Concatenation:** Joins all intermediate grid files into a single, wide GeoDataFrame.
    - **Cleanup:** Handles any remaining `NaN` values, checks data types, and saves the final data cube.
- **Output:** `FINAL_DATA_CUBE.gpkg`

| Layer Name                 | Spatial Operation         | Statistical Aggregation Rule |
| :------------------------- | :------------------------ | :--------------------------- |
| `h3_grid_with_dtm_stats`   | Direct Attribute Merge    | Direct 1-to-1 Join.          |
