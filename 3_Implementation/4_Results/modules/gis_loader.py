"""Cached GeoPackage loading and Folium map construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # The Info and Insights tabs remain usable without GIS extras.
    gpd = None  # type: ignore[assignment]

GEOMETRY_COLUMN = "geometry"
JOIN_KEY = "index"
VARIABLES = (
    "total_failure_chains",
    "unique_technical_triggers",
    "patent_count",
    "causal_socio_economic_index",
)
PALETTES = ("viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "RdYlGn")
MAX_RENDER_FEATURES = 60_000


def discover_geo_packages(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.gpkg"), key=lambda path: path.name.lower())


def discover_tabular_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".xlsx"}
    )


def list_layers(package: Path) -> list[str]:
    if gpd is None:
        return []
    try:
        if hasattr(gpd, "list_layers"):
            return list(gpd.list_layers(package)["name"])
        import fiona

        return list(fiona.listlayers(package))
    except (OSError, ValueError, ImportError):
        return []


def load_layer(package: Path, layer_name: str) -> Any:
    if gpd is None:
        return None
    try:
        return gpd.read_file(package, layer=layer_name)
    except (OSError, ValueError, ImportError):
        try:
            return gpd.read_file(package, layer=layer_name, ignore_geometry=True)
        except (OSError, TypeError, ValueError, ImportError):
            return None


def _prioritize_features(frame: Any, variable: str | None) -> Any:
    """Keep target-bearing features before empty/background features when sampling."""
    if len(frame) <= MAX_RENDER_FEATURES or not variable or variable not in frame.columns:
        return frame

    values = pd.to_numeric(frame[variable], errors="coerce")
    target_rows = values.notna() & values.ne(0)
    target_positions = frame.index[target_rows]
    if len(target_positions) >= MAX_RENDER_FEATURES:
        positions = pd.Series(range(len(target_positions))).iloc[
            [round(position) for position in pd.Series(range(MAX_RENDER_FEATURES))
             .mul((len(target_positions) - 1) / max(1, MAX_RENDER_FEATURES - 1))]
        ].tolist()
        return frame.loc[target_positions[positions]].copy()

    remaining = frame.index[~target_rows]
    remaining_count = MAX_RENDER_FEATURES - len(target_positions)
    return frame.loc[target_positions.union(remaining[:remaining_count])].copy()


def load_join_table(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_excel(path)
    except (OSError, ValueError, ImportError, pd.errors.ParserError):
        return None


def preferred_layer(layers: list[str]) -> str | None:
    for candidate in ("final_data_cube", "final_datacube", "h310_grid_final_datacube"):
        match = next((name for name in layers if candidate.lower() in name.lower()), None)
        if match:
            return match
    return layers[0] if layers else None


def select_variable(frame: Any) -> str | None:
    if frame is None:
        return None
    available = [field for field in VARIABLES if field in frame.columns]
    if available:
        return available[0]
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return next((field for field in numeric if field != JOIN_KEY), None)


def join_on_index(frame: Any, join_table: pd.DataFrame | None) -> Any:
    """Merge a tabular causal source into a GeoDataFrame using the explicit index field."""
    if frame is None or join_table is None or JOIN_KEY not in frame.columns or JOIN_KEY not in join_table.columns:
        return frame
    right = join_table.drop_duplicates(JOIN_KEY)
    return frame.merge(right, on=JOIN_KEY, how="left", suffixes=("", "_causal"))


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _feature_properties(frame: Any) -> list[dict[str, Any]]:
    properties = frame.drop(columns=[GEOMETRY_COLUMN], errors="ignore")
    return [
        {str(key): _safe_value(value) for key, value in row.items()}
        for row in properties.to_dict("records")
    ]


def _map_fields(frame: Any, variable: str | None) -> list[str]:
    """Keep interactive maps compact while retaining spatial and causal context."""
    available = [field for field in frame.columns if field != GEOMETRY_COLUMN]
    preferred = [JOIN_KEY, *VARIABLES]
    preferred.extend(
        field for field in available
        if any(token in field.lower() for token in ("causal", "technical", "trigger", "failure", "patent"))
    )
    if variable:
        preferred.insert(0, variable)
    fields = list(dict.fromkeys(field for field in preferred if field in available))
    return fields[:8] or available[:6]


def _classified_style(frame: Any, variable: str, classification: str, palette_name: str, opacity: float):
    import branca.colormap as cm

    values = pd.to_numeric(frame[variable], errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return lambda feature: {"fillColor": "#38bdf8", "color": "#dbeafe", "weight": 0.6, "fillOpacity": opacity}

    breaks: list[float]
    if classification == "Quantiles":
        breaks = sorted(set(float(value) for value in finite.quantile([0, .2, .4, .6, .8, 1]).tolist()))
    elif classification == "Natural breaks":
        try:
            import mapclassify

            breaks = sorted(set(float(value) for value in mapclassify.NaturalBreaks(finite, k=min(5, finite.nunique())).bins))
            breaks.insert(0, float(finite.min()))
        except (ImportError, ValueError):
            breaks = sorted(set(float(value) for value in finite.quantile([0, .25, .5, .75, 1]).tolist()))
    else:
        breaks = sorted(set(float(value) for value in finite.tolist()))

    minimum, maximum = float(finite.min()), float(finite.max())
    if minimum == maximum:
        maximum = minimum + 1
    palette = getattr(cm.linear, palette_name).scale(minimum, maximum)

    def style(feature: dict[str, Any]) -> dict[str, Any]:
        value = feature["properties"].get(variable)
        try:
            color = palette(float(value)) if value is not None else "#94a3b8"
        except (TypeError, ValueError):
            color = "#94a3b8"
        return {"fillColor": color, "color": "#dbeafe", "weight": 0.6, "fillOpacity": opacity}

    return style


def build_map(
    layers: dict[str, Any],
    *,
    variable: str,
    classification: str,
    palette_name: str,
    opacity_by_layer: dict[str, float],
    join_table: pd.DataFrame | None = None,
) -> str:
    """Return a self-contained Folium map with joined feature tooltips."""
    import folium

    if not layers:
        raise ValueError("At least one map layer is required.")
    spatial_layers = {
        name: frame for name, frame in layers.items()
        if hasattr(frame, "to_crs") and GEOMETRY_COLUMN in frame.columns
    }
    if not spatial_layers:
        raise ValueError("The selected layers contain no geometry.")
    tabular_context = [
        frame for frame in layers.values()
        if not hasattr(frame, "to_crs") and JOIN_KEY in frame.columns
    ]
    if tabular_context:
        enriched = spatial_layers.copy()
        base_name = preferred_layer(list(enriched)) or next(iter(enriched))
        base = enriched[base_name]
        for context in tabular_context:
            base = base.merge(
                context.drop_duplicates(JOIN_KEY),
                on=JOIN_KEY,
                how="left",
                suffixes=("", "_context"),
            )
        spatial_layers[base_name] = base
    first = next(iter(spatial_layers.values())).to_crs(4326)
    bounds = first.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap", control_scale=True)

    for name, frame in reversed(list(spatial_layers.items())):
        current = join_on_index(frame.to_crs(4326), join_table)
        current = _prioritize_features(current, variable)
        fields = _map_fields(current, variable)
        current = current[[*fields, GEOMETRY_COLUMN]]
        current[GEOMETRY_COLUMN] = current[GEOMETRY_COLUMN].simplify(0.00001, preserve_topology=False)
        chosen = variable if variable in current.columns else select_variable(current)
        style = _classified_style(
            current, chosen, classification, palette_name, opacity_by_layer.get(name, 0.75)
        ) if chosen else lambda feature: {"fillColor": "#38bdf8", "color": "#dbeafe", "weight": 0.6, "fillOpacity": opacity_by_layer.get(name, 0.75)}
        folium.GeoJson(
            current,
            name=name,
            style_function=style,
            highlight_function=lambda feature: {"weight": 2, "color": "#f8fafc", "fillOpacity": 0.9},
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=[field.replace("_", " ").title() for field in fields], sticky=False),
        ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap.get_root().render()
