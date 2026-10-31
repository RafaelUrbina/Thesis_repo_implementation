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


def _prioritize_features(
    frame: Any,
    variable: str | None,
    limit: int = MAX_RENDER_FEATURES,
) -> Any:
    """Keep target-bearing features before empty/background features when sampling."""
    limit = max(1, min(int(limit), MAX_RENDER_FEATURES))
    if len(frame) <= limit or not variable or variable not in frame.columns:
        return frame

    values = frame[variable]
    numeric_values = pd.to_numeric(values, errors="coerce")
    target_rows = numeric_values.notna() & numeric_values.ne(0)
    categorical_values = values.astype("string").str.strip()
    target_rows |= (
        numeric_values.isna()
        & categorical_values.notna()
        & categorical_values.ne("")
        & categorical_values.ne("0")
    )
    target_positions = frame.index[target_rows]
    if len(target_positions) >= limit:
        positions = pd.Series(range(len(target_positions))).iloc[
            [round(position) for position in pd.Series(range(limit))
             .mul((len(target_positions) - 1) / max(1, limit - 1))]
        ].tolist()
        return frame.loc[target_positions[positions]].copy()

    return frame.loc[target_positions].copy()


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


def discover_variables(*frames: Any) -> list[str]:
    """Return selectable data fields from spatial layers and joined tables."""
    available = {
        str(field)
        for frame in frames
        if frame is not None
        for field in getattr(frame, "columns", [])
        if field not in {GEOMETRY_COLUMN, JOIN_KEY}
    }
    if len(frames) > 1 and frames[-1] is not None:
        base_fields = {
            str(field)
            for frame in frames[:-1]
            if frame is not None
            for field in getattr(frame, "columns", [])
        }
        join_fields = {
            str(field)
            for field in getattr(frames[-1], "columns", [])
            if field not in {GEOMETRY_COLUMN, JOIN_KEY}
        }
        available.update(f"{field}_causal" for field in join_fields.intersection(base_fields))
    return [
        *[field for field in VARIABLES if field in available],
        *sorted(available.difference(VARIABLES), key=str.casefold),
    ]


def join_on_index(frame: Any, join_table: pd.DataFrame | None) -> Any:
    """Merge a tabular causal source into a GeoDataFrame using the explicit index field."""
    if frame is None or join_table is None or JOIN_KEY not in frame.columns or JOIN_KEY not in join_table.columns:
        return frame
    right = join_table.drop(columns=[GEOMETRY_COLUMN], errors="ignore").drop_duplicates(JOIN_KEY)
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
    """Expose the spatial and joined attributes in feature tooltips."""
    available = [field for field in frame.columns if field != GEOMETRY_COLUMN]
    preferred = [variable, JOIN_KEY, *VARIABLES] if variable else [JOIN_KEY, *VARIABLES]
    return list(dict.fromkeys(field for field in [*preferred, *available] if field in available))


def _classified_style(frame: Any, variable: str, classification: str, palette_name: str, opacity: float):
    import branca.colormap as cm

    values = pd.to_numeric(frame[variable], errors="coerce")
    finite = values.dropna()
    if finite.empty:
        categories = frame[variable].dropna().astype(str).unique().tolist()
        category_palette = getattr(cm.linear, palette_name).scale(0, max(1, len(categories) - 1))
        category_colors = {
            category: category_palette(position)
            for position, category in enumerate(categories)
        }

        def categorical_style(feature: dict[str, Any]) -> dict[str, Any]:
            value = feature["properties"].get(variable)
            color = category_colors.get(str(value), "#94a3b8")
            return {"fillColor": color, "color": "#dbeafe", "weight": 0.6, "fillOpacity": opacity}

        return categorical_style

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
    render_limit: int = MAX_RENDER_FEATURES,
    show_basemap: bool = True,
    basemap_opacity: float = 1.0,
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
    fmap = folium.Map(location=center, zoom_start=9, tiles=None, control_scale=True)
    if show_basemap:
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            opacity=max(0.0, min(1.0, float(basemap_opacity))),
            control=True,
        ).add_to(fmap)

    for name, frame in reversed(list(spatial_layers.items())):
        current = join_on_index(frame.to_crs(4326), join_table)
        current = _prioritize_features(current, variable, render_limit)
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
