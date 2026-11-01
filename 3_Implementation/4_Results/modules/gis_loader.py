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
    "causal_socioeconomic_index",
    "causal_socio_economic_index",
)
PALETTES = ("viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "RdYlGn", "Spectral")
MAX_RENDER_FEATURES = 100_000


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
        import fiona
        return list(fiona.listlayers(package))
    except (OSError, ValueError, ImportError, LookupError):
        try:
            if hasattr(gpd, "list_layers"):
                res = gpd.list_layers(package)
                if "name" in res.columns:
                    return list(res["name"])
                return list(res.iloc[:, 0])
        except Exception:
            pass
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
    if len(frame) <= MAX_RENDER_FEATURES:
        return frame

    limit = max(1, min(int(limit), MAX_RENDER_FEATURES))
    if variable and variable in frame.columns:
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
    else:
        target_rows = pd.Series(False, index=frame.index)
    target_positions = frame.index[target_rows]
    background_positions = frame.index[~target_rows]
    target_count = min(len(target_positions), limit)
    target_indices = [
        round(position)
        for position in pd.Series(range(target_count)).mul(
            (len(target_positions) - 1) / max(1, target_count - 1)
        )
    ]
    selected_positions = list(target_positions[target_indices])

    remaining = limit - len(selected_positions)
    if remaining > 0 and len(background_positions):
        background_indices = [
            round(position)
            for position in pd.Series(range(min(remaining, len(background_positions)))).mul(
                (len(background_positions) - 1) / max(1, min(remaining, len(background_positions)) - 1)
            )
        ]
        selected_positions.extend(background_positions[background_indices])
    return frame.loc[selected_positions].copy()


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


def _map_fields(
    frame: Any,
    variable: str | None,
    joined_fields: set[str] | None = None,
) -> list[str]:
    """Expose the core attributes and fields originating from joined tables."""
    available = [field for field in frame.columns if field != GEOMETRY_COLUMN]
    core_fields = [
        JOIN_KEY,
        variable,
        "total_failure_chains",
        "unique_technical_triggers",
        "all_unique_attributes",
        "all_unique_failures",
    ]
    fields: list[str] = []
    for field in core_fields:
        if field and field in available and field not in fields:
            fields.append(field)
    joined_fields = joined_fields or set()
    fields.extend(
        field for field in available
        if field not in fields
        and (
            field in joined_fields
            or str(field).casefold().endswith(("_causal", "_context"))
        )
    )
    return fields


def _palette_scale(palette_name: str, minimum: float, maximum: float):
    import branca.colormap as cm

    palette = getattr(cm.linear, palette_name, None)
    if palette is None:
        palette = getattr(cm.linear, f"{palette_name}_11", None)
    if palette is None:
        import matplotlib

        matplotlib_palette = matplotlib.colormaps[palette_name]
        palette = cm.LinearColormap(
            [matplotlib_palette(position / 255) for position in range(256)],
            vmin=minimum,
            vmax=maximum,
        )
    return palette.scale(minimum, maximum)


def _classified_style(
    frame: Any,
    variable: str,
    classification: str,
    palette_name: str,
    opacity: float,
    bin_count: int = 5,
):
    values = pd.to_numeric(frame[variable], errors="coerce")
    finite = values.dropna()
    if finite.empty:
        categories = frame[variable].dropna().astype(str).unique().tolist()
        category_palette = _palette_scale(palette_name, 0, max(1, len(categories) - 1))
        category_colors = {
            category: category_palette(position)
            for position, category in enumerate(categories)
        }

        def categorical_style(feature: dict[str, Any]) -> dict[str, Any]:
            value = feature["properties"].get(variable)
            color = category_colors.get(str(value), "#94a3b8")
            return {"fillColor": color, "color": "#1e293b", "weight": 0.6, "fillOpacity": opacity}

        return categorical_style

    bin_count = max(3, min(int(bin_count), 10))
    breaks: list[float]
    if classification == "Quantiles":
        probabilities = pd.Series(range(bin_count + 1), dtype="float64") / bin_count
        breaks = sorted(set(float(value) for value in finite.quantile(probabilities).tolist()))
    elif classification == "Natural breaks":
        try:
            import mapclassify

            breaks = sorted(
                set(
                    float(value)
                    for value in mapclassify.NaturalBreaks(
                        finite, k=min(bin_count, finite.nunique())
                    ).bins
                )
            )
            breaks.insert(0, float(finite.min()))
        except (ImportError, ValueError):
            probabilities = pd.Series(range(bin_count + 1), dtype="float64") / bin_count
            breaks = sorted(set(float(value) for value in finite.quantile(probabilities).tolist()))
    else:
        breaks = sorted(set(float(value) for value in finite.tolist()))

    minimum, maximum = float(finite.min()), float(finite.max())
    if minimum == maximum:
        maximum = minimum + 1
    palette = _palette_scale(palette_name, minimum, maximum)
    if len(breaks) < 2:
        breaks = [minimum, maximum]
    colors = [
        palette((lower + upper) / 2)
        for lower, upper in zip(breaks[:-1], breaks[1:])
    ]

    def style(feature: dict[str, Any]) -> dict[str, Any]:
        value = feature["properties"].get(variable)
        try:
            numeric_value = float(value)
            bin_index = min(
                len(colors) - 1,
                max(0, next(
                    (index for index, boundary in enumerate(breaks[1:])
                     if numeric_value <= boundary),
                    len(colors) - 1,
                )),
            )
            color = colors[bin_index]
        except (TypeError, ValueError):
            color = "#94a3b8"
        return {"fillColor": color, "color": "#1e293b", "weight": 0.6, "fillOpacity": opacity}

    return style


def build_map(
    layers: dict[str, Any],
    *,
    variable: str | dict[str, Any] | None = None,
    classification: str | dict[str, Any] = "Quantiles",
    palette_name: str | dict[str, Any] = "viridis",
    bin_count: int | dict[str, Any] = 5,
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
    joined_source_fields = {
        field
        for context in tabular_context
        for field in context.columns
        if field not in {JOIN_KEY, GEOMETRY_COLUMN}
    }
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

    def layer_setting(
        setting: str | dict[str, Any] | None,
        layer_name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(setting, dict):
            return setting.get(layer_name, default)
        return setting

    for name, frame in reversed(list(spatial_layers.items())):
        current = join_on_index(frame.to_crs(4326), join_table)
        layer_variable = layer_setting(variable, name)
        layer_classification = layer_setting(classification, name, "Quantiles") or "Quantiles"
        layer_palette = layer_setting(palette_name, name, "viridis") or "viridis"
        layer_bin_count = layer_setting(bin_count, name, 5) or 5
        current = _prioritize_features(current, layer_variable, render_limit)
        joined_fields = {
            candidate
            for source_field in joined_source_fields.union(
                join_table.columns
                if join_table is not None
                else []
            )
            if source_field not in {JOIN_KEY, GEOMETRY_COLUMN}
            for candidate in (
                source_field,
                f"{source_field}_causal",
                f"{source_field}_context",
            )
            if candidate in current.columns
        }
        fields = _map_fields(current, layer_variable, joined_fields)
        current = current[[*fields, GEOMETRY_COLUMN]]
        current[GEOMETRY_COLUMN] = current[GEOMETRY_COLUMN].simplify(
            0.00005, preserve_topology=True
        )
        chosen = layer_variable if layer_variable in current.columns else select_variable(current)
        style = _classified_style(
            current, chosen, layer_classification, layer_palette,
            opacity_by_layer.get(name, 0.75), layer_bin_count,
        ) if chosen else lambda feature: {"fillColor": "#38bdf8", "color": "#1e293b", "weight": 0.6, "fillOpacity": opacity_by_layer.get(name, 0.75)}
        folium.GeoJson(
            current,
            name=name,
            style_function=style,
            highlight_function=lambda feature: {"weight": 2, "color": "#f8fafc", "fillOpacity": 0.9},
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=[field.replace("_", " ").title() for field in fields], sticky=False),
        ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap.get_root().render()
