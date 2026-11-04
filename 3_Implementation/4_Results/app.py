"""Interactive thesis dashboard: From Patents to Places."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_sortables import sort_items
except ImportError:
    sort_items = None

from modules.gis_loader import (
    GEOMETRY_COLUMN,
    MAX_RENDER_FEATURES,
    PALETTES,
    build_map,
    discover_geo_packages,
    discover_tabular_files,
    discover_variables,
    list_layers,
    load_join_table,
    load_layer,
    preferred_layer,
    select_variable,
)
from modules.plot_manager import discover_plots, load_plot_asset, render_plot

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAP_CONTENTS = ROOT / "map_contents"
PLOTS = ROOT / "insights_plots"
CAROUSEL = ROOT / "carousel"
THESIS = ROOT / "thesis"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

st.set_page_config(
    page_title="From Patents to Places",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def initialize_state() -> None:
    st.session_state.setdefault("background_name", None)
    st.session_state.setdefault("map_layer_order", [])
    st.session_state.setdefault("plot_order", [])
    st.session_state.setdefault("selected_plot", None)
    st.session_state.setdefault("active_page", "landpage")
    st.session_state.setdefault("thesis_page_select", 1)


@st.cache_data(ttl="15m", max_entries=8)
def cached_layer_names(package: str) -> list[str]:
    return list_layers(Path(package))


@st.cache_data(ttl="15m", max_entries=24)
def cached_layer(package: str, layer_name: str):
    return load_layer(Path(package), layer_name)


@st.cache_data(ttl="15m", max_entries=16)
def cached_join_table(path: str):
    return load_join_table(Path(path))


@st.cache_data(ttl="15m", max_entries=8)
def cached_geo_packages(folder: str) -> list[Path]:
    return discover_geo_packages(Path(folder))


@st.cache_data(ttl="15m", max_entries=8)
def cached_tabular_files(folder: str) -> list[Path]:
    return discover_tabular_files(Path(folder))


@st.cache_data(ttl="15m", max_entries=8)
def cached_plot_paths(folder: str) -> list[Path]:
    return discover_plots(Path(folder))


def preload_plot_assets(folder: str) -> None:
    for path in cached_plot_paths(folder):
        load_plot_asset(str(path))


@st.cache_data(ttl="15m", max_entries=4, show_spinner=False)
def cached_pdf(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(ttl="15m", max_entries=4, show_spinner=False)
def cached_pdf_pages(path: str) -> list[bytes]:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    pages = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(dpi=150)
        pages.append(pix.tobytes("png"))
    return pages


@st.cache_data(ttl="15m", max_entries=16)
def image_data_uri(path_string: str) -> str:
    path = Path(path_string)
    mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}.get(path.suffix.lower(), "png")
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def asset_images(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    ) if folder.exists() else []


def apply_theme() -> None:
    backgrounds = asset_images(ASSETS / "backgrounds")
    names = [path.name for path in backgrounds]
    with st.sidebar:
        selected_name = st.selectbox(
            "Background theme", ["Automatic"] + names, key="background_name",
            help="Choose one of the supplied background images.",
        )
    selected = next((path for path in backgrounds if path.name == selected_name), backgrounds[0] if backgrounds else None)
    if selected:
        try:
            background = image_data_uri(str(selected))
            background_css = (
                f"background-image: linear-gradient(rgba(7, 14, 30, .74), rgba(7, 14, 30, .92)), "
                f"url('{background}');"
            )
        except OSError:
            background_css = "background: linear-gradient(135deg, #081225 0%, #142b48 100%);"
    else:
        background_css = "background: linear-gradient(135deg, #081225 0%, #142b48 100%);"
    
    background_uris = [image_data_uri(str(path)) for path in backgrounds]
    uris_js = json.dumps(background_uris)

    st.markdown(
        f"""
        <style>
        :root {{ --glass: rgba(15, 30, 53, .70); --line: rgba(180, 218, 255, .18); }}
        .stApp {{ {background_css} background-size: cover; background-attachment: fixed; color: #edf6ff; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: rgba(5, 14, 29, .82); border-right: 1px solid var(--line); }}
        .block-container {{ padding-top: 2rem; max-width: 1440px; }}
        .glass-card {{
            background: var(--glass); border: 1px solid var(--line); border-radius: 18px;
            padding: 1.25rem 1.5rem; box-shadow: 0 12px 40px rgba(0,0,0,.2);
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); margin-bottom: 1rem;
            width: 100% !important;
            box-sizing: border-box !important;
        }}
        .plot-media-frame, .plot-media-frame img {{
            width: 100% !important; max-width: 100% !important; height: auto !important;
            overflow: hidden !important;
        }}
        .eyebrow {{ color: #7dd3fc; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
        h1, h2, h3 {{ letter-spacing: -.02em; }}

        /* Structural full 100% width navbar reset */
        div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]) {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        [data-testid="stRadio"] {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        [data-testid="stRadio"] > div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            width: 100% !important;
            max-width: 100% !important;
            background: var(--glass);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.4rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 12px 40px rgba(0,0,0,.3);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-sizing: border-box !important;
        }}
        [data-testid="stRadio"] > div[role="radiogroup"] > label {{
            flex: 1 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            background: transparent !important;
            border-radius: 14px !important;
            padding: 0.7rem 1rem !important;
            margin: 0 !important;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            z-index: 2;
        }}
        [data-testid="stRadio"] > div[role="radiogroup"] > label:not(:last-child)::after {{
            content: "";
            position: absolute;
            right: 0;
            top: 20%;
            height: 60%;
            width: 1px;
            background: linear-gradient(to bottom, transparent, rgba(180, 218, 255, 0.4), transparent);
        }}
        [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
            background: rgba(125, 211, 252, 0.15) !important;
        }}
        [data-testid="stRadio"] input[type="radio"] {{
            display: none !important;
        }}
        [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {{
            font-weight: 600 !important;
            font-size: 0.98rem !important;
            letter-spacing: 0.04em !important;
            color: #edf6ff !important;
            margin: 0 !important;
            text-transform: capitalize;
        }}

        /* Navbar background cycling animation layers */
        @keyframes nav-bg-cycle {{
            0%, 3% {{ opacity: 0; }}
            15%, 25% {{ opacity: 1; }}
            35%, 100% {{ opacity: 0; }}
        }}
        .nav-bg-layer {{
            position: absolute;
            inset: 0;
            z-index: -2;
            background-size: cover;
            background-position: center;
            opacity: 0;
        }}
        </style>
        <script>
        (function() {{
            const uris = {uris_js};
            if (uris && uris.length > 0) {{
                const interval = setInterval(() => {{
                    const group = document.querySelector('[data-testid="stRadio"] > div[role="radiogroup"]');
                    if (group) {{
                        clearInterval(interval);
                        group.style.width = '100%';
                        group.style.maxWidth = '100%';
                        group.style.boxSizing = 'border-box';
                        if (!group.querySelector('.nav-bg-layer')) {{
                            group.style.position = 'relative';
                            group.style.overflow = 'hidden';
                            group.style.zIndex = '1';
                            uris.forEach((uri, index) => {{
                                const div = document.createElement('div');
                                div.className = 'nav-bg-layer';
                                div.style.position = 'absolute';
                                div.style.inset = '0';
                                div.style.zIndex = '-2';
                                div.style.backgroundSize = 'cover';
                                div.style.backgroundPosition = 'center';
                                div.style.backgroundImage = 'linear-gradient(rgba(7, 14, 30, .75), rgba(7, 14, 30, .90)), url(' + uri + ')';
                                div.style.opacity = '0';
                                div.style.animation = 'nav-bg-cycle ' + (uris.length * 10) + 's infinite';
                                div.style.animationDelay = (index * 10) + 's';
                                group.insertBefore(div, group.firstChild);
                            }});
                        }}
                    }}
                }}, 50);
            }}
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="glass-card">
          <div class="eyebrow">Master's thesis · Decision-support prototype</div>
          <h1>From Patents to Places</h1>
          <p style="font-size:1.15rem; margin-bottom:0">
            A feasibility study on integrating linguistic and GIS data for water risk.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_in_order(
    selected: list[str],
    *,
    state_key: str,
    default_items: list[str] | None = None,
) -> list[str]:
    current = list(st.session_state.get(state_key, []))
    current = [item for item in current if item in selected]
    for item in (default_items or selected):
        if item in selected and item not in current:
            current.append(item)
    st.session_state[state_key] = current
    return current


def render_sortable_order(
    items: list[str],
    *,
    state_key: str,
    empty_message: str,
) -> list[str]:
    if not items:
        st.caption(empty_message)
        st.session_state[state_key] = []
        return []
    if sort_items is None:
        st.error(
            "Drag-and-drop ordering is unavailable because streamlit-sortables is not installed.",
            icon=":material/extension_off:",
        )
        return selected_in_order(items, state_key=state_key, default_items=items)
    current = selected_in_order(items, state_key=state_key, default_items=items)
    sorted_items = sort_items(
        current,
        custom_style="""
        .sortable-component {
            background: rgba(15, 30, 53, .42);
            border: 1px solid rgba(180, 218, 255, .18);
            border-radius: 12px;
            padding: .35rem;
        }
        .sortable-item {
            background: rgba(61, 110, 155, .45);
            color: #edf6ff;
            border: 1px solid rgba(180, 218, 255, .22);
            border-radius: 8px;
            margin: .25rem;
        }
        """,
    )
    ordered = [item for item in sorted_items if item in items]
    st.session_state[state_key] = ordered
    return ordered


def render_layer_manager(layer_names: list[str], default_layers: list[str]) -> list[str]:
    """Single collapsible drag-and-drop container for layer activation and stacking order."""
    if not layer_names:
        st.warning("No layers found in the selected GeoPackage.", icon=":material/warning:")
        st.session_state["map_layer_order"] = []
        st.session_state["active_map_layers"] = []
        return []

    cube_layers = [name for name in layer_names if "final_data_cube" in name.lower()]
    excluded_layers = {
        "h3_triplet_summary",
        "h3_summary_table",
        "h3_extreme_pairs",
        "traceable_causal_chains",
    }
    other_layer_names = [
        name for name in layer_names
        if name not in cube_layers and name.casefold() not in excluded_layers
    ]

    with st.expander("Layer Manager & Stacking Order", expanded=True):
        if cube_layers:
            st.markdown("**Fixed Top Layer:**")
            for cl in cube_layers:
                st.markdown(f"🔒 `{cl}` (Static Top)")
            st.markdown("---")

        if sort_items is None:
            selected_others = st.multiselect("Active additional layers", options=other_layer_names, default=[l for l in default_layers if l in other_layer_names], key="active_map_layers_multiselect")
            active_ordered = cube_layers + selected_others
            st.session_state["active_map_layers"] = active_ordered
            st.session_state["map_layer_order"] = active_ordered
            return active_ordered

        stored_active = st.session_state.get("active_map_layers", default_layers)
        active_others = [name for name in stored_active if name in other_layer_names]
        for name in default_layers:
            if name in other_layer_names and name not in active_others:
                active_others.append(name)

        inactive_layers = [name for name in other_layer_names if name not in active_others]

        st.caption("Drag additional layers between lists to activate/deactivate, and reorder active layers:")

        sorted_containers = sort_items(
            [
                {"header": "Active Additional Layers", "items": active_others},
                {"header": "Inactive Layers", "items": inactive_layers},
            ],
            multi_containers=True,
            custom_style="""
            .sortable-component {
                background: rgba(15, 30, 53, .42);
                border: 1px solid rgba(180, 218, 255, .18);
                border-radius: 12px;
                padding: .5rem;
            }
            .sortable-container {
                background: rgba(5, 14, 29, .6);
                border: 1px solid rgba(180, 218, 255, .15);
                border-radius: 10px;
                padding: 0.5rem;
                margin: 0.25rem;
            }
            .sortable-item {
                background: rgba(61, 110, 155, .55);
                color: #edf6ff;
                border: 1px solid rgba(180, 218, 255, .3);
                border-radius: 8px;
                padding: 0.4rem 0.6rem;
                margin: .3rem 0;
                cursor: grab;
            }
            """,
        )

        if sorted_containers and len(sorted_containers) >= 2:
            active_others_ordered = [item for item in sorted_containers[0]["items"] if item in other_layer_names]
        else:
            active_others_ordered = active_others

        final_active = cube_layers + active_others_ordered

        st.session_state["active_map_layers"] = final_active
        st.session_state["map_layer_order"] = final_active

        return final_active


def render_map_tab() -> None:
    st.subheader("Geospatial intelligence")
    packages = cached_geo_packages(str(MAP_CONTENTS))
    if not packages:
        st.error("No GeoPackage was found in map_contents.", icon=":material/error:")
        return

    with st.sidebar:
        st.header("Map settings")
        package = packages[0]
        layer_names = cached_layer_names(str(package))
        default_layers = [
            name for name in layer_names
            if "final_data_cube" in name.lower()
        ] or [
            preferred
            for preferred in [preferred_layer(layer_names)]
            if preferred and preferred.casefold() not in {
                "h3_triplet_summary",
                "h3_summary_table",
                "h3_extreme_pairs",
                "traceable_causal_chains",
            }
        ]
        ordered = render_layer_manager(layer_names, default_layers)
        loaded_candidates = {
            name: cached_layer(str(package), name)
            for name in ordered
        }
        spatial_layer_names = [
            name for name, frame in loaded_candidates.items()
            if frame is not None
            and not frame.empty
            and hasattr(frame, "to_crs")
            and GEOMETRY_COLUMN in frame.columns
            and frame[GEOMETRY_COLUMN].notna().any()
        ]
        opacity_by_layer = {
            name: st.slider(
                f"Opacity · {name}",
                0.0,
                1.0,
                1.0 if "final_data_cube" in name.lower() else 0.25,
                0.05,
                key=f"opacity_{name}",
            )
            for name in spatial_layer_names
        }
        sampled_features_limit = st.slider(
            "Sampled features limit", min_value=1_000, max_value=MAX_RENDER_FEATURES,
            value=10_000, step=1_000, key="sampled_features_limit",
            help="Target-bearing features are retained first when a layer exceeds this limit.",
        )
        show_basemap = st.checkbox("Show OpenStreetMap basemap", value=True, key="show_basemap")
        basemap_opacity = st.slider(
            "Basemap opacity", min_value=0.0, max_value=1.0, value=0.40,
            step=0.05, key="basemap_opacity",
        )

    if not ordered:
        st.info("Select at least one layer to render the map.", icon=":material/layers:")
        return

    loaded = loaded_candidates
    loaded = {name: frame for name, frame in loaded.items() if frame is not None and not frame.empty}
    if not loaded:
        st.error("The selected GeoPackage layers could not be loaded.", icon=":material/error:")
        return

    join_table = None

    variable_by_layer: dict[str, str | None] = {}
    classification_by_layer: dict[str, str] = {}
    palette_by_layer: dict[str, str] = {}
    bin_count_by_layer: dict[str, int] = {}
    classification_options = ["Quantiles", "Natural breaks", "Unique values"]

    st.subheader("Layer styling")
    st.caption("Choose mapped data and styling independently for each active spatial layer.")
    
    if "all_styles_expanded" not in st.session_state:
        st.session_state["all_styles_expanded"] = True
    if "style_expand_version" not in st.session_state:
        st.session_state["style_expand_version"] = 0

    is_currently_expanded = st.session_state["all_styles_expanded"]
    toggle_label = "Collapse all styles" if is_currently_expanded else "Open all styles"
    toggle_icon = ":material/unfold_less:" if is_currently_expanded else ":material/unfold_more:"

    if st.button(toggle_label, icon=toggle_icon, use_container_width=True, key="btn_toggle_all_styles"):
        st.session_state["all_styles_expanded"] = not is_currently_expanded
        st.session_state["style_expand_version"] += 1
        st.rerun()

    expand_version = st.session_state["style_expand_version"]
    default_expanded = st.session_state["all_styles_expanded"]

    for layer_name, frame in loaded.items():
        if (
            not hasattr(frame, "to_crs")
            or GEOMETRY_COLUMN not in frame.columns
            or not frame[GEOMETRY_COLUMN].notna().any()
        ):
            continue
        layer_variables = discover_variables(frame, join_table)
        layer_variable_options = ["No mapped variable", *layer_variables]
        layer_name_lower = layer_name.casefold()
        report_bivariate_variables = tuple(
            field
            for field in layer_variables
            if field.casefold().startswith("bv_")
            and field.casefold().endswith("_cluster")
        )
        forced_variables = (
            ("capacita_duso_e_fertilita", ("fertilita",)),
            ("com_wgs84", ("causal_socioeconomic_index", "causal_socio_economic_index")),
            ("fertilita_chimica", ("lcc_classe",)),
        )
        if "report_bivariate_lisa" in layer_name_lower:
            forced_variable_options = report_bivariate_variables
        else:
            forced_variable_options = next(
                (
                    variables
                    for token, variables in forced_variables
                    if token in layer_name_lower
                    or (
                        token == "com_wgs84"
                        and "com" in layer_name_lower
                        and "wgs84" in layer_name_lower
                    )
                ),
                None,
            )
        forced_variable = next(
            (
                variable for variable in (forced_variable_options or ())
                if variable in layer_variables or variable in frame.columns
            ),
            forced_variable_options[0] if forced_variable_options else None,
        )
        default_variable = forced_variable or select_variable(frame)
        if forced_variable and forced_variable not in layer_variable_options:
            layer_variable_options.append(forced_variable)
        default_variable_label = default_variable if default_variable in layer_variable_options else "No mapped variable"

        with st.expander(f"Style: {layer_name}", expanded=default_expanded, key=f"style_expander_{layer_name}_{expand_version}"):
            if forced_variable:
                selected_variable = forced_variable
                st.caption(f"Mapped variable: `{forced_variable}`")
            else:
                selected_variable = st.selectbox(
                    "Mapped variable",
                    layer_variable_options,
                    index=layer_variable_options.index(default_variable_label),
                    key=f"mapped_variable_{layer_name}",
                )
            variable_by_layer[layer_name] = (
                None if selected_variable == "No mapped variable" else selected_variable
            )
            classification = st.segmented_control(
                "Classification method",
                classification_options,
                default="Quantiles",
                key=f"map_classification_{layer_name}",
            ) or "Quantiles"
            classification_by_layer[layer_name] = classification
            bin_count_by_layer[layer_name] = st.slider(
                "Number of groups/bins", min_value=3, max_value=10, value=5,
                step=1, key=f"map_bin_count_{layer_name}",
                disabled=classification == "Unique values",
                help="Controls the number of classes for Quantiles and Natural breaks.",
            )
            palette_by_layer[layer_name] = st.selectbox(
                "Colour palette",
                PALETTES,
                key=f"map_palette_{layer_name}",
            )

    if any(len(frame) > sampled_features_limit for frame in loaded.values()):
        st.caption(
            f"Hover a polygon to inspect its spatial fields and the attributes joined through the `index` key. "
            f"The largest H3 layer is sampled to {sampled_features_limit:,} target-bearing features for responsive browser rendering."
        )
    else:
        st.caption("Hover a polygon to inspect its spatial fields and the attributes joined through the `index` key.")
    try:
        html = build_map(
            loaded, variable=variable_by_layer, classification=classification_by_layer,
            palette_name=palette_by_layer, bin_count=bin_count_by_layer,
            opacity_by_layer=opacity_by_layer, _join_table=join_table,
            render_limit=sampled_features_limit, show_basemap=show_basemap,
            basemap_opacity=basemap_opacity,
        )
        html = f"<!-- tab-instance: Map -->\n" + html
        st.caption("Interactive map built for showing the risk prone areas. Use the layer manager to toggle layers and adjust their order. Other settings are available in the sidebar.")
        components.html(html, height=650, scrolling=True)
    except (ImportError, OSError, ValueError, KeyError) as error:
        st.error(f"Could not render the map: {error}", icon=":material/error:")


def render_insights_tab() -> None:
    st.subheader("Multi-plot visualizer")
    plots = cached_plot_paths(str(PLOTS))
    if not plots:
        st.error("No PNG, JPEG, SVG, or HTML insight files were found.", icon=":material/error:")
        return
    with st.sidebar:
        st.header("Insights settings")
        plot_names = [path.name for path in plots]
        current_order = selected_in_order(
            plot_names,
            state_key="plot_order",
            default_items=plot_names[:3],
        )
        selected_names = st.multiselect(
            "Plots in display order", [path.name for path in plots],
            default=current_order,
            key="selected_plots",
            help="Select plots, then drag the cards below to change their display order.",
        )
        current_order = selected_in_order(
            selected_names,
            state_key="plot_order",
            default_items=selected_names,
        )
        st.caption("Display order")
        current_order = render_sortable_order(
            current_order,
            state_key="plot_order",
            empty_message="Select plots to configure their display order.",
        )
    plot_by_name = {path.name: path for path in plots}
    selected = [plot_by_name[name] for name in current_order if name in plot_by_name]
    if not selected:
        st.info("Select one or more plots from the sidebar.", icon=":material/insert_chart:")
        return
    plot_descriptions = {
        "filtered_forward_patent_edge_cases_pos_sankey": "Sankey diagram of forward patent edge cases, showing how technical terms connect to attributes and failure concepts.",
        "filtered_patent_edge_cases_pos_sankey": "Sankey diagram of filtered patent edge cases, highlighting the strongest observed causal term transitions.",
        "incomplete_sankey_diagram": "Sankey diagram of incomplete causal links that remain unresolved in the patent-derived network.",
        "heatmap_top20_attr_vs_fail": "Clustered NPMI heatmap showing co-occurrence strength between the top attribute and failure terms.",
        "heatmap_top20_tech_vs_attr": "Clustered NPMI heatmap showing co-occurrence strength between technical terms and attribute terms.",
        "report_bivariate_lisa_census_pop_vs_inflow_total": "Bivariate LISA map comparing census population with total inflow and identifying significant spatial associations.",
        "report_bivariate_lisa_idw_rain_mm_avg_monthly_vs_idw_wind_speed_max_95p": "Bivariate LISA map comparing average monthly rainfall with the 95th-percentile wind speed.",
        "report_bivariate_lisa_idw_temp_thermal_range_vs_dtmidcnt_mean": "Bivariate LISA map comparing temperature thermal range with mean elevation.",
        "top_attribute_terms": "Horizontal bar chart ranking the twenty most frequent attribute or variable terms.",
        "top_failure_terms": "Horizontal bar chart ranking the twenty most frequent failure terms.",
        "top_technical_terms": "Horizontal bar chart ranking the twenty most frequent technical terms.",
    }

    columns = 1 if len(selected) == 1 else 2 if len(selected) >= 4 else len(selected)
    for start in range(0, len(selected), columns):
        row = st.columns(columns)
        for column, plot in zip(row, selected[start:start + columns]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{plot.stem.replace('_', ' ').title()}**")
                    render_plot(plot)
                    description = plot_descriptions.get(
                        plot.stem,
                        "Analytical visualization summarizing a distinct result from the thesis study.",
                    )
                    st.markdown(f"<p style='font-size:0.85rem; color:#94a3b8; margin-top:0.5rem;'>{description}</p>", unsafe_allow_html=True)
                    st.button(
                        "Open full size",
                        key=f"open_plot_{plot.name}",
                        icon=":material/zoom_in:",
                        on_click=lambda path=str(plot): st.session_state.__setitem__("selected_plot", path),
                    )


@st.dialog("Plot preview", width="large")
def render_plot_dialog(path_string: str) -> None:
    path = Path(path_string)
    st.caption(path.stem.replace("_", " ").title())
    render_plot(path, height=900)


def render_info_tab() -> None:
    st.subheader("Project information")
    logos = list(reversed([
        path for path in asset_images(ASSETS / "images" / "info")
        if "qgis" not in path.name.casefold()
    ]))
    if logos:
        columns = st.columns([1, 2, 2, 1])
        for column, image in zip(columns[1:3], logos[:2]):
            with column:
                with st.container(border=True):
                    st.image(str(image), width="stretch")
                    st.caption(image.stem.replace("_", " ").title())
    else:
        st.warning("No images were found in assets/images/info.", icon=":material/image_not_supported:")

    with st.container(border=True):
        st.markdown("### About the study")
        st.write("**Author:** Rafael Ignacio Urbina Hincapie")
        st.write("**Affiliation:** Università di Pisa")
        st.write("**Scope:** linking patent-derived causal chains with H3-indexed spatial water-risk indicators.")

    st.markdown("### Compiled thesis reader")
    pdf_path = THESIS / "report.pdf"
    if pdf_path.exists():
        try:
            pdf_bytes = cached_pdf(str(pdf_path))
            st.download_button(
                label="Download full thesis PDF",
                data=pdf_bytes,
                file_name="report.pdf",
                mime="application/pdf",
                icon=":material/picture_as_pdf:",
                key="download_thesis_pdf",
            )
            
            pages = cached_pdf_pages(str(pdf_path))
            st.caption(f"Document reader · {len(pages)} pages total:")
            
            total_p = len(pages)
            st.session_state.setdefault("thesis_page_select", 1)

            # Define callbacks to update state cleanly before widget instantiation
            def go_prev():
                if st.session_state["thesis_page_select"] > 1:
                    st.session_state["thesis_page_select"] -= 1

            def go_next():
                if st.session_state["thesis_page_select"] < total_p:
                    st.session_state["thesis_page_select"] += 1
            
            col_prev, col_sel, col_next = st.columns([1, 4, 1])
            with col_prev:
                st.button("◀ Previous", use_container_width=True, disabled=(st.session_state["thesis_page_select"] <= 1), on_click=go_prev, key="btn_pdf_prev")
            with col_sel:
                st.selectbox(
                    "Select page",
                    options=list(range(1, total_p + 1)),
                    format_func=lambda p: f"Page {p} of {total_p}",
                    key="thesis_page_select",
                    label_visibility="collapsed",
                )
            with col_next:
                st.button("Next ▶", use_container_width=True, disabled=(st.session_state["thesis_page_select"] >= total_p), on_click=go_next, key="btn_pdf_next")
            
            if pages:
                current_page_idx = st.session_state["thesis_page_select"] - 1
                st.image(
                    pages[current_page_idx],
                    caption=f"Page {st.session_state['thesis_page_select']} of {total_p}",
                    use_container_width=True,
                )
                
        except Exception as error:
            st.error(f"Could not render the compiled thesis PDF: {error}", icon=":material/error:")
    else:
        st.info("The compiled thesis PDF is not available yet.", icon=":material/info:")


def render_landpage() -> None:
    backgrounds = asset_images(ASSETS / "backgrounds")
    carousel_images = asset_images(CAROUSEL)
    background_uris = [image_data_uri(str(path)) for path in backgrounds]
    carousel_uris = [image_data_uri(str(path)) for path in carousel_images]
    background_layers = "".join(
        f'<div class="landpage-bg" style="background-image:url({uri})"></div>'
        for uri in background_uris
    )
    carousel_layers = "".join(
        f'<img class="landpage-slide" src="{uri}" alt="Dashboard overview image">'
        for uri in carousel_uris
    )
    st.markdown(
        f"""
        <style>
        .landpage {{
            position: relative; overflow: hidden; border-radius: 24px; padding: 5rem 8%;
            min-height: 760px; display: flex; flex-direction: column; align-items: center;
            justify-content: center; text-align: center; isolation: isolate;
        }}
        .landpage-bg {{
            position: absolute; inset: 0; z-index: -2; background-size: cover;
            background-position: center; opacity: 0; animation: landpage-bg-cycle 60s infinite;
        }}
        .landpage-bg:nth-child(1) {{ animation-delay: 0s; }}
        .landpage-bg:nth-child(2) {{ animation-delay: 10s; }}
        .landpage-bg:nth-child(3) {{ animation-delay: 20s; }}
        .landpage-bg:nth-child(4) {{ animation-delay: 30s; }}
        .landpage-bg:nth-child(5) {{ animation-delay: 40s; }}
        .landpage-bg:nth-child(6) {{ animation-delay: 50s; }}
        .landpage::after {{
            content: ""; position: absolute; inset: 0; z-index: -1;
            background: linear-gradient(135deg, rgba(5,14,29,.82), rgba(7,14,30,.55));
        }}
        .landpage h1 {{ font-size: clamp(2.5rem, 6vw, 5rem); margin: 0 0 1.25rem; }}
        .landpage p {{ max-width: 900px; font-size: 1.15rem; line-height: 1.7; }}
        .landpage-carousel {{
            width: min(100%, 980px); height: 330px; margin-top: 2rem; position: relative;
            overflow: hidden; border-radius: 18px; border: 1px solid rgba(180,218,255,.25);
            box-shadow: 0 18px 50px rgba(0,0,0,.35);
        }}
        .landpage-slide {{
            position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
            opacity: 0; animation: landpage-slide-cycle 30s infinite;
        }}
        .landpage-slide:nth-child(1) {{ animation-delay: 0s; }}
        .landpage-slide:nth-child(2) {{ animation-delay: 10s; }}
        .landpage-slide:nth-child(3) {{ animation-delay: 20s; }}
        @keyframes landpage-bg-cycle {{
            0%, 3% {{ opacity: 0; }} 15%, 25% {{ opacity: 1; }} 35%, 100% {{ opacity: 0; }}
        }}
        @keyframes landpage-slide-cycle {{
            0%, 3% {{ opacity: 0; }} 15%, 25% {{ opacity: 1; }} 35%, 100% {{ opacity: 0; }}
        }}
        </style>
        <section class="landpage">
          {background_layers}
          <h1>From Patents to Places</h1>
          <p>
            This interactive dashboard connects patent-derived causal knowledge with
            H3-indexed spatial indicators to explore how technical systems, environmental
            conditions, and water-risk factors relate across the study area. Use the map
            to inspect spatial patterns, the insights tab to examine analytical plots, and
            the project information tab to access the study context and compiled thesis.
          </p>
          <div class="landpage-carousel">{carousel_layers}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    initialize_state()
    apply_theme()
    preload_plot_assets(str(PLOTS))
    
    # Silent background preload & cache of PDF
    pdf_path = THESIS / "report.pdf"
    if pdf_path.exists():
        cached_pdf(str(pdf_path))
        cached_pdf_pages(str(pdf_path))

    active_tab = st.radio(
        "Navigation",
        ["landpage", "Map", "Insights", "Info"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_page",
    )
    if active_tab != "landpage":
        render_header()
    
    st.markdown("---")

    if active_tab == "landpage":
        render_landpage()
    elif active_tab == "Map":
        render_map_tab()
    elif active_tab == "Insights":
        render_insights_tab()
        if st.session_state.get("selected_plot"):
            render_plot_dialog(st.session_state.selected_plot)
            st.session_state.selected_plot = None
    elif active_tab == "Info":
        render_info_tab()


if __name__ == "__main__":
    main()