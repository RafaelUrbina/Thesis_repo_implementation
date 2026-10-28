"""Interactive thesis dashboard: From Patents to Places."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from modules.gis_loader import (
    MAX_RENDER_FEATURES,
    PALETTES,
    build_map,
    discover_geo_packages,
    discover_tabular_files,
    join_on_index,
    list_layers,
    load_join_table,
    load_layer,
    preferred_layer,
    select_variable,
)
from modules.plot_manager import discover_plots, render_plot, slots_for_layout
from modules.thesis_reader import discover_tex_files, latex_to_plain_text, read_tex_file

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAP_CONTENTS = ROOT / "map_contents"
PLOTS = ROOT / "insights_plots"
THESIS = ROOT / "thesis"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
LAYOUTS = ["1 Plot", "2 Plots (Side-by-Side)", "3 Plots (Row)", "2x2 Grid (4 Plots)"]

st.set_page_config(
    page_title="From Patents to Places",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    st.session_state.setdefault("background_name", None)
    st.session_state.setdefault("map_layers", [])
    st.session_state.setdefault("plot_layout", LAYOUTS[0])


@st.cache_data(ttl="15m", max_entries=8)
def cached_layer_names(package: str) -> list[str]:
    return list_layers(Path(package))


@st.cache_data(ttl="15m", max_entries=24)
def cached_layer(package: str, layer_name: str):
    return load_layer(Path(package), layer_name)


@st.cache_data(ttl="15m", max_entries=16)
def cached_join_table(path: str):
    return load_join_table(Path(path))


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
        }}
        .eyebrow {{ color: #7dd3fc; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
        h1, h2, h3 {{ letter-spacing: -.02em; }}
        </style>
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


def render_map_tab() -> None:
    st.subheader("Geospatial intelligence")
    packages = discover_geo_packages(MAP_CONTENTS)
    tables = discover_tabular_files(MAP_CONTENTS)
    if not packages:
        st.error("No GeoPackage was found in map_contents.", icon=":material/error:")
        return

    with st.sidebar:
        st.header("Map settings")
        package = st.selectbox("GeoPackage", packages, format_func=lambda path: path.name, key="map_package")
        layer_names = cached_layer_names(str(package))
        default_layers = [
            name for name in layer_names
            if any(token in name.lower() for token in ("final_data_cube", "triplet", "extreme_pairs"))
        ] or ([preferred_layer(layer_names)] if preferred_layer(layer_names) else [])
        selected = st.multiselect("Active layers", layer_names, default=default_layers, key="active_map_layers")
        ordered = st.multiselect(
            "Layer order (top to bottom)", selected,
            default=[name for name in selected if name in default_layers] + [name for name in selected if name not in default_layers],
            key="ordered_map_layers",
            help="The first item is rendered above the later items.",
        )
        opacity_by_layer = {
            name: st.slider(f"Opacity · {name}", 0.0, 1.0, 0.78, 0.05, key=f"opacity_{name}")
            for name in ordered
        }

    if not ordered:
        st.info("Select at least one layer to render the map.", icon=":material/layers:")
        return

    loaded = {name: cached_layer(str(package), name) for name in ordered}
    loaded = {name: frame for name, frame in loaded.items() if frame is not None and not frame.empty}
    if not loaded:
        st.error("The selected GeoPackage layers could not be loaded.", icon=":material/error:")
        return

    primary = loaded.get(preferred_layer(list(loaded)))
    if primary is None:
        primary = next(iter(loaded.values()))
    variables = [field for field in ("total_failure_chains", "unique_technical_triggers", "patent_count", "causal_socio_economic_index") if field in primary.columns]
    variable = st.selectbox("Mapped variable", variables or [select_variable(primary) or "No numeric variable"], key="mapped_variable")
    classification = st.segmented_control(
        "Classification", ["Quantiles", "Natural breaks", "Unique values"], default="Quantiles", key="map_classification"
    )
    palette = st.selectbox("Colour palette", PALETTES, key="map_palette")
    join_options = ["None"] + [path.name for path in tables]
    join_name = st.selectbox("Additional causal table", join_options, key="causal_table")
    join_table = cached_join_table(str(MAP_CONTENTS / join_name)) if join_name != "None" else None

    if any(len(frame) > MAX_RENDER_FEATURES for frame in loaded.values()):
        st.caption(
            f"Hover a polygon to inspect its spatial fields and the attributes joined through the `index` key. "
            f"The largest H3 layer is sampled to {MAX_RENDER_FEATURES:,} features for responsive browser rendering."
        )
    else:
        st.caption("Hover a polygon to inspect its spatial fields and the attributes joined through the `index` key.")
    try:
        html = build_map(
            loaded, variable=variable, classification=classification or "Quantiles",
            palette_name=palette, opacity_by_layer=opacity_by_layer, join_table=join_table,
        )
        st.html(html, unsafe_allow_javascript=True)
    except (ImportError, OSError, ValueError, KeyError) as error:
        st.error(f"Could not render the map: {error}", icon=":material/error:")


def render_insights_tab() -> None:
    st.subheader("Multi-plot visualizer")
    plots = discover_plots(PLOTS)
    if not plots:
        st.error("No PNG, JPEG, SVG, or HTML insight files were found.", icon=":material/error:")
        return
    with st.sidebar:
        st.header("Insights settings")
        layout = st.selectbox("Plot layout", LAYOUTS, key="plot_layout")
        ordered_names = st.multiselect(
            "Plots in display order", [path.name for path in plots],
            default=[path.name for path in plots[:slots_for_layout(layout)]],
            key="ordered_plots",
        )
    selected = [path for name in ordered_names if (path := next((item for item in plots if item.name == name), None))]
    selected = selected[:slots_for_layout(layout)]
    if not selected:
        st.info("Select one or more plots from the sidebar.", icon=":material/insert_chart:")
        return
    columns = 2 if layout == "2x2 Grid (4 Plots)" else len(selected)
    for start in range(0, len(selected), columns):
        row = st.columns(columns)
        for column, plot in zip(row, selected[start:start + columns]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{plot.stem.replace('_', ' ').title()}**")
                    render_plot(plot)
                    with st.expander("Magnify / inspect", icon=":material/zoom_in:"):
                        render_plot(plot, height=850)


def render_info_tab() -> None:
    st.subheader("Project information")
    logos = asset_images(ASSETS / "images" / "info")
    if logos:
        columns = st.columns(3)
        for column, image in zip(columns, logos[:3]):
            with column:
                with st.container(border=True):
                    st.image(str(image), width="stretch")
                    st.caption(image.stem.replace("_", " ").title())
    else:
        st.warning("No images were found in assets/images/info.", icon=":material/image_not_supported:")

    abstract_path = THESIS / "0-Abstract.tex"
    abstract = latex_to_plain_text(read_tex_file(abstract_path)) if abstract_path.exists() else "The thesis abstract is not available."
    with st.container(border=True):
        st.markdown("### About the study")
        st.write("**Author:** Rafael Ignacio Urbina Hincapie")
        st.write("**Affiliation:** Università di Pisa")
        st.write("**Scope:** linking patent-derived causal chains with H3-indexed spatial water-risk indicators.")
        st.markdown("**Abstract**")
        st.write(abstract)

    st.markdown("### LaTeX thesis reader")
    files = discover_tex_files(THESIS)
    if not files:
        st.warning("No .tex files were found in thesis.", icon=":material/description:")
        return
    selected = st.selectbox(
        "Source file", files,
        format_func=lambda path: str(path.relative_to(THESIS)),
        key="thesis_file",
    )
    content = read_tex_file(selected)
    st.caption(f"{len(content):,} characters · source preview")
    st.code(content, language="latex", line_numbers=True)


def main() -> None:
    initialize_state()
    apply_theme()
    render_header()
    map_tab, insights_tab, info_tab = st.tabs(
        [":material/map: Map", ":material/analytics: Insights", ":material/info: Info"]
    )
    with map_tab:
        render_map_tab()
    with insights_tab:
        render_insights_tab()
    with info_tab:
        render_info_tab()


if __name__ == "__main__":
    main()
