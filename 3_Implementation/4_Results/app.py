"""Interactive thesis dashboard: From Patents to Places."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_sortables import sort_items
except ImportError:
    sort_items = None

from modules.gis_loader import (
    MAX_RENDER_FEATURES,
    PALETTES,
    VARIABLES,
    build_map,
    discover_geo_packages,
    discover_tabular_files,
    list_layers,
    load_join_table,
    load_layer,
    preferred_layer,
    select_variable,
)
from modules.plot_manager import discover_plots, render_plot

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAP_CONTENTS = ROOT / "map_contents"
PLOTS = ROOT / "insights_plots"
THESIS = ROOT / "thesis"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

st.set_page_config(
    page_title="From Patents to Places",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    st.session_state.setdefault("background_name", None)
    st.session_state.setdefault("map_layer_order", [])
    st.session_state.setdefault("plot_order", [])
    st.session_state.setdefault("selected_plot", None)
    st.session_state.setdefault("active_page", "Map")


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


@st.cache_data(ttl="15m", max_entries=4)
def cached_pdf(path: str) -> bytes:
    return Path(path).read_bytes()


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
        .plot-media-frame, .plot-media-frame img {{
            width: 100% !important; max-width: 100% !important; height: auto !important;
            overflow: hidden !important;
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


def render_map_tab() -> None:
    st.subheader("Geospatial intelligence")
    packages = cached_geo_packages(str(MAP_CONTENTS))
    tables = cached_tabular_files(str(MAP_CONTENTS))
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
        ordered = selected_in_order(
            selected,
            state_key="map_layer_order",
            default_items=[
                name for name in selected if name in default_layers
            ] + [name for name in selected if name not in default_layers],
        )
        st.caption("Layer order (top to bottom)")
        ordered = render_sortable_order(
            ordered,
            state_key="map_layer_order",
            empty_message="Select active layers to configure their stacking order.",
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
    variables = [field for field in VARIABLES if field in primary.columns]
    variable = st.selectbox("Mapped variable", variables or [select_variable(primary) or "No numeric variable"], key="mapped_variable")
    classification = st.segmented_control(
        "Classification", ["Quantiles", "Natural breaks", "Unique values"], default="Quantiles", key="map_classification"
    )
    palette = st.selectbox("Colour palette", PALETTES, key="map_palette")
    table_by_name = {
        str(path.relative_to(MAP_CONTENTS)): path
        for path in tables
    }
    causal_layer_names = [
        name for name in cached_layer_names(str(package))
        if any(token in name.lower() for token in (
            "h3_triplet_summary",
            "h3_extreme_pairs",
            "traceable_causal_chains",
        ))
    ]
    table_options = list(dict.fromkeys([*table_by_name, *causal_layer_names]))
    join_name = st.selectbox(
        "Additional causal table",
        ["None", *table_options],
        key="causal_table",
        help="The selected table is merged with the spatial layer on the `index` column.",
    )
    if join_name == "None":
        join_table = None
    elif join_name in table_by_name:
        join_table = cached_join_table(str(table_by_name[join_name]))
    else:
        join_table = cached_layer(str(package), join_name)

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
    columns = 1 if len(selected) == 1 else 2 if len(selected) >= 4 else len(selected)
    for start in range(0, len(selected), columns):
        row = st.columns(columns)
        for column, plot in zip(row, selected[start:start + columns]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{plot.stem.replace('_', ' ').title()}**")
                    render_plot(plot)
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
    logos = asset_images(ASSETS / "images" / "info")
    if logos:
        columns = st.columns(3)
        for column, image in zip(columns, logos[:3]):
            with column:
                with st.container(border=True):
                    st.image(str(image), use_container_width=True)
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
            # Provide a direct download button
            pdf_bytes = cached_pdf(str(pdf_path))
            st.download_button(
                label="Download full thesis PDF",
                data=pdf_bytes,
                file_name="report.pdf",
                mime="application/pdf",
                icon=":material/picture_as_pdf:",
                key="download_thesis_pdf",
            )
            
            # Render PDF pages natively as images to bypass browser iframe blocks
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            
            st.caption(f"Displaying {len(doc)} pages:")
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(dpi=150)  # Good balance of clarity and performance
                st.image(
                    pix.tobytes("png"),
                    caption=f"Page {page_idx + 1} of {len(doc)}",
                    use_container_width=True,
                )
                st.markdown("---")
                
        except Exception as error:
            st.error(f"Could not render the compiled thesis PDF: {error}", icon=":material/error:")
    else:
        st.info("The compiled thesis PDF is not available yet.", icon=":material/info:")

def main() -> None:
    initialize_state()
    apply_theme()
    render_header()
    
    # Main page navigation tabs placed right below the header
    active_tab = st.radio(
        "Navigation",
        ["Map", "Insights", "Info"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_page",
    )
    
    st.markdown("---")

    if active_tab == "Map":
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