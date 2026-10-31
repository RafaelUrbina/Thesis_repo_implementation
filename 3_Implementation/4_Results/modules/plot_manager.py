"""Discovery and safe rendering helpers for insight assets."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".html", ".htm"}


def discover_plots(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.glob("*") if path.is_file() and path.suffix.lower() in SUPPORTED),
        key=lambda path: path.name.lower(),
    ) if folder.exists() else []


def render_plot(path: Path, *, height: int = 420) -> None:
    suffix = path.suffix.lower()
    
    if suffix in {".html", ".htm"}:
        try:
            html_content = path.read_text(encoding="utf-8")
            
            # CSS Transform approach: Scale the entire HTML view down to fit the iframe height perfectly
            wrapped_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                <meta charset="utf-8">
                <style>
                    html, body {{
                        width: 100% !important;
                        height: 100% !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        overflow: hidden !important;
                        background: transparent !important;
                    }}
                    #scaler-wrapper {{
                        width: 100%;
                        height: 100%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        overflow: hidden;
                    }}
                    #scaled-content {{
                        transform-origin: center center;
                        transition: transform 0.2s ease-in-out;
                    }}
                </style>
                </head>
                <body>
                    <div id="scaler-wrapper">
                        <div id="scaled-content">
                            {html_content}
                        </div>
                    </div>
                    <script>
                        function autoScale() {{
                            const wrapper = document.getElementById('scaler-wrapper');
                            const content = document.getElementById('scaled-content');
                            
                            // Find inner Plotly div or target document bounds
                            const target = content.querySelector('.plotly-graph-div') || content.firstElementChild || content;
                            
                            const contentWidth = target.scrollWidth || target.offsetWidth || 800;
                            const contentHeight = target.scrollHeight || target.offsetHeight || 500;
                            
                            const containerWidth = wrapper.clientWidth;
                            const containerHeight = wrapper.clientHeight;
                            
                            if (contentWidth > 0 && contentHeight > 0 && containerWidth > 0 && containerHeight > 0) {{
                                const scaleX = containerWidth / contentWidth;
                                const scaleY = containerHeight / contentHeight;
                                // Choose uniform scale to fit inside container box completely
                                const scale = Math.min(scaleX, scaleY, 1.0);
                                
                                content.style.transform = `scale(${{scale}})`;
                                content.style.width = `${{contentWidth}}px`;
                                content.style.height = `${{contentHeight}}px`;
                            }}
                        }}

                        window.addEventListener('load', () => {{
                            setTimeout(autoScale, 200);
                            setTimeout(autoScale, 600); // Secondary check for delayed Plotly execution
                        }});
                        window.addEventListener('resize', autoScale);
                    </script>
                </body>
                </html>
            """
            components.html(wrapped_html, height=height, scrolling=False)
        except (OSError, UnicodeDecodeError) as error:
            st.error(f"Could not render {path.name}: {error}", icon=":material/error:")
    else:
        try:
            mime_map = {
                ".jpg": "jpeg",
                ".jpeg": "jpeg",
                ".png": "png",
                ".webp": "webp",
                ".svg": "svg+xml",
            }
            mime = mime_map.get(suffix, "png")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            
            image_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                <style>
                    html, body {{
                        width: 100%;
                        height: 100%;
                        margin: 0;
                        padding: 0;
                        background: transparent;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        overflow: hidden;
                    }}
                    img {{
                        max-width: 100%;
                        max-height: 100%;
                        object-fit: contain;
                        display: block;
                    }}
                </style>
                </head>
                <body>
                    <img src="data:image/{mime};base64,{encoded}" />
                </body>
                </html>
            """
            components.html(image_html, height=height, scrolling=False)
        except Exception as error:
            st.error(f"Could not render image {path.name}: {error}", icon=":material/error:")