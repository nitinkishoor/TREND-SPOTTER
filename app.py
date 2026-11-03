"""
Stock Chart Pattern & Edge Spotter - Streamlit Web Application
Upload any stock/crypto chart screenshot to detect edges, identify chart patterns
(Double Top / "Double Header", Head and Shoulders, Double Bottom, etc.), and view
exact matching percentages and annotations.
"""

import os
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from detector import (
    ChartAnnotator,
    ChartPatternDetector,
    ChartPreprocessor,
    ExtremaDetector,
    PatternMatcher,
)

# Page configuration
st.set_page_config(
    page_title="TREND SPOTTER",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Custom CSS Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8c9ba5;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .pattern-card {
        background: #181c27;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00d2ff;
    }
    .pattern-card-bearish {
        border-left: 4px solid #f04455;
    }
    .pattern-card-bullish {
        border-left: 4px solid #26a69a;
    }
    .pattern-card-continuation {
        border-left: 4px solid #4285f4;
    }
    .badge-score {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def main():
    st.markdown(
        '<div class="main-title">📈 TREND SPOTTER</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Upload a chart screenshot to detect price edges.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("⚙️ Configuration & Inputs")

    # Image source selection
    input_source = st.sidebar.radio(
        "Choose Chart Source:",
        ("Upload Screenshot", "Use Sample Chart"),
        index=0,
    )

    image_to_process = None
    image_name = "chart.png"

    if input_source == "Upload Screenshot":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Chart Screenshot (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload any TradingView, broker, or crypto exchange screenshot.",
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
    else:
        sample_options = {
            "Double Top ('Double Header') - Bearish Reversal": "samples/sample_double_top.png",
            "Head and Shoulders - Classic Bearish Reversal": "samples/sample_head_and_shoulders.png",
            "Double Bottom ('W' Pattern) - Bullish Reversal": "samples/sample_double_bottom.png",
        }
        selected_sample_label = st.sidebar.selectbox(
            "Select Sample Chart:", list(sample_options.keys())
        )
        sample_path = sample_options[selected_sample_label]
        if os.path.exists(sample_path):
            image_to_process = Image.open(sample_path).convert("RGB")
            image_name = os.path.basename(sample_path)
        else:
            st.sidebar.error("Sample file not found. Please run generator.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Pattern Filters")

    enable_double_top = st.sidebar.checkbox(
        "Double Top ('Double Header')", value=True
    )
    enable_head_shoulders = st.sidebar.checkbox(
        "Head and Shoulders", value=True
    )
    enable_double_bottom = st.sidebar.checkbox(
        "Double Bottom ('W')", value=True
    )
    enable_inverse_hs = st.sidebar.checkbox(
        "Inverse Head & Shoulders", value=True
    )
    enable_triangles = st.sidebar.checkbox(
        "Triangles (Symmetrical/Ascending/Descending)", value=False
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Detection Sensitivity")

    min_match_pct = st.sidebar.slider(
        "Minimum Match Percentage (%)",
        min_value=50,
        max_value=95,
        value=65,
        step=5,
        help="Only display patterns meeting or exceeding this confidence threshold.",
    )

    prominence_ratio = st.sidebar.slider(
        "Peak & Valley Sensitivity",
        min_value=0.02,
        max_value=0.08,
        value=0.035,
        step=0.005,
        help="Higher values detect only major swings; lower values detect finer swings.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ Visual Annotations")
    draw_trajectory = st.sidebar.checkbox(
        "Draw Detected Price Trajectory Line", value=True
    )
    draw_all_extrema = st.sidebar.checkbox(
        "Show All Extrema Points", value=False
    )
    show_cv_diagnostics = st.sidebar.checkbox(
        "Show Edge Detection & Price Mask Diagnostics", value=True
    )

    # If no image is available yet
    if image_to_process is None:
        st.info(
            "👈 Please upload a screenshot using the sidebar, or select 'Use Sample Chart' to test immediately."
        )
        st.markdown(
            """
            JUST UPLODE  THE SCREENSHOT
            """
        )
        return

    # Run Detection Pipeline
    with st.spinner("Analyzing screenshot and detecting patterns..."):
        detector = ChartPatternDetector(
            min_match_percentage=float(min_match_pct),
            prominence_ratio=float(prominence_ratio),
        )

        res = detector.detect_from_image(
            image_to_process,
            draw_trajectory=draw_trajectory,
            draw_all_extrema=draw_all_extrema,
        )

        # Apply pattern type filter
        all_detected = res["patterns"]
        filtered_patterns = []
        for p in all_detected:
            if "Double Top" in p.name and not enable_double_top:
                continue
            if p.name == "Head and Shoulders" and not enable_head_shoulders:
                continue
            if "Double Bottom" in p.name and not enable_double_bottom:
                continue
            if p.name == "Inverse Head and Shoulders" and not enable_inverse_hs:
                continue
            if "Triangle" in p.name and not enable_triangles:
                continue
            filtered_patterns.append(p)

        # Re-annotate with filtered patterns
        annotator = ChartAnnotator()
        annotated_bgr = annotator.annotate(
            cv2.cvtColor(np.array(image_to_process), cv2.COLOR_RGB2BGR),
            patterns=filtered_patterns,
            extrema=res["extrema"],
            x_series=res["x_series"],
            y_series=res["y_series"],
            draw_trajectory=draw_trajectory,
            draw_all_extrema=draw_all_extrema,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Patterns Found", len(filtered_patterns))
    with m2:
        top_score = (
            f"{filtered_patterns[0].match_percentage:.1f}%"
            if filtered_patterns
            else "N/A"
        )
        st.metric("Top Match Score", top_score)
    with m3:
        top_name = (
            filtered_patterns[0].name.split("(")[0].strip()
            if filtered_patterns
            else "None"
        )
        st.metric("Dominant Pattern", top_name)
    with m4:
        theme = "Dark Mode" if res["is_dark_theme"] else "Light Mode"
        st.metric("Chart Theme", theme)

    st.markdown("---")

    # Main Visualization
    st.subheader("🎯 Annotated Chart with Detected Patterns")
    st.image(
        annotated_rgb,
        caption=f"Annotated: {image_name} with marked patterns and matching percentages",
        use_container_width=True,
    )

    # Download Button
    buf = BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    btn = st.download_button(
        label="💾 Download Annotated Screenshot (PNG)",
        data=buf.getvalue(),
        file_name=f"detected_{image_name}",
        mime="image/png",
    )

    st.markdown("---")

    # Scoreboard & Pattern Breakdown
    st.subheader("📊 Pattern Matching Scoreboard & Technical Analysis")

    if not filtered_patterns:
        st.warning(
            f"No chart patterns found matching $\\ge {min_match_pct}\\%$ confidence with current settings. Try lowering the 'Minimum Match Percentage' in the sidebar or adjusting peak sensitivity."
        )
    else:
        for idx, pat in enumerate(filtered_patterns, 1):
            theme_class = "pattern-card"
            if pat.pattern_type == "bearish_reversal":
                theme_class += " pattern-card-bearish"
                badge_bg = "#f04455"
                action_text = "🔻 Bearish Signal: Potential downward breakout or short opportunity below neckline."
            elif pat.pattern_type == "bullish_reversal":
                theme_class += " pattern-card-bullish"
                badge_bg = "#26a69a"
                action_text = "🟢 Bullish Signal: Potential upward breakout or buy opportunity above neckline."
            else:
                theme_class += " pattern-card-continuation"
                badge_bg = "#4285f4"
                action_text = (
                    "🔷 Continuation Signal: Trend consolidation within converging boundaries."
                )

            with st.container():
                st.markdown(
                    f"""
                    <div class="{theme_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #ffffff;">{idx}. {pat.name}</h3>
                            <span class="badge-score" style="background-color: {badge_bg}; color: white;">
                                {pat.match_percentage:.1f}% MATCH
                            </span>
                        </div>
                        <p style="color: #c5cbd3; margin-top: 0.5rem; margin-bottom: 0.3rem;">
                            <strong>Description:</strong> {pat.description}
                        </p>
                        <p style="color: #90caf9; margin-bottom: 0.5rem;">
                            <em>{action_text}</em>
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                # Sub-metrics progress bars & Coordinates
                col_scores, col_coords = st.columns([1, 1])

                with col_scores:
                    st.write("**Matching Metrics Breakdown:**")
                    for m_name, m_val in pat.metrics.items():
                        st.write(f"• **{m_name}**: `{m_val:.1f}%`")
                        st.progress(min(1.0, max(0.0, m_val / 100.0)))

                with col_coords:
                    st.write("**Key Pattern Vertices & Neckline:**")
                    coord_data = [
                        {
                            "Marker": pt.label,
                            "X (px)": pt.x,
                            "Y (px)": pt.y,
                            "Type": "Peak (High)"
                            if pt.is_peak
                            else "Valley (Low)",
                        }
                        for pt in pat.points
                    ]
                    st.dataframe(pd.DataFrame(coord_data), hide_index=True)

                    if pat.neckline is not None:
                        p1, p2 = pat.neckline
                        st.write(
                            f"**Neckline Coordinate**: `({p1[0]}, {p1[1]})` to `({p2[0]}, {p2[1]})`"
                        )

    # Computer Vision Diagnostics
    if show_cv_diagnostics:
        st.markdown("---")
        st.subheader(
            "🔬 Computer Vision & Edge Detection Diagnostics"
        )
        st.caption(
            "Under-the-hood steps: Original chart -> Canny edges with gridline removal -> Price Color Mask -> Reconstructed continuous price curve."
        )

        d1, d2 = st.columns(2)
        with d1:
            st.image(
                res["edges"],
                caption="Cleaned Edges (Canny + Long Gridline Suppression)",
                use_container_width=True,
            )
        with d2:
            st.image(
                res["price_mask"],
                caption="Filtered Price Region Mask (Candles & Trendlines)",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
