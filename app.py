# app.py
# FitCaddie Spec-Range MVP
# Streamlit Community Cloud app

import streamlit as st
import pandas as pd
from hosel_db import get_supported_brands, get_brand_systems, list_settings, translate_setting

from fit_engine import (
    load_gspro_csv,
    analyze,
    recommend_titleist_surefit,
    sure_fit_options
)

st.set_page_config(
    page_title="FitCaddie",
    page_icon="🏌️",
    layout="wide"
)

st.title("🏌️ FitCaddie — Spec Range Fitting")
st.caption("Upload a GSPro CSV export to analyze your clubs and receive fitting recommendations.")

# =============================================================================
# File Upload
# =============================================================================

uploaded_file = st.file_uploader(
    "Upload GSPro CSV export",
    type=["csv"]
)

st.subheader("Driver Setup")

brand = st.selectbox(
    "Driver Brand",
    get_supported_brands()
)

systems = get_brand_systems(brand)

system_name = st.selectbox(
    "Hosel System",
    [s.system_name for s in systems]
)

handedness = st.selectbox(
    "Handedness",
    ["RH", "LH"]
)

settings = list_settings(brand, system_name, handedness)

setting = st.selectbox(
    "Hosel Setting",
    settings
)

delta = translate_setting(
    brand,
    system_name,
    setting,
    handedness
)

st.write("Hosel Adjustment Translation:", delta)

if uploaded_file is None:
    st.info("Upload a GSPro CSV to begin analysis.")
    st.stop()

# =============================================================================
# Load + analyze
# =============================================================================

df = load_gspro_csv(uploaded_file)
analysis = analyze(df)

if not analysis:
    st.warning("No Driver, Fairway, or Hybrid shots detected in this file.")
    st.stop()

# =============================================================================
# Helper functions
# =============================================================================

def display_summary(summary):
    cols = st.columns(4)

    cols[0].metric("Club Speed", f"{summary['club_speed_mph']:.1f} mph")
    cols[1].metric("Ball Speed", f"{summary['ball_speed_mph']:.1f} mph")
    cols[2].metric("Launch", f"{summary['launch_deg']:.1f}°")
    cols[3].metric("Spin", f"{summary['spin_rpm']:.0f} rpm")

    cols = st.columns(4)

    cols[0].metric("Carry", f"{summary['carry_yd']:.1f} yd")
    cols[1].metric("Total", f"{summary['total_yd']:.1f} yd")
    cols[2].metric("Offline", f"{summary['offline_yd']:.1f} yd")
    cols[3].metric("Smash", f"{summary['smash_factor']:.2f}")


def display_variability(var):
    cols = st.columns(3)

    cols[0].metric("Offline Std Dev", f"{var['offline_std_yd']:.1f} yd")
    cols[1].metric("Carry Std Dev", f"{var['carry_std_yd']:.1f} yd")
    cols[2].metric("Spin Std Dev", f"{var['spin_std_rpm']:.0f} rpm")


def display_limiting(limiting):
    if not limiting:
        st.success("No major limiting factors detected.")
        return

    for item in limiting:
        if item["severity"] == "High":
            st.error(f"⚠ {item['title']}")
        elif item["severity"] == "Med":
            st.warning(f"⚠ {item['title']}")
        else:
            st.info(item["title"])

        st.write(item["detail"])


def display_recommendations(recs):
    for r in recs:
        st.write(f"• {r}")


# =============================================================================
# Hosel recommendation block
# =============================================================================

def render_hosel_block(bucket, bucket_analysis):

    st.subheader("Titleist SureFit Hosel Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        handedness = st.selectbox(
            f"{bucket} Handedness",
            ["RH", "LH"],
            key=f"{bucket}_hand"
        )

    with col2:
        options = sure_fit_options(bucket, handedness)

        current_setting = st.selectbox(
            "Current SureFit Setting",
            options,
            index=0,
            key=f"{bucket}_setting"
        )

    with col3:
        miss = st.selectbox(
            "Miss Tendency",
            ["RIGHT", "STRAIGHT", "LEFT"],
            index=1,
            key=f"{bucket}_miss"
        )

    summary = bucket_analysis.summary

    rec = recommend_titleist_surefit(
        bucket=bucket,
        handedness=handedness,
        club_speed_mph=summary["club_speed_mph"],
        launch_deg=summary["launch_deg"],
        spin_rpm=summary["spin_rpm"],
        miss_tendency=miss,
        current_setting=current_setting
    )

    st.markdown(
        f"""
### Recommended Hosel Change

**{rec.current or "—"} → {rec.recommended}**

**Loft Change:** {rec.loft_delta:+.2f}°  
**Lie Change:** {rec.lie_delta:+.2f}°
"""
    )

    with st.expander("Why this setting was chosen", expanded=True):
        for line in rec.rationale:
            st.write(f"- {line}")


# =============================================================================
# Club analysis tabs
# =============================================================================

tabs = st.tabs(["Driver", "Fairway Wood", "Hybrid"])

# =============================================================================
# DRIVER TAB
# =============================================================================

with tabs[0]:

    if "DR" not in analysis:
        st.info("No Driver data detected.")
    else:

        bucket = analysis["DR"]

        st.header("Driver Overview")
        display_summary(bucket.summary)

        st.header("Variability")
        display_variability(bucket.variability)

        st.header("Limiting Factors")
        display_limiting(bucket.limiting)

        st.header("Recommendations")
        display_recommendations(bucket.recs)

        render_hosel_block("DR", bucket)

        with st.expander("Driver Shot Data"):
            st.dataframe(bucket.df)


# =============================================================================
# FAIRWAY TAB
# =============================================================================

with tabs[1]:

    if "3W" not in analysis:
        st.info("No 3W / Fairway data detected.")
    else:

        bucket = analysis["3W"]

        st.header("Fairway Overview")
        display_summary(bucket.summary)

        st.header("Variability")
        display_variability(bucket.variability)

        st.header("Limiting Factors")
        display_limiting(bucket.limiting)

        st.header("Recommendations")
        display_recommendations(bucket.recs)

        render_hosel_block("3W", bucket)

        with st.expander("Fairway Shot Data"):
            st.dataframe(bucket.df)


# =============================================================================
# HYBRID TAB
# =============================================================================

with tabs[2]:

    if "HY" not in analysis:
        st.info("No Hybrid data detected.")
    else:

        bucket = analysis["HY"]

        st.header("Hybrid Overview")
        display_summary(bucket.summary)

        st.header("Variability")
        display_variability(bucket.variability)

        st.header("Limiting Factors")
        display_limiting(bucket.limiting)

        st.header("Recommendations")
        display_recommendations(bucket.recs)

        render_hosel_block("HY", bucket)

        with st.expander("Hybrid Shot Data"):
            st.dataframe(bucket.df)


# =============================================================================
# Export Section
# =============================================================================

st.divider()
st.header("Export Filtered Shot Data")

club_options = list(analysis.keys())

export_club = st.selectbox(
    "Select club dataset to export",
    club_options
)

export_df = analysis[export_club].df

csv = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    f"fitcaddie_{export_club}_filtered.csv",
    "text/csv"
)
