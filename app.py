# app.py
import pandas as pd
import streamlit as st

from fit_engine import analyze_dataframe, session_to_dict

st.set_page_config(page_title="GSPro Fitting Engine (MVP)", layout="wide")

st.title("GSPro Club Fitting Engine (Spec-Range MVP)")
st.caption("Upload a GSPro CSV export and get spec-range recommendations (settings → shaft → head category).")

uploaded = st.file_uploader("Upload GSPro CSV", type=["csv"])

st.sidebar.header("Options")
show_raw = st.sidebar.checkbox("Show raw uploaded data", value=False)
show_cleaning_notes = st.sidebar.checkbox("Show cleaning notes", value=True)

if uploaded is None:
    st.info("Upload a GSPro CSV to begin.")
    st.stop()

# Read CSV
df = pd.read_csv(uploaded)
st.write("Detected Columns:")
if show_raw:
    st.subheader("Raw Data (first 200 rows)")
    st.dataframe(df.head(200), use_container_width=True)

# Run engine
try:
    result = analyze_dataframe(df)
    out = session_to_dict(result)
except Exception as e:
    st.error(f"Could not analyze file: {e}")
    st.stop()

clubs = out["clubs"]
if not clubs:
    st.warning("No supported clubs found (v1 supports: Driver (DR), 3-Wood (3W), Hybrids (H* / HY)).")
    st.stop()

# Overview
st.subheader("Club Overview")
cols = st.columns(len(clubs))
for i, (club, payload) in enumerate(clubs.items()):
    s = payload["summary"]
    m = s["metrics"]
    with cols[i]:
        st.metric(f"{club} (Confidence: {s['confidence']})", f"{s['n_used']}/{s['n_total']} shots used")
        st.write(
            f"- **Carry:** {m.get('carry', float('nan')):.1f}\n"
            f"- **Ball Speed:** {m.get('ball_speed', float('nan')):.1f}\n"
            f"- **Club Speed:** {m.get('club_speed', float('nan')):.1f}\n"
            f"- **Launch:** {m.get('launch_vla', float('nan')):.1f}°\n"
            f"- **Spin:** {m.get('spin', float('nan')):.0f} rpm\n"
            f"- **Offline Mean:** {m.get('offline_mean', float('nan')):.1f}"
        )

st.divider()

# Club detail tabs
tab_names = list(clubs.keys())
tabs = st.tabs([f"{c} Details" for c in tab_names])

for tab, club in zip(tabs, tab_names):
    payload = clubs[club]
    s = payload["summary"]
    m = s["metrics"]
    v = s["variability"]

    with tab:
        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"### {club} Summary")
            st.write(f"**Shots used:** {s['n_used']} / {s['n_total']}  |  **Confidence:** {s['confidence']}")
            st.markdown("**Key metrics**")
            metric_table = pd.DataFrame(
                {
                    "Metric": ["Carry", "Total", "Ball Speed", "Club Speed", "Smash", "Launch (VLA)", "Spin", "AoA", "Path", "Face-to-Path", "Descent"],
                    "Value": [
                        m.get("carry"), m.get("total"), m.get("ball_speed"), m.get("club_speed"), m.get("smash"),
                        m.get("launch_vla"), m.get("spin"), m.get("aoa"), m.get("path"), m.get("face_to_path"), m.get("descent")
                    ],
                }
            )
            st.dataframe(metric_table, hide_index=True, use_container_width=True)

            st.markdown("**Variability (consistency)**")
            var_table = pd.DataFrame(
                {
                    "Metric": ["Offline Std", "Launch Std", "Spin Std", "Face-to-Path Std", "Path Std", "Ball Speed Std"],
                    "Std Dev": [
                        v.get("offline_std"), v.get("launch_std"), v.get("spin_std"),
                        v.get("face_to_path_std"), v.get("path_std"), v.get("ball_speed_std"),
                    ],
                }
            )
            st.dataframe(var_table, hide_index=True, use_container_width=True)

        with right:
            st.markdown("### Limiting Factors")
            lf = payload["limiting_factors"]
            if lf:
                for item in lf:
                    st.write(f"• {item}")
            else:
                st.success("No major limiting factors flagged for v1 targets.")

            st.markdown("### Recommendations (Spec Ranges)")
            for r in payload["recommendations"]:
                with st.expander(f"{r['priority']}. {r['title']}  ({r['confidence']})", expanded=(r["priority"] <= 2)):
                    st.write(r["rationale"])
                    st.code(r["spec"], language="json")

        st.divider()
        st.markdown("### Quick Plots")

        # Basic plots if columns exist in raw df
        # We'll map club rows from uploaded df for simple visualizations
        # Note: v1 keeps plots simple; later you can add dispersion ellipses.
        df_local = df.copy()
        if "Club" in df_local.columns:
            club_col = "Club"
        elif "club" in df_local.columns:
            club_col = "club"
        else:
            club_col = None

        if club_col:
            df_local[club_col] = df_local[club_col].astype(str).str.upper().str.strip()

            # Attempt to select same bucket
            if club == "DR":
                df_plot = df_local[df_local[club_col] == "DR"]
            elif club == "3W":
                df_plot = df_local[df_local[club_col].isin(["3W", "FW", "W3"])]
            else:  # HY
                df_plot = df_local[df_local[club_col].str.startswith("H") | df_local[club_col].str.contains("HY", na=False)]

            # Canonicalize plot columns
            from fit_engine import _canonicalize_columns  # internal
            df_plot = _canonicalize_columns(df_plot)
            for col in ["carry", "offline", "spin", "vla", "ball_speed", "club_speed"]:
                if col in df_plot.columns:
                    df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

            plot_cols = st.columns(2)

            with plot_cols[0]:
                if "offline" in df_plot.columns and "carry" in df_plot.columns:
                    st.scatter_chart(df_plot[["offline", "carry"]].dropna(), x="offline", y="carry")
                else:
                    st.info("Need Offline + Carry columns for dispersion plot.")

            with plot_cols[1]:
                if "spin" in df_plot.columns and "vla" in df_plot.columns:
                    st.scatter_chart(df_plot[["vla", "spin"]].dropna(), x="vla", y="spin")
                else:
                    st.info("Need VLA + Spin columns for launch/spin plot.")
        else:
            st.info("Could not detect club column for plotting.")

# Download JSON output
st.divider()
st.subheader("Export")
st.download_button(
    label="Download analysis as JSON",
    data=pd.Series(out).to_json(),
    file_name="fit_analysis.json",
    mime="application/json",
)
