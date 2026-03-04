# app.py — FitCaddie (GSPro Club Fitting Engine MVP)
# Works with the full fit_engine.py I provided (imports: analyze_dataframe, session_to_dict)

import streamlit as st
import pandas as pd
import json

from fit_engine import analyze_dataframe, session_to_dict


# -----------------------------
# Page config + simple styling
# -----------------------------
st.set_page_config(
    page_title="FitCaddie",
    page_icon="⛳",
    layout="wide",
)

st.title("FitCaddie")
st.caption("Upload a GSPro CSV export and get spec-range recommendations (settings → shaft → head category).")

st.markdown(
    """
**What you’ll get:**
- Key averages (carry, ball speed, launch, spin, dispersion)
- Variability (consistency)
- Limiting factors (what’s holding you back)
- “What to test” spec-range recommendations

*Current MVP supports DR (driver), 3W (fairway), HY (hybrid) from GSPro CSV exports.*
"""
)


# -----------------------------
# Helpers
# -----------------------------
def _fmt_num(x, nd=1):
    try:
        if x is None:
            return "—"
        if isinstance(x, str):
            return x
        if pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _fmt_int(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{int(round(float(x)))}"
    except Exception:
        return "—"


def _club_label(code: str) -> str:
    return {"DR": "Driver", "3W": "Fairway Wood", "HY": "Hybrid"}.get(code, code)


def _confidence_badge(conf: str) -> str:
    conf = (conf or "").upper()
    if conf == "HIGH":
        return "🟢 HIGH"
    if conf == "MED":
        return "🟠 MED"
    return "🔴 LOW"


# -----------------------------
# Upload
# -----------------------------
st.subheader("Upload GSPro CSV")
uploaded = st.file_uploader("Upload a GSPro CSV export", type=["csv"])

if not uploaded:
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Run engine
try:
    result = analyze_dataframe(df)
    payload = session_to_dict(result)
except Exception as e:
    st.error(f"Could not analyze file: {e}")
    st.stop()

clubs = payload.get("clubs", {})
if not clubs:
    st.warning("No supported clubs found in this CSV (supported: DR, 3W, HY).")
    st.stop()


# -----------------------------
# Club Overview
# -----------------------------
st.subheader("Club Overview")

cols = st.columns(len(clubs))
for idx, (club_code, club_data) in enumerate(sorted(clubs.items(), key=lambda kv: kv[0])):
    summary = club_data.get("summary", {})
    metrics = summary.get("metrics", {})
    conf = summary.get("confidence", "LOW")
    n_used = summary.get("n_used", 0)
    n_total = summary.get("n_total", 0)

    with cols[idx]:
        st.markdown(f"#### {_club_label(club_code)} ({_confidence_badge(conf)})")
        st.markdown(f"**{n_used}/{n_total} shots used**")

        carry = metrics.get("carry")
        ball_speed = metrics.get("ball_speed")
        club_speed = metrics.get("club_speed")

        # launch may be under launch or launch_vla
        launch = metrics.get("launch")
        if launch is None or (isinstance(launch, float) and pd.isna(launch)):
            launch = metrics.get("launch_vla")

        spin = metrics.get("spin")
        offline_mean = metrics.get("offline_mean")

        st.write(f"• **Carry:** {_fmt_num(carry, 1)} yd")
        st.write(f"• **Ball Speed:** {_fmt_num(ball_speed, 1)} mph")
        st.write(f"• **Club Speed:** {_fmt_num(club_speed, 1)} mph")
        st.write(f"• **Launch:** {_fmt_num(launch, 1)}°")
        st.write(f"• **Spin:** {_fmt_int(spin)} rpm")
        st.write(f"• **Offline Mean:** {_fmt_num(offline_mean, 1)} yd")


# -----------------------------
# Details section
# -----------------------------
st.divider()
tabs = st.tabs([f"{code} Details" for code in sorted(clubs.keys())])

for tab, club_code in zip(tabs, sorted(clubs.keys())):
    club_data = clubs[club_code]
    summary = club_data.get("summary", {})
    metrics = summary.get("metrics", {})
    variability = summary.get("variability", {})
    limiting = club_data.get("limiting_factors", [])
    recs = club_data.get("recommendations", [])

    with tab:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown(f"### {_club_label(club_code)} Summary")
            st.caption(f"Shots used: {summary.get('n_used', 0)} / {summary.get('n_total', 0)}  |  Confidence: {summary.get('confidence', 'LOW')}")

            # Key metrics table
            key_rows = [
                ("Carry (yd)", metrics.get("carry")),
                ("Total (yd)", metrics.get("total")),
                ("Ball Speed (mph)", metrics.get("ball_speed")),
                ("Club Speed (mph)", metrics.get("club_speed")),
                ("Smash", metrics.get("smash")),
                ("Launch (VLA°)", metrics.get("launch_vla") if metrics.get("launch_vla") is not None else metrics.get("launch")),
                ("Spin (rpm)", metrics.get("spin")),
                ("Spin Axis", metrics.get("spin_axis")),
                ("AoA", metrics.get("aoa")),
                ("Path", metrics.get("path")),
                ("Face-to-Path", metrics.get("face_to_path")),
                ("Descent", metrics.get("descent")),
                ("Peak Height", metrics.get("peak_height")),
                ("Offline Mean (yd)", metrics.get("offline_mean")),
                ("Offline |abs| Mean (yd)", metrics.get("offline_abs_mean")),
            ]
            key_df = pd.DataFrame(key_rows, columns=["Metric", "Value"])
            key_df["Value"] = key_df["Value"].apply(lambda v: _fmt_num(v, 2))
            st.dataframe(key_df, use_container_width=True, hide_index=True)

            st.markdown("### Variability (consistency)")
            var_rows = [(k, v) for k, v in variability.items()]
            if var_rows:
                var_df = pd.DataFrame(var_rows, columns=["Metric", "Std Dev"])
                var_df["Std Dev"] = var_df["Std Dev"].apply(lambda v: _fmt_num(v, 2))
                st.dataframe(var_df, use_container_width=True, hide_index=True)
            else:
                st.info("No variability metrics available yet.")

        what_first = club_data.get("what_to_change_first", {}) or {}
        badge = what_first.get("badge", "—")
        why = what_first.get("why", "")

        st.markdown("### What to change first")
        st.info(f"**{badge}**\n\n{why}")

        with right:

            # ------------------------------
            # Driver Specs + SureFit Tool
            # ------------------------------
            if club_code == "DR":

                st.markdown("### Driver Specs")

                handed = st.selectbox("Handedness", ["RH", "LH"])
                current_setting = st.selectbox(
                    "Current SureFit Setting",
                    ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4","D1","D2","D3","D4"]
                )

                miss = st.selectbox(
                    "Primary Miss Tendency",
                    ["RIGHT", "LEFT", "BOTH", "NOT SURE"]
                )

                from fit_engine import recommend_titleist_surefit_driver, ClubSummary

                summary_obj = ClubSummary(
                    club=summary.get("club", "DR"),
                    n_total=summary.get("n_total", 0),
                    n_used=summary.get("n_used", 0),
                    confidence=summary.get("confidence", "LOW"),
                    metrics=summary.get("metrics", {}),
                    variability=summary.get("variability", {}),
                )

                hosel_rec = recommend_titleist_surefit_driver(
                    summary_obj,
                    handed,
                    current_setting,
                    miss
                )

                st.markdown("### Hosel Setting Recommendation")

                if hosel_rec["action"] == "change_setting":
                    st.success(f"Change SureFit: **{hosel_rec['from']} → {hosel_rec['to']}**")
                    st.write(hosel_rec["why"])
                    st.code(hosel_rec["expected"], language="python")

                else:
                    st.info(f"Stay at **{hosel_rec['from']}** — {hosel_rec['why']}")

            # ------------------------------
            # Limiting Factors (existing)
            # ------------------------------
            st.markdown("### Limiting Factors")
            if limiting:
                for f in limiting:
                    st.write(f"• {f}")
            else:
                st.write("• No major limiting factors detected.")

            st.markdown("### Recommendations (Spec Ranges)")
            if not recs:
                st.info("No recommendations available yet.")
            else:
                # sort by priority if present
                recs_sorted = sorted(recs, key=lambda r: r.get("priority", 999))
                for r in recs_sorted:
                    title = r.get("title", "Recommendation")
                    rationale = r.get("rationale", "")
                    conf = r.get("confidence", "")
                    spec = r.get("spec", {})

                    with st.expander(f"{title} ({conf})", expanded=True):
                        if rationale:
                            st.write(rationale)
                        if spec:
                            st.code(spec, language="python")


# -----------------------------
# Export
# -----------------------------
st.divider()
st.subheader("Export")
st.download_button(
    label="Download analysis JSON",
    data=json.dumps(payload, indent=2, default=str),
    file_name="fitcaddie_analysis.json",
    mime="application/json",
)
