from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from viz import render_dispersion, render_compare_dispersion

from hosel_db import (
    get_supported_brands,
    get_brand_systems,
    list_settings,
    translate_setting,
    system_ranges,
)

from fit_engine import (
    DriverUserSetup,
    build_driver_recommendations,
    canonicalize,
    club_family,
    compare_driver_setups,
    estimate_launch_spin_change,
    miss_tendency,
    pick_one_hosel_setting,
    smash_flag_driver,
    summarize_by_club,
    targets_for_club,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="FitCaddie", layout="wide")
st.title("FitCaddie")
st.caption("A cleaner fitter workflow: one club at a time, clearer next steps, easier setup testing.")


# -----------------------------
# Theme / Styling
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --fc-blue: #1f77d0;
        --fc-blue-dark: #103e6e;
        --fc-blue-soft: #eff6ff;
        --fc-border: #d7e6f7;
        --fc-text: #16324f;
        --fc-green-bg: #edf9f1;
        --fc-green-border: #87d4a1;
        --fc-yellow-bg: #fff8e8;
        --fc-yellow-border: #e8c96b;
        --fc-red-bg: #fff1f1;
        --fc-red-border: #e09a9a;
        --fc-card-bg: #ffffff;
        --fc-panel-bg: #f8fbff;
    }

    .fc-hero {
        background: linear-gradient(135deg, #1f77d0 0%, #245f9c 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(16, 62, 110, 0.12);
    }

    .fc-card {
        background: var(--fc-card-bg);
        border: 1px solid var(--fc-border);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(31, 119, 208, 0.06);
        margin-bottom: 14px;
    }

    .fc-card h3, .fc-card h4 {
        margin-top: 0;
        color: var(--fc-blue-dark);
    }

    .fc-mini {
        background: var(--fc-panel-bg);
        border: 1px solid var(--fc-border);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .fc-rec-green, .fc-rec-yellow, .fc-rec-red {
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 2px solid;
    }

    .fc-rec-green {
        background: var(--fc-green-bg);
        border-color: var(--fc-green-border);
    }

    .fc-rec-yellow {
        background: var(--fc-yellow-bg);
        border-color: var(--fc-yellow-border);
    }

    .fc-rec-red {
        background: var(--fc-red-bg);
        border-color: var(--fc-red-border);
    }

    .fc-status {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .fc-status-green {
        background: #d9f3e2;
        color: #17653a;
    }

    .fc-status-yellow {
        background: #fff0c8;
        color: #7a5b00;
    }

    .fc-status-red {
        background: #ffd7d7;
        color: #8a1f1f;
    }

    .fc-verdict {
        background: var(--fc-blue-soft);
        border: 2px solid var(--fc-blue);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    .fc-section-title {
        color: var(--fc-blue-dark);
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state defaults
# -----------------------------
def _init_state():
    defaults = {
        "analysis_mode": "Single Club Analysis",
        "k_loft_to_dynamic": 1.0,
        "min_shots": 5,
        "show_raw": False,
        "selected_focus_family": "Driver",
        "selected_focus_club": "DR",

        # Single mode
        "single_driver_brand": "Titleist",
        "single_driver_model": "TSR3",
        "single_driver_loft": 10.0,
        "single_driver_hosel": "A1",
        "single_driver_shaft_model": "HZRDUS Black",
        "single_driver_shaft_weight": 60.0,
        "single_driver_shaft_flex": "6.0",

        # Compare mode A
        "cmpA_driver_brand": "Titleist",
        "cmpA_driver_model": "TSR3",
        "cmpA_driver_loft": 10.0,
        "cmpA_driver_hosel": "A1",
        "cmpA_driver_shaft_model": "HZRDUS Black",
        "cmpA_driver_shaft_weight": 60.0,
        "cmpA_driver_shaft_flex": "6.0",

        # Compare mode B
        "cmpB_driver_brand": "Titleist",
        "cmpB_driver_model": "TSR3",
        "cmpB_driver_loft": 10.0,
        "cmpB_driver_hosel": "A2",
        "cmpB_driver_shaft_model": "HZRDUS Smoke Red RDX",
        "cmpB_driver_shaft_weight": 60.0,
        "cmpB_driver_shaft_flex": "6.0",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# -----------------------------
# Helpers
# -----------------------------
def _fmt(value, decimals=1, suffix=""):
    if value is None:
        return "—"
    try:
        if np.isnan(value):
            return "—"
    except Exception:
        pass
    return f"{value:.{decimals}f}{suffix}"


def _reset_system_and_settings_for_club(club_id: str):
    brand_key = f"{club_id}_brand"
    sys_key = f"{club_id}_sys"
    cur_key = f"{club_id}_cur"
    new_key = f"{club_id}_new"
    hand_key = f"{club_id}_hand"

    brand = st.session_state.get(brand_key, None)
    handedness = st.session_state.get(hand_key, "RH")

    systems = get_brand_systems(brand) if brand else []
    system_names = [s.system_name for s in systems]

    if not system_names:
        st.session_state[sys_key] = "(no systems found)"
        st.session_state[cur_key] = "STD"
        st.session_state[new_key] = "STD"
        return

    st.session_state[sys_key] = system_names[0]
    settings = list_settings(brand, st.session_state[sys_key], handedness)
    first_setting = settings[0] if settings else "STD"
    st.session_state[cur_key] = first_setting
    st.session_state[new_key] = first_setting


def _reset_settings_for_club(club_id: str):
    brand_key = f"{club_id}_brand"
    sys_key = f"{club_id}_sys"
    cur_key = f"{club_id}_cur"
    new_key = f"{club_id}_new"
    hand_key = f"{club_id}_hand"

    brand = st.session_state.get(brand_key, None)
    system_name = st.session_state.get(sys_key, None)
    handedness = st.session_state.get(hand_key, "RH")

    settings = list_settings(brand, system_name, handedness) if (brand and system_name) else []
    first_setting = settings[0] if settings else "STD"
    st.session_state[cur_key] = first_setting
    st.session_state[new_key] = first_setting


def _driver_setup_from_prefix(prefix: str) -> DriverUserSetup:
    return DriverUserSetup(
        brand=st.session_state[f"{prefix}_driver_brand"],
        model=st.session_state[f"{prefix}_driver_model"],
        loft_deg=float(st.session_state[f"{prefix}_driver_loft"]),
        hosel_setting=st.session_state[f"{prefix}_driver_hosel"],
        shaft_model=st.session_state[f"{prefix}_driver_shaft_model"],
        shaft_weight_g=float(st.session_state[f"{prefix}_driver_shaft_weight"]),
        shaft_flex=st.session_state[f"{prefix}_driver_shaft_flex"],
    )


def _render_driver_setup(prefix: str, title: str):
    st.markdown(f'<div class="fc-card"><h3>{title}</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        brand_options = ["Titleist", "Ping", "Other"]
        current_brand = st.session_state[f"{prefix}_driver_brand"]
        if current_brand not in brand_options:
            current_brand = "Other"
        st.session_state[f"{prefix}_driver_brand"] = st.selectbox(
            "Driver Brand",
            brand_options,
            index=brand_options.index(current_brand),
            key=f"{prefix}_driver_brand_select",
        )

    with c2:
        brand = st.session_state[f"{prefix}_driver_brand"]
        if brand == "Titleist":
            model_options = ["TSR2", "TSR3", "GT2", "GT3", "Other"]
        elif brand == "Ping":
            model_options = ["G430 Max", "G430 LST", "G440 Max", "G440 LST", "Other"]
        else:
            model_options = ["Other"]

        current_model = st.session_state[f"{prefix}_driver_model"]
        if current_model not in model_options:
            current_model = model_options[0]

        st.session_state[f"{prefix}_driver_model"] = st.selectbox(
            "Head Model",
            model_options,
            index=model_options.index(current_model),
            key=f"{prefix}_driver_model_select",
        )

    with c3:
        st.session_state[f"{prefix}_driver_loft"] = st.number_input(
            "Loft (°)",
            min_value=7.0,
            max_value=14.0,
            value=float(st.session_state[f"{prefix}_driver_loft"]),
            step=0.5,
            key=f"{prefix}_driver_loft_input",
        )

    with c4:
        st.session_state[f"{prefix}_driver_hosel"] = st.text_input(
            "Hosel Setting",
            value=st.session_state[f"{prefix}_driver_hosel"],
            key=f"{prefix}_driver_hosel_input",
        )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.session_state[f"{prefix}_driver_shaft_model"] = st.text_input(
            "Shaft Model",
            value=st.session_state[f"{prefix}_driver_shaft_model"],
            key=f"{prefix}_driver_shaft_model_input",
        )
    with s2:
        st.session_state[f"{prefix}_driver_shaft_weight"] = st.number_input(
            "Shaft Weight (g)",
            min_value=40.0,
            max_value=90.0,
            value=float(st.session_state[f"{prefix}_driver_shaft_weight"]),
            step=1.0,
            key=f"{prefix}_driver_shaft_weight_input",
        )
    with s3:
        st.session_state[f"{prefix}_driver_shaft_flex"] = st.text_input(
            "Shaft Flex",
            value=st.session_state[f"{prefix}_driver_shaft_flex"],
            key=f"{prefix}_driver_shaft_flex_input",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _status_html(tone: str) -> str:
    if tone == "green":
        return '<span class="fc-status fc-status-green">Positive change</span>'
    if tone == "yellow":
        return '<span class="fc-status fc-status-yellow">Test carefully</span>'
    return '<span class="fc-status fc-status-red">Avoid first</span>'


def _render_recommendation_cards(bundle):
    for block in [bundle.swing, bundle.driver_settings, bundle.equipment_adjustment]:
        css_class = {
            "green": "fc-rec-green",
            "yellow": "fc-rec-yellow",
            "red": "fc-rec-red",
        }.get(block.tone, "fc-rec-yellow")

        st.markdown(
            f"""
            <div class="{css_class}">
                {_status_html(block.tone)}
                <h4>{block.title}</h4>
                <p><strong>Suggestion:</strong> {block.suggestion}</p>
                <p><strong>Why:</strong> {block.why}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_summary_cards(summary):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Club Speed", _fmt(summary.club_speed_avg, 1))
    c2.metric("Ball Speed", _fmt(summary.ball_speed_avg, 1))
    c3.metric("Smash", _fmt(summary.smash_avg, 2))
    c4.metric("Carry", _fmt(summary.carry_avg, 1))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Offline", _fmt(summary.offline_avg, 1))
    c6.metric("Launch", _fmt(summary.vla_avg, 1))
    c7.metric("Spin", _fmt(summary.spin_avg, 0))
    c8.metric("AoA", _fmt(summary.aoa_avg, 1))


def _render_focus_picker(selected_clubs: List[str]):
    families_present = []
    if any(c == "DR" for c in selected_clubs):
        families_present.append("Driver")
    if any(c.endswith("W") for c in selected_clubs):
        families_present.append("Fairway Wood")
    if any(c.endswith("H") for c in selected_clubs):
        families_present.append("Hybrid")

    if not families_present:
        st.warning("No supported driver, fairway wood, or hybrid data found.")
        st.stop()

    if st.session_state["selected_focus_family"] not in families_present:
        st.session_state["selected_focus_family"] = families_present[0]

    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.subheader("Choose Fitting Focus")

    st.session_state["selected_focus_family"] = st.radio(
        "What are we fitting today?",
        families_present,
        horizontal=True,
        index=families_present.index(st.session_state["selected_focus_family"]),
    )

    if st.session_state["selected_focus_family"] == "Driver":
        available = [c for c in selected_clubs if c == "DR"]
    elif st.session_state["selected_focus_family"] == "Fairway Wood":
        available = [c for c in selected_clubs if c.endswith("W")]
    else:
        available = [c for c in selected_clubs if c.endswith("H")]

    if st.session_state["selected_focus_club"] not in available:
        st.session_state["selected_focus_club"] = available[0]

    st.session_state["selected_focus_club"] = st.selectbox(
        "Choose club",
        available,
        index=available.index(st.session_state["selected_focus_club"]),
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state["selected_focus_club"]


def _render_hosel_block(club_id: str, title: str, k_loft_to_dynamic: float) -> Dict[str, Dict]:
    hosel_configs: Dict[str, Dict] = {}

    st.markdown(f'<div class="fc-card"><h3>{title}</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.0, 1.3, 1.2, 1.1])

    with c1:
        handedness = st.selectbox(
            f"{club_id} Hand",
            ["RH", "LH"],
            index=0,
            key=f"{club_id}_hand",
            on_change=_reset_settings_for_club,
            args=(club_id,),
        )

    with c2:
        brand = st.selectbox(
            f"{club_id} Brand",
            get_supported_brands(),
            index=0,
            key=f"{club_id}_brand",
            on_change=_reset_system_and_settings_for_club,
            args=(club_id,),
        )

    systems = get_brand_systems(brand)
    system_names = [s.system_name for s in systems] if systems else ["(no systems found)"]

    with c3:
        system_name = st.selectbox(
            f"{club_id} Hosel System",
            system_names,
            index=0,
            key=f"{club_id}_sys",
            on_change=_reset_settings_for_club,
            args=(club_id,),
        )

    with c4:
        if club_id == "DR":
            default_loft = 10.0
        elif club_id.endswith("W"):
            default_loft = 15.0
        elif club_id.endswith("H"):
            default_loft = 18.0
        else:
            default_loft = 10.0

        stated_loft = st.number_input(
            f"{club_id} Stated Loft (°)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_loft),
            step=0.5,
            key=f"{club_id}_loft",
        )

    settings = list_settings(brand, system_name, handedness)

    s1, s2 = st.columns(2)
    with s1:
        current_setting = st.selectbox(
            f"{club_id} Current Setting",
            settings if settings else ["STD"],
            key=f"{club_id}_cur",
        )
    with s2:
        proposed_setting = st.selectbox(
            f"{club_id} Proposed Setting",
            settings if settings else ["STD"],
            key=f"{club_id}_new",
        )

    cur_delta = translate_setting(brand, system_name, current_setting, handedness)
    new_delta = translate_setting(brand, system_name, proposed_setting, handedness)

    cur_loft = getattr(cur_delta, "loft_deg", None)
    new_loft = getattr(new_delta, "loft_deg", None)

    st.markdown("#### Projected Change")
    if (cur_loft is not None) and (new_loft is not None) and (proposed_setting != current_setting):
        delta_static_loft = new_loft - cur_loft
        est = estimate_launch_spin_change(delta_static_loft, k_loft_to_dynamic, club_id)
        st.info(
            f"Estimated launch change: **{est.launch_change_deg:+.1f}°** "
            f"(range {est.launch_range_deg[0]:+.1f}° to {est.launch_range_deg[1]:+.1f}°)\n\n"
            f"Estimated spin change: **{est.spin_change_rpm:+d} rpm** "
            f"(range {est.spin_range_rpm[0]:+d} to {est.spin_range_rpm[1]:+d})"
        )
        st.caption(est.notes)
    elif proposed_setting != current_setting:
        ranges = system_ranges(brand, system_name)
        st.warning(
            "Exact deltas for these settings are not encoded yet.\n\n"
            f"System ranges: loft={ranges.get('loft_range_deg')}, lie={ranges.get('lie_range_deg')}."
        )
    else:
        st.caption("Choose a different proposed setting to see projected launch and spin changes.")

    st.markdown("</div>", unsafe_allow_html=True)

    hosel_configs[club_id] = {
        "club_id": club_id,
        "handedness": handedness,
        "brand": brand,
        "system_name": system_name,
        "stated_loft": stated_loft,
        "current_setting": current_setting,
        "proposed_setting": proposed_setting,
        "cur_delta": asdict(cur_delta),
        "new_delta": asdict(new_delta),
    }
    return hosel_configs


def _render_advanced_analysis(club_id: str, canon_df: pd.DataFrame, hosel_configs: Dict[str, Dict], k_loft_to_dynamic: float):
    summary = summarize_by_club(canon_df)[club_id]
    t = targets_for_club(club_id, summary.club_speed_avg)
    launch_lo, launch_hi = t["launch"]
    spin_lo, spin_hi = t["spin"]

    with st.expander("Advanced Analysis"):
        st.markdown("### Limiting Factors")
        lim: List[str] = []

        miss = miss_tendency(summary.offline_avg)
        lim.append(f"Miss tendency: **{miss}**" if miss != "Unknown" else "Miss tendency: Unknown")

        if club_id == "DR":
            smash_msg = smash_flag_driver(summary.smash_avg)
            if smash_msg:
                lim.append(smash_msg)

        if not np.isnan(summary.vla_avg) and (summary.vla_avg < launch_lo or summary.vla_avg > launch_hi):
            lim.append(f"Launch window miss: {summary.vla_avg:.1f}° vs target {launch_lo:.1f}–{launch_hi:.1f}°.")
        if not np.isnan(summary.spin_avg) and (summary.spin_avg < spin_lo or summary.spin_avg > spin_hi):
            lim.append(f"Spin window miss: {summary.spin_avg:.0f} rpm vs target {spin_lo:.0f}–{spin_hi:.0f} rpm.")

        for item in lim:
            st.write("•", item)

        st.markdown("### Settings Recommendation")
        recos: List[str] = []

        if club_id in hosel_configs:
            h = hosel_configs[club_id]
            brand = h["brand"]
            system_name = h["system_name"]
            handedness = h["handedness"]
            current_setting = h["current_setting"]

            needed_loft = 0.0
            needed_lie = 0.0

            if not np.isnan(summary.vla_avg):
                launch_per_static_loft = 0.85 * k_loft_to_dynamic
                if summary.vla_avg < launch_lo:
                    needed_loft = min(2.0, (launch_lo - summary.vla_avg) / max(0.05, launch_per_static_loft))
                elif summary.vla_avg > launch_hi:
                    needed_loft = max(-2.0, -(summary.vla_avg - launch_hi) / max(0.05, launch_per_static_loft))

            miss = miss_tendency(summary.offline_avg)
            if miss == "Right miss tendency":
                needed_lie = +0.75
            elif miss == "Left miss tendency":
                needed_lie = -0.75

            if not np.isnan(summary.spin_avg):
                if summary.spin_avg < spin_lo:
                    needed_loft = max(needed_loft, +0.75)
                elif summary.spin_avg > spin_hi:
                    needed_loft = min(needed_loft, -0.75)

            settings = list_settings(brand, system_name, handedness)
            reco = pick_one_hosel_setting(
                settings=settings,
                translate_fn=translate_setting,
                brand=brand,
                system_name=system_name,
                handedness=handedness,
                current_setting=current_setting,
                needed_loft_delta=needed_loft,
                needed_lie_delta=needed_lie,
            )

            if abs(needed_loft) < 0.25 and abs(needed_lie) < 0.25:
                recos.append("Current setting looks reasonable for launch, spin, and start-line tendencies.")
            else:
                recos.append(f"Hosel goal: loft Δ **{needed_loft:+.2f}°**, lie Δ **{needed_lie:+.2f}°**.")
                if reco["type"] == "exact":
                    r = reco["recommended"]
                    recos.append(
                        f"Suggested change: **{reco['current']} → {r['setting']}** "
                        f"(loft {r['loft_delta']:+.2f}°, lie {r['lie_delta']:+.2f}°)."
                    )
                else:
                    recos.append(reco["message"])
        else:
            recos.append("Configure this club in Club Settings to enable a setting recommendation.")

        for r in recos:
            st.write("•", r)

        st.markdown("### Shot Table")
        show_cols = [
            "club_raw", "club_id",
            "club_speed_mph", "ball_speed_mph", "smash",
            "carry_yd", "total_yd", "offline_yd",
            "vla_deg", "backspin_rpm", "aoa_deg",
            "club_path_deg", "face_to_path_deg", "face_to_target_deg",
        ]
        cols_present = [c for c in show_cols if c in canon_df.columns]
        st.dataframe(
            canon_df[canon_df["club_id"] == club_id][cols_present].reset_index(drop=True),
            use_container_width=True,
        )


def _render_driver_recommendations(driver_df: pd.DataFrame, driver_setup: DriverUserSetup):
    summaries = summarize_by_club(driver_df)
    if "DR" not in summaries:
        st.info("No valid driver summary available.")
        return

    offline_valid = driver_df["offline_yd"].dropna()
    fairway_pct = float((offline_valid.abs() <= 15).mean() * 100.0) if len(offline_valid) else np.nan

    bundle = build_driver_recommendations(
        summary=summaries["DR"],
        user_setup=driver_setup,
        fairway_hit_pct=fairway_pct if not np.isnan(fairway_pct) else None,
    )

    st.markdown('<div class="fc-card"><h3>Fitter Recommendations</h3></div>', unsafe_allow_html=True)
    _render_recommendation_cards(bundle)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode")
    st.session_state["analysis_mode"] = st.radio(
        "Choose analysis type",
        ["Single Club Analysis", "Compare Driver Setups"],
        index=0 if st.session_state["analysis_mode"] == "Single Club Analysis" else 1,
    )

    st.divider()
    st.header("Model")
    st.session_state["k_loft_to_dynamic"] = st.slider(
        "Loft → delivered loft multiplier (k)",
        min_value=0.6,
        max_value=1.6,
        value=float(st.session_state["k_loft_to_dynamic"]),
        step=0.05,
    )

    st.divider()
    st.header("Filters")
    st.session_state["min_shots"] = st.slider(
        "Min shots per club",
        5,
        50,
        int(st.session_state["min_shots"]),
        1,
    )

    st.divider()
    st.header("Debug")
    st.session_state["show_raw"] = st.checkbox("Show raw tables", value=bool(st.session_state["show_raw"]))


analysis_mode = st.session_state["analysis_mode"]
k_loft_to_dynamic = float(st.session_state["k_loft_to_dynamic"])
min_shots = int(st.session_state["min_shots"])
show_raw = bool(st.session_state["show_raw"])


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="fc-hero">
        <h2 style="margin:0 0 8px 0;">Cleaner fitting workflow</h2>
        <div>Upload data, pick one club, get clearer next steps. Use compare mode separately when you want to test two driver setups.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Single Club Analysis
# -----------------------------
if analysis_mode == "Single Club Analysis":
    with st.sidebar:
        st.header("Upload")
        uploaded = st.file_uploader("Upload GSPro CSV", type=["csv"], key="single_upload")

    if not uploaded:
        st.info("Upload a GSPro CSV to begin.")
        st.stop()

    raw_df = pd.read_csv(uploaded)
    canon_df, fmt = canonicalize(raw_df)

    st.success(f"Loaded {len(canon_df)} shots. Detected export format: **{fmt}**")

    if show_raw:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Raw CSV")
            st.dataframe(raw_df.head(200), use_container_width=True)
        with c2:
            st.subheader("Canonicalized")
            st.dataframe(canon_df.head(200), use_container_width=True)

    club_counts = canon_df["club_id"].value_counts().to_dict()
    club_ids_all = [c for c in club_counts.keys() if c != "OTHER"]
    club_ids = [c for c in club_ids_all if club_counts.get(c, 0) >= min_shots]

    if not club_ids:
        st.warning(f"No clubs have at least {min_shots} shots. Try lowering the filter or collect more data.")
        st.stop()

    selected_clubs = st.multiselect(
        "Detected clubs in this upload",
        options=club_ids,
        default=club_ids,
    )

    if not selected_clubs:
        st.stop()

    canon_df = canon_df[canon_df["club_id"].isin(selected_clubs)].copy()
    focus_club = _render_focus_picker(selected_clubs)
    focus_df = canon_df[canon_df["club_id"] == focus_club].copy()

    summaries = summarize_by_club(focus_df)
    if focus_club not in summaries:
        st.warning("No valid data for the selected club.")
        st.stop()

    focus_summary = summaries[focus_club]

    top1, top2 = st.columns([1.35, 1.0])

    with top1:
        st.markdown('<div class="fc-card"><h3>Dispersion</h3>', unsafe_allow_html=True)
        render_dispersion(focus_df, key_prefix="single_focus")
        st.markdown("</div>", unsafe_allow_html=True)

    with top2:
        st.markdown(f'<div class="fc-card"><h3>{focus_club} Overview</h3>', unsafe_allow_html=True)
        _render_summary_cards(focus_summary)
        st.markdown("</div>", unsafe_allow_html=True)

        if focus_club == "DR":
            _render_driver_setup("single", "Driver Setup")
            _render_driver_recommendations(focus_df, _driver_setup_from_prefix("single"))

    if focus_club != "DR":
        st.markdown('<div class="fc-card"><h3>Fitter Guidance</h3>', unsafe_allow_html=True)
        miss = miss_tendency(focus_summary.offline_avg)
        family = club_family(focus_club)

        guidance_lines = []

        if family == "Fairway Wood":
            if not np.isnan(focus_summary.vla_avg) and focus_summary.vla_avg < targets_for_club(focus_club, focus_summary.club_speed_avg)["launch"][0]:
                guidance_lines.append("This club appears to need more launch first. Try more loft or a more launch-friendly setup before chasing lower spin.")
            if miss == "Right miss tendency":
                guidance_lines.append("Your miss pattern trends right, so test a slightly more upright or draw-help setting before changing flex.")
            if np.isnan(focus_summary.vla_avg) or np.isnan(focus_summary.spin_avg):
                guidance_lines.append("Keep collecting shots so the app can get cleaner launch and spin trends.")
        elif family == "Hybrid":
            if miss == "Centered":
                guidance_lines.append("This hybrid looks fairly playable. Keep settings stable and focus on repeatability.")
            elif miss == "Right miss tendency":
                guidance_lines.append("If this hybrid leaks right, test a slightly more upright setting or slightly softer-feeling profile before going stiffer.")
            else:
                guidance_lines.append("If this hybrid turns over too much, test a more neutral setting before changing shaft.")
        else:
            guidance_lines.append("Collect more shots for cleaner guidance.")

        for line in guidance_lines:
            st.write("•", line)
        st.markdown("</div>", unsafe_allow_html=True)

    hosel_configs = _render_hosel_block(
        club_id=focus_club,
        title=f"Club Settings — {focus_club}",
        k_loft_to_dynamic=k_loft_to_dynamic,
    )

    _render_advanced_analysis(
        club_id=focus_club,
        canon_df=focus_df,
        hosel_configs=hosel_configs,
        k_loft_to_dynamic=k_loft_to_dynamic,
    )


# -----------------------------
# Compare Driver Setups
# -----------------------------
else:
    with st.sidebar:
        st.header("Compare Uploads")
        uploaded_a = st.file_uploader("Upload Setup A CSV", type=["csv"], key="compare_upload_a")
        uploaded_b = st.file_uploader("Upload Setup B CSV", type=["csv"], key="compare_upload_b")

    st.markdown('<div class="fc-card"><h3>Compare Two Driver Setups</h3><p>Best for A1 vs A2, shaft vs shaft, or head vs head testing.</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        _render_driver_setup("cmpA", "Setup A")
    with c2:
        _render_driver_setup("cmpB", "Setup B")

    if not uploaded_a or not uploaded_b:
        st.info("Upload both Setup A and Setup B CSV files to compare them.")
        st.stop()

    raw_a = pd.read_csv(uploaded_a)
    raw_b = pd.read_csv(uploaded_b)

    canon_a, fmt_a = canonicalize(raw_a)
    canon_b, fmt_b = canonicalize(raw_b)

    st.success(
        f"Loaded Setup A: {len(canon_a)} shots (**{fmt_a}**) | "
        f"Setup B: {len(canon_b)} shots (**{fmt_b}**)"
    )

    if show_raw:
        ra, rb = st.columns(2)
        with ra:
            st.subheader("Setup A — Canonicalized")
            st.dataframe(canon_a.head(200), use_container_width=True)
        with rb:
            st.subheader("Setup B — Canonicalized")
            st.dataframe(canon_b.head(200), use_container_width=True)

    dr_a = canon_a[canon_a["club_id"] == "DR"].copy()
    dr_b = canon_b[canon_b["club_id"] == "DR"].copy()

    if dr_a.empty or dr_b.empty:
        st.warning("Both uploaded files need driver shots (DR) for compare mode.")
        st.stop()

    compare = compare_driver_setups(dr_a, dr_b, "Setup A", "Setup B")

    map_a, map_b = st.columns(2)
    with map_a:
        st.markdown('<div class="fc-card"><h3>Setup A Dispersion</h3>', unsafe_allow_html=True)
        render_dispersion(dr_a)
        st.markdown("</div>", unsafe_allow_html=True)
    with map_b:
        st.markdown('<div class="fc-card"><h3>Setup B Dispersion</h3>', unsafe_allow_html=True)
        render_dispersion(dr_b)
        st.markdown("</div>", unsafe_allow_html=True)

    a = compare["a"]
    b = compare["b"]

    st.markdown('<div class="fc-card"><h3>Side-by-Side Metrics</h3>', unsafe_allow_html=True)
    metric_df = pd.DataFrame(
        {
            "Metric": [
                "Shots", "Club Speed", "Ball Speed", "Smash",
                "Carry", "Total", "Launch", "Spin", "AoA",
                "Avg Abs Offline", "Fairway %"
            ],
            "Setup A": [
                a.shots,
                _fmt(a.club_speed, 1),
                _fmt(a.ball_speed, 1),
                _fmt(a.smash, 2),
                _fmt(a.carry, 1),
                _fmt(a.total, 1),
                _fmt(a.launch, 1),
                _fmt(a.spin, 0),
                _fmt(a.aoa, 1),
                _fmt(a.offline, 1),
                _fmt(a.fairway_pct, 0, "%"),
            ],
            "Setup B": [
                b.shots,
                _fmt(b.club_speed, 1),
                _fmt(b.ball_speed, 1),
                _fmt(b.smash, 2),
                _fmt(b.carry, 1),
                _fmt(b.total, 1),
                _fmt(b.launch, 1),
                _fmt(b.spin, 0),
                _fmt(b.aoa, 1),
                _fmt(b.offline, 1),
                _fmt(b.fairway_pct, 0, "%"),
            ],
        }
    )
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    winners = compare["winners"]
    st.markdown(
        f"""
        <div class="fc-verdict">
            <h3 style="margin-top:0;">Comparison Verdict</h3>
            <p><strong>Longest carry:</strong> {winners['longest_carry']}</p>
            <p><strong>Fastest ball speed:</strong> {winners['fastest_ball_speed']}</p>
            <p><strong>Straightest:</strong> {winners['straightest']}</p>
            <p><strong>Most fairways:</strong> {winners['most_fairways']}</p>
            <p><strong>Best overall gamer:</strong> {winners['best_overall']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_a = _driver_setup_from_prefix("cmpA")
    setup_b = _driver_setup_from_prefix("cmpB")

    summaries_a = summarize_by_club(dr_a)
    summaries_b = summarize_by_club(dr_b)

    rec_a = build_driver_recommendations(
        summary=summaries_a["DR"],
        user_setup=setup_a,
        fairway_hit_pct=float((dr_a["offline_yd"].dropna().abs() <= 15).mean() * 100.0) if len(dr_a["offline_yd"].dropna()) else None,
    )
    rec_b = build_driver_recommendations(
        summary=summaries_b["DR"],
        user_setup=setup_b,
        fairway_hit_pct=float((dr_b["offline_yd"].dropna().abs() <= 15).mean() * 100.0) if len(dr_b["offline_yd"].dropna()) else None,
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<div class="fc-card"><h3>Setup A Recommendation</h3></div>', unsafe_allow_html=True)
        _render_recommendation_cards(rec_a)
    with rc2:
        st.markdown('<div class="fc-card"><h3>Setup B Recommendation</h3></div>', unsafe_allow_html=True)
        _render_recommendation_cards(rec_b)

    best_overall = winners["best_overall"]
    interpretation = []
    if best_overall == "Setup A":
        interpretation.append("Setup A looks like the stronger overall gamer from this test.")
    elif best_overall == "Setup B":
        interpretation.append("Setup B looks like the stronger overall gamer from this test.")
    else:
        interpretation.append("This comparison is close enough that neither setup clearly dominates overall.")

    if winners["longest_carry"] != winners["straightest"]:
        interpretation.append("One setup appears better for distance while the other appears better for control.")
    else:
        interpretation.append("The same setup appears to be winning both carry and dispersion, which is a strong sign.")

    interpretation.append("Repeat the test on another day with 8–10 fresh shots per setup to confirm the winner.")

    st.markdown(
        f"""
        <div class="fc-card">
            <h3>Best Overall Interpretation</h3>
            <p>{" ".join(interpretation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Next smart upgrade: save preferred setups permanently, then add face-to-path pattern labels like push fade, pull fade, push draw, and pull hook.")
