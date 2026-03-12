from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from viz import render_dispersion

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
    compare_driver_setups,
    estimate_launch_spin_change,
    miss_tendency,
    pick_one_hosel_setting,
    smash_flag_driver,
    summarize_by_club,
    targets_for_club,
)

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="FitCaddie — Spec-Range MVP", layout="wide")
st.title("FitCaddie — Spec-Range MVP")
st.caption("Upload GSPro CSV data → analyze clubs → compare driver setups → get fitter-style recommendations.")


# -----------------------------
# Session state defaults
# -----------------------------
def _init_state():
    defaults = {
        "analysis_mode": "Single Upload",
        "k_loft_to_dynamic": 1.0,
        "min_shots": 10,
        "show_raw": False,

        # Single mode driver setup
        "single_driver_brand": "Titleist",
        "single_driver_model": "TSR3",
        "single_driver_loft": 10.0,
        "single_driver_hosel": "A1",
        "single_driver_shaft_model": "HZRDUS Black",
        "single_driver_shaft_weight": 60.0,
        "single_driver_shaft_flex": "6.0",

        # Compare mode setup A
        "cmpA_driver_brand": "Titleist",
        "cmpA_driver_model": "TSR3",
        "cmpA_driver_loft": 10.0,
        "cmpA_driver_hosel": "A1",
        "cmpA_driver_shaft_model": "HZRDUS Black",
        "cmpA_driver_shaft_weight": 60.0,
        "cmpA_driver_shaft_flex": "6.0",

        # Compare mode setup B
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
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .fit-box {
        padding: 16px 18px;
        border-radius: 14px;
        border: 2px solid #2e8b57;
        background: #f4fff7;
        margin-bottom: 12px;
    }
    .fit-box h4 {
        margin: 0 0 8px 0;
        color: #1f3d2e;
    }
    .compare-box {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid #d6d6d6;
        background: #fafafa;
        margin-bottom: 10px;
    }
    .winner-box {
        padding: 14px 16px;
        border-radius: 14px;
        border: 2px solid #1f77b4;
        background: #f4f9ff;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Dependent-dropdown reset helpers
# -----------------------------
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


# -----------------------------
# Utility helpers
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


def _metric_row(summary):
    return {
        "Shots": summary.n,
        "Club Speed": _fmt(summary.club_speed_avg, 1),
        "Ball Speed": _fmt(summary.ball_speed_avg, 1),
        "Smash": _fmt(summary.smash_avg, 2),
        "Carry": _fmt(summary.carry_avg, 1),
        "Offline": _fmt(summary.offline_avg, 1),
        "Launch": _fmt(summary.vla_avg, 1),
        "Spin": _fmt(summary.spin_avg, 0),
        "AoA": _fmt(summary.aoa_avg, 1),
    }


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
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state[f"{prefix}_driver_brand"] = st.selectbox(
            "Driver Brand",
            ["Titleist", "Ping", "Other"],
            key=f"{prefix}_driver_brand_select",
            index=["Titleist", "Ping", "Other"].index(st.session_state[f"{prefix}_driver_brand"])
            if st.session_state[f"{prefix}_driver_brand"] in ["Titleist", "Ping", "Other"] else 0,
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


def _render_recommendation_cards(bundle):
    for block in [bundle.swing, bundle.driver_settings, bundle.equipment_adjustment]:
        st.markdown(
            f"""
            <div class="fit-box">
                <h4>{block.title}</h4>
                <p><strong>Suggestion:</strong> {block.suggestion}</p>
                <p><strong>Why:</strong> {block.why}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_hosel_block(club_id: str, title: str, hosel_configs: Dict[str, Dict], k_loft_to_dynamic: float):
    st.subheader(title)

    c1, c2, c3, c4 = st.columns([1.1, 1.4, 1.2, 1.4])

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
        elif club_id.endswith("H"):
            default_loft = 18.0
        elif club_id.endswith("W"):
            default_loft = 15.0
        else:
            default_loft = 0.0

        stated_loft = st.number_input(
            f"{club_id} Stated Loft (°)",
            min_value=0.0,
            max_value=30.0,
            value=default_loft,
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
            "Exact loft deltas for these settings aren’t encoded yet (range-only).\n\n"
            f"System ranges: loft={ranges.get('loft_range_deg')}, lie={ranges.get('lie_range_deg')}."
        )

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


def _render_club_analysis(canon_df: pd.DataFrame, min_shots: int, hosel_configs: Dict[str, Dict], k_loft_to_dynamic: float):
    st.divider()
    st.header("Club-Specific Analysis")

    summaries = summarize_by_club(canon_df)

    order = {
        "DR": 0,
        "2W": 1, "3W": 2, "4W": 3, "5W": 4, "7W": 5, "9W": 6,
        "2H": 10, "3H": 11, "4H": 12, "5H": 13, "6H": 14, "7H": 15
    }
    club_list_sorted = sorted(summaries.keys(), key=lambda c: order.get(c, 99))

    for club_id in club_list_sorted:
        s = summaries[club_id]
        if s.n < min_shots:
            continue

        st.subheader(f"{club_id} — Overview ({s.n} shots)")

        mcols = st.columns(4)
        mcols[0].metric("Club Speed (mph)", _fmt(s.club_speed_avg, 1), None if np.isnan(s.club_speed_std) else f"±{s.club_speed_std:.1f}")
        mcols[1].metric("Ball Speed (mph)", _fmt(s.ball_speed_avg, 1), None if np.isnan(s.ball_speed_std) else f"±{s.ball_speed_std:.1f}")
        mcols[2].metric("Smash", _fmt(s.smash_avg, 2), None if np.isnan(s.smash_std) else f"±{s.smash_std:.2f}")
        mcols[3].metric("Carry (yd)", _fmt(s.carry_avg, 1), None if np.isnan(s.carry_std) else f"±{s.carry_std:.1f}")

        mcols2 = st.columns(4)
        mcols2[0].metric("Offline (yd)", _fmt(s.offline_avg, 1), None if np.isnan(s.offline_std) else f"±{s.offline_std:.1f}")
        mcols2[1].metric("Launch / VLA (°)", _fmt(s.vla_avg, 1), None if np.isnan(s.vla_std) else f"±{s.vla_std:.1f}")
        mcols2[2].metric("Backspin (rpm)", _fmt(s.spin_avg, 0), None if np.isnan(s.spin_std) else f"±{s.spin_std:.0f}")
        mcols2[3].metric("AoA (°)", _fmt(s.aoa_avg, 1), None if np.isnan(s.aoa_std) else f"±{s.aoa_std:.1f}")

        st.markdown("### Limiting Factors")
        lim: List[str] = []

        miss = miss_tendency(s.offline_avg)
        lim.append(f"Miss tendency: **{miss}**" if miss != "Unknown" else "Miss tendency: Unknown")

        if club_id == "DR":
            smash_msg = smash_flag_driver(s.smash_avg)
            if smash_msg:
                lim.append(smash_msg)

        t = targets_for_club(club_id, s.club_speed_avg)
        launch_lo, launch_hi = t["launch"]
        spin_lo, spin_hi = t["spin"]

        if not np.isnan(s.vla_avg) and (s.vla_avg < launch_lo or s.vla_avg > launch_hi):
            lim.append(f"Launch window miss: {s.vla_avg:.1f}° vs target {launch_lo:.1f}–{launch_hi:.1f}°.")
        if not np.isnan(s.spin_avg) and (s.spin_avg < spin_lo or s.spin_avg > spin_hi):
            lim.append(f"Spin window miss: {s.spin_avg:.0f} rpm vs target {spin_lo:.0f}–{spin_hi:.0f} rpm.")

        for item in lim:
            st.write("•", item)

        st.markdown("### Hosel Recommendation")
        recos: List[str] = []

        if club_id in hosel_configs:
            h = hosel_configs[club_id]
            brand = h["brand"]
            system_name = h["system_name"]
            handedness = h["handedness"]
            current_setting = h["current_setting"]

            needed_loft = 0.0
            needed_lie = 0.0

            if not np.isnan(s.vla_avg):
                launch_per_static_loft = 0.85 * k_loft_to_dynamic
                if s.vla_avg < launch_lo:
                    needed_loft = min(2.0, (launch_lo - s.vla_avg) / max(0.05, launch_per_static_loft))
                elif s.vla_avg > launch_hi:
                    needed_loft = max(-2.0, -(s.vla_avg - launch_hi) / max(0.05, launch_per_static_loft))

            if miss == "Right miss tendency":
                needed_lie = +0.75
            elif miss == "Left miss tendency":
                needed_lie = -0.75

            if not np.isnan(s.spin_avg):
                if s.spin_avg < spin_lo:
                    needed_loft = max(needed_loft, +0.75)
                elif s.spin_avg > spin_hi:
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
                recos.append("Hosel: current setting looks fine for launch, miss pattern, and spin.")
            else:
                recos.append(f"Hosel goal: loft Δ **{needed_loft:+.2f}°**, then lie Δ **{needed_lie:+.2f}°**.")
                if reco["type"] == "exact":
                    r = reco["recommended"]
                    recos.append(
                        f"Hosel recommendation: change **{reco['current']} → {r['setting']}** "
                        f"(loft {r['loft_delta']:+.2f}°, lie {r['lie_delta']:+.2f}°)."
                    )
                else:
                    recos.append(reco["message"])
        else:
            recos.append("Configure this club in the Hosel Setup section to enable setting recommendations.")

        for r in recos:
            st.write("•", r)

        with st.expander(f"Show shot rows for {club_id}"):
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
                use_container_width=True
            )


def _render_driver_recommendation_section(canon_df: pd.DataFrame, driver_setup: DriverUserSetup):
    driver_df = canon_df[canon_df["club_id"] == "DR"].copy()
    if driver_df.empty:
        st.info("No driver shots found in this upload, so driver-specific recommendations are not shown.")
        return

    summaries = summarize_by_club(driver_df)
    if "DR" not in summaries:
        st.info("No valid driver summary available yet.")
        return

    offline_valid = driver_df["offline_yd"].dropna()
    fairway_pct = float((offline_valid.abs() <= 15).mean() * 100.0) if len(offline_valid) else np.nan

    bundle = build_driver_recommendations(
        summary=summaries["DR"],
        user_setup=driver_setup,
        fairway_hit_pct=fairway_pct if not np.isnan(fairway_pct) else None,
    )

    st.divider()
    st.header("🎯 Fitter Recommendations")
    _render_recommendation_cards(bundle)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode")
    st.session_state["analysis_mode"] = st.radio(
        "Choose analysis type",
        ["Single Upload", "Compare Driver Setups"],
        index=0 if st.session_state["analysis_mode"] == "Single Upload" else 1,
    )

    st.divider()
    st.header("Hosel Estimation Model")
    st.session_state["k_loft_to_dynamic"] = st.slider(
        "Loft → delivered loft multiplier (k)",
        min_value=0.6,
        max_value=1.6,
        value=float(st.session_state["k_loft_to_dynamic"]),
        step=0.05,
        help="Higher = adding loft tends to raise launch more for your delivery / strike.",
    )

    st.divider()
    st.header("Filters")
    st.session_state["min_shots"] = st.slider(
        "Min shots per Club",
        5,
        50,
        int(st.session_state["min_shots"]),
        1,
    )

    st.divider()
    st.header("Debug")
    st.session_state["show_raw"] = st.checkbox("Show raw tables", value=bool(st.session_state["show_raw"]))


# -----------------------------
# Main modes
# -----------------------------
analysis_mode = st.session_state["analysis_mode"]
k_loft_to_dynamic = float(st.session_state["k_loft_to_dynamic"])
min_shots = int(st.session_state["min_shots"])
show_raw = bool(st.session_state["show_raw"])


if analysis_mode == "Single Upload":
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
            st.subheader("Raw CSV (first 200)")
            st.dataframe(raw_df.head(200), use_container_width=True)
        with c2:
            st.subheader("Canonicalized (first 200)")
            st.dataframe(canon_df.head(200), use_container_width=True)

    club_counts = canon_df["club_id"].value_counts().to_dict()
    club_ids_all = [c for c in club_counts.keys() if c != "OTHER"]
    club_ids = [c for c in club_ids_all if club_counts.get(c, 0) >= min_shots]

    if not club_ids:
        st.warning(f"No clubs have at least {min_shots} shots. Try lowering the filter or collect more data.")
        st.stop()

    st.subheader("Detected Clubs")
    cols = st.columns(min(6, len(club_ids)))
    for i, c in enumerate(club_ids):
        cols[i % len(cols)].metric(c, club_counts.get(c, 0))

    selected_clubs = st.multiselect(
        "Choose which clubs to analyze",
        options=club_ids,
        default=club_ids,
    )

    if not selected_clubs:
        st.stop()

    canon_df = canon_df[canon_df["club_id"].isin(selected_clubs)].copy()

    st.divider()
    st.header("Shot Dispersion Map")
    render_dispersion(canon_df)

    st.divider()
    st.header("Driver Setup")
    _render_driver_setup("single", "Current Driver Setup")

    single_driver_setup = _driver_setup_from_prefix("single")
    _render_driver_recommendation_section(canon_df, single_driver_setup)

    st.divider()
    st.header("Hosel Setup (Driver / Fairways / Hybrids)")
    st.caption("Configure hosel settings by actual club. Settings stay in place while you continue working in the app.")

    detected_fw = [c for c in selected_clubs if c.endswith("W")]
    detected_hy = [c for c in selected_clubs if c.endswith("H")]
    detected_dr = [c for c in selected_clubs if c == "DR"]

    hosel_configs: Dict[str, Dict] = {}

    if len(detected_dr) > 0:
        _render_hosel_block("DR", "Driver", hosel_configs, k_loft_to_dynamic)
    else:
        st.caption("No driver detected / selected (DR).")

    if len(detected_fw) == 0:
        st.caption("No fairway woods detected / selected (e.g., 3W / 5W / 7W).")
    elif len(detected_fw) == 1:
        _render_hosel_block(detected_fw[0], f"Fairway: {detected_fw[0]}", hosel_configs, k_loft_to_dynamic)
    else:
        st.subheader("Fairway Woods")
        max_fw = min(6, len(detected_fw))
        default_fw = min(2, max_fw)

        fw_slider_key = f"n_fw__{'_'.join(detected_fw)}"
        n_fw = st.slider(
            "How many fairway woods to configure?",
            min_value=1,
            max_value=max_fw,
            value=default_fw,
            step=1,
            key=fw_slider_key,
        )

        for i in range(n_fw):
            club_choice = st.selectbox(
                f"Fairway slot {i+1}: which club?",
                detected_fw,
                key=f"fw_slot_{i}__{'_'.join(detected_fw)}"
            )
            _render_hosel_block(club_choice, f"Fairway: {club_choice}", hosel_configs, k_loft_to_dynamic)

    if len(detected_hy) == 0:
        st.caption("No hybrids detected / selected.")
    elif len(detected_hy) == 1:
        _render_hosel_block(detected_hy[0], f"Hybrid: {detected_hy[0]}", hosel_configs, k_loft_to_dynamic)
    else:
        st.subheader("Hybrids")
        max_hy = min(6, len(detected_hy))
        default_hy = min(2, max_hy)

        hy_slider_key = f"n_hy__{'_'.join(detected_hy)}"
        n_hy = st.slider(
            "How many hybrids to configure?",
            min_value=1,
            max_value=max_hy,
            value=default_hy,
            step=1,
            key=hy_slider_key,
        )

        for i in range(n_hy):
            club_choice = st.selectbox(
                f"Hybrid slot {i+1}: which club?",
                detected_hy,
                key=f"hy_slot_{i}__{'_'.join(detected_hy)}"
            )
            _render_hosel_block(club_choice, f"Hybrid: {club_choice}", hosel_configs, k_loft_to_dynamic)

    _render_club_analysis(canon_df, min_shots, hosel_configs, k_loft_to_dynamic)

else:
    with st.sidebar:
        st.header("Compare Uploads")
        uploaded_a = st.file_uploader("Upload Setup A CSV", type=["csv"], key="compare_upload_a")
        uploaded_b = st.file_uploader("Upload Setup B CSV", type=["csv"], key="compare_upload_b")

    st.header("Compare Driver Setups")
    st.caption("Use this to compare two different driver settings, heads, or shafts.")

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
        st.warning("Both uploaded files need driver shots (DR) for comparison mode.")
        st.stop()

    setup_a = _driver_setup_from_prefix("cmpA")
    setup_b = _driver_setup_from_prefix("cmpB")

    compare = compare_driver_setups(dr_a, dr_b, "Setup A", "Setup B")

    st.divider()
    st.header("Dispersion Comparison")
    map_a, map_b = st.columns(2)
    with map_a:
        st.subheader("Setup A")
        render_dispersion(dr_a)
    with map_b:
        st.subheader("Setup B")
        render_dispersion(dr_b)

    st.divider()
    st.header("Side-by-Side Driver Metrics")

    a = compare["a"]
    b = compare["b"]

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

    winners = compare["winners"]
    st.markdown(
        f"""
        <div class="winner-box">
            <h4>Comparison Verdict</h4>
            <p><strong>Longest carry:</strong> {winners['longest_carry']}</p>
            <p><strong>Fastest ball speed:</strong> {winners['fastest_ball_speed']}</p>
            <p><strong>Straightest:</strong> {winners['straightest']}</p>
            <p><strong>Most fairways:</strong> {winners['most_fairways']}</p>
            <p><strong>Best overall gamer:</strong> {winners['best_overall']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.divider()
    st.header("🎯 Compare-Mode Recommendations")

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        st.subheader("Setup A Recommendation")
        _render_recommendation_cards(rec_a)
    with cmp2:
        st.subheader("Setup B Recommendation")
        _render_recommendation_cards(rec_b)

    st.divider()
    st.subheader("Best Overall Interpretation")

    overall_lines: List[str] = []
    best_overall = winners["best_overall"]

    if best_overall == "Setup A":
        overall_lines.append("Setup A looks like the stronger overall gamer from this test.")
    elif best_overall == "Setup B":
        overall_lines.append("Setup B looks like the stronger overall gamer from this test.")
    else:
        overall_lines.append("This test is close enough that neither setup clearly dominates overall.")

    if winners["longest_carry"] != winners["straightest"]:
        overall_lines.append("One setup appears better for distance, while the other appears better for control.")
    else:
        overall_lines.append("The same setup appears to be winning both carry and dispersion, which is a strong sign.")

    overall_lines.append("Keep the winning setup honest by repeating the test with a fresh 8–10 driver shots per setup on another day.")

    st.markdown(
        f"""
        <div class="fit-box">
            <h4>Equipment Adjustment</h4>
            <p>{" ".join(overall_lines)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Next useful upgrade: save preferred setup(s) permanently and add face-to-path pattern classification like push fade vs pull hook.")
