# app.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from hosel_db import (
    get_supported_brands,
    get_brand_systems,
    list_settings,
    translate_setting,
    system_ranges,
)

from fit_engine import (
    canonicalize,
    summarize_by_club,
    targets_for_club,
    estimate_launch_spin_change,
    miss_tendency,
    smash_flag_driver,
    pick_one_hosel_setting,
)

# -----------------------------
# Dependent-dropdown reset helpers (brand -> system -> settings)
# -----------------------------
def _reset_system_and_settings_for_club(club_id: str):
    """When brand changes: reset hosel system + current/proposed setting to valid first options."""
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

    # reset system to first valid
    st.session_state[sys_key] = system_names[0]

    # reset settings to first valid for (brand, system, hand)
    settings = list_settings(brand, st.session_state[sys_key], handedness)
    first_setting = settings[0] if settings else "STD"
    st.session_state[cur_key] = first_setting
    st.session_state[new_key] = first_setting


def _reset_settings_for_club(club_id: str):
    """When system OR handedness changes: reset current/proposed to first valid option."""
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
# Streamlit config
# -----------------------------
st.set_page_config(page_title="FitCaddie — Spec-Range MVP", layout="wide")
st.title("FitCaddie — Spec-Range MVP")
st.caption("Upload a GSPro CSV export → club-specific analysis (DR/3W/5W/3H/etc.) → hosel guidance.")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload GSPro CSV", type=["csv"])

    st.divider()
    st.header("Hosel Estimation Model")
    k_loft_to_dynamic = st.slider(
        "Loft → delivered loft multiplier (k)",
        min_value=0.6,
        max_value=1.6,
        value=1.0,
        step=0.05,
        help="Higher = adding loft tends to raise launch more for your delivery/strike."
    )

    st.divider()
    st.header("Filters")
    # min=10, max=50, default=10
    min_shots = st.slider("Min shots per Club", 10, 50, 10, 1)

    st.divider()
    st.header("Debug")
    show_raw = st.checkbox("Show raw tables", value=False)

if not uploaded:
    st.info("Upload a GSPro CSV to begin.")
    st.stop()

# -----------------------------
# Load + canonicalize
# -----------------------------
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

# -----------------------------
# Club-specific detection
# -----------------------------
club_counts = canon_df["club_id"].value_counts().to_dict()
club_ids_all = [c for c in club_counts.keys() if c != "OTHER"]
club_ids = [c for c in club_ids_all if club_counts.get(c, 0) >= min_shots]

if not club_ids:
    st.warning(f"No clubs have at least {min_shots} shots. Try lowering the filter or collect more data.")
    st.stop()

st.subheader("Detected Clubs (club-specific)")
cols = st.columns(min(6, len(club_ids)))
for i, c in enumerate(club_ids):
    cols[i % len(cols)].metric(c, club_counts.get(c, 0))

selected_clubs = st.multiselect(
    "Choose which clubs to analyze",
    options=club_ids,
    default=club_ids
)

if not selected_clubs:
    st.stop()

canon_df = canon_df[canon_df["club_id"].isin(selected_clubs)].copy()

# -----------------------------
# Hosel Setup UI
# -----------------------------
st.divider()
st.header("Hosel Setup (Driver / Fairways / Hybrids)")
st.caption("Configure hosel settings by actual club (3W/5W/3H/etc.). Changing brand resets system + settings automatically.")

detected_fw = [c for c in selected_clubs if c.endswith("W")]
detected_hy = [c for c in selected_clubs if c.endswith("H")]
detected_dr = [c for c in selected_clubs if c == "DR"]

hosel_configs: Dict[str, Dict] = {}


def hosel_block(club_id: str, title: str):
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
        stated_loft = st.number_input(
            f"{club_id} Stated Loft (°)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
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

    # Estimate deltas for proposed change when exact deltas exist
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


# Driver
if len(detected_dr) > 0:
    hosel_block("DR", "Driver")
else:
    st.caption("No driver detected/selected (DR).")

# Fairways (bulletproof slider)
# Fairways (bulletproof slider + state clamp)
if len(detected_fw) > 0:
    st.subheader("Fairway Woods")

    max_fw = min(6, len(detected_fw))
    default_fw = min(2, max_fw)

    fw_slider_key = "n_fw"

    if fw_slider_key in st.session_state:
        st.session_state[fw_slider_key] = int(
            max(1, min(max_fw, st.session_state[fw_slider_key]))
        )

    n_fw = st.slider(
        "How many fairway woods to configure?",
        min_value=1,
        max_value=max_fw,
        value=st.session_state.get(fw_slider_key, default_fw),
        step=1,
        key=fw_slider_key,
    )

    for i in range(n_fw):
        club_choice = st.selectbox(
            f"Fairway slot {i+1}: which club?",
            detected_fw,
            key=f"fw_slot_{i}"
        )
        hosel_block(club_choice, f"Fairway: {club_choice}")
else:
    st.caption("No fairway woods detected/selected (e.g., 3W/5W/7W).")
    for i in range(n_fw):
        club_choice = st.selectbox(f"Fairway slot {i+1}: which club?", detected_fw, key=f"fw_slot_{i}")
        hosel_block(club_choice, f"Fairway: {club_choice}")
else:
    st.caption("No fairway woods detected/selected (e.g., 3W/5W/7W).")

# Hybrids (bulletproof slider)
# Hybrids (bulletproof slider + state clamp)
if len(detected_hy) > 0:
    st.subheader("Hybrids")

    max_hy = min(6, len(detected_hy))
    default_hy = min(2, max_hy)

    hy_slider_key = "n_hy"

    if hy_slider_key in st.session_state:
        st.session_state[hy_slider_key] = int(
            max(1, min(max_hy, st.session_state[hy_slider_key]))
        )

    n_hy = st.slider(
        "How many hybrids to configure?",
        min_value=1,
        max_value=max_hy,
        value=st.session_state.get(hy_slider_key, default_hy),
        step=1,
        key=hy_slider_key,
    )

    for i in range(n_hy):
        club_choice = st.selectbox(
            f"Hybrid slot {i+1}: which club?",
            detected_hy,
            key=f"hy_slot_{i}"
        )
        hosel_block(club_choice, f"Hybrid: {club_choice}")
else:
    st.caption("No hybrids detected/selected (e.g., 3H/4H/5H).")
    for i in range(n_hy):
        club_choice = st.selectbox(f"Hybrid slot {i+1}: which club?", detected_hy, key=f"hy_slot_{i}")
        hosel_block(club_choice, f"Hybrid: {club_choice}")
else:
    st.caption("No hybrids detected/selected (e.g., 3H/4H/5H).")

# -----------------------------
# Club-specific analysis
# -----------------------------
st.divider()
st.header("Club-Specific Analysis & Recommendations")

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

    # Metrics
    mcols = st.columns(4)
    mcols[0].metric(
        "Club Speed (mph)",
        "—" if np.isnan(s.club_speed_avg) else f"{s.club_speed_avg:.1f}",
        None if np.isnan(s.club_speed_std) else f"±{s.club_speed_std:.1f}",
    )
    mcols[1].metric(
        "Ball Speed (mph)",
        "—" if np.isnan(s.ball_speed_avg) else f"{s.ball_speed_avg:.1f}",
        None if np.isnan(s.ball_speed_std) else f"±{s.ball_speed_std:.1f}",
    )
    mcols[2].metric(
        "Smash",
        "—" if np.isnan(s.smash_avg) else f"{s.smash_avg:.2f}",
        None if np.isnan(s.smash_std) else f"±{s.smash_std:.2f}",
    )
    mcols[3].metric(
        "Carry (yd)",
        "—" if np.isnan(s.carry_avg) else f"{s.carry_avg:.1f}",
        None if np.isnan(s.carry_std) else f"±{s.carry_std:.1f}",
    )

    mcols2 = st.columns(4)
    mcols2[0].metric(
        "Offline (yd)",
        "—" if np.isnan(s.offline_avg) else f"{s.offline_avg:.1f}",
        None if np.isnan(s.offline_std) else f"±{s.offline_std:.1f}",
    )
    mcols2[1].metric(
        "Launch / VLA (°)",
        "—" if np.isnan(s.vla_avg) else f"{s.vla_avg:.1f}",
        None if np.isnan(s.vla_std) else f"±{s.vla_std:.1f}",
    )
    mcols2[2].metric(
        "Backspin (rpm)",
        "—" if np.isnan(s.spin_avg) else f"{s.spin_avg:.0f}",
        None if np.isnan(s.spin_std) else f"±{s.spin_std:.0f}",
    )
    mcols2[3].metric(
        "AoA (°)",
        "—" if np.isnan(s.aoa_avg) else f"{s.aoa_avg:.1f}",
        None if np.isnan(s.aoa_std) else f"±{s.aoa_std:.1f}",
    )

    # Limiting factors
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

    # Recommendations
    st.markdown("### Recommendations")
    recos: List[str] = []

    if club_id in hosel_configs:
        h = hosel_configs[club_id]
        brand = h["brand"]
        system_name = h["system_name"]
        handedness = h["handedness"]
        current_setting = h["current_setting"]

        needed_loft = 0.0
        needed_lie = 0.0

        # 1) launch window (use VLA)
        if not np.isnan(s.vla_avg):
            launch_per_static_loft = 0.85 * k_loft_to_dynamic
            if s.vla_avg < launch_lo:
                needed_loft = min(2.0, (launch_lo - s.vla_avg) / max(0.05, launch_per_static_loft))
            elif s.vla_avg > launch_hi:
                needed_loft = max(-2.0, -(s.vla_avg - launch_hi) / max(0.05, launch_per_static_loft))

        # 2) miss -> lie
        if miss == "Right miss tendency":
            needed_lie = +0.75
        elif miss == "Left miss tendency":
            needed_lie = -0.75

        # 3) spin safety -> nudge loft
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
            recos.append("Hosel: current setting looks fine for launch/miss/spin (no change suggested).")
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
        recos.append("Hosel: configure this club in the Hosel Setup section to enable setting recommendations.")

    if miss == "Right miss tendency":
        recos.append("Directional: right bias → try more upright / draw setting before changing shafts.")
    elif miss == "Left miss tendency":
        recos.append("Directional: left bias → try flatter / neutral setting before changing shafts.")

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
        st.dataframe(
            canon_df[canon_df["club_id"] == club_id][show_cols].reset_index(drop=True),
            use_container_width=True
        )

st.divider()
st.caption("Next upgrade after this: dispersion charts + face-to-path pattern classification (push-slice vs pull-hook).")
