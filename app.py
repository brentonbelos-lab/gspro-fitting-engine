# app.py
from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

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
    estimate_launch_spin_change,
    driver_targets,
    pick_one_hosel_setting,
)

st.set_page_config(page_title="FitCaddie — Spec-Range MVP", layout="wide")
st.title("FitCaddie — Spec-Range MVP")
st.caption("Upload a GSPro CSV export to get club-by-club insights and spec-range fitting recommendations.")

# -----------------------------
# Helpers: robust numeric parsing
# -----------------------------
def _extract_float(s: str) -> Optional[float]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    txt = str(s).strip()
    if txt == "":
        return None
    m = re.search(r"-?\d+(\.\d+)?", txt)
    return float(m.group(0)) if m else None

def parse_dir_value(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    num = _extract_float(s)
    if num is None:
        return None

    s_upper = s.upper()
    if re.search(r"\bL\b", s_upper): return -abs(num)
    if re.search(r"\bR\b", s_upper): return abs(num)
    if re.search(r"\bU\b", s_upper): return abs(num)
    if re.search(r"\bD\b", s_upper): return -abs(num)
    if "I-O" in s_upper: return abs(num)
    if "O-I" in s_upper: return -abs(num)
    if re.search(r"\bC\b", s_upper): return -abs(num)
    if re.search(r"\bO\b", s_upper): return abs(num)
    return num

def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.dropna().mean()) if x.notna().any() else float("nan")

def safe_std(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.dropna().std(ddof=1)) if x.dropna().shape[0] >= 2 else float("nan")

# -----------------------------
# Canonicalize GSPro exports
# -----------------------------
CANON = [
    "club","club_speed_mph","ball_speed_mph","smash",
    "carry_yd","total_yd","offline_yd","peak_height_yd",
    "descent_deg","hla_deg","vla_deg","backspin_rpm","spin_axis_deg",
    "aoa_deg","club_path_deg","face_to_path_deg","face_to_target_deg",
]

def detect_format(columns: List[str]) -> str:
    cols = set(columns)
    if "Club Name" in cols and "Club Speed (mph)" in cols:
        return "portal"
    if "Club" in cols and "ClubSpeed" in cols and "BallSpeed" in cols:
        return "software"
    return "unknown"

def canonicalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    fmt = detect_format(df.columns.tolist())
    out = pd.DataFrame(index=df.index)
    for c in CANON:
        out[c] = np.nan

    if fmt == "portal":
        out["club"] = df.get("Club Name")
        out["club_speed_mph"] = pd.to_numeric(df.get("Club Speed (mph)"), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("Ball Speed (mph)"), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry Dist (yd)"), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("Total Dist (yd)"), errors="coerce")
        out["offline_yd"] = df.get("Offline (yd)").apply(parse_dir_value)
        out["peak_height_yd"] = pd.to_numeric(df.get("Peak Height (yd)"), errors="coerce")
        out["descent_deg"] = df.get("Desc Angle").apply(parse_dir_value)
        out["hla_deg"] = df.get("HLA").apply(parse_dir_value)
        out["vla_deg"] = df.get("VLA").apply(parse_dir_value)
        out["backspin_rpm"] = pd.to_numeric(df.get("Back Spin"), errors="coerce")
        out["spin_axis_deg"] = df.get("Spin Axis").apply(parse_dir_value)
        out["aoa_deg"] = df.get("Club AoA").apply(parse_dir_value)
        out["club_path_deg"] = df.get("Club Path").apply(parse_dir_value)
        out["face_to_path_deg"] = df.get("Face to Path").apply(parse_dir_value)
        out["face_to_target_deg"] = df.get("Face to Target").apply(parse_dir_value)
        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    elif fmt == "software":
        out["club"] = df.get("Club")
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed"), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed"), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry"), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance"), errors="coerce")
        out["offline_yd"] = pd.to_numeric(df.get("Offline"), errors="coerce")
        out["peak_height_yd"] = pd.to_numeric(df.get("PeakHeight"), errors="coerce")
        out["descent_deg"] = pd.to_numeric(df.get("Decent"), errors="coerce")
        out["hla_deg"] = pd.to_numeric(df.get("HLA"), errors="coerce")
        out["vla_deg"] = pd.to_numeric(df.get("VLA"), errors="coerce")
        out["backspin_rpm"] = pd.to_numeric(df.get("BackSpin"), errors="coerce")
        out["spin_axis_deg"] = pd.to_numeric(df.get("rawSpinAxis", df.get("SideSpin")), errors="coerce")
        out["aoa_deg"] = pd.to_numeric(df.get("AoA"), errors="coerce")
        out["club_path_deg"] = pd.to_numeric(df.get("Path"), errors="coerce")
        out["face_to_path_deg"] = pd.to_numeric(df.get("FaceToPath"), errors="coerce")
        out["face_to_target_deg"] = pd.to_numeric(df.get("FaceToTarget"), errors="coerce")
        out["smash"] = pd.to_numeric(df.get("SmashFactor"), errors="coerce")
        if out["smash"].isna().all():
            out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    else:
        out["club"] = df.get("Club", df.get("Club Name", np.nan))
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed", df.get("Club Speed (mph)")), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed", df.get("Ball Speed (mph)")), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry", df.get("Carry Dist (yd)")), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance", df.get("Total Dist (yd)")), errors="coerce")
        out["offline_yd"] = df.get("Offline", df.get("Offline (yd)")).apply(parse_dir_value)
        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    out["club"] = out["club"].astype(str).str.strip().replace({"nan": np.nan})
    return out, fmt

def bucket_club(club: str) -> str:
    if club is None or (isinstance(club, float) and np.isnan(club)):
        return "OTHER"
    c = str(club).upper().strip()
    if c in {"DR", "D", "DRIVER"} or c.startswith("DR"):
        return "DR"
    if "W" in c and any(n in c for n in ["2", "3", "4", "5", "7", "9"]):
        return "FW"
    if c.startswith("H") or "HY" in c:
        return "HY"
    return "OTHER"

def miss_tendency(offline_avg: float) -> str:
    if np.isnan(offline_avg): return "Unknown"
    if offline_avg > 5: return "Right miss tendency"
    if offline_avg < -5: return "Left miss tendency"
    return "Centered"

def spin_safety(backspin_avg: float) -> str:
    if np.isnan(backspin_avg): return "Unknown"
    if backspin_avg < 1800: return "Low spin risk"
    if backspin_avg > 3300: return "High spin risk"
    return "Spin in a safe range"

def smash_flag_driver(smash_avg: float) -> Optional[str]:
    if np.isnan(smash_avg): return None
    if smash_avg < 1.42:
        return f"Smash factor is low ({smash_avg:.2f}). Efficiency/contact is a limiting factor."
    if smash_avg < 1.45:
        return f"Smash factor is slightly low ({smash_avg:.2f}). There’s still efficiency left."
    return None

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
        help="How much a sleeve loft change alters delivered dynamic loft. 1.0 assumes delivery unchanged."
    )

    st.divider()
    st.header("Options")
    show_raw = st.checkbox("Show raw tables", value=False)
    min_shots = st.slider("Min shots per bucket", 3, 20, 5, 1)

if not uploaded:
    st.info("Upload a GSPro CSV to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded)
canon_df, fmt = canonicalize(raw_df)
canon_df["bucket"] = canon_df["club"].apply(bucket_club)

st.success(f"Loaded {len(canon_df)} shots. Detected export format: **{fmt}**")

if show_raw:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw CSV (first 200)")
        st.dataframe(raw_df.head(200), use_container_width=True)
    with c2:
        st.subheader("Canonicalized (first 200)")
        st.dataframe(canon_df.head(200), use_container_width=True)

bucket_counts = canon_df["bucket"].value_counts(dropna=False).to_dict()
available_buckets = [b for b, n in bucket_counts.items() if n >= min_shots and b != "OTHER"]
if not available_buckets:
    st.warning(f"No club buckets have at least {min_shots} shots. Lower the minimum in the sidebar.")
    st.stop()

st.subheader("Detected Club Buckets")
cols = st.columns(len(available_buckets))
for i, b in enumerate(available_buckets):
    cols[i].metric(b, bucket_counts.get(b, 0))

selected_buckets = st.multiselect("Choose which buckets to analyze", options=available_buckets, default=available_buckets)
if not selected_buckets:
    st.stop()

df = canon_df[canon_df["bucket"].isin(selected_buckets)].copy()

# -----------------------------
# Hosel Setup UI (DR + multiple FW/HY)
# -----------------------------
st.divider()
st.header("Hosel & Club Setup")

def club_setup_ui(section_title: str, club_type: str, n: int, default_labels: List[str]):
    st.subheader(section_title)
    setups = []

    for i in range(n):
        st.markdown(f"**{default_labels[i] if i < len(default_labels) else f'{club_type}-{i+1}'}**")

        cc1, cc2, cc3, cc4, cc5 = st.columns([1.0, 1.2, 0.9, 1.3, 1.3])
        with cc1:
            label = st.text_input("Label", value=(default_labels[i] if i < len(default_labels) else f"{club_type}-{i+1}"), key=f"{club_type}_label_{i}")
        with cc2:
            stated_loft = st.number_input("Stated Loft (°)", min_value=0.0, max_value=30.0, value=(0.0), step=0.5, key=f"{club_type}_loft_{i}")
        with cc3:
            handedness = st.selectbox("Hand", ["RH", "LH"], index=0, key=f"{club_type}_hand_{i}")
        with cc4:
            brand = st.selectbox("Brand", get_supported_brands(), index=0, key=f"{club_type}_brand_{i}")

        systems = get_brand_systems(brand)
        system_names = [s.system_name for s in systems] if systems else ["(no systems found)"]
        with cc5:
            system_name = st.selectbox("Hosel System", system_names, index=0, key=f"{club_type}_sys_{i}")

        settings = list_settings(brand, system_name, handedness)
        s1, s2 = st.columns(2)
        with s1:
            current_setting = st.selectbox("Current Setting", settings if settings else ["STD"], key=f"{club_type}_cur_{i}")
        with s2:
            proposed_setting = st.selectbox("Proposed Setting", settings if settings else ["STD"], key=f"{club_type}_new_{i}")

        cur_delta = translate_setting(brand, system_name, current_setting, handedness)
        new_delta = translate_setting(brand, system_name, proposed_setting, handedness)

        cur_loft = getattr(cur_delta, "loft_deg", None)
        new_loft = getattr(new_delta, "loft_deg", None)

        # Compute delta between settings when possible
        if cur_loft is not None and new_loft is not None:
            delta_static_loft = (new_loft - cur_loft)
            est = estimate_launch_spin_change(delta_static_loft, k_loft_to_dynamic, club_type)
            st.info(
                f"Estimated launch change: **{est.launch_change_deg:+.1f}°** "
                f"(range {est.launch_range_deg[0]:+.1f}° to {est.launch_range_deg[1]:+.1f}°)\n\n"
                f"Estimated spin change: **{est.spin_change_rpm:+d} rpm** "
                f"(range {est.spin_range_rpm[0]:+d} to {est.spin_range_rpm[1]:+d})"
            )
        else:
            ranges = system_ranges(brand, system_name)
            st.warning(
                "Exact loft deltas for these settings aren’t encoded yet, so FitCaddie can’t compute a numeric change.\n\n"
                f"System ranges: loft={ranges.get('loft_range_deg')}, lie={ranges.get('lie_range_deg')}."
            )

        setups.append({
            "label": label,
            "club_type": club_type,
            "stated_loft": stated_loft,
            "brand": brand,
            "system_name": system_name,
            "handedness": handedness,
            "current_setting": current_setting,
            "proposed_setting": proposed_setting,
            "cur_delta": asdict(cur_delta),
            "new_delta": asdict(new_delta),
        })

        st.divider()

    return setups

hosel_setups = {"DR": [], "FW": [], "HY": []}

if "DR" in selected_buckets:
    hosel_setups["DR"] = club_setup_ui("Driver", "DR", 1, ["Driver"])

if "FW" in selected_buckets:
    n_fw = st.slider("How many fairway woods do you want to configure?", 1, 4, 1, 1)
    hosel_setups["FW"] = club_setup_ui("Fairway Woods", "FW", n_fw, ["3W", "5W", "7W", "9W"])

if "HY" in selected_buckets:
    n_hy = st.slider("How many hybrids do you want to configure?", 1, 4, 1, 1)
    hosel_setups["HY"] = club_setup_ui("Hybrids", "HY", n_hy, ["3H", "4H", "5H", "6H"])

# -----------------------------
# Analysis & Recommendations
# -----------------------------
st.divider()
st.header("Analysis & Recommendations")

for bucket in selected_buckets:
    bucket_df = df[df["bucket"] == bucket].copy()
    if len(bucket_df) < min_shots:
        continue

    st.subheader(f"{bucket} — Overview ({len(bucket_df)} shots)")

    agg = {
        "Club Speed (mph)": (safe_mean(bucket_df["club_speed_mph"]), safe_std(bucket_df["club_speed_mph"])),
        "Ball Speed (mph)": (safe_mean(bucket_df["ball_speed_mph"]), safe_std(bucket_df["ball_speed_mph"])),
        "Smash": (safe_mean(bucket_df["smash"]), safe_std(bucket_df["smash"])),
        "Carry (yd)": (safe_mean(bucket_df["carry_yd"]), safe_std(bucket_df["carry_yd"])),
        "Total (yd)": (safe_mean(bucket_df["total_yd"]), safe_std(bucket_df["total_yd"])),
        "Offline (yd)": (safe_mean(bucket_df["offline_yd"]), safe_std(bucket_df["offline_yd"])),
        "VLA (deg)": (safe_mean(bucket_df["vla_deg"]), safe_std(bucket_df["vla_deg"])),
        "Backspin (rpm)": (safe_mean(bucket_df["backspin_rpm"]), safe_std(bucket_df["backspin_rpm"])),
        "AoA (deg)": (safe_mean(bucket_df["aoa_deg"]), safe_std(bucket_df["aoa_deg"])),
    }

    mcols = st.columns(4)
    keys = list(agg.keys())
    for i, k in enumerate(keys):
        mean_v, std_v = agg[k]
        value = "—" if np.isnan(mean_v) else f"{mean_v:.2f}"
        delta_txt = None if np.isnan(std_v) else f"±{std_v:.2f}"
        mcols[i % 4].metric(k, value, delta_txt)

    st.markdown("### Limiting Factors")
    limiting: List[str] = []

    smash_avg = agg["Smash"][0]
    if bucket == "DR":
        msg = smash_flag_driver(smash_avg)
        if msg:
            limiting.append(msg)

    off_avg = agg["Offline (yd)"][0]
    limiting.append(f"Miss tendency: **{miss_tendency(off_avg)}**" if not np.isnan(off_avg) else "Miss tendency: Unknown")

    spin_avg = agg["Backspin (rpm)"][0]
    limiting.append(f"Spin safety: **{spin_safety(spin_avg)}**" if not np.isnan(spin_avg) else "Spin safety: Unknown")

    for item in limiting:
        st.write("•", item)

    st.markdown("### Recommendations")
    recos: List[str] = []

    # Driver hosel recommendation (ONE setting)
    if bucket == "DR" and hosel_setups["DR"]:
        h = hosel_setups["DR"][0]  # driver config
        brand, system_name, handedness = h["brand"], h["system_name"], h["handedness"]
        current_setting = h["current_setting"]

        cs_avg = agg["Club Speed (mph)"][0]
        vla_avg = agg["VLA (deg)"][0]

        needed_loft = 0.0
        needed_lie = 0.0

        # 1) Launch window (speed adjusted)
        if not np.isnan(cs_avg) and not np.isnan(vla_avg):
            t = driver_targets(cs_avg)
            launch_lo, launch_hi = t["launch"]

            launch_per_static_loft = 0.85 * k_loft_to_dynamic  # based on 85% dynamic loft relationship
            if vla_avg < launch_lo:
                needed_loft = min(2.0, (launch_lo - vla_avg) / max(0.05, launch_per_static_loft))
            elif vla_avg > launch_hi:
                needed_loft = max(-2.0, -(vla_avg - launch_hi) / max(0.05, launch_per_static_loft))

        # 2) Miss tendency (upright/flatter)
        if not np.isnan(off_avg):
            if off_avg > 5:
                needed_lie = +0.75
            elif off_avg < -5:
                needed_lie = -0.75

        # 3) Spin safety check
        if not np.isnan(spin_avg):
            if spin_avg < 1800:
                needed_loft = max(needed_loft, +0.75)
            elif spin_avg > 3300:
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
            recos.append("Hosel: your current setting looks fine relative to launch/miss/spin (no change suggested).")
        else:
            recos.append(
                f"Hosel goal (priority): loft Δ **{needed_loft:+.2f}°**, then lie Δ **{needed_lie:+.2f}°**."
            )

            if reco["type"] == "exact":
                r = reco["recommended"]
                recos.append(
                    f"Hosel recommendation: change **{reco['current']} → {r['setting']}** "
                    f"(loft {r['loft_delta']:+.2f}°, lie {r['lie_delta']:+.2f}°)."
                )
            else:
                recos.append(reco["message"])

    # FW/HY message (hosel recos can be added later once charts are encoded)
    if bucket in {"FW", "HY"}:
        recos.append("Hosel: FW/HY settings are captured in the UI. Exact one-setting recommendations will improve as we encode more model-specific charts.")

    # General advice
    if bucket == "DR" and not np.isnan(smash_avg) and smash_avg < 1.45:
        recos.append("Efficiency: focus on center contact (tee height, strike). Low smash is a limiting factor.")

    for r in recos:
        st.write("•", r)

    with st.expander("See shot table for this bucket"):
        show_cols = [
            "club","club_speed_mph","ball_speed_mph","smash",
            "carry_yd","total_yd","offline_yd","vla_deg","backspin_rpm","aoa_deg",
            "club_path_deg","face_to_path_deg","face_to_target_deg",
        ]
        st.dataframe(bucket_df[show_cols].reset_index(drop=True), use_container_width=True)
