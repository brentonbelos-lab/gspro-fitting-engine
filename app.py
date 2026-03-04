# app.py
# FitCaddie (Spec-Range MVP) — Streamlit app
# Upload GSPro CSV (Web Portal or Software export) → clean + analyze by club bucket → recommendations
#
# Adds: Brand → Hosel system → Handedness → Hosel setting (dynamic), and feeds hosel deltas into recos.
#
# You can copy/paste this whole file as-is.

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Optional hosel database import
# -----------------------------
# If you created hosel_db.py in the same folder, this will work.
# If not, the app still runs with a small fallback list so you aren't blocked.
try:
    from hosel_db import (
        get_supported_brands,
        get_brand_systems,
        list_settings,
        translate_setting,
    )
except Exception:
    def get_supported_brands() -> List[str]:
        return ["Titleist", "Callaway", "TaylorMade", "PING", "Cobra", "Srixon", "Mizuno", "PXG", "Wilson"]

    def get_brand_systems(brand: str):
        class _S:
            def __init__(self, name): self.system_name = name
        systems = {
            "Titleist": [_S("Titleist SureFit (Driver/Fairway)")],
            "Callaway": [_S("Callaway OptiFit")],
            "TaylorMade": [_S("TaylorMade Loft Sleeve (12-position)")],
            "PING": [_S("PING Trajectory Tuning 2.0 (8-position)")],
            "Cobra": [_S("Cobra MyFly (8 loft settings)"), _S("Cobra FutureFit33 (33 unique settings)")],
            "Srixon": [_S("12-position sleeve (STD / -1.5 / +1.5 / STD FL)")],
            "Mizuno": [_S("Mizuno Quick Switch")],
            "PXG": [_S("PXG Adapter (8 settings)")],
            "Wilson": [_S("Wilson Fast Fit (6 settings)")],
        }
        return systems.get(brand, [])

    def list_settings(brand: str, system_name: str, handedness: str) -> List[str]:
        if "Titleist" in system_name:
            return ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4","D1","D2","D3","D4"]
        if "OptiFit" in system_name:
            return ["-1 N","S N","+1 N","+2 N","-1 D","S D","+1 D","+2 D"]
        if "PING" in system_name:
            return ["STD","+0.5","+1.0","+1.5","-0.5","-1.0","-1.5","FLAT"]
        return ["STD"]

    def translate_setting(brand: str, system_name: str, setting: str, handedness: str):
        # Return a dict-like object to keep UI consistent
        return {"loft_deg": None, "lie_deg": None, "face_deg": None, "note": "hosel_db.py not found; using fallback."}


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="FitCaddie — Spec-Range MVP",
    layout="wide",
)

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
    """
    Parses strings like:
      '11.4 R'  -> +11.4
      '2.6° L'  -> -2.6
      '4.1° U'  -> +4.1
      '2.2° D'  -> -2.2
      '0.8° I-O'-> +0.8 (inside-out)
      '0.8° O-I'-> -0.8 (outside-in)
      '2.3° O'  -> +2.3 (open)
      '2.3° C'  -> -2.3 (closed)
    If it's already numeric, returns float(value).
    """
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

    # left/right
    if re.search(r"\bL\b", s_upper):
        return -abs(num)
    if re.search(r"\bR\b", s_upper):
        return abs(num)

    # up/down (AoA)
    if re.search(r"\bU\b", s_upper):
        return abs(num)
    if re.search(r"\bD\b", s_upper):
        return -abs(num)

    # in-out / out-in (club path)
    if "I-O" in s_upper:
        return abs(num)
    if "O-I" in s_upper:
        return -abs(num)

    # open/closed (face)
    # note: some exports use 'O' for open. We'll treat 'O' as +, 'C' as -.
    if re.search(r"\bC\b", s_upper):
        return -abs(num)
    if re.search(r"\bO\b", s_upper):
        return abs(num)

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
    "club",
    "club_speed_mph",
    "ball_speed_mph",
    "smash",
    "carry_yd",
    "total_yd",
    "offline_yd",
    "peak_height_yd",
    "descent_deg",
    "hla_deg",
    "vla_deg",
    "backspin_rpm",
    "spin_axis_deg",
    "aoa_deg",
    "club_path_deg",
    "face_to_path_deg",
    "face_to_target_deg",
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

        # smash may not exist in portal export
        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    elif fmt == "software":
        out["club"] = df.get("Club")
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed"), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed"), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry"), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance"), errors="coerce")

        out["offline_yd"] = pd.to_numeric(df.get("Offline"), errors="coerce")
        out["peak_height_yd"] = pd.to_numeric(df.get("PeakHeight"), errors="coerce")
        out["descent_deg"] = pd.to_numeric(df.get("Decent"), errors="coerce")  # yes, "Decent" in file
        out["hla_deg"] = pd.to_numeric(df.get("HLA"), errors="coerce")
        out["vla_deg"] = pd.to_numeric(df.get("VLA"), errors="coerce")

        out["backspin_rpm"] = pd.to_numeric(df.get("BackSpin"), errors="coerce")
        # prefer rawSpinAxis when available
        if "rawSpinAxis" in df.columns:
            out["spin_axis_deg"] = pd.to_numeric(df.get("rawSpinAxis"), errors="coerce")
        else:
            out["spin_axis_deg"] = pd.to_numeric(df.get("SideSpin"), errors="coerce")

        out["aoa_deg"] = pd.to_numeric(df.get("AoA"), errors="coerce")
        out["club_path_deg"] = pd.to_numeric(df.get("Path"), errors="coerce")
        out["face_to_path_deg"] = pd.to_numeric(df.get("FaceToPath"), errors="coerce")
        out["face_to_target_deg"] = pd.to_numeric(df.get("FaceToTarget"), errors="coerce")

        if "SmashFactor" in df.columns:
            out["smash"] = pd.to_numeric(df.get("SmashFactor"), errors="coerce")
        else:
            out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    else:
        # Best-effort: try common names
        # (App still runs, but may have many NaNs depending on file.)
        out["club"] = df.get("Club", df.get("Club Name", np.nan))
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed", df.get("Club Speed (mph)")), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed", df.get("Ball Speed (mph)")), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry", df.get("Carry Dist (yd)")), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance", df.get("Total Dist (yd)")), errors="coerce")
        out["offline_yd"] = df.get("Offline", df.get("Offline (yd)")).apply(parse_dir_value)
        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    # clean club strings
    out["club"] = out["club"].astype(str).str.strip().replace({"nan": np.nan})

    return out, fmt


# -----------------------------
# Club bucketing
# -----------------------------
def bucket_club(club: str) -> str:
    if club is None or (isinstance(club, float) and np.isnan(club)):
        return "OTHER"
    c = str(club).upper().strip()

    # common driver labels
    if c in {"DR", "D", "DRIVER"} or c.startswith("DR"):
        return "DR"
    # fairway woods
    if "W" in c and any(n in c for n in ["3", "4", "5", "7"]):
        return "FW"
    # hybrids
    if c.startswith("H") or "HY" in c:
        return "HY"
    return "OTHER"


# -----------------------------
# Recommendation logic (simple MVP)
# -----------------------------
def driver_targets(club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
    """
    Returns target windows for driver VLA (launch) and spin based on club speed.
    These are "safe MVP heuristics" (not a TrackMan optimizer).
    """
    if np.isnan(club_speed_mph):
        return {"launch": (11.0, 14.0), "spin": (2000.0, 3000.0)}

    cs = club_speed_mph
    if cs < 90:
        return {"launch": (14.0, 17.0), "spin": (2600.0, 3400.0)}
    if cs < 100:
        return {"launch": (12.5, 15.5), "spin": (2300.0, 3200.0)}
    if cs < 110:
        return {"launch": (11.0, 14.0), "spin": (2000.0, 2900.0)}
    return {"launch": (10.0, 13.0), "spin": (1800.0, 2700.0)}


def smash_flag_driver(smash_avg: float) -> Optional[str]:
    if np.isnan(smash_avg):
        return None
    # Driver smash: ~1.45+ is generally "solid"; <1.42 usually efficiency issue.
    if smash_avg < 1.42:
        return f"Smash factor is low ({smash_avg:.2f}). Efficiency/contact is a limiting factor."
    if smash_avg < 1.45:
        return f"Smash factor is slightly low ({smash_avg:.2f}). There’s still speed-to-ball conversion left."
    return None


def miss_tendency(offline_avg: float) -> str:
    if np.isnan(offline_avg):
        return "Unknown"
    if offline_avg > 5:
        return "Right miss tendency"
    if offline_avg < -5:
        return "Left miss tendency"
    return "Centered"


def spin_safety(backspin_avg: float) -> str:
    if np.isnan(backspin_avg):
        return "Unknown"
    if backspin_avg < 1800:
        return "Low spin risk (knuckle / drop-out)"
    if backspin_avg > 3300:
        return "High spin risk (balloon / lose distance)"
    return "Spin in a safe range"


def pick_hosel_reco(
    brand: str,
    system_name: str,
    handedness: str,
    current_setting: str,
    needed_loft_delta: float,
    needed_lie_delta: float,
) -> Dict[str, object]:
    """
    Try to recommend a better hosel setting.
    Works best when hosel_db has exact per-setting deltas (Titleist RH, Callaway).
    Otherwise returns a range-only guidance.
    """
    settings = list_settings(brand, system_name, handedness)
    if not settings:
        return {"type": "none", "message": "No settings available for this brand/system."}

    # Try to score each candidate by closeness to desired deltas.
    scored = []
    for s in settings:
        d = translate_setting(brand, system_name, s, handedness)
        # d might be dataclass or dict fallback
        loft = getattr(d, "loft_deg", None) if not isinstance(d, dict) else d.get("loft_deg")
        lie = getattr(d, "lie_deg", None) if not isinstance(d, dict) else d.get("lie_deg")

        if loft is None or lie is None:
            continue

        score = abs(loft - needed_loft_delta) * 1.5 + abs(lie - needed_lie_delta) * 1.0
        scored.append((score, s, loft, lie))

    if not scored:
        # We don't have exact mapping stored → give guidance only
        direction = []
        if needed_loft_delta > 0.25:
            direction.append("add loft")
        elif needed_loft_delta < -0.25:
            direction.append("reduce loft")

        if needed_lie_delta > 0.25:
            direction.append("more upright (draw-bias)")
        elif needed_lie_delta < -0.25:
            direction.append("flatter (fade-bias)")

        if not direction:
            direction = ["stay near current setting"]

        return {
            "type": "guidance",
            "message": f"Your hosel chart isn’t encoded for exact setting math yet. Guidance: {', '.join(direction)}.",
        }

    scored.sort(key=lambda x: x[0])
    top = scored[:3]

    return {
        "type": "exact",
        "current": current_setting,
        "top": [{"setting": s, "loft_delta": loft, "lie_delta": lie, "score": score} for score, s, loft, lie in top],
    }


# -----------------------------
# UI — Upload
# -----------------------------
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload GSPro CSV", type=["csv"])

    st.divider()
    st.header("Options")
    show_raw = st.checkbox("Show raw tables", value=False)
    min_shots = st.slider("Min shots per club bucket", 3, 20, 5, 1)


if not uploaded:
    st.info("Upload a GSPro CSV to begin. (Web Portal export or Software export both work.)")
    st.stop()

raw_df = pd.read_csv(uploaded)
canon_df, fmt = canonicalize(raw_df)

canon_df["bucket"] = canon_df["club"].apply(bucket_club)

st.success(f"Loaded {len(canon_df)} shots. Detected export format: **{fmt}**")

if show_raw:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw CSV (first 200 rows)")
        st.dataframe(raw_df.head(200), use_container_width=True)
    with c2:
        st.subheader("Canonicalized (first 200 rows)")
        st.dataframe(canon_df.head(200), use_container_width=True)

# Filter to buckets with enough shots
bucket_counts = canon_df["bucket"].value_counts(dropna=False).to_dict()
available_buckets = [b for b, n in bucket_counts.items() if n >= min_shots and b != "OTHER"]
if not available_buckets:
    st.warning(f"No club buckets have at least {min_shots} shots. Try lowering the minimum in the sidebar.")
    st.stop()

st.subheader("Detected Club Buckets")
cols = st.columns(len(available_buckets))
for i, b in enumerate(available_buckets):
    cols[i].metric(b, bucket_counts.get(b, 0))

selected_buckets = st.multiselect(
    "Choose which buckets to analyze",
    options=available_buckets,
    default=available_buckets,
)

if not selected_buckets:
    st.stop()

df = canon_df[canon_df["bucket"].isin(selected_buckets)].copy()


# -----------------------------
# Hosel UI (Driver-focused for MVP)
# -----------------------------
st.divider()
st.header("Hosel & Club Setup (feeds recommendations)")

driver_present = "DR" in selected_buckets
hosel_context = {
    "driver": None,
}

if driver_present:
    st.subheader("Driver Setup")

    c1, c2, c3, c4 = st.columns([1.1, 1.4, 0.9, 1.2])
    with c1:
        brand = st.selectbox("Driver Brand", get_supported_brands(), index=0)
    systems = get_brand_systems(brand)
    system_names = [s.system_name for s in systems] if systems else ["(no systems found)"]

    with c2:
        system_name = st.selectbox("Hosel System", system_names, index=0)
    with c3:
        handedness = st.selectbox("Handedness", ["RH", "LH"], index=0)
    settings = list_settings(brand, system_name, handedness)
    with c4:
        current_setting = st.selectbox("Current Hosel Setting", settings if settings else ["STD"], index=0)

    delta = translate_setting(brand, system_name, current_setting, handedness)
    # delta may be dataclass; convert to dict for storage
    delta_dict = asdict(delta) if hasattr(delta, "__dataclass_fields__") else (delta if isinstance(delta, dict) else {"note": str(delta)})

    st.caption("Hosel translation (what FitCaddie *thinks* this setting does). If loft/lie are blank, we haven’t encoded the exact chart yet.")
    st.json(delta_dict)

    hosel_context["driver"] = {
        "brand": brand,
        "system_name": system_name,
        "handedness": handedness,
        "setting": current_setting,
        "delta": delta_dict,
    }
else:
    st.info("No Driver bucket selected, so hosel inputs are hidden. Select DR above to enable driver hosel recommendations.")


# -----------------------------
# Analysis + Recommendations
# -----------------------------
st.divider()
st.header("Analysis & Recommendations")

for bucket in selected_buckets:
    bucket_df = df[df["bucket"] == bucket].copy()
    if len(bucket_df) < min_shots:
        continue

    st.subheader(f"{bucket} — Overview ({len(bucket_df)} shots)")

    # Basic aggregates
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
        "Face→Target (deg)": (safe_mean(bucket_df["face_to_target_deg"]), safe_std(bucket_df["face_to_target_deg"])),
    }

    # Display metrics in 4 columns
    mcols = st.columns(4)
    keys = list(agg.keys())
    for i, k in enumerate(keys):
        mean_v, std_v = agg[k]
        label = k
        value = "—" if np.isnan(mean_v) else f"{mean_v:.2f}"
        delta_txt = None if np.isnan(std_v) else f"±{std_v:.2f}"
        mcols[i % 4].metric(label, value, delta_txt)

    # Limiting factors + recos
    st.markdown("### Limiting Factors")
    limiting: List[str] = []

    smash_avg = agg["Smash"][0]
    smash_msg = smash_flag_driver(smash_avg) if bucket == "DR" else None
    if smash_msg:
        limiting.append(smash_msg)

    off_avg = agg["Offline (yd)"][0]
    miss = miss_tendency(off_avg)
    limiting.append(f"Miss tendency: **{miss}** (avg offline {off_avg:.1f} yd)" if not np.isnan(off_avg) else "Miss tendency: Unknown")

    spin_avg = agg["Backspin (rpm)"][0]
    limiting.append(f"Spin safety: **{spin_safety(spin_avg)}** (avg {spin_avg:.0f} rpm)" if not np.isnan(spin_avg) else "Spin safety: Unknown")

    vla_avg = agg["VLA (deg)"][0]
    cs_avg = agg["Club Speed (mph)"][0]

    if bucket == "DR" and not np.isnan(cs_avg):
        t = driver_targets(cs_avg)
        launch_lo, launch_hi = t["launch"]
        spin_lo, spin_hi = t["spin"]

        if not np.isnan(vla_avg) and (vla_avg < launch_lo or vla_avg > launch_hi):
            limiting.append(f"Launch window miss: VLA {vla_avg:.1f}° vs target {launch_lo:.1f}–{launch_hi:.1f}° for your speed.")
        if not np.isnan(spin_avg) and (spin_avg < spin_lo or spin_avg > spin_hi):
            limiting.append(f"Spin window miss: {spin_avg:.0f} rpm vs target {spin_lo:.0f}–{spin_hi:.0f} rpm for your speed.")

    for item in limiting:
        st.write("•", item)

    st.markdown("### Recommendations")

    # Priority order you specified:
    # 1) Launch window (speed adjusted)
    # 2) Miss tendency (upright/flatter)
    # 3) Spin safety check
    recos: List[str] = []

    if bucket == "DR":
        # Determine needed loft change for launch (rough heuristic: +1° loft ~ +0.8° VLA)
        needed_loft = 0.0
        needed_lie = 0.0

        if not np.isnan(cs_avg):
            t = driver_targets(cs_avg)
            launch_lo, launch_hi = t["launch"]
            if not np.isnan(vla_avg):
                if vla_avg < launch_lo:
                    needed_loft = min(2.0, (launch_lo - vla_avg) / 0.8)
                elif vla_avg > launch_hi:
                    needed_loft = max(-2.0, -(vla_avg - launch_hi) / 0.8)

        # Miss tendency → lie adjustment
        # Right miss: more upright. Left miss: flatter.
        if not np.isnan(off_avg):
            if off_avg > 5:
                needed_lie = +0.75
            elif off_avg < -5:
                needed_lie = -0.75

        # Spin safety → tweak loft if extreme spin (fallback)
        if not np.isnan(spin_avg):
            if spin_avg < 1800:
                needed_loft = max(needed_loft, +0.75)
            elif spin_avg > 3300:
                needed_loft = min(needed_loft, -0.75)

        # If hosel context exists, attempt setting suggestion
        h = hosel_context.get("driver")
        if h:
            brand = h["brand"]
            system_name = h["system_name"]
            handedness = h["handedness"]
            current_setting = h["setting"]

            reco = pick_hosel_reco(
                brand=brand,
                system_name=system_name,
                handedness=handedness,
                current_setting=current_setting,
                needed_loft_delta=needed_loft,
                needed_lie_delta=needed_lie,
            )

            # Human-friendly output
            if abs(needed_loft) < 0.25 and abs(needed_lie) < 0.25:
                recos.append("Hosel: your current setting looks fine relative to launch/miss/spin (no change suggested).")
            else:
                recos.append(
                    f"Hosel goals (priority order): **loft Δ {needed_loft:+.2f}°** to improve launch window, "
                    f"then **lie Δ {needed_lie:+.2f}°** to manage miss tendency."
                )

                if reco["type"] == "exact":
                    tops = reco["top"]
                    lines = []
                    for r in tops:
                        lines.append(f"{r['setting']} (loft {r['loft_delta']:+.2f}°, lie {r['lie_delta']:+.2f}°)")
                    recos.append(
                        f"Best matching settings (from your brand chart): **{', '.join(lines)}**. "
                        f"Current: {reco['current']}."
                    )
                else:
                    recos.append(reco["message"])
        else:
            recos.append("Hosel: select the **DR** bucket to enable driver hosel recommendations.")

        # Non-hosel recos (shaft/head hints are kept conservative)
        if smash_avg is not None and not np.isnan(smash_avg) and smash_avg < 1.45:
            recos.append("Efficiency: focus on center contact (tee height, strike location). If contact is heel/toe-biased, consider a head with more stability (MOI) or a shaft profile that improves face delivery.")

        if miss == "Right miss tendency":
            recos.append("Directional: if you’re consistently right, check grip + face control; equipment-wise, upright lie / draw setting is your first lever before changing shafts.")
        elif miss == "Left miss tendency":
            recos.append("Directional: if you’re consistently left, a slightly flatter setting can help; also check face-to-path (hook pattern) before reducing loft.")

        # Spin advice
        if spin_safety(spin_avg).startswith("Low"):
            recos.append("Spin safety: avoid chasing even lower spin—prioritize launch and strike. Too-low spin can make dispersion worse.")
        elif spin_safety(spin_avg).startswith("High"):
            recos.append("Spin safety: if launch is fine but spin is high, a lower-spin head/ball or a more forward CG setting may help (after verifying strike isn’t low-face).")

    else:
        # For FW/HY/OTHER we keep it simple for MVP:
        recos.append("MVP note: hosel recommendations are currently driver-first. Fairway/Hybrid support can be added next.")
        if not np.isnan(off_avg):
            recos.append(f"Dispersion: average offline {off_avg:.1f} yd — work on start line (HLA) and face-to-target consistency.")
        if not np.isnan(smash_avg) and smash_avg < 1.40:
            recos.append("Efficiency: low smash suggests strike/loft mismatch — consider loft gapping, lie, and contact pattern.")

    for r in recos:
        st.write("•", r)

    with st.expander("See shot table for this bucket"):
        show_cols = [
            "club",
            "club_speed_mph",
            "ball_speed_mph",
            "smash",
            "carry_yd",
            "total_yd",
            "offline_yd",
            "vla_deg",
            "backspin_rpm",
            "aoa_deg",
            "club_path_deg",
            "face_to_path_deg",
            "face_to_target_deg",
        ]
        st.dataframe(bucket_df[show_cols].reset_index(drop=True), use_container_width=True)

st.divider()
st.caption("If you want, I can also rewrite fit_engine.py to match this structure exactly (so app.py stays clean).")
