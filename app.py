from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FitCaddie",
    page_icon="⛳",
    layout="wide",
)


# =========================================================
# STYLES
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        .fc-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .fc-subtitle {
            color: #5f6b6d;
            margin-bottom: 1rem;
        }

        .fc-card {
            background: #ffffff;
            border: 1px solid #e8ecea;
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            margin-bottom: 1rem;
        }

        .fc-card h3 {
            margin-top: 0;
            margin-bottom: 0.7rem;
            font-size: 1.05rem;
        }

        .fc-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        .fc-pill-good {
            background: #e9f8ef;
            color: #1f7a3f;
            border: 1px solid #cfeeda;
        }

        .fc-pill-warn {
            background: #fff6e7;
            color: #9a6500;
            border: 1px solid #f2dfb4;
        }

        .fc-pill-alert {
            background: #fdeeee;
            color: #aa2e2e;
            border: 1px solid #f1cccc;
        }

        .fc-small {
            color: #5f6b6d;
            font-size: 0.9rem;
        }

        .metric-note {
            color: #667085;
            font-size: 0.85rem;
            margin-top: -0.3rem;
        }

        .section-gap {
            height: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class ClubSummary:
    club_id: str
    n_shots: int

    club_speed_avg: float
    ball_speed_avg: float
    smash_avg: float
    carry_avg: float
    total_avg: float
    offline_avg: float
    offline_abs_avg: float
    hla_avg: float
    vla_avg: float
    backspin_avg: float
    peak_height_avg: float
    descent_avg: float
    spin_axis_avg: float
    aoa_avg: float
    path_avg: float
    ftp_avg: float
    ftt_avg: float

    carry_std: float
    offline_std: float
    ball_speed_std: float
    vla_std: float
    spin_std: float

    start_dir_bias: str
    curve_bias: str
    shot_shape: str
    shot_shape_confidence: str


@dataclass
class DistancePotential:
    expected_carry_yd: float
    actual_carry_yd: float
    delta_yd: float
    label: str


# =========================================================
# HELPERS
# =========================================================
def _fmt(x: Optional[float], digits: int = 1, suffix: str = "") -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}{suffix}"


def _safe_mean(series: pd.Series) -> float:
    if series is None or len(series.dropna()) == 0:
        return float("nan")
    return float(series.mean())


def _safe_std(series: pd.Series) -> float:
    if series is None or len(series.dropna()) <= 1:
        return float("nan")
    return float(series.std(ddof=1))


def _num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s == "":
        return np.nan

    s = s.replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return np.nan
    return float(m.group(0))


def _signed_dir_value(x, right_tokens=("R", "O", "U"), left_tokens=("L", "I-O", "D")) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip().upper()
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return np.nan
    val = float(m.group(0))

    # Explicit left/right markers
    if " L" in s or s.endswith("L") or "° L" in s:
        return -abs(val)
    if " R" in s or s.endswith("R") or "° R" in s:
        return abs(val)

    # Face/path style helpers
    if "I-O" in s:
        return -abs(val)
    if "O-I" in s:
        return abs(val)
    if " U" in s:
        return abs(val)
    if " D" in s:
        return -abs(val)

    return val


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace('="', "", regex=False).str.replace('"', "", regex=False)

    return out


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_text_columns(df.copy())

    mapping = {
        "Club Name": "club_raw",
        "Club": "club_raw",
        "Club Speed (mph)": "club_speed_mph",
        "ClubSpeed": "club_speed_mph",
        "Ball Speed (mph)": "ball_speed_mph",
        "BallSpeed": "ball_speed_mph",
        "Carry Dist (yd)": "carry_yd",
        "Carry": "carry_yd",
        "Total Dist (yd)": "total_yd",
        "TotalDistance": "total_yd",
        "Offline (yd)": "offline_yd",
        "Offline": "offline_yd",
        "Peak Height (yd)": "peak_height_yd",
        "PeakHeight": "peak_height_yd",
        "Desc Angle": "descent_deg",
        "Decent": "descent_deg",
        "HLA": "hla_deg",
        "VLA": "vla_deg",
        "Back Spin": "backspin_rpm",
        "BackSpin": "backspin_rpm",
        "Spin Axis": "spin_axis_deg",
        "rawSpinAxis": "spin_axis_deg",
        "Club AoA": "aoa_deg",
        "AoA": "aoa_deg",
        "Club Path": "path_deg",
        "Path": "path_deg",
        "Face to Path": "face_to_path_deg",
        "FaceToPath": "face_to_path_deg",
        "Face to Target": "face_to_target_deg",
        "FaceToTarget": "face_to_target_deg",
        "SmashFactor": "smash",
    }

    out = pd.DataFrame()
    for src, dst in mapping.items():
        if src in df.columns:
            out[dst] = df[src]

    # Preserve a few extra fields if present
    for extra in ["Shot Time", "Round Date", "Shot Key", "Global Shot Number"]:
        if extra in df.columns:
            out[extra] = df[extra]

    # Numeric conversions
    numeric_direct = [
        "club_speed_mph",
        "ball_speed_mph",
        "carry_yd",
        "total_yd",
        "peak_height_yd",
        "backspin_rpm",
        "smash",
    ]
    for c in numeric_direct:
        if c in out.columns:
            out[c] = out[c].apply(_num)

    signed_fields = [
        "offline_yd",
        "hla_deg",
        "vla_deg",
        "descent_deg",
        "spin_axis_deg",
        "aoa_deg",
        "path_deg",
        "face_to_path_deg",
        "face_to_target_deg",
    ]
    for c in signed_fields:
        if c in out.columns:
            if c == "vla_deg" or c == "descent_deg":
                out[c] = out[c].apply(_num)
            else:
                out[c] = out[c].apply(_signed_dir_value)

    if "smash" not in out.columns and {"ball_speed_mph", "club_speed_mph"}.issubset(out.columns):
        out["smash"] = out["ball_speed_mph"] / out["club_speed_mph"].replace(0, np.nan)

    return out


def bucket_club(club_raw: str) -> str:
    if pd.isna(club_raw):
        return "UNKNOWN"

    s = str(club_raw).strip().upper()

    # Normalize common names
    s = s.replace("HYBRID", "H").replace("HY", "H").replace("FW", "W").replace("WOOD", "W")

    if s in {"DR", "D", "DRIVER"}:
        return "DR"

    if s in {"3W", "4W", "5W", "7W", "9W"}:
        return s

    if re.fullmatch(r"[3579]W", s):
        return s

    if s in {"H2", "2H", "H3", "3H", "H4", "4H", "H5", "5H", "H6", "6H"}:
        digits = re.findall(r"\d+", s)
        return f"{digits[0]}H" if digits else "HY"

    # Irons
    if re.fullmatch(r"[2-9]I", s):
        return s
    if re.fullmatch(r"I[2-9]", s):
        return f"{s[1]}I"

    # Wedges
    if s in {"PW", "GW", "AW", "SW", "LW"}:
        return s

    # Loose fallback
    if "WEDGE" in s:
        if "LOB" in s:
            return "LW"
        if "SAND" in s:
            return "SW"
        if "APPROACH" in s or "GAP" in s:
            return "GW"
        return "PW"

    if "IRON" in s:
        digits = re.findall(r"\d+", s)
        if digits:
            return f"{digits[0]}I"

    if "HY" in s or s.endswith("H"):
        digits = re.findall(r"\d+", s)
        return f"{digits[0]}H" if digits else "HY"

    if s.endswith("W"):
        digits = re.findall(r"\d+", s)
        return f"{digits[0]}W" if digits else "FW"

    return s


def club_family(club_id: str) -> str:
    if club_id == "DR":
        return "driver"
    if club_id.endswith("W"):
        return "fairway"
    if club_id.endswith("H") or club_id == "HY":
        return "hybrid"
    if club_id.endswith("I"):
        return "iron"
    if club_id in {"PW", "GW", "AW", "SW", "LW"}:
        return "wedge"
    return "other"


def prepare_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df_raw)

    if "club_raw" not in df.columns:
        raise ValueError("Could not find a club column in this CSV.")

    df["club_id"] = df["club_raw"].apply(bucket_club)
    df["club_family"] = df["club_id"].apply(club_family)

    # Keep only rows with enough launch data to matter
    req = ["club_id", "carry_yd", "offline_yd", "ball_speed_mph", "club_speed_mph"]
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    df = df.dropna(subset=["club_id"])
    df = df.reset_index(drop=True)
    df["_row_i"] = np.arange(len(df))

    return df


# =========================================================
# FLIGHT WINDOWS / SHOT SHAPE / DISTANCE POTENTIAL
# =========================================================
def speed_adjusted_launch_window(summary: ClubSummary) -> Tuple[float, float]:
    fam = club_family(summary.club_id)
    cs = summary.club_speed_avg

    if fam == "driver":
        if cs < 90:
            return 12.0, 16.5
        if cs < 100:
            return 11.0, 15.5
        return 10.0, 14.5

    if fam == "fairway":
        return 10.0, 15.5

    if fam == "hybrid":
        return 13.0, 18.5

    if fam == "iron":
        return 14.0, 20.0

    if fam == "wedge":
        return 24.0, 34.0

    return 12.0, 18.0


def spin_window(summary: ClubSummary) -> Tuple[float, float]:
    fam = club_family(summary.club_id)
    if fam == "driver":
        return 1800, 3200
    if fam == "fairway":
        return 2800, 4500
    if fam == "hybrid":
        return 3500, 5500
    if fam == "iron":
        return 4000, 7000
    if fam == "wedge":
        return 7000, 10500
    return 2500, 6000


def good_window_buffer(summary: ClubSummary) -> float:
    fam = club_family(summary.club_id)
    if fam == "hybrid":
        return 1.0
    if fam == "fairway":
        return 0.9
    if fam == "driver":
        return 0.8
    return 0.7


def classify_direction(x: float, mild: float = 1.2, strong: float = 3.0) -> str:
    if pd.isna(x):
        return "neutral"
    if x <= -strong:
        return "left"
    if x < -mild:
        return "slight_left"
    if x >= strong:
        return "right"
    if x > mild:
        return "slight_right"
    return "neutral"


def detect_shot_shape(summary: ClubSummary) -> Tuple[str, str, str, str]:
    start = classify_direction(summary.hla_avg, mild=0.8, strong=2.0)
    curve = classify_direction(summary.spin_axis_avg, mild=1.0, strong=3.0)

    # Confidence
    conf_points = 0
    if abs(summary.hla_avg) >= 1.5:
        conf_points += 1
    if abs(summary.spin_axis_avg) >= 2.0:
        conf_points += 1
    if summary.n_shots >= 8:
        conf_points += 1
    if summary.offline_std <= 12:
        conf_points += 1

    if conf_points >= 4:
        confidence = "high"
    elif conf_points >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Pattern naming
    if start in {"right", "slight_right"} and curve in {"right", "slight_right"}:
        shot_shape = "fade / slice bias"
    elif start in {"left", "slight_left"} and curve in {"left", "slight_left"}:
        shot_shape = "draw / hook bias"
    elif start in {"left", "slight_left"} and curve in {"right", "slight_right"}:
        shot_shape = "pull-fade"
    elif start in {"right", "slight_right"} and curve in {"left", "slight_left"}:
        shot_shape = "push-draw"
    elif curve in {"right", "slight_right"}:
        shot_shape = "curve-right bias"
    elif curve in {"left", "slight_left"}:
        shot_shape = "curve-left bias"
    else:
        shot_shape = "fairly neutral"

    return start, curve, shot_shape, confidence


def expected_carry_from_ball_speed(summary: ClubSummary) -> float:
    fam = club_family(summary.club_id)
    bs = summary.ball_speed_avg

    # Simple practical estimates for sim fitting
    if fam == "driver":
        factor = 1.73
    elif fam == "fairway":
        factor = 1.52
    elif fam == "hybrid":
        factor = 1.46
    elif fam == "iron":
        factor = 1.38
    elif fam == "wedge":
        factor = 1.15
    else:
        factor = 1.42

    return bs * factor


def distance_potential_for_summary(summary: ClubSummary) -> DistancePotential:
    expected = expected_carry_from_ball_speed(summary)
    actual = summary.carry_avg
    delta = actual - expected

    if delta >= 6:
        label = "beating expected carry"
    elif delta >= -5:
        label = "right around expected carry"
    elif delta >= -12:
        label = "slightly below expected carry"
    else:
        label = "well below expected carry"

    return DistancePotential(
        expected_carry_yd=expected,
        actual_carry_yd=actual,
        delta_yd=delta,
        label=label,
    )


# =========================================================
# SUMMARIES
# =========================================================
def build_summary(df_club: pd.DataFrame, club_id: str) -> ClubSummary:
    base = ClubSummary(
        club_id=club_id,
        n_shots=len(df_club),
        club_speed_avg=_safe_mean(df_club.get("club_speed_mph")),
        ball_speed_avg=_safe_mean(df_club.get("ball_speed_mph")),
        smash_avg=_safe_mean(df_club.get("smash")),
        carry_avg=_safe_mean(df_club.get("carry_yd")),
        total_avg=_safe_mean(df_club.get("total_yd")),
        offline_avg=_safe_mean(df_club.get("offline_yd")),
        offline_abs_avg=_safe_mean(df_club.get("offline_yd", pd.Series(dtype=float)).abs()),
        hla_avg=_safe_mean(df_club.get("hla_deg")),
        vla_avg=_safe_mean(df_club.get("vla_deg")),
        backspin_avg=_safe_mean(df_club.get("backspin_rpm")),
        peak_height_avg=_safe_mean(df_club.get("peak_height_yd")),
        descent_avg=_safe_mean(df_club.get("descent_deg")),
        spin_axis_avg=_safe_mean(df_club.get("spin_axis_deg")),
        aoa_avg=_safe_mean(df_club.get("aoa_deg")),
        path_avg=_safe_mean(df_club.get("path_deg")),
        ftp_avg=_safe_mean(df_club.get("face_to_path_deg")),
        ftt_avg=_safe_mean(df_club.get("face_to_target_deg")),
        carry_std=_safe_std(df_club.get("carry_yd")),
        offline_std=_safe_std(df_club.get("offline_yd")),
        ball_speed_std=_safe_std(df_club.get("ball_speed_mph")),
        vla_std=_safe_std(df_club.get("vla_deg")),
        spin_std=_safe_std(df_club.get("backspin_rpm")),
        start_dir_bias="neutral",
        curve_bias="neutral",
        shot_shape="fairly neutral",
        shot_shape_confidence="low",
    )

    start, curve, shape, conf = detect_shot_shape(base)
    base.start_dir_bias = start
    base.curve_bias = curve
    base.shot_shape = shape
    base.shot_shape_confidence = conf
    return base


def summarize_by_club(df: pd.DataFrame) -> Dict[str, ClubSummary]:
    out: Dict[str, ClubSummary] = {}
    for club_id, g in df.groupby("club_id"):
        if len(g) < 1:
            continue
        out[club_id] = build_summary(g.copy(), club_id)
    return out


# =========================================================
# RECOMMENDATION GATING
# =========================================================
def current_setup_looks_good(summary: ClubSummary) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    vla_lo, vla_hi = speed_adjusted_launch_window(summary)
    spin_lo, spin_hi = spin_window(summary)
    buffer = good_window_buffer(summary)

    in_launch = (summary.vla_avg >= vla_lo - buffer) and (summary.vla_avg <= vla_hi + buffer)
    in_spin = (summary.backspin_avg >= spin_lo - 250) and (summary.backspin_avg <= spin_hi + 250)

    if in_launch:
        reasons.append("launch is in a good playable window")
    if in_spin:
        reasons.append("spin is in a safe playable window")
    if summary.offline_abs_avg <= 12:
        reasons.append("dispersion is reasonably controlled")
    if summary.ball_speed_std <= 5:
        reasons.append("speed consistency is acceptable")
    if abs(summary.spin_axis_avg) <= 3:
        reasons.append("curvature bias is mild")

    # Stronger good-gate for hybrids / woods so the app does not overreact
    fam = club_family(summary.club_id)
    threshold = 4 if fam in {"hybrid", "fairway"} else 3

    return len(reasons) >= threshold, reasons


def build_limiting_factors(summary: ClubSummary) -> List[str]:
    lim: List[str] = []

    vla_lo, vla_hi = speed_adjusted_launch_window(summary)
    spin_lo, spin_hi = spin_window(summary)

    if summary.vla_avg < vla_lo - 1.2:
        lim.append(f"Launch is low for this speed/club type ({summary.vla_avg:.1f}°).")
    elif summary.vla_avg > vla_hi + 1.2:
        lim.append(f"Launch is high for this speed/club type ({summary.vla_avg:.1f}°).")

    if summary.backspin_avg < spin_lo - 400:
        lim.append(f"Spin is low for this club type ({summary.backspin_avg:.0f} rpm).")
    elif summary.backspin_avg > spin_hi + 400:
        lim.append(f"Spin is high for this club type ({summary.backspin_avg:.0f} rpm).")

    if summary.offline_abs_avg > 15:
        lim.append(f"Average directional miss is meaningful ({summary.offline_abs_avg:.1f} yd offline).")

    if summary.offline_std > 16:
        lim.append(f"Shot pattern is wide ({summary.offline_std:.1f} yd offline std dev).")

    if club_family(summary.club_id) == "driver" and summary.smash_avg < 1.44:
        lim.append(f"Smash factor is low for driver ({summary.smash_avg:.2f}).")
    elif club_family(summary.club_id) in {"fairway", "hybrid"} and summary.smash_avg < 1.40:
        lim.append(f"Smash factor is a little low ({summary.smash_avg:.2f}).")
    elif club_family(summary.club_id) == "iron" and summary.smash_avg < 1.32:
        lim.append(f"Strike efficiency looks low ({summary.smash_avg:.2f}).")

    dp = distance_potential_for_summary(summary)
    if dp.delta_yd <= -10:
        lim.append(
            f"Distance potential: expected carry about {dp.expected_carry_yd:.1f} yd, "
            f"actual {dp.actual_carry_yd:.1f} yd."
        )

    return lim


def build_recommendations(summary: ClubSummary) -> List[str]:
    recs: List[str] = []

    looks_good, good_reasons = current_setup_looks_good(summary)
    fam = club_family(summary.club_id)
    vla_lo, vla_hi = speed_adjusted_launch_window(summary)
    spin_lo, spin_hi = spin_window(summary)

    # Philosophy: default to good unless strong evidence says otherwise
    if looks_good:
        recs.append("Current setup looks good. No clear spec change is required from this sample.")
        if good_reasons:
            recs.append("Why: " + "; ".join(good_reasons[:3]) + ".")
        return recs

    # Two-signal confirmation rule before spec change
    low_launch = summary.vla_avg < vla_lo - 1.0
    high_launch = summary.vla_avg > vla_hi + 1.0
    low_spin = summary.backspin_avg < spin_lo - 350
    high_spin = summary.backspin_avg > spin_hi + 350
    right_bias = summary.hla_avg > 1.5 and summary.spin_axis_avg > 1.5
    left_bias = summary.hla_avg < -1.5 and summary.spin_axis_avg < -1.5

    # 1) Hosel / loft adjustment
    if fam in {"driver", "fairway", "hybrid"}:
        if low_launch and low_spin:
            recs.append("Test a slightly higher loft / more lofted hosel setting first.")
            recs.append("Expected effect: a little more launch, a little more spin, slightly higher peak height.")
            return recs

        if high_launch and high_spin:
            recs.append("Test a slightly lower loft / less lofted hosel setting first.")
            recs.append("Expected effect: a little less launch, a little less spin, and a flatter peak flight.")
            return recs

        if right_bias:
            recs.append("If your hosel allows lie/face adjustment, test a slightly more upright or draw-biased setting.")
            recs.append("Expected effect: start line and curvature may move a bit less right.")
            return recs

        if left_bias:
            recs.append("If your hosel allows it, test a slightly flatter or fade-biased setting.")
            recs.append("Expected effect: start line and curvature may move a bit less left.")
            return recs

    # 2) Setup / strike note
    if summary.smash_avg < 1.40 and fam in {"driver", "fairway", "hybrid"}:
        recs.append("Before changing parts, verify strike location and impact consistency.")
        recs.append("The data suggests some carry may be left on the table from strike efficiency, not just spec fit.")
        return recs

    # 3) Shaft profile test
    if low_launch and not low_spin:
        recs.append("If you test a shaft, start with a slightly softer-launch profile before making a bigger head change.")
        return recs

    if high_launch and not high_spin:
        recs.append("If you test a shaft, start with a slightly firmer/lower-launch profile rather than changing the club head.")
        return recs

    # 4) Head / club type change
    if fam == "hybrid" and summary.peak_height_avg < 22 and summary.descent_avg < 36:
        recs.append("If this club must hold greens, a higher-launch hybrid or weak-loft option could be worth testing.")
        return recs

    recs.append("Current setup is close enough that no strong spec recommendation stands out from this sample.")
    recs.append("Collect a few more shots before making a parts change.")
    return recs


# =========================================================
# CHARTS
# =========================================================
def render_dispersion(df_club: pd.DataFrame, key_prefix: str = "disp") -> None:
    d = df_club.copy()

    if "offline_yd" not in d.columns or "carry_yd" not in d.columns:
        st.info("Dispersion chart needs offline and carry columns.")
        return

    d = d.dropna(subset=["offline_yd", "carry_yd"]).copy()
    if len(d) == 0:
        st.info("No valid dispersion data.")
        return

    d["_row_i"] = np.arange(len(d))

    hover_cols = []
    hover_bits = []

    def add_hover(col: str, label: str, fmt: str):
        if col in d.columns:
            hover_cols.append(col)
            hover_bits.append(f"{label}: %{{customdata[{len(hover_cols) - 1}]{fmt}}}")

    add_hover("club_speed_mph", "Club Speed", ":.1f")
    add_hover("ball_speed_mph", "Ball Speed", ":.1f")
    add_hover("smash", "Smash", ":.2f")
    add_hover("vla_deg", "Launch", ":.1f")
    add_hover("backspin_rpm", "Spin", ":.0f")

    customdata = d[hover_cols].to_numpy() if hover_cols else None
    hovertemplate = "Carry: %{y:.1f} yd<br>Offline: %{x:.1f} yd"
    if hover_bits:
        hovertemplate += "<br>" + "<br>".join(hover_bits)
    hovertemplate += "<extra></extra>"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["offline_yd"],
            y=d["carry_yd"],
            mode="markers",
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="Shots",
            marker=dict(size=10, opacity=0.82),
        )
    )

    # Centerline
    fig.add_vline(x=0, line_width=1, line_dash="dash")

    # Average marker
    fig.add_trace(
        go.Scatter(
            x=[d["offline_yd"].mean()],
            y=[d["carry_yd"].mean()],
            mode="markers",
            marker=dict(size=16, symbol="diamond"),
            name="Average",
            hovertemplate="Average<br>Offline: %{x:.1f} yd<br>Carry: %{y:.1f} yd<extra></extra>",
        )
    )

    x_pad = max(8, abs(d["offline_yd"]).max() * 0.15)
    y_pad = max(8, d["carry_yd"].max() * 0.08)

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Offline (left / right, yd)",
        yaxis_title="Carry distance (yd)",
        xaxis=dict(range=[d["offline_yd"].min() - x_pad, d["offline_yd"].max() + x_pad]),
        yaxis=dict(range=[0, d["carry_yd"].max() + y_pad]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")


# =========================================================
# UI RENDERERS
# =========================================================
def render_summary_cards(summary: ClubSummary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Club Speed", _fmt(summary.club_speed_avg, 1))
    c2.metric("Ball Speed", _fmt(summary.ball_speed_avg, 1))
    c3.metric("Smash", _fmt(summary.smash_avg, 2))
    c4.metric("Carry", _fmt(summary.carry_avg, 1))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Offline", _fmt(summary.offline_avg, 1))
    c6.metric("Launch", _fmt(summary.vla_avg, 1, "°"))
    c7.metric("Spin", _fmt(summary.backspin_avg, 0, " rpm"))
    c8.metric("Peak Height", _fmt(summary.peak_height_avg, 1))


def render_flight_window(summary: ClubSummary) -> None:
    vla_lo, vla_hi = speed_adjusted_launch_window(summary)
    spin_lo, spin_hi = spin_window(summary)

    st.markdown("**Flight Window**")
    a, b = st.columns(2)

    with a:
        st.metric("Launch Avg", _fmt(summary.vla_avg, 1, "°"))
        st.caption(f"Target window: {vla_lo:.1f}° to {vla_hi:.1f}°")

    with b:
        st.metric("Spin Avg", _fmt(summary.backspin_avg, 0, " rpm"))
        st.caption(f"Target window: {spin_lo:.0f} to {spin_hi:.0f} rpm")


def render_shot_shape(summary: ClubSummary) -> None:
    conf_class = {
        "high": "fc-pill-good",
        "medium": "fc-pill-warn",
        "low": "fc-pill-alert",
    }.get(summary.shot_shape_confidence, "fc-pill-warn")

    st.markdown("**Shot Shape**", unsafe_allow_html=True)
    st.markdown(
        f"""
        <span class="fc-pill fc-pill-good">{summary.shot_shape}</span>
        <span class="fc-pill {conf_class}">confidence: {summary.shot_shape_confidence}</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Start line avg: {_fmt(summary.hla_avg, 1, '°')} | "
        f"Spin axis avg: {_fmt(summary.spin_axis_avg, 1, '°')}"
    )


def render_distance_potential(summary: ClubSummary) -> None:
    dp = distance_potential_for_summary(summary)

    st.markdown("**Distance Potential**")
    a, b, c = st.columns(3)
    a.metric("Expected Carry", _fmt(dp.expected_carry_yd, 1))
    b.metric("Actual Carry", _fmt(dp.actual_carry_yd, 1))
    c.metric("Delta", _fmt(dp.delta_yd, 1))
    st.caption(dp.label)


def render_variability(summary: ClubSummary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Carry Std Dev", _fmt(summary.carry_std, 1))
    c2.metric("Offline Std Dev", _fmt(summary.offline_std, 1))
    c3.metric("Ball Speed Std Dev", _fmt(summary.ball_speed_std, 1))
    c4.metric("Spin Std Dev", _fmt(summary.spin_std, 0))


def render_list_card(title: str, items: List[str], good: bool = False) -> None:
    st.markdown(f'<div class="fc-card"><h3>{title}</h3>', unsafe_allow_html=True)

    if not items:
        if good:
            st.success("No meaningful issue flagged from this sample.")
        else:
            st.write("Nothing major to flag from this sample.")
    else:
        for item in items:
            st.write(f"• {item}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_focus_picker(selected_clubs: List[str]) -> str:
    if not selected_clubs:
        st.stop()

    families_present = []
    if any(c == "DR" for c in selected_clubs):
        families_present.append("Driver")
    if any(c.endswith("W") for c in selected_clubs):
        families_present.append("Fairway Wood")
    if any(c.endswith("H") for c in selected_clubs):
        families_present.append("Hybrid")
    if any(c.endswith("I") for c in selected_clubs):
        families_present.append("Iron")
    if any(c in {"PW", "GW", "AW", "SW", "LW"} for c in selected_clubs):
        families_present.append("Wedge")

    st.markdown("### Club Focus")
    focus = st.selectbox("Choose a club to inspect", options=selected_clubs, index=0)

    if families_present:
        st.caption("Families found in file: " + ", ".join(families_present))

    return focus


# =========================================================
# MAIN APP
# =========================================================
def main() -> None:
    st.markdown('<div class="fc-title">FitCaddie</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fc-subtitle">Upload a GSPro CSV and get a practical fitting read on launch, spin, dispersion, shot shape, and spec direction.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Filters")
        min_shots = st.slider("Minimum shots per club", min_value=3, max_value=20, value=5, step=1)
        st.caption("Higher values reduce noise but may hide clubs with smaller samples.")

    uploaded = st.file_uploader("Upload GSPro CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a GSPro CSV export to begin.")
        return

    try:
        raw = pd.read_csv(uploaded)
        canon_df = prepare_dataset(raw)
    except Exception as e:
        st.error(f"Could not read this CSV: {e}")
        return

    if canon_df.empty:
        st.warning("No usable shot data found in this file.")
        return

    shot_counts = canon_df["club_id"].value_counts().sort_index()
    club_ids = [c for c in shot_counts.index.tolist() if shot_counts[c] >= min_shots]

    if not club_ids:
        st.warning(f"No clubs have at least {min_shots} shots. Try lowering the filter.")
        return

    focus_club = render_focus_picker(club_ids)
    focus_df = canon_df[canon_df["club_id"] == focus_club].copy()

    summaries = summarize_by_club(focus_df)
    if focus_club not in summaries:
        st.warning("No valid summary available for that club.")
        return

    summary = summaries[focus_club]
    limiting = build_limiting_factors(summary)
    recs = build_recommendations(summary)
    looks_good, good_reasons = current_setup_looks_good(summary)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    top1, top2 = st.columns([1.35, 1.0])

    with top1:
        st.markdown('<div class="fc-card"><h3>Dispersion</h3>', unsafe_allow_html=True)
        render_dispersion(focus_df, key_prefix="focus")
        st.markdown("</div>", unsafe_allow_html=True)

    with top2:
        st.markdown(f'<div class="fc-card"><h3>{focus_club} Overview</h3>', unsafe_allow_html=True)
        render_summary_cards(summary)
        st.caption(f"Shots used: {summary.n_shots}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fc-card"><h3>Shot Shape & Flight</h3>', unsafe_allow_html=True)
        render_shot_shape(summary)
        st.divider()
        render_flight_window(summary)
        st.divider()
        render_distance_potential(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    mid1, mid2 = st.columns(2)

    with mid1:
        st.markdown('<div class="fc-card"><h3>Variability</h3>', unsafe_allow_html=True)
        render_variability(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    with mid2:
        st.markdown('<div class="fc-card"><h3>Setup Confidence</h3>', unsafe_allow_html=True)
        if looks_good:
            st.success("Current setup looks good.")
            for r in good_reasons:
                st.write(f"• {r}")
        else:
            st.warning("No strong 'leave it alone' signal yet.")
            st.caption("The sample shows enough mixed signals that a small test change may be worth it.")
        st.markdown("</div>", unsafe_allow_html=True)

    render_list_card("Limiting Factors", limiting, good=False)
    render_list_card("Recommendations", recs, good=looks_good)

    with st.expander("Preview cleaned shot data"):
        preview_cols = [
            c for c in [
                "club_id",
                "club_speed_mph",
                "ball_speed_mph",
                "smash",
                "carry_yd",
                "total_yd",
                "offline_yd",
                "hla_deg",
                "vla_deg",
                "backspin_rpm",
                "spin_axis_deg",
                "peak_height_yd",
                "descent_deg",
                "aoa_deg",
                "path_deg",
                "face_to_path_deg",
                "face_to_target_deg",
            ] if c in focus_df.columns
        ]
        st.dataframe(focus_df[preview_cols], use_container_width=True)

    csv_export = focus_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned focus-club CSV",
        data=csv_export,
        file_name=f"fitcaddie_{focus_club.lower()}_cleaned.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
