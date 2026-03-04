# fit_engine.py
# FitCaddie Spec-Range MVP engine (drop-in replacement)
#
# Includes:
# - Robust GSPro CSV parsing/canonicalization (handles "11.4 R", "40.1°", "181.28 yds", missing/renamed cols)
# - Outlier filtering
# - Club summaries + variability
# - Limiting factors (smash factor flagged clearly; driver smash < ~1.44)
# - Recommendations
# - Titleist SureFit tables (Driver/Fairway RH+LH; Hybrid RH+LH)
# - Generalized hosel function: recommend_titleist_surefit(...) for DR/3W/HY and RH/LH
#
# SureFit charts are encoded as OEM setting grids (A1..D4) mapped to loft/lie deltas:
# Driver/Fairway: 0.75° increments; Hybrid: 1° increments.
#
# Titleist reference:
# https://www.titleist.com/fitting/golf-club-fitting/surefit

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import re

import numpy as np
import pandas as pd


# =============================================================================
# Column canonicalization / parsing
# =============================================================================

_CANON_MAP: Dict[str, List[str]] = {
    "club": ["club", "club name", "club_name", "clubname", "club type", "clubtype"],
    "club_speed_mph": ["club speed", "club speed (mph)", "clubspeed", "club_speed", "club head speed", "clubhead speed"],
    "ball_speed_mph": ["ball speed", "ball speed (mph)", "ballspeed", "ball_speed"],
    "carry_yd": ["carry dist", "carry dist (yd)", "carry distance", "carry", "carry (yd)"],
    "total_yd": ["total dist", "total dist (yd)", "total distance", "total", "total (yd)"],
    "offline_yd": ["offline", "offline (yd)", "offline dist", "offline distance"],
    "launch_deg": ["vla", "launch", "launch angle", "launch angle (deg)", "launch (deg)", "vertical launch", "vertical launch angle"],
    "h_launch_deg": ["hla", "horizontal launch", "horizontal launch angle", "horizontal launch angle (deg)", "h launch", "h launch (deg)"],
    "spin_rpm": ["back spin", "backspin", "spin", "spin (rpm)", "back spin (rpm)"],
    "spin_axis_deg": ["spin axis", "spinaxis", "axis", "spin axis (deg)"],
    "aoa_deg": ["club aoa", "aoa", "attack angle", "attack", "club attack angle"],
    "club_path_deg": ["club path", "path", "clubpath"],
    "face_to_path_deg": ["face to path", "face_to_path", "face-path", "ftp"],
    "face_to_target_deg": ["face to target", "face_to_target", "face-target", "ftt"],
    "peak_height_yd": ["peak height", "peak height (yd)", "apex", "apex height", "max height"],
    "desc_angle_deg": ["desc angle", "descent angle", "descent", "descent angle (deg)"],
    "smash_factor": ["smash factor", "smash", "sf"],
}


def _norm_col(c: str) -> str:
    c = c.strip().lower()
    c = re.sub(r"[\(\)\[\]\{\}]", "", c)
    c = re.sub(r"[^a-z0-9]+", " ", c).strip()
    return c


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_float_any(v) -> float:
    """Parse numeric from strings like '40.1°', '181.28 yds', '  10.2 ', '', NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return np.nan
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group(0)) if m else np.nan


def _parse_spin_axis_signed(v) -> float:
    """
    Parse spin axis strings like:
      '11.4 R' -> +11.4
      '8.2 L'  -> -8.2
      '-6.0'   -> -6.0
    """
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return np.nan

    s = str(v).strip().upper()
    if s == "" or s in {"NAN", "NONE", "NULL"}:
        return np.nan

    val = _parse_float_any(s)
    if np.isnan(val):
        return np.nan

    if " L" in s or s.endswith("L"):
        return -abs(val)
    if " R" in s or s.endswith("R"):
        return abs(val)
    return val


def canonicalize_gspro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns into canonical schema; tolerate missing/renamed columns."""
    cols_norm = {_norm_col(c): c for c in df.columns}
    rename: Dict[str, str] = {}

    for canon, aliases in _CANON_MAP.items():
        found = None
        for a in aliases:
            key = _norm_col(a)
            if key in cols_norm:
                found = cols_norm[key]
                break
        if found:
            rename[found] = canon

    out = df.rename(columns=rename).copy()

    # Ensure all canonical columns exist
    for canon in _CANON_MAP.keys():
        if canon not in out.columns:
            out[canon] = np.nan

    # Standardize club labels
    out["club"] = out["club"].astype(str).str.strip()

    # Parse numerics
    numeric_cols = [
        "club_speed_mph",
        "ball_speed_mph",
        "carry_yd",
        "total_yd",
        "offline_yd",
        "launch_deg",
        "h_launch_deg",
        "spin_rpm",
        "aoa_deg",
        "club_path_deg",
        "face_to_path_deg",
        "face_to_target_deg",
        "peak_height_yd",
        "desc_angle_deg",
        "smash_factor",
    ]
    for c in numeric_cols:
        out[c] = out[c].apply(_parse_float_any)

    out["spin_axis_deg"] = out["spin_axis_deg"].apply(_parse_spin_axis_signed)

    # Compute smash if missing
    mask = out["smash_factor"].isna() & out["club_speed_mph"].notna() & out["ball_speed_mph"].notna()
    out.loc[mask, "smash_factor"] = out.loc[mask, "ball_speed_mph"] / out.loc[mask, "club_speed_mph"]

    return out


def load_gspro_csv(file_like) -> pd.DataFrame:
    """Read CSV robustly from Streamlit upload or file path."""
    try:
        df = pd.read_csv(file_like)
    except Exception:
        df = pd.read_csv(file_like, engine="python")
    return canonicalize_gspro_columns(df)


# =============================================================================
# Club bucketing (DR / 3W / HY)
# =============================================================================

def bucket_club(club_str: str) -> Optional[str]:
    if club_str is None:
        return None
    s = str(club_str).strip().upper()

    if s in {"DR", "DRIVER"}:
        return "DR"

    # Fairway / 3W
    if "3W" in s or "3 W" in s or "3WOOD" in s or "3 WOOD" in s:
        return "3W"
    if s in {"FW", "FAIRWAY"}:
        return "3W"

    # Hybrid
    if s in {"HY", "HYBRID"} or "HYBRID" in s:
        return "HY"
    if re.search(r"\b[2-7]H\b", s):  # e.g., 3H
        return "HY"

    return None


def add_bucket_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bucket"] = out["club"].apply(bucket_club)
    return out


# =============================================================================
# Outlier filtering
# =============================================================================

@dataclass
class FilterReport:
    n_in: int
    n_out: int
    reason_counts: Dict[str, int]


def filter_outliers(
    df: pd.DataFrame,
    bucket: str,
    z: float = 3.5,
    iqr_k: float = 2.0,
) -> Tuple[pd.DataFrame, FilterReport]:
    """
    Practical filtering:
    - remove impossible values
    - robust z-score (MAD) on key metrics
    - IQR bounds for carry/offline/spin where present
    """
    d = df[df["bucket"] == bucket].copy()
    n_in = len(d)
    reasons: Dict[str, int] = {}

    def _flag(mask: pd.Series, reason: str) -> None:
        nonlocal d
        c = int(mask.sum())
        if c > 0:
            reasons[reason] = reasons.get(reason, 0) + c
        d = d[~mask]

    # Hard sanity bounds
    _flag((d["club_speed_mph"] < 40) | (d["club_speed_mph"] > 140), "club_speed_bounds")
    _flag((d["ball_speed_mph"] < 50) | (d["ball_speed_mph"] > 220), "ball_speed_bounds")
    _flag((d["smash_factor"] < 1.0) | (d["smash_factor"] > 1.65), "smash_bounds")
    _flag((d["launch_deg"] < -5) | (d["launch_deg"] > 25), "launch_bounds")
    _flag((d["spin_rpm"] < 300) | (d["spin_rpm"] > 6000), "spin_bounds")
    _flag((d["carry_yd"] < 50) | (d["carry_yd"] > 350), "carry_bounds")
    _flag((d["offline_yd"].abs() > 120), "offline_bounds")

    def robust_z(series: pd.Series) -> pd.Series:
        # Returns a series with same index as input
        x = series.dropna()
        if len(x) < 6:
            return pd.Series(index=series.index, data=np.zeros(len(series)))
        med = np.nanmedian(series.values)
        mad = np.nanmedian(np.abs(series.values - med))
        if mad == 0 or np.isnan(mad):
            return pd.Series(index=series.index, data=np.zeros(len(series)))
        return 0.6745 * (series - med) / mad

    for col in ["club_speed_mph", "ball_speed_mph", "carry_yd", "spin_rpm", "launch_deg"]:
        rz = robust_z(d[col])
        _flag(rz.abs() > z, f"robust_z_{col}")

    for col in ["carry_yd", "spin_rpm", "offline_yd"]:
        x = d[col].dropna()
        if len(x) >= 8:
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
            _flag((d[col] < lo) | (d[col] > hi), f"iqr_{col}")

    n_out = n_in - len(d)
    return d, FilterReport(n_in=n_in, n_out=n_out, reason_counts=reasons)


# =============================================================================
# Summaries / variability
# =============================================================================

def summarize_bucket(df_bucket: pd.DataFrame) -> Dict[str, float]:
    keys = [
        "club_speed_mph",
        "ball_speed_mph",
        "smash_factor",
        "launch_deg",
        "spin_rpm",
        "carry_yd",
        "total_yd",
        "offline_yd",
        "h_launch_deg",
        "spin_axis_deg",
        "aoa_deg",
        "club_path_deg",
        "face_to_path_deg",
        "face_to_target_deg",
        "peak_height_yd",
        "desc_angle_deg",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.nanmean(df_bucket[k].values)) if k in df_bucket.columns else np.nan
    out["n_shots"] = int(len(df_bucket))
    return out


def variability_bucket(df_bucket: pd.DataFrame) -> Dict[str, float]:
    out = {
        "offline_std_yd": float(np.nanstd(df_bucket["offline_yd"].values, ddof=1)) if len(df_bucket) > 1 else np.nan,
        "offline_p95_abs_yd": float(np.nanpercentile(np.abs(df_bucket["offline_yd"].values), 95)) if len(df_bucket) > 0 else np.nan,
        "carry_std_yd": float(np.nanstd(df_bucket["carry_yd"].values, ddof=1)) if len(df_bucket) > 1 else np.nan,
        "spin_std_rpm": float(np.nanstd(df_bucket["spin_rpm"].values, ddof=1)) if len(df_bucket) > 1 else np.nan,
        "launch_std_deg": float(np.nanstd(df_bucket["launch_deg"].values, ddof=1)) if len(df_bucket) > 1 else np.nan,
        "smash_std": float(np.nanstd(df_bucket["smash_factor"].values, ddof=1)) if len(df_bucket) > 1 else np.nan,
    }
    return out


# =============================================================================
# Limiting factors / heuristics
# =============================================================================

def _driver_smash_threshold(club_speed_mph: float) -> float:
    # Your requested anchor: driver smash < ~1.44 should flag clearly.
    if np.isnan(club_speed_mph):
        return 1.44
    if club_speed_mph >= 110:
        return 1.46
    if club_speed_mph >= 100:
        return 1.44
    return 1.42


def limiting_factors(bucket: str, merged_stats: Dict[str, float]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []

    smash = merged_stats.get("smash_factor", np.nan)
    cs = merged_stats.get("club_speed_mph", np.nan)

    if bucket == "DR":
        th = _driver_smash_threshold(cs)
        if not np.isnan(smash) and smash < th:
            issues.append(
                {
                    "title": "Smash factor is limiting ball speed",
                    "severity": "High",
                    "detail": f"Avg smash {smash:.2f} (flag < {th:.2f}). Strike efficiency/contact is likely costing distance.",
                }
            )

    off_std = merged_stats.get("offline_std_yd", np.nan)
    if not np.isnan(off_std) and off_std >= 18:
        issues.append(
            {
                "title": "High left/right dispersion",
                "severity": "Med" if off_std < 26 else "High",
                "detail": f"Offline std dev ≈ {off_std:.1f} yd.",
            }
        )

    spin = merged_stats.get("spin_rpm", np.nan)
    launch = merged_stats.get("launch_deg", np.nan)

    if bucket == "DR" and (not np.isnan(spin)) and (not np.isnan(cs)):
        if cs >= 105 and spin > 3100:
            issues.append({"title": "Spin is high (balloon risk)", "severity": "Med", "detail": f"Avg spin {spin:.0f} rpm at {cs:.0f} mph."})
        if cs >= 95 and spin > 3400:
            issues.append({"title": "Spin is very high (balloon risk)", "severity": "High", "detail": f"Avg spin {spin:.0f} rpm at {cs:.0f} mph."})

    if bucket == "DR" and (not np.isnan(cs)) and (not np.isnan(launch)):
        if cs >= 100 and launch < 10.0:
            issues.append({"title": "Launch is low for speed", "severity": "Med", "detail": f"Avg launch {launch:.1f}° at {cs:.0f} mph."})

    return issues


# =============================================================================
# Launch window targets (swing-speed adjusted)
# =============================================================================

@dataclass(frozen=True)
class LaunchTarget:
    launch_lo: float
    launch_hi: float
    spin_lo: float
    spin_hi: float


def target_windows(bucket: str, club_speed_mph: float) -> LaunchTarget:
    cs = club_speed_mph if not np.isnan(club_speed_mph) else 100.0

    if bucket == "DR":
        if cs >= 110:
            return LaunchTarget(launch_lo=11.0, launch_hi=14.0, spin_lo=1900, spin_hi=2600)
        if cs >= 100:
            return LaunchTarget(launch_lo=11.0, launch_hi=14.0, spin_lo=2100, spin_hi=2800)
        if cs >= 90:
            return LaunchTarget(launch_lo=12.0, launch_hi=15.0, spin_lo=2300, spin_hi=3100)
        return LaunchTarget(launch_lo=13.0, launch_hi=16.0, spin_lo=2400, spin_hi=3300)

    if bucket == "3W":
        if cs >= 100:
            return LaunchTarget(launch_lo=11.5, launch_hi=15.0, spin_lo=2800, spin_hi=3800)
        if cs >= 90:
            return LaunchTarget(launch_lo=12.5, launch_hi=16.0, spin_lo=3000, spin_hi=4200)
        return LaunchTarget(launch_lo=13.0, launch_hi=17.0, spin_lo=3200, spin_hi=4500)

    # HY
    if cs >= 95:
        return LaunchTarget(launch_lo=12.0, launch_hi=16.0, spin_lo=3200, spin_hi=4800)
    if cs >= 85:
        return LaunchTarget(launch_lo=13.0, launch_hi=17.0, spin_lo=3400, spin_hi=5200)
    return LaunchTarget(launch_lo=14.0, launch_hi=18.0, spin_lo=3600, spin_hi=5600)


# =============================================================================
# Titleist SureFit tables (OEM grids -> loft/lie deltas)
# =============================================================================

# Driver/Fairway RH:
# Rows (top->bottom) loft: +1.5, +0.75, STD, -0.75
# Cols (left->right) lie: +1.5 upright, +0.75 upright, STD, -0.75 flat
_SUREFIT_DF_RH_GRID = [
    ["A3", "B3", "A4", "B4"],
    ["D3", "C3", "D4", "C4"],
    ["A2", "B2", "A1", "B1"],
    ["D2", "C2", "D1", "C1"],
]
_SUREFIT_DF_RH_LOFTS = [+1.5, +0.75, 0.0, -0.75]
_SUREFIT_DF_RH_LIES = [+1.5, +0.75, 0.0, -0.75]

# Driver/Fairway LH:
# Rows loft: +1.5, +0.75, STD, -0.75
# Cols lie: -0.75 flat, STD, +0.75 upright, +1.5 upright
_SUREFIT_DF_LH_GRID = [
    ["C1", "D1", "C2", "D2"],
    ["B1", "A1", "B2", "A2"],
    ["C4", "D4", "C3", "D3"],
    ["B4", "A4", "B3", "A3"],
]
_SUREFIT_DF_LH_LOFTS = [+1.5, +0.75, 0.0, -0.75]
_SUREFIT_DF_LH_LIES = [-0.75, 0.0, +0.75, +1.5]

# Hybrid RH:
# Rows loft: +2, +1, STD, -1
# Cols lie: +2 upright, +1 upright, STD, -1 flat
_SUREFIT_HY_RH_GRID = [
    ["A3", "B3", "A4", "B4"],
    ["D3", "C3", "D4", "C4"],
    ["A2", "B2", "A1", "B1"],
    ["D2", "C2", "D1", "C1"],
]
_SUREFIT_HY_RH_LOFTS = [+2.0, +1.0, 0.0, -1.0]
_SUREFIT_HY_RH_LIES = [+2.0, +1.0, 0.0, -1.0]

# Hybrid LH:
# Rows loft: +2, +1, STD, -1
# Cols lie: -1 flat, STD, +1 upright, +2 upright
_SUREFIT_HY_LH_GRID = [
    ["C1", "D1", "C2", "D2"],
    ["B1", "A1", "B2", "A2"],
    ["C4", "D4", "C3", "D3"],
    ["B4", "A4", "B3", "A3"],
]
_SUREFIT_HY_LH_LOFTS = [+2.0, +1.0, 0.0, -1.0]
_SUREFIT_HY_LH_LIES = [-1.0, 0.0, +1.0, +2.0]


def _grid_to_map(grid: List[List[str]], loft_rows: List[float], lie_cols: List[float]) -> Dict[str, Dict[str, float]]:
    m: Dict[str, Dict[str, float]] = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            code = grid[r][c]
            m[code] = {"loft_delta": float(loft_rows[r]), "lie_delta": float(lie_cols[c])}
    return m


_SUREFIT_MAPS: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {
    ("DR", "RH"): _grid_to_map(_SUREFIT_DF_RH_GRID, _SUREFIT_DF_RH_LOFTS, _SUREFIT_DF_RH_LIES),
    ("3W", "RH"): _grid_to_map(_SUREFIT_DF_RH_GRID, _SUREFIT_DF_RH_LOFTS, _SUREFIT_DF_RH_LIES),
    ("DR", "LH"): _grid_to_map(_SUREFIT_DF_LH_GRID, _SUREFIT_DF_LH_LOFTS, _SUREFIT_DF_LH_LIES),
    ("3W", "LH"): _grid_to_map(_SUREFIT_DF_LH_GRID, _SUREFIT_DF_LH_LOFTS, _SUREFIT_DF_LH_LIES),
    ("HY", "RH"): _grid_to_map(_SUREFIT_HY_RH_GRID, _SUREFIT_HY_RH_LOFTS, _SUREFIT_HY_RH_LIES),
    ("HY", "LH"): _grid_to_map(_SUREFIT_HY_LH_GRID, _SUREFIT_HY_LH_LOFTS, _SUREFIT_HY_LH_LIES),
}


def sure_fit_options(bucket: str, handedness: str) -> List[str]:
    """Convenience for UI dropdowns."""
    m = _SUREFIT_MAPS.get((bucket.upper(), handedness.upper()))
    if not m:
        return []
    return sorted(m.keys())


# =============================================================================
# Hosel recommender (required priority: launch -> miss -> spin safety)
# =============================================================================

@dataclass
class SureFitRecommendation:
    recommended: str
    current: Optional[str]
    loft_delta: float
    lie_delta: float
    score: float
    rationale: List[str]


def _launch_effect_from_loft(bucket: str, loft_delta: float) -> float:
    # MVP heuristic translating mechanical loft changes into expected launch changes
    if bucket in {"DR", "3W"}:
        return 0.65 * loft_delta
    return 0.75 * loft_delta


def _lie_direction_effect(handedness: str, lie_delta: float) -> float:
    """
    Positive value = more LEFT start bias for RH.
    For LH, sign flips (upright tends to start more RIGHT).
    """
    if handedness.upper() == "RH":
        return lie_delta
    return -lie_delta


def recommend_titleist_surefit(
    bucket: str,
    handedness: str,
    club_speed_mph: float,
    launch_deg: float,
    spin_rpm: float,
    miss_tendency: str,
    current_setting: Optional[str] = None,
) -> SureFitRecommendation:
    """
    REQUIRED priority order:
      1) Launch window (swing-speed adjusted)
      2) Miss tendency (upright / flatter) from user input
      3) Spin safety check (avoid ballooning if spin already high)
      Then pick closest SureFit setting from the OEM table.
    """
    bucket = bucket.upper().strip()
    handedness = handedness.upper().strip()
    miss = (miss_tendency or "STRAIGHT").strip().upper()

    if (bucket, handedness) not in _SUREFIT_MAPS:
        raise ValueError(f"Unsupported combo bucket={bucket} handedness={handedness}")

    sure_map = _SUREFIT_MAPS[(bucket, handedness)]
    tgt = target_windows(bucket, club_speed_mph)
    target_launch = 0.5 * (tgt.launch_lo + tgt.launch_hi)

    # Miss direction objective in RH-left-positive space
    # RH: RIGHT miss -> want more LEFT -> +lie
    # LH: RIGHT miss -> want more LEFT -> -lie (because upright pushes start right for LH)
    if miss == "RIGHT":
        desired_lie_dir = +1.0
    elif miss == "LEFT":
        desired_lie_dir = -1.0
    else:
        desired_lie_dir = 0.0

    spin_high = (not np.isnan(spin_rpm)) and (spin_rpm > tgt.spin_hi)

    # Weights follow your required priority order
    W_LAUNCH = 1.00
    W_MISS = 0.35
    W_SPIN = 0.45

    best_code: Optional[str] = None
    best_score = float("inf")
    best_meta = None

    for code, adj in sure_map.items():
        loft_d = adj["loft_delta"]
        lie_d = adj["lie_delta"]

        # 1) Launch fit
        pred_launch = launch_deg + _launch_effect_from_loft(bucket, loft_d)
        launch_err = abs(pred_launch - target_launch)

        # 2) Miss fit
        lie_effect = _lie_direction_effect(handedness, lie_d)  # RH-left-positive space

        if desired_lie_dir == 0.0:
            miss_err = abs(lie_effect) * 0.35
        else:
            desired_mag = 1.0 if bucket == "HY" else 0.75
            miss_err = 0.0
            if math.copysign(1.0, lie_effect) != math.copysign(1.0, desired_lie_dir):
                miss_err += 2.0  # wrong direction is heavily penalized
            miss_err += abs(abs(lie_effect) - desired_mag)

        # 3) Spin safety
        spin_pen = 0.0
        if spin_high and loft_d > 0:
            spin_pen = 1.0 + 0.6 * loft_d

        score = (W_LAUNCH * launch_err) + (W_MISS * miss_err) + (W_SPIN * spin_pen)

        if best_code is None or score < best_score - 1e-6:
            best_code = code
            best_score = score
            best_meta = (pred_launch, launch_err, miss_err, spin_pen, loft_d, lie_d)
        elif abs(score - best_score) <= 1e-6 and current_setting:
            # Tie-break: prefer staying at current if truly equal (fitter-like minimal change)
            if code == current_setting.strip().upper():
                best_code = code
                best_score = score
                best_meta = (pred_launch, launch_err, miss_err, spin_pen, loft_d, lie_d)

    assert best_code is not None and best_meta is not None
    pred_launch, launch_err, miss_err, spin_pen, loft_d, lie_d = best_meta

    rationale = [
        f"Launch target window ≈ {tgt.launch_lo:.1f}–{tgt.launch_hi:.1f}° (mid {target_launch:.1f}°) based on speed {club_speed_mph:.1f} mph.",
        f"Chosen setting predicts launch ≈ {pred_launch:.1f}° (error {launch_err:.2f}°).",
    ]

    if miss in {"LEFT", "RIGHT"}:
        if handedness == "RH":
            rationale.append(f"Miss input: {miss}. Lie {lie_d:+.2f}° biases start {'LEFT' if lie_d>0 else 'RIGHT'} for RH.")
        else:
            rationale.append(f"Miss input: {miss}. Lie {lie_d:+.2f}° biases start {'RIGHT' if lie_d>0 else 'LEFT'} for LH.")
    else:
        rationale.append("Miss input: STRAIGHT. Preference given to more neutral lie unless launch required otherwise.")

    if spin_high:
        rationale.append(f"Spin safety active: spin {spin_rpm:.0f} rpm is above upper band {tgt.spin_hi:.0f}; penalized adding loft.")
    else:
        rationale.append("Spin safety: no penalty (spin not above upper band).")

    return SureFitRecommendation(
        recommended=best_code,
        current=(current_setting.strip().upper() if current_setting else None),
        loft_delta=float(loft_d),
        lie_delta=float(lie_d),
        score=float(best_score),
        rationale=rationale,
    )


# =============================================================================
# Recommendations (MVP level)
# =============================================================================

def recommendations(bucket: str, merged_stats: Dict[str, float]) -> List[str]:
    recs: List[str] = []

    cs = merged_stats.get("club_speed_mph", np.nan)
    smash = merged_stats.get("smash_factor", np.nan)
    spin = merged_stats.get("spin_rpm", np.nan)
    launch = merged_stats.get("launch_deg", np.nan)

    if bucket == "DR":
        th = _driver_smash_threshold(cs)
        if not np.isnan(smash) and smash < th:
            recs.append("Priority: improve strike efficiency (center contact). Check tee height, ball position, and impact location.")
            recs.append("Use impact tape/foot spray to confirm strike pattern before buying a new head/shaft.")

        if not np.isnan(spin) and not np.isnan(cs) and cs >= 95 and spin > 3200:
            recs.append("Spin is on the high side: avoid adding loft if you’re already ballooning. Consider a lower-spin head/ball or reducing delivered loft.")

        if not np.isnan(launch) and not np.isnan(cs) and cs >= 100 and launch < 10.5:
            recs.append("Launch looks low for your speed: a small loft increase can help if spin remains in check.")

    elif bucket == "3W":
        recs.append("Fairway fitting: prioritize launch + landing angle for playability off turf. If launch is low, consider more loft before chasing low-spin.")

    elif bucket == "HY":
        recs.append("Hybrid fitting: prioritize consistent start line and height. Lie tweaks can be very effective for left/right misses.")

    if not recs:
        recs.append("Baseline looks solid for MVP spec-ranges. Next step: validate with strike pattern and target-based testing.")

    return recs


# =============================================================================
# End-to-end analysis helper
# =============================================================================

@dataclass
class BucketAnalysis:
    bucket: str
    df: pd.DataFrame
    summary: Dict[str, float]
    variability: Dict[str, float]
    limiting: List[Dict[str, str]]
    recs: List[str]
    filter_report: FilterReport


def analyze(df: pd.DataFrame) -> Dict[str, BucketAnalysis]:
    """
    Full pipeline:
    - add bucket labels
    - filter outliers per bucket
    - compute summary/variability/limiting/recommendations
    """
    df2 = add_bucket_column(df)
    results: Dict[str, BucketAnalysis] = {}

    for bucket in ["DR", "3W", "HY"]:
        d0 = df2[df2["bucket"] == bucket].copy()
        if len(d0) == 0:
            continue

        d, rep = filter_outliers(df2, bucket=bucket)
        s = summarize_bucket(d)
        v = variability_bucket(d)
        merged = {**s, **v}

        lim = limiting_factors(bucket, merged)
        recs = recommendations(bucket, merged)

        results[bucket] = BucketAnalysis(
            bucket=bucket,
            df=d,
            summary=s,
            variability=v,
            limiting=lim,
            recs=recs,
            filter_report=rep,
        )

    return results
