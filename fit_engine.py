# fit_engine.py — FitCaddie (GSPro Club Fitting Engine)
# Drop-in replacement file. Delete your current fit_engine.py and paste this whole file.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Data models
# ============================================================

@dataclass
class ClubSummary:
    club: str                 # "DR", "3W", "HY"
    n_total: int
    n_used: int
    confidence: str           # "LOW" | "MED" | "HIGH"
    metrics: Dict[str, Any]
    variability: Dict[str, Any]


@dataclass
class Recommendation:
    priority: int
    title: str
    rationale: str
    confidence: str           # "LOW" | "MED" | "HIGH"
    spec: Dict[str, Any]


@dataclass
class ClubAnalysis:
    summary: ClubSummary
    limiting_factors: List[str]
    recommendations: List[Recommendation]


@dataclass
class SessionResult:
    club_results: Dict[str, ClubAnalysis]


# ============================================================
# Titleist compatibility notes (for UI later)
# ============================================================

TITLEIST_COMPATIBILITY = {
    "driver_shafts_interchangeable": ["GT", "TSR", "TSi", "TS", "917", "915", "913", "910"],
    "fairway_shafts_interchangeable": ["GT", "TSR", "TSi", "TS", "917", "915", "913"],
    "surefit_driver_fairway_step_deg": 0.75,
    "surefit_hybrid_step_deg": 1.0,
}


# ============================================================
# Titleist SureFit tables (from your charts)
# Values = (loft_change_deg, lie_change_deg) where upright is +
# ============================================================

# DRIVER / FAIRWAY (RIGHT HAND)
SUREFIT_DRIVER_FW_RH: Dict[str, Tuple[float, float]] = {
    "A1": (0.0, 0.0),
    "A2": (0.0, 1.5),
    "A3": (1.5, 1.5),
    "A4": (1.5, 0.0),

    "B1": (0.0, -0.75),
    "B2": (0.0, 0.75),
    "B3": (1.5, 0.75),
    "B4": (1.5, -0.75),

    "C1": (-0.75, -0.75),
    "C2": (-0.75, 0.75),
    "C3": (0.75, 0.75),
    "C4": (0.75, -0.75),

    "D1": (-0.75, 0.0),
    "D2": (-0.75, 1.5),
    "D3": (0.75, 1.5),
    "D4": (0.75, 0.0),
}

# DRIVER / FAIRWAY (LEFT HAND)
SUREFIT_DRIVER_FW_LH: Dict[str, Tuple[float, float]] = {
    "A1": (0.75, 0.0),
    "A2": (0.75, 1.5),
    "A3": (-0.75, 1.5),
    "A4": (-0.75, 0.0),

    "B1": (0.75, -0.75),
    "B2": (0.75, 0.75),
    "B3": (-0.75, 0.75),
    "B4": (-0.75, -0.75),

    "C1": (1.5, -0.75),
    "C2": (1.5, 0.75),
    "C3": (0.0, 0.75),
    "C4": (0.0, -0.75),

    "D1": (1.5, 0.0),
    "D2": (1.5, 1.5),
    "D3": (0.0, 1.5),
    "D4": (0.0, 0.0),
}

# HYBRID (RIGHT HAND) — 1° increments
SUREFIT_HYBRID_RH: Dict[str, Tuple[float, float]] = {
    "A1": (0.0, 0.0),
    "A2": (0.0, 2.0),
    "A3": (2.0, 2.0),
    "A4": (2.0, 0.0),

    "B1": (0.0, -1.0),
    "B2": (0.0, 1.0),
    "B3": (2.0, 1.0),
    "B4": (2.0, -1.0),

    "C1": (-1.0, -1.0),
    "C2": (-1.0, 1.0),
    "C3": (1.0, 1.0),
    "C4": (1.0, -1.0),

    "D1": (-1.0, 0.0),
    "D2": (-1.0, 2.0),
    "D3": (1.0, 2.0),
    "D4": (1.0, 0.0),
}

# HYBRID (LEFT HAND) — 1° increments
SUREFIT_HYBRID_LH: Dict[str, Tuple[float, float]] = {
    "A1": (1.0, 0.0),
    "A2": (1.0, 2.0),
    "A3": (-1.0, 2.0),
    "A4": (-1.0, 0.0),

    "B1": (1.0, -1.0),
    "B2": (1.0, 1.0),
    "B3": (-1.0, 1.0),
    "B4": (-1.0, -1.0),

    "C1": (2.0, -1.0),
    "C2": (2.0, 1.0),
    "C3": (0.0, 1.0),
    "C4": (0.0, -1.0),

    "D1": (2.0, 0.0),
    "D2": (2.0, 2.0),
    "D3": (0.0, 2.0),
    "D4": (0.0, 0.0),
}


# ============================================================
# Parsing helpers
# ============================================================

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_numeric_lr(series: pd.Series) -> pd.Series:
    """
    Convert strings like:
      '19.8 R' -> 19.8
      '14.0 L' -> -14.0
      '9.4°'   -> 9.4
      '4.0° L' -> -4.0
      '0.8° I-O' -> 0.8  (we ignore I-O/O-I direction tags)
    """
    s = series.astype(str).str.strip()

    # remove known non-numeric tokens
    s = s.str.replace("°", "", regex=False)
    s = s.str.replace("yds", "", regex=False).str.replace("yd", "", regex=False)
    s = s.str.replace(",", "", regex=False)

    # identify Left marker at end or standalone
    is_left = s.str.contains(r"(^L\b|\bL$|\bL\b)", regex=True)

    # strip common direction tokens
    s = s.str.replace(r"\b[LR]\b", "", regex=True)
    s = s.str.replace("L", "", regex=False)
    s = s.str.replace("R", "", regex=False)

    # strip face/path direction labels like I-O, O-I
    s = s.str.replace("I-O", "", regex=False)
    s = s.str.replace("O-I", "", regex=False)

    s = s.str.strip()

    num = pd.to_numeric(s, errors="coerce")
    num = np.where(is_left, -num, num)
    return pd.Series(num, index=series.index)


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    med = np.median(x)
    return np.median(np.abs(x - med))


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = _mad(x)
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def _confidence_label(n_used: int) -> str:
    if n_used >= 10:
        return "HIGH"
    if n_used >= 6:
        return "MED"
    return "LOW"


# ============================================================
# Column canonicalization (handles GSPro variations)
# ============================================================

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize headers aggressively
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_")
        .str.lower()
    )

    # map common GSPro names -> internal names
    rename_map = {
        # club id
        "club_name": "club",
        "club": "club",

        # speeds
        "club_speed_mph": "club_speed",
        "club_speed": "club_speed",
        "ball_speed_mph": "ball_speed",
        "ball_speed": "ball_speed",

        # distances
        "carry_dist_yd": "carry",
        "carry": "carry",
        "total_dist_yd": "total",
        "total": "total",
        "totaldistance": "total",
        "total_dist": "total",

        # dispersion
        "offline_yd": "offline",
        "offline": "offline",

        # launch/angles
        "hla": "hla",
        "vla": "vla",
        "desc_angle": "descent",
        "descent_angle": "descent",
        "descent": "descent",

        # height
        "peak_height_yd": "peak_height",
        "peakheight": "peak_height",
        "peak_height": "peak_height",

        # spin
        "back_spin": "spin",
        "backspin": "spin",
        "spin": "spin",
        "spin_axis": "spin_axis",
        "rawspinaxis": "spin_axis",
        "side_spin": "side_spin",
        "sidespin": "side_spin",

        # swing delivery
        "club_aoa": "aoa",
        "aoa": "aoa",
        "club_path": "path",
        "path": "path",
        "face_to_path": "face_to_path",
        "facetopath": "face_to_path",
        "face_to_target": "face_to_target",
        "facetotarget": "face_to_target",

        # optional
        "distancetop in": "distance_to_pin",
        "distance_to_pin": "distance_to_pin",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # ensure club column exists even if weird casing
    if "club" not in df.columns:
        # attempt: some exports keep "club_name" but got missed
        for c in df.columns:
            if c.replace("_", "") in ("clubname",):
                df = df.rename(columns={c: "club"})
                break

    return df


# ============================================================
# Club bucket mapping
# ============================================================

def _club_bucket(club_raw: str) -> str:
    s = str(club_raw).upper().strip()

    # GSPro often uses "DR", "H3", "HY", "3W" etc.
    if s in {"DR", "DRIVER"}:
        return "DR"
    if s in {"3W", "FW", "FAIRWAY", "W3"}:
        return "3W"
    # hybrids
    if s.startswith("H") and len(s) >= 2 and s[1].isdigit():
        return "HY"
    if s in {"HY", "HYBRID"}:
        return "HY"

    return s


# ============================================================
# Outlier handling
# ============================================================

def flag_outliers_per_club(df_club: pd.DataFrame, bucket: str) -> pd.Series:
    """
    Robust outlier detection using carry + ball_speed.
    Returns boolean series True for outliers.
    """
    n = len(df_club)
    if n < 6:
        return pd.Series([False] * n, index=df_club.index)

    carry = df_club["carry"].to_numpy(dtype=float) if "carry" in df_club.columns else np.full(n, np.nan)
    bs = df_club["ball_speed"].to_numpy(dtype=float) if "ball_speed" in df_club.columns else np.full(n, np.nan)

    z_c = _robust_z(np.nan_to_num(carry, nan=np.nanmedian(carry)))
    z_b = _robust_z(np.nan_to_num(bs, nan=np.nanmedian(bs)))

    # slightly looser threshold for small samples
    thresh = 3.5
    out = (np.abs(z_c) > thresh) | (np.abs(z_b) > thresh)

    return pd.Series(out, index=df_club.index)


# ============================================================
# Summaries + recommendations
# ============================================================

def _safe_smash(df_used: pd.DataFrame) -> float:
    if "ball_speed" in df_used.columns and "club_speed" in df_used.columns:
        bs = df_used["ball_speed"].mean(skipna=True)
        cs = df_used["club_speed"].mean(skipna=True)
        if pd.notna(bs) and pd.notna(cs) and cs != 0:
            return float(bs / cs)
    return float("nan")


def summarize_club(df_used: pd.DataFrame, bucket: str, n_total: int) -> ClubSummary:
    def mean(col: str) -> float:
        return float(df_used[col].mean(skipna=True)) if col in df_used.columns else float("nan")

    def std(col: str) -> float:
        return float(df_used[col].std(skipna=True)) if col in df_used.columns else float("nan")

    metrics: Dict[str, Any] = {
        "carry": mean("carry"),
        "total": mean("total"),
        "offline_mean": mean("offline"),
        "offline_abs_mean": float(df_used["offline"].abs().mean(skipna=True)) if "offline" in df_used.columns else float("nan"),
        "ball_speed": mean("ball_speed"),
        "club_speed": mean("club_speed"),
        "smash": mean("smash") if "smash" in df_used.columns else _safe_smash(df_used),
        "launch_vla": mean("vla"),
        # convenience alias (some code expects "launch")
        "launch": mean("vla"),
        "spin": mean("spin"),
        "spin_axis": mean("spin_axis"),
        "aoa": mean("aoa"),
        "path": mean("path"),
        "face_to_path": mean("face_to_path"),
        "face_to_target": mean("face_to_target"),
        "descent": mean("descent"),
        "peak_height": mean("peak_height"),
    }

    variability: Dict[str, Any] = {
        "offline_std": std("offline"),
        "launch_std": std("vla"),
        "spin_std": std("spin"),
        "face_to_path_std": std("face_to_path"),
        "path_std": std("path"),
        "ball_speed_std": std("ball_speed"),
    }

    n_used = int(len(df_used))
    return ClubSummary(
        club=bucket,
        n_total=int(n_total),
        n_used=n_used,
        confidence=_confidence_label(n_used),
        metrics=metrics,
        variability=variability,
    )


def limiting_factors(summary: ClubSummary) -> List[str]:
    m = summary.metrics
    v = summary.variability
    factors: List[str] = []

    offline_std = v.get("offline_std", float("nan"))
    smash = m.get("smash", float("nan"))
    launch = m.get("launch", m.get("launch_vla", float("nan")))
    spin = m.get("spin", float("nan"))

    # dispersion
    if np.isfinite(offline_std) and offline_std >= 20:
        factors.append("Dispersion is wide (offline variability high).")

    # smash
    if np.isfinite(smash) and smash < 1.44:
        factors.append("Smash factor is low (strike efficiency below potential).")

    # launch/spin (driver emphasis)
    if summary.club == "DR":
        if np.isfinite(launch) and launch < 11:
            factors.append("Launch is low for this swing speed.")
        if np.isfinite(spin) and spin > 3000:
            factors.append("Spin is high (can reduce distance / add curvature).")

    # hybrid spin can be higher naturally, but flag extreme
    if summary.club == "HY":
        if np.isfinite(spin) and spin > 4800:
            factors.append("Hybrid spin is very high (may cost distance / balloon).")

    return factors


def _driver_launch_window_by_speed(club_speed: float) -> Tuple[float, float]:
    # Your spec ranges:
    # Fast (>105): 10–14
    # Average (90–105): 12–15
    # Slow (<90): 14–19
    if not np.isfinite(club_speed):
        return 12.0, 15.0
    if club_speed < 90:
        return 14.0, 19.0
    if 90 <= club_speed <= 105:
        return 12.0, 15.0
    return 10.0, 14.0


def _fairway_launch_window_by_speed(club_speed: float) -> Tuple[float, float]:
    # Simple MVP windows (tunable later)
    if not np.isfinite(club_speed):
        return 11.0, 15.0
    if club_speed < 85:
        return 13.0, 17.0
    if 85 <= club_speed <= 100:
        return 11.5, 15.5
    return 10.5, 14.5


def _hybrid_launch_window_by_speed(club_speed: float) -> Tuple[float, float]:
    # Simple MVP windows
    if not np.isfinite(club_speed):
        return 14.0, 18.0
    if club_speed < 80:
        return 16.0, 20.0
    if 80 <= club_speed <= 95:
        return 14.5, 18.5
    return 13.5, 17.5


def recommend_for_club(summary: ClubSummary) -> List[Recommendation]:
    m = summary.metrics
    v = summary.variability
    recs: List[Recommendation] = []

    club_speed = m.get("club_speed", float("nan"))
    launch = m.get("launch", m.get("launch_vla", float("nan")))
    spin = m.get("spin", float("nan"))
    smash = m.get("smash", float("nan"))
    offline_std = v.get("offline_std", float("nan"))

    # Smash flag (you asked to flag smash)
    if np.isfinite(smash) and smash < 1.44:
        recs.append(
            Recommendation(
                priority=1,
                title="Improve strike efficiency (Smash)",
                rationale="Your smash factor is below typical potential for this speed. Before buying equipment, verify centered contact and consistent strike.",
                confidence=summary.confidence,
                spec={"target_smash": "≥1.46 (driver) if possible", "tip": "Check impact location; higher/center strikes often improve launch + reduce spin."},
            )
        )

    if summary.club == "DR":
        low, high = _driver_launch_window_by_speed(club_speed)
        if np.isfinite(launch) and launch < (low - 1.0):
            recs.append(
                Recommendation(
                    priority=2,
                    title="Test higher loft via hosel (+0.75° to +1.5°)",
                    rationale=f"Your launch ({launch:.1f}°) is below your target window ({low:.0f}–{high:.0f}°) for ~{club_speed:.0f} mph club speed.",
                    confidence=summary.confidence,
                    spec={"adapter_loft_change_deg": "+0.75 to +1.5", "goal": {"launch_deg": f"{low:.0f}–{high:.0f}", "spin_rpm": "avoid ballooning"}},
                )
            )

        if np.isfinite(spin) and spin > 3000:
            recs.append(
                Recommendation(
                    priority=3,
                    title="Bias toward a lower-spin build",
                    rationale="Driver spin is above a typical window; lowering spin can improve carry/roll and reduce curvature.",
                    confidence=summary.confidence,
                    spec={"shaft_profile": "low–mid launch / low spin, tip-stiff, lower torque", "goal": {"spin_rpm": "-200 to -500"}},
                )
            )

        if np.isfinite(offline_std) and offline_std >= 20:
            recs.append(
                Recommendation(
                    priority=4,
                    title="Tighten dispersion before chasing distance",
                    rationale="Offline variability is high. Hosel/face-angle tweaks and strike/face control tend to deliver the biggest gains first.",
                    confidence=summary.confidence,
                    spec={"goal": {"offline_std_yd": "< 18"}, "notes": ["Try your recommended hosel move, then retest 10–15 shots."]},
                )
            )

    if summary.club == "3W":
        low, high = _fairway_launch_window_by_speed(club_speed)
        if np.isfinite(launch) and launch < (low - 1.0):
            recs.append(
                Recommendation(
                    priority=2,
                    title="Test higher loft via hosel (+0.75°)",
                    rationale=f"Fairway launch ({launch:.1f}°) is low vs window ({low:.1f}–{high:.1f}°).",
                    confidence=summary.confidence,
                    spec={"adapter_loft_change_deg": "+0.75", "goal": {"launch_deg": f"{low:.1f}–{high:.1f}"}},
                )
            )
        if np.isfinite(spin) and spin < 2500:
            recs.append(
                Recommendation(
                    priority=3,
                    title="Increase spin/launch slightly (fairway hold)",
                    rationale="Fairway spin looks quite low; you may struggle to hold greens.",
                    confidence=summary.confidence,
                    spec={"goal": {"spin_rpm": "+300 to +700"}, "notes": ["Try more loft or higher-launch shaft profile."]},
                )
            )

    if summary.club == "HY":
        low, high = _hybrid_launch_window_by_speed(club_speed)
        if np.isfinite(launch) and launch < (low - 1.0):
            recs.append(
                Recommendation(
                    priority=2,
                    title="Test higher loft via hosel (+1°)",
                    rationale=f"Hybrid launch ({launch:.1f}°) is low vs window ({low:.1f}–{high:.1f}°).",
                    confidence=summary.confidence,
                    spec={"adapter_loft_change_deg": "+1", "goal": {"launch_deg": f"{low:.1f}–{high:.1f}"}},
                )
            )
        if np.isfinite(spin) and spin > 4800:
            recs.append(
                Recommendation(
                    priority=3,
                    title="Reduce hybrid spin slightly",
                    rationale="Hybrid spin is very high; a slightly lower-spin shaft profile or more neutral setting can help.",
                    confidence=summary.confidence,
                    spec={"goal": {"spin_rpm": "-200 to -600"}},
                )
            )

    # Keep list stable/sorted
    recs = sorted(recs, key=lambda r: r.priority)
    return recs


# ============================================================
# Hosel recommendation engine (Titleist SureFit)
# Priority order you requested:
# 1) Launch window (speed adjusted)
# 2) Miss tendency (upright / flatter)  [user input required]
# 3) Spin safety check
# Then pick closest setting
# ============================================================

def _get_surefit_table(club_type: str, handedness: str) -> Dict[str, Tuple[float, float]]:
    club_type = (club_type or "").upper().strip()
    handedness = (handedness or "RH").upper().strip()

    if club_type in {"DR", "3W"}:
        return SUREFIT_DRIVER_FW_LH if handedness == "LH" else SUREFIT_DRIVER_FW_RH

    if club_type == "HY":
        return SUREFIT_HYBRID_LH if handedness == "LH" else SUREFIT_HYBRID_RH

    # default fallback
    return SUREFIT_DRIVER_FW_LH if handedness == "LH" else SUREFIT_DRIVER_FW_RH


def _launch_window(club_type: str, club_speed: float) -> Tuple[float, float]:
    if club_type == "DR":
        return _driver_launch_window_by_speed(club_speed)
    if club_type == "3W":
        return _fairway_launch_window_by_speed(club_speed)
    if club_type == "HY":
        return _hybrid_launch_window_by_speed(club_speed)
    return 12.0, 15.0


def recommend_titleist_surefit(
    summary: ClubSummary,
    club_type: str,
    handedness: str,
    current_setting: str,
    miss_tendency: str,   # "RIGHT" / "LEFT" / "BOTH" / "NOT SURE"
) -> Dict[str, Any]:
    club_type = (club_type or summary.club or "").upper().strip()
    handedness = (handedness or "RH").upper().strip()
    current_setting = (current_setting or "").upper().strip()
    miss_tendency = (miss_tendency or "NOT SURE").upper().strip()

    table = _get_surefit_table(club_type, handedness)

    if current_setting not in table:
        return {
            "action": "unknown_current_setting",
            "from": current_setting,
            "to": None,
            "why": "Current SureFit setting not recognized.",
            "expected": {},
        }

    # --- read metrics robustly ---
    club_speed = float(summary.metrics.get("club_speed", np.nan))
    launch = summary.metrics.get("launch", summary.metrics.get("launch_vla", np.nan))
    launch = float(launch) if launch is not None else np.nan
    spin = float(summary.metrics.get("spin", np.nan))

    # --- 1) Launch window (speed adjusted) ---
    target_low, target_high = _launch_window(club_type, club_speed)

    loft_need = 0.0
    if np.isfinite(launch):
        if launch < (target_low - 1.0):
            loft_need = +0.75 if club_type in {"DR", "3W"} else +1.0
            if launch < (target_low - 2.0):
                loft_need = +1.5 if club_type in {"DR", "3W"} else +2.0
        elif launch > (target_high + 1.0):
            loft_need = -0.75 if club_type in {"DR", "3W"} else -1.0

    # --- 2) Miss tendency (upright / flatter) ---
    # Use the player's words as primary (your choice B).
    # RIGHT miss -> more upright (draw bias)
    # LEFT miss  -> flatter (fade bias)
    lie_need = 0.0
    if miss_tendency == "RIGHT":
        lie_need = +1.5 if club_type in {"DR", "3W"} else +2.0
    elif miss_tendency == "LEFT":
        lie_need = -0.75 if club_type in {"DR", "3W"} else -1.0

    # --- 3) Spin safety check (soft guardrails) ---
    # If spin is already high, don't pile on too much loft.
    if np.isfinite(spin) and loft_need > 0:
        if club_type == "DR" and spin > 3100:
            loft_need = +0.75 if loft_need > +0.75 else loft_need
        if club_type == "3W" and spin > 4800:
            loft_need = +0.75 if loft_need > +0.75 else loft_need
        if club_type == "HY" and spin > 5200:
            loft_need = +1.0 if loft_need > +1.0 else loft_need

    # --- Build target deltas relative to current ---
    curr_loft, curr_lie = table[current_setting]
    target_loft = curr_loft + loft_need
    target_lie = curr_lie + lie_need

    # clamp to available ranges for each table type
    if club_type in {"DR", "3W"}:
        target_loft = float(np.clip(target_loft, -0.75, +1.5))
        target_lie = float(np.clip(target_lie, -0.75, +1.5))
    else:
        target_loft = float(np.clip(target_loft, -1.0, +2.0))
        target_lie = float(np.clip(target_lie, -1.0, +2.0))

    # --- Choose closest setting (least squares distance) ---
    best = current_setting
    best_dist = 1e9
    for setting, (ld, lied) in table.items():
        dist = (ld - target_loft) ** 2 + (lied - target_lie) ** 2
        if dist < best_dist:
            best_dist = dist
            best = setting

    # if no change
    if best == current_setting:
        return {
            "action": "no_change",
            "from": current_setting,
            "to": best,
            "why": "Your current setting is already the closest match to your launch window + miss tendency.",
            "expected": {"launch_window": f"{target_low:.1f}–{target_high:.1f}°"},
        }

    # explanation
    best_loft, best_lie = table[best]
    d_loft = best_loft - curr_loft
    d_lie = best_lie - curr_lie

    why_parts: List[str] = []
    if np.isfinite(launch):
        why_parts.append(f"Launch {launch:.1f}° vs target {target_low:.1f}–{target_high:.1f}°")
    if loft_need > 0:
        why_parts.append("Add loft to raise launch")
    if loft_need < 0:
        why_parts.append("Reduce loft to lower launch")
    if miss_tendency == "RIGHT":
        why_parts.append("Miss right → more upright (draw bias)")
    if miss_tendency == "LEFT":
        why_parts.append("Miss left → flatter (fade bias)")

    expected: Dict[str, Any] = {}
    if d_loft != 0:
        expected["loft_change_deg"] = f"{d_loft:+.2f}"
        expected["launch_expectation"] = "+1–2°" if d_loft > 0 else "-1–2°"
    if d_lie != 0:
        expected["lie_change_deg"] = f"{d_lie:+.2f}"
        expected["direction_bias"] = "More draw bias" if d_lie > 0 else "More fade bias"
    if np.isfinite(spin) and d_loft > 0:
        if club_type == "DR" and spin > 3100:
            expected["spin_watchout"] = "Spin already high—if spin climbs, keep loft and move to a lower-spin shaft profile instead."
        if club_type == "3W" and spin > 4800:
            expected["spin_watchout"] = "Spin already high—if spin climbs, keep loft and test a lower-spin shaft profile."
        if club_type == "HY" and spin > 5200:
            expected["spin_watchout"] = "Spin already high—if spin climbs, avoid more loft and test a lower-spin shaft profile."

    return {
        "action": "change_setting",
        "from": current_setting,
        "to": best,
        "why": "; ".join(why_parts) if why_parts else "Tune loft/lie toward your target window.",
        "expected": expected,
    }


# ============================================================
# Main analyzer
# ============================================================

def analyze_dataframe(df_raw: pd.DataFrame) -> SessionResult:
    df = _canonicalize_columns(df_raw.copy())

    # required
    if "club" not in df.columns:
        raise ValueError("CSV missing club column (could not find Club/Club Name).")

    # normalize club strings
    df["club"] = df["club"].astype(str).str.upper().str.strip()

    # Clean DistanceToPin like "181.28 yds" -> 181.28 (optional)
    if "distance_to_pin" in df.columns:
        df["distance_to_pin"] = (
            df["distance_to_pin"].astype(str)
            .str.replace("yds", "", regex=False)
            .str.replace("yd", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["distance_to_pin"] = pd.to_numeric(df["distance_to_pin"], errors="coerce")

    # convert numeric columns (strings->numbers)
    numeric_cols = [
        "carry", "total", "offline",
        "ball_speed", "club_speed", "smash",
        "vla", "hla", "peak_height", "descent",
        "spin", "spin_axis", "side_spin",
        "aoa", "path", "face_to_path", "face_to_target",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("yds", "", regex=False)
                .str.replace("yd", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fields that often have L/R markers
    for col in ["offline", "hla", "spin_axis", "face_to_target", "vla", "aoa", "path", "face_to_path"]:
        if col in df.columns:
            df[col] = _to_numeric_lr(df[col])

    club_results: Dict[str, ClubAnalysis] = {}

    for club_raw, df_club in df.groupby("club", dropna=False):
        bucket = _club_bucket(club_raw)
        if bucket not in {"DR", "3W", "HY"}:
            continue

        n_total = len(df_club)
        outliers = flag_outliers_per_club(df_club, bucket)
        df_used = df_club.loc[~outliers].copy()

        # ensure smash exists
        if "smash" not in df_used.columns and "ball_speed" in df_used.columns and "club_speed" in df_used.columns:
            df_used["smash"] = df_used["ball_speed"] / df_used["club_speed"]

        # repeat numeric conversion at club level (robust against dtype changes)
        for col in numeric_cols:
            if col in df_used.columns:
                df_used[col] = pd.to_numeric(df_used[col], errors="coerce")

        for col in ["offline", "hla", "spin_axis", "face_to_target", "vla", "aoa", "path", "face_to_path"]:
            if col in df_used.columns:
                df_used[col] = _to_numeric_lr(df_used[col])

        summary = summarize_club(df_used, bucket, n_total=n_total)
        factors = limiting_factors(summary)
        recs = recommend_for_club(summary)

        club_results[bucket] = ClubAnalysis(
            summary=summary,
            limiting_factors=factors,
            recommendations=recs,
        )

    return SessionResult(club_results=club_results)


# ============================================================
# Streamlit-friendly export
# ============================================================

def session_to_dict(result: SessionResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {"clubs": {}}
    for club, analysis in result.club_results.items():
        out["clubs"][club] = {
            "summary": {
                "club": analysis.summary.club,
                "n_total": analysis.summary.n_total,
                "n_used": analysis.summary.n_used,
                "confidence": analysis.summary.confidence,
                "metrics": analysis.summary.metrics,
                "variability": analysis.summary.variability,
            },
            "limiting_factors": analysis.limiting_factors,
            "recommendations": [
                {
                    "priority": r.priority,
                    "title": r.title,
                    "rationale": r.rationale,
                    "confidence": r.confidence,
                    "spec": r.spec,
                }
                for r in analysis.recommendations
            ],
            "titleist_compatibility": TITLEIST_COMPATIBILITY,
        }
    return out
