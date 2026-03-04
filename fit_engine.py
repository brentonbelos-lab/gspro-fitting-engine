# fit_engine.py
# Rules-based fitting recommendation engine for GSPro CSV exports.
# Spec-range recommendations (no product-specific calls).

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import re
import pandas as pd


CANON_COLS = {
    # distances
    "carry": ["Carry", "Carry Dist (yd)", "Carry Dist", "CarryDist", "Carry Distance", "Carry Distance (yd)"],
    "total": ["TotalDistance", "Total Dist (yd)", "Total Dist", "TotalDist", "Total", "Total Distance (yd)"],
    "offline": ["Offline", "Offline (yd)", "offline", "LeftRight", "Dispersion"],

    # speed
    "ball_speed": ["BallSpeed", "Ball Speed (mph)", "Ball Speed", "ball_speed"],
    "club_speed": ["ClubSpeed", "Club Speed (mph)", "Club Speed", "club_speed"],
    "smash": ["SmashFactor", "Smash Factor", "Smash", "smash"],

    # launch
    "vla": ["VLA", "Vla", "LaunchAngle", "VerticalLaunch", "launch"],
    "hla": ["HLA", "Hla", "HorizontalLaunch", "StartLine"],

    # height / landing
    "peak_height": ["PeakHeight", "Peak Height (yd)", "Peak Height", "Apex"],
    "descent": ["Decent", "Desc Angle", "Descent", "DescentAngle", "LandingAngle", "DescAngle"],

    # spin
    "spin": ["BackSpin", "Back Spin", "Spin", "TotalSpin"],
    "spin_axis": ["rawSpinAxis", "Spin Axis", "SpinAxis", "Axis"],
    "side_spin": ["SideSpin", "Side Spin"],

    # delivery
    "aoa": ["AoA", "Club AoA", "AngleOfAttack"],
    "path": ["Path", "Club Path"],
    "face_to_path": ["FaceToPath", "Face to Path", "F2P"],
    "face_to_target": ["FaceToTarget", "Face to Target", "FaceAngle"],

    # identifiers
    "club": ["Club", "Club Name", "ClubName", "club"],
}

TARGETS = {
    "DR": {"launch_deg": (12.0, 15.0), "spin_rpm": (2000.0, 2800.0), "smash": (1.45, 1.50)},
    "3W": {"launch_deg": (11.0, 14.0), "spin_rpm": (2800.0, 3800.0), "descent_deg": (38.0, 45.0)},
    "HY": {"launch_deg": (14.0, 18.0), "spin_rpm": (3500.0, 5000.0), "descent_deg": (40.0, 50.0)},
}

SHAFT_WEIGHT_RULES = [
    (0, 92, (50, 60)),
    (92, 102, (55, 65)),
    (102, 110, (60, 70)),
    (110, 999, (65, 75)),
]


@dataclass
class ClubSummary:
    club: str
    n_total: int
    n_used: int
    confidence: str
    metrics: Dict[str, float]
    variability: Dict[str, float]


@dataclass
class Recommendation:
    club: str
    priority: int
    title: str
    rationale: str
    confidence: str
    spec: Dict[str, object]


@dataclass
class ClubAnalysis:
    summary: ClubSummary
    limiting_factors: List[str]
    recommendations: List[Recommendation]


@dataclass
class SessionResult:
    club_results: Dict[str, ClubAnalysis]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from original headers first
    df.columns = [c.strip() for c in df.columns]

    rename = {}
    for canon, candidates in CANON_COLS.items():
        col = _find_col(df, candidates)
        if col is not None:
            rename[col] = canon

    df = df.rename(columns=rename)

    # Final strip (just in case)
    df.columns = [c.strip() for c in df.columns]
    return df


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_numeric_lr(series: pd.Series) -> pd.Series:
    """
    Robust parse for GSPro fields that may include directions/units.
    Examples:
      '11.4 R'     ->  11.4
      '14.0 L'     -> -14.0
      '2.6° R'     ->  2.6
      '0.8° I-O'   ->  0.8
      '4.1° U'     ->  4.1
      '40.1°'      -> 40.1
    """
    s = series.astype(str).str.strip()

    # Determine sign: treat explicit ' L' or 'L ' or trailing 'L' as left (negative)
    is_left = s.str.contains(r"(^L\b|\bL$|\bL\b)", regex=True)

    # Remove commas in thousands (just in case)
    s = s.str.replace(",", "", regex=False)

    # Extract the first numeric value in the string (handles degrees and extra tokens)
    num = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    num = pd.to_numeric(num, errors="coerce")

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


def _club_bucket(club: str) -> str:
    c = (club or "").upper().strip()
    if c == "DR":
        return "DR"
    if c in {"3W", "FW", "W3"}:
        return "3W"
    if c.startswith("H") or "HY" in c:
        return "HY"
    return c


def _safe_smash(df: pd.DataFrame) -> float:
    if "ball_speed" in df.columns and "club_speed" in df.columns:
        bs = df["ball_speed"].mean(skipna=True)
        cs = df["club_speed"].mean(skipna=True)
        if pd.notna(bs) and pd.notna(cs) and cs and cs > 0:
            return float(bs / cs)
    return float("nan")


def flag_outliers_per_club(df_club: pd.DataFrame, bucket: str) -> pd.Series:
    out = pd.Series(False, index=df_club.index)

    for col in ["carry", "ball_speed"]:
        if col in df_club.columns:
            z = _robust_z(df_club[col].to_numpy())
            out |= (np.abs(z) > 3.5)

    if bucket == "DR" and "smash" in df_club.columns:
        out |= (df_club["smash"] < 1.25)

    if "carry" in df_club.columns:
        med = df_club["carry"].median(skipna=True)
        mad = _mad(df_club["carry"].to_numpy())
        if np.isfinite(med) and np.isfinite(mad) and mad > 0:
            out |= (df_club["carry"] < (med - 2.5 * mad))

    if out.mean() > 0.35:
        out[:] = False

    return out


def summarize_club(df_used: pd.DataFrame, bucket: str, n_total: int) -> ClubSummary:

    # FINAL SAFETY: force numeric conversion for columns that may contain L/R or degree symbols
    for col in ["offline", "vla", "hla", "spin_axis", "face_to_target", "face_to_path"]:
        if col in df_used.columns:
            df_used[col] = _to_numeric_lr(df_used[col])

    def mean(col: str) -> float:
    if col not in df_used.columns:
        return float("nan")
    s = pd.to_numeric(df_used[col], errors="coerce")
    return float(s.mean(skipna=True))

    def std(col: str) -> float:
    if col not in df_used.columns:
        return float("nan")
    s = pd.to_numeric(df_used[col], errors="coerce")
    return float(s.std(skipna=True))

    metrics = {
        "carry": mean("carry"),
        "total": mean("total"),

        # offline
        "offline_mean": mean("offline"),
        "offline_abs_mean": float(pd.to_numeric(df_used["offline"], errors="coerce").abs().mean(skipna=True))
            if "offline" in df_used.columns else float("nan"),

        "ball_speed": mean("ball_speed"),
        "club_speed": mean("club_speed"),
        "smash": mean("smash") if "smash" in df_used.columns else _safe_smash(df_used),

        # launch (store both names so UI always finds it)
        "launch_vla": mean("vla"),
        "launch": mean("vla"),

        "spin": mean("spin"),
        "spin_axis": mean("spin_axis"),
        "aoa": mean("aoa"),
        "path": mean("path"),
        "face_to_path": mean("face_to_path"),
        "descent": mean("descent"),
        "peak_height": mean("peak_height"),
    }

    variability = {
        "offline_std": std("offline"),
        "launch_std": std("vla"),
        "spin_std": std("spin"),
        "face_to_path_std": std("face_to_path"),
        "path_std": std("path"),
        "ball_speed_std": std("ball_speed"),
        "smash_std": std("smash") if "smash" in df_used.columns else float("nan"),
    }

    n_used = len(df_used)
    return ClubSummary(
        club=bucket,
        n_total=int(n_total),
        n_used=int(n_used),
        confidence=_confidence_label(n_used),
        metrics=metrics,
        variability=variability,
    )


def limiting_factors(summary: ClubSummary) -> List[str]:
    c = summary.club
    m = summary.metrics
    v = summary.variability
    factors: List[str] = []

    if np.isfinite(v.get("offline_std", np.nan)) and v["offline_std"] > 18:
        factors.append("Dispersion is wide (offline variability high).")

    if np.isfinite(v.get("face_to_path_std", np.nan)) and v["face_to_path_std"] > 4:
        factors.append("Face-to-path is inconsistent (face control / timing).")

    t = TARGETS.get(c)
    if t:
        launch = m.get("launch_vla", np.nan)
        spin = m.get("spin", np.nan)
        if np.isfinite(launch):
            lo, hi = t.get("launch_deg", (None, None))
            if lo is not None and launch < lo:
                factors.append("Launch is low for this club.")
            if hi is not None and launch > hi:
                factors.append("Launch is high for this club.")
        if np.isfinite(spin):
            slo, shi = t.get("spin_rpm", (None, None))
            if slo is not None and spin < slo:
                factors.append("Spin is low (can reduce carry / stopping).")
            if shi is not None and spin > shi:
                factors.append("Spin is high (can reduce distance / add curvature).")

    smash = m.get("smash", np.nan)
    if c == "DR" and np.isfinite(smash) and smash < 1.43:
        factors.append("Strike efficiency is below potential (smash factor low).")

    if np.isfinite(v.get("spin_std", np.nan)) and v["spin_std"] > 450:
        factors.append("Spin is volatile (strike / dynamic loft variability).")

    off_mean = m.get("offline_mean", np.nan)
    if np.isfinite(off_mean) and abs(off_mean) > 12:
        factors.append("Directional bias: right of target." if off_mean > 0 else "Directional bias: left of target.")

    return factors


def _driver_weight_range(club_speed: float) -> Tuple[int, int]:
    if not np.isfinite(club_speed):
        return (60, 70)
    for lo, hi, wr in SHAFT_WEIGHT_RULES:
        if lo <= club_speed < hi:
            return wr
    return (60, 70)


def _driver_flex_range(club_speed: float) -> str:
    if not np.isfinite(club_speed):
        return "stiff (test x if aggressive transition)"
    if club_speed < 95:
        return "regular to stiff"
    if club_speed < 105:
        return "stiff"
    if club_speed < 112:
        return "stiff (test x if transition is aggressive)"
    return "x-stiff"


def recommend_for_club(summary: ClubSummary) -> List[Recommendation]:
    c = summary.club
    m = summary.metrics
    v = summary.variability
    conf = summary.confidence

    recs: List[Recommendation] = []
    priority = 1

    def add(title: str, rationale: str, spec: Dict[str, object], confidence: Optional[str] = None):
        nonlocal priority
        recs.append(
            Recommendation(
                club=c,
                priority=priority,
                title=title,
                rationale=rationale,
                confidence=confidence or conf,
                spec=spec,
            )
        )
        priority += 1

    if c == "DR":
        launch = m.get("launch_vla", np.nan)
        spin = m.get("spin", np.nan)
        cs = m.get("club_speed", np.nan)

        if np.isfinite(launch) and launch < 12:
            add(
                "Test a higher loft setting (+0.75° to +1.5°) first",
                "Launch is low; a small loft increase often raises launch without major swing changes. Confirm spin doesn’t balloon.",
                {"adapter_loft_change_deg": "+0.75 to +1.5", "goal": {"launch_deg": "+1–2", "spin_rpm": "stay in target"}},
                confidence="MED" if conf == "LOW" else conf,
            )

        if np.isfinite(spin) and spin > 2800:
            add(
                "Bias toward a lower-spin build",
                "Spin is above the typical driver window; lowering spin can improve carry/roll and reduce curvature.",
                {
                    "shaft_profile": "low–mid launch / low spin, tip-stiff, lower torque",
                    "head_category": "neutral low-spin OR forgiving low-spin (avoid ultra-low if launch is already low)",
                    "goal": {"spin_rpm": "-200 to -500"},
                },
            )

        weight_range = _driver_weight_range(cs)
        flex_range = _driver_flex_range(cs)

        notes = []
        if np.isfinite(v.get("face_to_path_std", np.nan)) and v["face_to_path_std"] > 4:
            notes.append("Prioritize lower torque and a stiffer tip to improve face control.")
        if np.isfinite(v.get("offline_std", np.nan)) and v["offline_std"] > 18:
            notes.append("Consider slightly heavier within range for timing (and/or shorten driver 0.5\").")

        add(
            "Shaft spec range to test (driver)",
            "Based on speed and consistency, start with a stable weight class and stiff flex; adjust after testing.",
            {
                "weight_g": f"{weight_range[0]}–{weight_range[1]}",
                "flex": flex_range,
                "profile": "tip-stiff, low–mid launch, low spin",
                "torque": "approx 3.0–3.6",
                "notes": notes,
            },
        )

        off_mean = m.get("offline_mean", np.nan)
        f2p = m.get("face_to_path", np.nan)
        if np.isfinite(off_mean) and off_mean > 12:
            add(
                "Reduce right-bias (settings first)",
                "Average miss is right. Slightly more upright/closed settings can help; then consider draw-biased head category if needed.",
                {
                    "setting_ideas": ["slightly more upright lie (if available)", "slightly more closed face setting"],
                    "head_category": "neutral-to-draw bias; higher MOI if strike is inconsistent",
                    "check_metrics": ["offline_mean", "face_to_path_mean"],
                    "notes": ["Spin axis can be noisy on some monitors; trust face-to-path more if available."],
                },
                confidence="MED" if conf == "LOW" else conf,
            )

        smash = m.get("smash", np.nan)
        if np.isfinite(smash) and smash < 1.43:
            add(
                "Improve strike efficiency (equipment that helps)",
                "Smash is below potential; a more forgiving head and/or slightly shorter build often improves centered contact.",
                {
                    "head_category": "higher MOI / forgiving",
                    "build_options": ["shorten length by 0.5 inch", "stay within 60–70g shaft class"],
                    "goal": {"smash": "+0.02 to +0.05"},
                },
                confidence="MED" if conf == "LOW" else conf,
            )

    elif c == "3W":
        add(
            "Shaft spec range to test (3W)",
            "Fairway woods typically work best slightly heavier than driver for control from turf.",
            {"weight_g": "65–75", "flex": "stiff (test x only if you swing very hard)", "profile": "mid launch / mid-low spin; stable tip"},
        )

    elif c == "HY":
        add(
            "Shaft spec range to test (hybrid)",
            "Hybrids commonly perform best with heavier shafts for consistent strike and start line.",
            {"weight_g": "80–95", "flex": "stiff (test x if aggressive)", "profile": "mid launch / mid spin; tip-stable"},
        )

    else:
        add(
            "Club type not covered in v1",
            "v1 focuses on driver/3W/hybrid. Add iron/wedge windows later.",
            {"next_step": "implement irons + gapping rules"},
            confidence="LOW",
        )

    return recs

def analyze_dataframe(df_raw: pd.DataFrame) -> SessionResult:
    df = _canonicalize_columns(df_raw.copy())

    # Optional: clean DistanceToPin like "181.28 yds" -> 181.28
    if "DistanceToPin" in df.columns:
        df["DistanceToPin"] = (
            df["DistanceToPin"].astype(str)
            .str.replace("yds", "", regex=False)
            .str.replace("yd", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["DistanceToPin"] = pd.to_numeric(df["DistanceToPin"], errors="coerce")

    # Make sure numeric columns are numeric at the session level
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
            
    # Fix columns that may include 'L'/'R' or degree symbols in GSPro exports
    for col in ["offline", "hla", "spin_axis", "face_to_target", "face_to_path", "vla"]:
        if col in df.columns:
            df[col] = _to_numeric_lr(df[col])

    if "club" not in df.columns:
        raise ValueError("CSV missing club column (could not find Club/Club Name).")

    df["club"] = df["club"].astype(str).str.upper().str.strip()

    club_results: Dict[str, ClubAnalysis] = {}

    for club_raw, df_club in df.groupby("club", dropna=False):
        bucket = _club_bucket(club_raw)

        # v1 supports only DR, 3W, HY
        if bucket not in {"DR", "3W", "HY"}:
            continue

        n_total = len(df_club)

        outliers = flag_outliers_per_club(df_club, bucket)
        df_used = df_club.loc[~outliers].copy()

        # Ensure smash exists if missing
        if "smash" not in df_used.columns and "ball_speed" in df_used.columns and "club_speed" in df_used.columns:
            df_used["smash"] = df_used["ball_speed"] / df_used["club_speed"]

        # Force numeric conversion inside df_used too (handles weird dtype/object cases)
        for col in numeric_cols:
            if col in df_used.columns:
                df_used[col] = (
                    df_used[col].astype(str)
                    .str.replace("yds", "", regex=False)
                    .str.replace("yd", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df_used[col] = pd.to_numeric(df_used[col], errors="coerce")

        # Fix columns that may include 'L'/'R' or degree symbols at the club level too
        for col in ["offline", "hla", "spin_axis", "face_to_target", "vla"]:
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

       
def session_to_dict(result: SessionResult) -> Dict[str, object]:
    """
    Convert dataclasses to JSON-serializable dict for Streamlit.
    (Minimal version — won’t crash your app.)
    """
    out: Dict[str, object] = {"clubs": {}}
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
        }
    return out
