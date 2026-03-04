import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any


# =========================
# Data structures
# =========================

@dataclass
class ClubSummary:
    club: str
    n_total: int
    n_used: int
    confidence: str
    metrics: Dict[str, float]
    variability: Dict[str, float]


@dataclass
class ClubAnalysis:
    summary: ClubSummary
    limiting_factors: List[str]
    recommendations: List[Dict[str, Any]]


@dataclass
class SessionResult:
    club_results: Dict[str, ClubAnalysis]


# =========================
# Column canonicalization
# =========================

CANON_MAP = {
    # Club identity
    "Club Name": "club",
    "Club": "club",

    # Speeds / distance
    "Club Speed (mph)": "club_speed",
    "Ball Speed (mph)": "ball_speed",
    "Carry Dist (yd)": "carry",
    "Total Dist (yd)": "total",

    # Direction / flight
    "Offline (yd)": "offline",
    "Peak Height (yd)": "peak_height",
    "Desc Angle": "descent",
    "HLA": "hla",
    "VLA": "vla",

    # Spin / delivery
    "Back Spin": "spin",
    "Spin Axis": "spin_axis",
    "Club AoA": "aoa",
    "Club Path": "path",
    "Face to Path": "face_to_path",
    "Face to Target": "face_to_target",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize headers aggressively
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("-", "_")
        .str.lower()
    )

    rename_map = {
        "club_name": "club",
        "club": "club",

        "club_speed_mph": "club_speed",
        "ball_speed_mph": "ball_speed",

        "carry_dist_yd": "carry",
        "total_dist_yd": "total",

        "offline_yd": "offline",
        "peak_height_yd": "peak_height",

        "desc_angle": "descent",

        "hla": "hla",
        "vla": "vla",

        "back_spin": "spin",
        "spin_axis": "spin_axis",

        "club_aoa": "aoa",
        "club_path": "path",

        "face_to_path": "face_to_path",
        "face_to_target": "face_to_target",
    }

    df = df.rename(columns=rename_map)

    return df
    rename = {src: dst for src, dst in CANON_MAP.items() if src in df.columns}
    df = df.rename(columns=rename)

    # Also allow already-canonical headers (some exports/tools do this)
    # Normalize common variants:
    lower = {c.lower(): c for c in df.columns}
    variants = {
        "club_name": "club",
        "clubspeed": "club_speed",
        "ballspeed": "ball_speed",
        "carrydist": "carry",
        "totaldist": "total",
    }
    for k, v in variants.items():
        if k in lower and v not in df.columns:
            df = df.rename(columns={lower[k]: v})

    return df


# =========================
# Parsing helpers
# =========================

def _confidence_label(n_used: int) -> str:
    if n_used >= 10:
        return "HIGH"
    if n_used >= 6:
        return "MED"
    return "LOW"


def _to_numeric_lr(series: pd.Series) -> pd.Series:
    """
    Robust parse for GSPro fields that may include directions/units:
      '11.4 R'     ->  11.4
      '14.0 L'     -> -14.0
      '2.6° R'     ->  2.6
      '0.8° I-O'   ->  0.8
      '4.1° U'     ->  4.1
      '40.1°'      -> 40.1
    Rule: extract first number; if contains standalone L token => negative.
    """
    s = series.astype(str).str.strip()

    # Left indicator (token L anywhere)
    is_left = s.str.contains(r"(^L\b|\bL\b|\bL$)", regex=True)

    # Remove commas (just in case)
    s = s.str.replace(",", "", regex=False)

    # Extract first numeric token
    num = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    num = pd.to_numeric(num, errors="coerce")

    num = np.where(is_left, -num, num)
    return pd.Series(num, index=series.index)


def _bucket_club(club_raw: Any) -> str:
    s = str(club_raw).upper().strip()
    # common driver labels
    if s in {"DR", "D", "DRIVER"}:
        return "DR"
    # fairway wood common labels
    if s in {"3W", "W3", "FW3", "3 WOOD", "3-WOOD"}:
        return "3W"
    # hybrids: H3, HY, HB, etc.
    if s.startswith("H") or s in {"HY", "HB", "HYBRID"}:
        return "HY"
    return "OTHER"


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def _flag_outliers(df: pd.DataFrame, bucket: str) -> pd.Series:
    """
    Light outlier filter to stabilize averages.
    Uses robust-z on carry and ball_speed if available.
    """
    if len(df) < 6:
        return pd.Series(False, index=df.index)

    flags = pd.Series(False, index=df.index)

    for col in ["carry", "ball_speed"]:
        if col in df.columns:
            z = _robust_z(pd.to_numeric(df[col], errors="coerce").to_numpy())
            flags = flags | (np.abs(z) > 3.5)

    # Extra: remove totally broken shots for driver
    if bucket == "DR" and "carry" in df.columns:
        carry = pd.to_numeric(df["carry"], errors="coerce")
        flags = flags | (carry < 120)

    return flags.fillna(False)


# =========================
# Core analysis
# =========================

def summarize_club(df_used: pd.DataFrame, bucket: str, n_total: int) -> ClubSummary:
    df_used = df_used.copy()

    # Force numeric parsing for core fields (handles "11.4 R", "2.6° R", etc.)
    for col in ["offline", "vla", "hla", "spin_axis", "face_to_target", "face_to_path", "descent", "peak_height"]:
        if col in df_used.columns:
            df_used[col] = _to_numeric_lr(df_used[col])

    # Force numeric for standard numeric fields
    for col in ["carry", "total", "ball_speed", "club_speed", "spin", "aoa", "path"]:
        if col in df_used.columns:
            df_used[col] = pd.to_numeric(df_used[col], errors="coerce")

    # Smash if missing
    if "smash" not in df_used.columns and "ball_speed" in df_used.columns and "club_speed" in df_used.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df_used["smash"] = df_used["ball_speed"] / df_used["club_speed"]

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
        "ball_speed": mean("ball_speed"),
        "club_speed": mean("club_speed"),
        "smash": mean("smash"),

        # launch (VLA)
        "launch_vla": mean("vla"),
        "launch": mean("vla"),  # alias for UI

        "spin": mean("spin"),
        "spin_axis": mean("spin_axis"),
        "aoa": mean("aoa"),
        "path": mean("path"),
        "face_to_path": mean("face_to_path"),

        "descent": mean("descent"),
        "peak_height": mean("peak_height"),

        # offline
        "offline_mean": mean("offline"),
        "offline_abs_mean": float(pd.to_numeric(df_used["offline"], errors="coerce").abs().mean(skipna=True))
        if "offline" in df_used.columns else float("nan"),
    }

    variability = {
        "offline_std": std("offline"),
        "launch_std": std("vla"),
        "spin_std": std("spin"),
        "face_to_path_std": std("face_to_path"),
        "ball_speed_std": std("ball_speed"),
    }

    n_used = int(df_used.shape[0])
    confidence = _confidence_label(n_used)

    return ClubSummary(
        club=bucket,
        n_total=int(n_total),
        n_used=n_used,
        confidence=confidence,
        metrics=metrics,
        variability=variability,
    )


def limiting_factors(summary: ClubSummary) -> List[str]:
    m = summary.metrics
    v = summary.variability

    factors: List[str] = []

    launch = m.get("launch", float("nan"))
    spin = m.get("spin", float("nan"))
    offline_std = v.get("offline_std", float("nan"))
    face_std = v.get("face_to_path_std", float("nan"))
    smash = m.get("smash", float("nan"))

    if np.isfinite(offline_std) and offline_std > 20:
        factors.append("Dispersion is wide (offline variability high).")

    if np.isfinite(face_std) and face_std > 3.5:
        factors.append("Face-to-path is inconsistent (timing / face control).")

    if summary.club == "DR":
        if np.isfinite(launch) and launch < 10:
            factors.append("Launch is low for this club.")
        if np.isfinite(spin) and spin > 3100:
            factors.append("Spin is high (can reduce distance / add curvature).")
        if np.isfinite(smash) and smash < 1.42:
            factors.append("Strike efficiency is below potential (smash factor low).")

    if summary.club == "HY":
        if np.isfinite(spin) and spin > 4500:
            factors.append("Spin is high for a hybrid (possible added height/balloon).")

    if not factors:
        factors.append("No major red flags detected; focus on consistency and gapping.")

    return factors


def recommend_for_club(summary: ClubSummary) -> List[Dict[str, Any]]:
    m = summary.metrics
    recs: List[Dict[str, Any]] = []

    if summary.club == "DR":
        launch = m.get("launch", float("nan"))
        spin = m.get("spin", float("nan"))
        offline_std = summary.variability.get("offline_std", float("nan"))

        if np.isfinite(launch) and launch < 10:
            recs.append({
                "priority": 1,
                "title": "Test a higher loft setting (+0.75° to +1.5°) first",
                "rationale": "Launch is low; a small loft increase often raises launch without major swing changes.",
                "confidence": summary.confidence,
                "spec": {"adapter_loft_change_deg": "+0.75 to +1.5", "goal": {"launch_deg": "+1–2"}}
            })

        if np.isfinite(spin) and spin > 3000:
            recs.append({
                "priority": 2,
                "title": "Bias toward a lower-spin build",
                "rationale": "Spin is above a typical driver window; lowering spin can improve carry/roll and reduce curvature.",
                "confidence": summary.confidence,
                "spec": {
                    "shaft_profile": "low–mid launch / low spin, tip-stiff, lower torque",
                    "goal": {"spin_rpm": "-200 to -500"}
                }
            })

        if np.isfinite(offline_std) and offline_std > 20:
            recs.append({
                "priority": 3,
                "title": "Reduce dispersion (settings first)",
                "rationale": "Wide dispersion suggests face/path variability. Start with sure-fit bias + strike location before buying parts.",
                "confidence": summary.confidence,
                "spec": {"goal": {"offline_std": "-10 yd"}}
            })

        if not recs:
            recs.append({
                "priority": 1,
                "title": "Stay in current window; test for consistency",
                "rationale": "Numbers look generally playable. Focus on tightening strike and start line.",
                "confidence": summary.confidence,
                "spec": {}
            })

    if summary.club == "HY":
        recs.append({
            "priority": 1,
            "title": "Hybrid spec check",
            "rationale": "Confirm gapping and whether launch/spin fit your approach-shot needs.",
            "confidence": summary.confidence,
            "spec": {"goal": {"consistent_carry_window": "stable", "dispersion": "tighten"}}
        })

    if summary.club == "3W":
        recs.append({
            "priority": 1,
            "title": "Fairway baseline",
            "rationale": "Establish launch/spin/carry baseline and fit for tee + turf use.",
            "confidence": summary.confidence,
            "spec": {}
        })

    return recs


def analyze_dataframe(df_raw: pd.DataFrame) -> SessionResult:
    df = _canonicalize_columns(df_raw)

    if "club" not in df.columns:
        raise ValueError("CSV missing club column. Expected GSPro 'Club Name' field.")

    # Normalize club strings
    df["club"] = df["club"].astype(str).str.strip()

    club_results: Dict[str, ClubAnalysis] = {}

    for club_raw, df_club in df.groupby("club", dropna=False):
        bucket = _bucket_club(club_raw)
        if bucket not in {"DR", "3W", "HY"}:
            continue

        n_total = len(df_club)

        outliers = _flag_outliers(df_club, bucket)
        df_used = df_club.loc[~outliers].copy()

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
            "recommendations": analysis.recommendations,
        }

    return out
