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
    what_to_change_first: Dict[str, Any]


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

def what_to_change_first(summary: ClubSummary) -> Dict[str, Any]:
    """
    Returns a single best next action (badge) based on the summary metrics.
    """
    m = summary.metrics
    v = summary.variability

    launch = m.get("launch", float("nan"))
    spin = m.get("spin", float("nan"))
    smash = m.get("smash", float("nan"))
    offline_std = v.get("offline_std", float("nan"))

    # Driver logic
    if summary.club == "DR":
        if np.isfinite(launch) and launch < 10.0:
            return {
                "badge": "Raise launch",
                "why": "Launch is low; test +0.75° to +1.5° loft (adapter) before buying shafts/heads.",
                "priority": 1,
            }
        if np.isfinite(spin) and spin > 3000:
            return {
                "badge": "Lower spin",
                "why": "Spin is high; test a lower-spin shaft profile (tip-stiff / lower torque) and avoid adding dynamic loft.",
                "priority": 2,
            }
        if np.isfinite(offline_std) and offline_std > 20:
            return {
                "badge": "Tighten dispersion",
                "why": "Dispersion is wide; settings/face control and strike location will help more than chasing distance.",
                "priority": 3,
            }
        if np.isfinite(smash) and smash < 1.42:
            return {
                "badge": "Improve strike (smash)",
                "why": "Strike efficiency is low; test strike-friendly tee height + ball position and verify centered contact.",
                "priority": 4,
            }
        return {
            "badge": "Keep testing",
            "why": "No major red flags. Use a simple A/B test: loft change → shaft profile → head category.",
            "priority": 5,
        }

    # Hybrid logic
    if summary.club == "HY":
        if np.isfinite(offline_std) and offline_std > 15:
            return {
                "badge": "Tighten dispersion",
                "why": "Hybrid dispersion is the main limiter. Test lie/face angle settings and shaft weight/length consistency.",
                "priority": 1,
            }
        if np.isfinite(spin) and spin > 4500:
            return {
                "badge": "Lower spin slightly",
                "why": "Hybrid spin is high; test a slightly lower-spin shaft profile or a more neutral head setting.",
                "priority": 2,
            }
        return {
            "badge": "Confirm gapping",
            "why": "Make sure carry gap to your next club is consistent (goal: predictable yardage window).",
            "priority": 3,
        }

    # Fairway wood (if/when present)
    if summary.club == "3W":
        if np.isfinite(offline_std) and offline_std > 18:
            return {"badge": "Tighten dispersion", "why": "Start with strike + face control; then tune loft/shaft.", "priority": 1}
        return {"badge": "Confirm launch window", "why": "Tune launch/spin for turf + tee use.", "priority": 2}

    return {"badge": "Next best test", "why": "Collect more shots to increase confidence.", "priority": 9}


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
            what_to_change_first=what_to_change_first(summary),
        )

    return SessionResult(club_results=club_results)


# =========================
# Titleist SureFit (Driver/FW) setting recommender
# =========================

SUREFIT_RH = {
    # loft_delta, lie_delta (upright positive)
    "A3": (+1.5, +1.5), "B3": (+1.5, +0.75), "A4": (+1.5, 0.0),  "B4": (+1.5, -0.75),
    "D3": (+0.75, +1.5), "C3": (+0.75, +0.75), "D4": (+0.75, 0.0), "C4": (+0.75, -0.75),
    "A2": (0.0, +1.5),  "B2": (0.0, +0.75),  "A1": (0.0, 0.0),  "B1": (0.0, -0.75),
    "D2": (-0.75, +1.5), "C2": (-0.75, +0.75), "D1": (-0.75, 0.0), "C1": (-0.75, -0.75),
}

SUREFIT_LH = {
    # loft_delta, lie_delta (upright positive)
    "C1": (+1.5, -0.75), "D1": (+1.5, 0.0),  "C2": (+1.5, +0.75), "D2": (+1.5, +1.5),
    "B1": (+0.75, -0.75), "A1": (+0.75, 0.0), "B2": (+0.75, +0.75), "A2": (+0.75, +1.5),
    "C4": (0.0, -0.75),  "D4": (0.0, 0.0),  "C3": (0.0, +0.75),  "D3": (0.0, +1.5),
    "B4": (-0.75, -0.75), "A4": (-0.75, 0.0), "B3": (-0.75, +0.75), "A3": (-0.75, +1.5),
}

def recommend_titleist_surefit_driver(
    summary: "ClubSummary",
    handedness: str,
    current_setting: str,
    miss_tendency: str,
) -> dict:
    """
    Strong, specific SureFit recommendation for Driver/FW:
    - Uses user's miss tendency (B) as primary for direction.
    - Uses launch/spin from summary to decide loft change.
    Returns dict with from/to + expected effect.
    """
    handedness = (handedness or "RH").upper().strip()
    current_setting = (current_setting or "").upper().strip()
    miss_tendency = (miss_tendency or "NOT SURE").upper().strip()

    table = SUREFIT_LH if handedness == "LH" else SUREFIT_RH

    if current_setting not in table:
        return {
            "action": "unknown_current_setting",
            "from": current_setting,
            "to": None,
            "why": "Current SureFit setting not recognized.",
            "expected": {},
        }

    launch = summary.metrics.get("launch", float("nan"))
    spin = summary.metrics.get("spin", float("nan"))

    # ---- Decide loft target from data ----
    # (simple + safe thresholds; you can tune later)
    loft_need = 0.0
    if np.isfinite(launch) and launch < 10.0:
        loft_need = +0.75
        if np.isfinite(launch) and launch < 9.0:
            loft_need = +1.5
    elif np.isfinite(launch) and launch > 14.5:
        loft_need = -0.75

    # ---- Decide direction target from user miss tendency ----
    # We interpret:
    # - miss RIGHT => add draw bias => more upright
    # - miss LEFT  => add fade bias => flatter
    lie_need = 0.0
    if miss_tendency == "RIGHT":
        lie_need = +0.75
        # if also high dispersion, nudge stronger
        if summary.variability.get("offline_std", 0) and summary.variability["offline_std"] > 20:
            lie_need = +1.5
    elif miss_tendency == "LEFT":
        lie_need = -0.75
    else:
        lie_need = 0.0  # BOTH / NOT SURE

    # ---- Build target deltas relative to current ----
    curr_loft, curr_lie = table[current_setting]
    target_loft = curr_loft + loft_need
    target_lie = curr_lie + lie_need

    # clamp to available
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    target_loft = _clamp(target_loft, -0.75, +1.5)
    target_lie = _clamp(target_lie, -0.75, +1.5)

    # ---- Choose closest setting in this handedness table ----
    best = None
    best_dist = 1e9
    for setting, (ld, lied) in table.items():
        dist = (ld - target_loft) ** 2 + (lied - target_lie) ** 2
        if dist < best_dist:
            best_dist = dist
            best = setting

    if best == current_setting:
        return {
            "action": "no_change",
            "from": current_setting,
            "to": best,
            "why": "Your current setting is already close to the best match for your miss tendency and launch window.",
            "expected": {"launch_deg": "≈0", "start_line": "≈0"},
        }

    # ---- Build explanation ----
    best_loft, best_lie = table[best]
    d_loft = best_loft - curr_loft
    d_lie = best_lie - curr_lie

    expected = {}
    if d_loft > 0:
        expected["launch_deg"] = f"+{d_loft:.2f} (square-face effective)"
        expected["note"] = "Higher loft can add launch and may add spin—confirm spin doesn’t balloon."
    elif d_loft < 0:
        expected["launch_deg"] = f"{d_loft:.2f} (square-face effective)"
        expected["note"] = "Lower loft can reduce launch/spin—confirm you don’t lose carry."

    if d_lie > 0:
        expected["direction_bias"] = "More draw bias (more upright)"
    elif d_lie < 0:
        expected["direction_bias"] = "More fade bias (flatter)"

    # extra spin caution
    if np.isfinite(spin) and spin > 3200 and d_loft > 0:
        expected["spin_watchout"] = "Spin already high—if spin rises further, switch to lower-spin shaft profile instead of more loft."

    why_parts = []
    if loft_need > 0:
        why_parts.append("Launch is low → add loft")
    if loft_need < 0:
        why_parts.append("Launch is high → reduce loft")
    if miss_tendency == "RIGHT":
        why_parts.append("Miss is right → add draw bias (upright lie)")
    if miss_tendency == "LEFT":
        why_parts.append("Miss is left → add fade bias (flatter lie)")

    return {
        "action": "change_setting",
        "from": current_setting,
        "to": best,
        "why": "; ".join(why_parts) if why_parts else "Tune loft/lie toward your target window.",
        "expected": expected,
    }


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
            "what_to_change_first": analysis.what_to_change_first,
            "limiting_factors": analysis.limiting_factors,
            "recommendations": analysis.recommendations,
        }

    return out
