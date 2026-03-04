# fit_engine.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Numeric parsing
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
      '11.4 R' -> +11.4
      '11.4 L' -> -11.4
      '2.6° U' -> +2.6
      '2.6° D' -> -2.6
      '3.2 I-O' -> +3.2
      '3.2 O-I' -> -3.2
      '2.0 O' -> +2.0 (open)
      '2.0 C' -> -2.0 (closed)
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

    if re.search(r"\bL\b", s_upper):
        return -abs(num)
    if re.search(r"\bR\b", s_upper):
        return abs(num)

    if re.search(r"\bU\b", s_upper):
        return abs(num)
    if re.search(r"\bD\b", s_upper):
        return -abs(num)

    if "I-O" in s_upper:
        return abs(num)
    if "O-I" in s_upper:
        return -abs(num)

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
# Club normalization (NEW)
# -----------------------------
_CLUB_PATTERNS = [
    ("DR", [r"\bDR\b", r"\bDRIVER\b", r"^D$"]),
    ("2W", [r"\b2W\b", r"\b2\s*WOOD\b"]),
    ("3W", [r"\b3W\b", r"\b3\s*WOOD\b"]),
    ("4W", [r"\b4W\b", r"\b4\s*WOOD\b"]),
    ("5W", [r"\b5W\b", r"\b5\s*WOOD\b"]),
    ("7W", [r"\b7W\b", r"\b7\s*WOOD\b"]),
    ("9W", [r"\b9W\b", r"\b9\s*WOOD\b"]),
    ("2H", [r"\b2H\b", r"\bH2\b", r"\b2\s*HY\b", r"\b2\s*HYBRID\b"]),
    ("3H", [r"\b3H\b", r"\bH3\b", r"\b3\s*HY\b", r"\b3\s*HYBRID\b"]),
    ("4H", [r"\b4H\b", r"\bH4\b", r"\b4\s*HY\b", r"\b4\s*HYBRID\b"]),
    ("5H", [r"\b5H\b", r"\bH5\b", r"\b5\s*HY\b", r"\b5\s*HYBRID\b"]),
    ("6H", [r"\b6H\b", r"\bH6\b", r"\b6\s*HY\b", r"\b6\s*HYBRID\b"]),
    ("7H", [r"\b7H\b", r"\bH7\b", r"\b7\s*HY\b", r"\b7\s*HYBRID\b"]),
]

def normalize_club_label(label: str) -> str:
    if not isinstance(label, str):
        return "OTHER"

    s = label.strip().upper()

    # Driver
    if s == "DR":
        return "DR"

    # Woods
    if s.startswith("W") and len(s) == 2:
        return f"{s[1]}W"

    # Hybrids
    if s.startswith("H") and len(s) == 2:
        return f"{s[1]}H"

    # Irons
    if s.startswith("I") and len(s) == 2:
        return f"{s[1]}I"

    # Wedges
    if s in ["PW", "GW", "SW", "LW"]:
        return s

    # Putter
    if s == "PT":
        return "PT"

    return "OTHER"


def club_family(club_id: str) -> str:
    if club_id == "DR":
        return "DR"
    if club_id.endswith("W"):
        return "FW"
    if club_id.endswith("H"):
        return "HY"
    return "OTHER"


# -----------------------------
# GSPro canonicalize
# -----------------------------
CANON = [
    "club_raw",
    "club_id",
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
        out["club_raw"] = df.get("Club Name")
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
        out["club_raw"] = df.get("Club")
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed"), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed"), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry"), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance"), errors="coerce")
        out["offline_yd"] = pd.to_numeric(df.get("Offline"), errors="coerce")
        out["peak_height_yd"] = pd.to_numeric(df.get("PeakHeight"), errors="coerce")
        out["descent_deg"] = pd.to_numeric(df.get("Decent"), errors="coerce")  # yes "Decent"
        out["hla_deg"] = pd.to_numeric(df.get("HLA"), errors="coerce")
        out["vla_deg"] = pd.to_numeric(df.get("VLA"), errors="coerce")
        out["backspin_rpm"] = pd.to_numeric(df.get("BackSpin"), errors="coerce")

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
        # best effort
        out["club_raw"] = df.get("Club", df.get("Club Name", np.nan))
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed", df.get("Club Speed (mph)")), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed", df.get("Ball Speed (mph)")), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry", df.get("Carry Dist (yd)")), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance", df.get("Total Dist (yd)")), errors="coerce")
        out["offline_yd"] = df.get("Offline", df.get("Offline (yd)")).apply(parse_dir_value)
        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    out["club_id"] = out["club_raw"].apply(normalize_club_label)
    return out, fmt


# -----------------------------
# Targets (club-specific)
# -----------------------------
def targets_for_club(club_id: str, club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
    """
    Practical MVP target windows per club. These are intentionally broad.
    Driver uses speed-adjusted windows; fairway/hybrids use fixed windows.
    """
    fam = club_family(club_id)

    if fam == "DR":
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

    # Fairway woods (broad windows by loft class)
    if fam == "FW":
        if club_id in {"2W", "3W"}:
            return {"launch": (11.0, 15.0), "spin": (2800.0, 3800.0)}
        if club_id in {"4W", "5W"}:
            return {"launch": (13.0, 17.0), "spin": (3200.0, 4500.0)}
        return {"launch": (14.0, 19.0), "spin": (3800.0, 5200.0)}

    # Hybrids (broad)
    if fam == "HY":
        if club_id in {"2H", "3H"}:
            return {"launch": (13.0, 18.0), "spin": (3500.0, 5200.0)}
        if club_id in {"4H", "5H"}:
            return {"launch": (15.0, 20.0), "spin": (3800.0, 5600.0)}
        return {"launch": (16.0, 22.0), "spin": (4200.0, 6200.0)}

    return {"launch": (0.0, 99.0), "spin": (0.0, 99999.0)}


# -----------------------------
# Hosel change estimate (NEW)
# -----------------------------
LAUNCH_FROM_DYNAMIC_WEIGHT = 0.85

@dataclass(frozen=True)
class LaunchSpinEstimate:
    launch_change_deg: float
    launch_range_deg: Tuple[float, float]
    spin_change_rpm: int
    spin_range_rpm: Tuple[int, int]
    notes: str

def estimate_launch_spin_change(delta_static_loft_deg: float, k_loft_to_dynamic: float, club_id: str) -> LaunchSpinEstimate:
    fam = club_family(club_id)

    launch_est = LAUNCH_FROM_DYNAMIC_WEIGHT * k_loft_to_dynamic * delta_static_loft_deg
    launch_unc = max(0.6, abs(launch_est) * 0.35)
    launch_low = launch_est - launch_unc
    launch_high = launch_est + launch_unc

    # conservative spin bands (rpm per 1° loft)
    if fam == "DR":
        center, lo, hi = 250, 150, 400
    elif fam == "FW":
        center, lo, hi = 300, 180, 450
    elif fam == "HY":
        center, lo, hi = 320, 200, 500
    else:
        center, lo, hi = 250, 150, 400

    spin_est = int(round(center * delta_static_loft_deg))
    spin_low = int(round(lo * delta_static_loft_deg))
    spin_high = int(round(hi * delta_static_loft_deg))
    spin_range = (min(spin_low, spin_high), max(spin_low, spin_high))

    return LaunchSpinEstimate(
        launch_change_deg=launch_est,
        launch_range_deg=(launch_low, launch_high),
        spin_change_rpm=spin_est,
        spin_range_rpm=spin_range,
        notes="Estimates assume similar delivery; real changes vary with strike location, face angle at address, and shaft lean.",
    )


# -----------------------------
# One-setting hosel recommendation (NEW)
# -----------------------------
def miss_tendency(offline_avg: float) -> str:
    if np.isnan(offline_avg):
        return "Unknown"
    if offline_avg > 5:
        return "Right miss tendency"
    if offline_avg < -5:
        return "Left miss tendency"
    return "Centered"


def smash_flag_driver(smash_avg: float) -> Optional[str]:
    if np.isnan(smash_avg):
        return None
    if smash_avg < 1.42:
        return f"Smash factor is low ({smash_avg:.2f}). Efficiency/contact is a limiting factor."
    if smash_avg < 1.45:
        return f"Smash factor is slightly low ({smash_avg:.2f}). There’s still efficiency left."
    return None


def pick_one_hosel_setting(
    settings: List[str],
    translate_fn,
    brand: str,
    system_name: str,
    handedness: str,
    current_setting: str,
    needed_loft_delta: float,
    needed_lie_delta: float,
) -> Dict[str, object]:
    """
    Returns ONE recommended hosel setting if exact deltas exist.
    Adds a strong penalty if the setting moves loft in the opposite direction
    of the required loft change (prevents 'lower loft' recommendations when
    launch is too low, and vice-versa).
    """
    scored = []

    for s in settings:
        d = translate_fn(brand, system_name, s, handedness)
        loft = getattr(d, "loft_deg", None)
        lie = getattr(d, "lie_deg", None)
        if loft is None or lie is None:
            continue

        # Base distance-to-goal score
        score = abs(loft - needed_loft_delta) * 1.5 + abs(lie - needed_lie_delta) * 1.0

        # Directional guardrail for loft
        # If we need more loft, strongly avoid negative loft settings (and vice versa)
        if needed_loft_delta > 0.25 and loft < -0.01:
            score += 100.0
        if needed_loft_delta < -0.25 and loft > 0.01:
            score += 100.0

        scored.append((score, s, loft, lie))

    if not scored:
        direction = []
        if needed_loft_delta > 0.25:
            direction.append("add loft")
        elif needed_loft_delta < -0.25:
            direction.append("reduce loft")
        if needed_lie_delta > 0.25:
            direction.append("more upright")
        elif needed_lie_delta < -0.25:
            direction.append("flatter")
        if not direction:
            direction = ["stay near current"]

        return {"type": "guidance", "message": f"Exact chart not encoded for this hosel. Guidance: {', '.join(direction)}."}

    scored.sort(key=lambda x: x[0])
    score, setting, loft, lie = scored[0]
    return {
        "type": "exact",
        "current": current_setting,
        "recommended": {"setting": setting, "loft_delta": loft, "lie_delta": lie, "score": score},
    }


# -----------------------------
# Aggregation per club (NEW)
# -----------------------------
@dataclass(frozen=True)
class ClubSummary:
    club_id: str
    n: int
    club_speed_avg: float
    club_speed_std: float
    ball_speed_avg: float
    ball_speed_std: float
    smash_avg: float
    smash_std: float
    carry_avg: float
    carry_std: float
    offline_avg: float
    offline_std: float
    vla_avg: float
    vla_std: float
    spin_avg: float
    spin_std: float
    aoa_avg: float
    aoa_std: float


def summarize_by_club(canon_df: pd.DataFrame) -> Dict[str, ClubSummary]:
    summaries: Dict[str, ClubSummary] = {}
    for club_id, g in canon_df.groupby("club_id"):
        if club_id == "OTHER":
            continue
        summaries[club_id] = ClubSummary(
            club_id=club_id,
            n=int(len(g)),
            club_speed_avg=safe_mean(g["club_speed_mph"]),
            club_speed_std=safe_std(g["club_speed_mph"]),
            ball_speed_avg=safe_mean(g["ball_speed_mph"]),
            ball_speed_std=safe_std(g["ball_speed_mph"]),
            smash_avg=safe_mean(g["smash"]),
            smash_std=safe_std(g["smash"]),
            carry_avg=safe_mean(g["carry_yd"]),
            carry_std=safe_std(g["carry_yd"]),
            offline_avg=safe_mean(g["offline_yd"]),
            offline_std=safe_std(g["offline_yd"]),
            vla_avg=safe_mean(g["vla_deg"]),
            vla_std=safe_std(g["vla_deg"]),
            spin_avg=safe_mean(g["backspin_rpm"]),
            spin_std=safe_std(g["backspin_rpm"]),
            aoa_avg=safe_mean(g["aoa_deg"]),
            aoa_std=safe_std(g["aoa_deg"]),
        )
    return summaries
