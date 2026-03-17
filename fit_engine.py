from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Numeric parsing
# =========================================================
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


def _is_nan(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


# =========================================================
# Club normalization
# =========================================================
def normalize_club_label(label: str) -> str:
    if not isinstance(label, str):
        return "OTHER"

    s = label.strip().upper()

    if s in {"DR", "PW", "GW", "SW", "LW", "PT"}:
        return s

    if s.startswith("W") and len(s) == 2 and s[1].isdigit():
        return f"{s[1]}W"
    if s.startswith("H") and len(s) == 2 and s[1].isdigit():
        return f"{s[1]}H"
    if s.startswith("I") and len(s) == 2 and s[1].isdigit():
        return f"{s[1]}I"

    aliases = {
        "DRIVER": "DR",
        "3 WOOD": "3W",
        "4 WOOD": "4W",
        "5 WOOD": "5W",
        "7 WOOD": "7W",
        "2 HYBRID": "2H",
        "3 HYBRID": "3H",
        "4 HYBRID": "4H",
        "5 HYBRID": "5H",
        "3 IRON": "3I",
        "4 IRON": "4I",
        "5 IRON": "5I",
        "6 IRON": "6I",
        "7 IRON": "7I",
        "8 IRON": "8I",
        "9 IRON": "9I",
        "P WEDGE": "PW",
        "G WEDGE": "GW",
        "S WEDGE": "SW",
        "L WEDGE": "LW",
    }
    if s in aliases:
        return aliases[s]

    return "OTHER"


def club_family(club_id: str) -> str:
    if club_id == "DR":
        return "Driver"
    if club_id.endswith("W"):
        return "Fairway Wood"
    if club_id.endswith("H"):
        return "Hybrid"
    if club_id.endswith("I"):
        return "Iron"
    if club_id in {"PW", "GW", "SW", "LW"}:
        return "Wedge"
    return "Other"


# =========================================================
# GSPro canonicalize
# =========================================================
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
        out["descent_deg"] = pd.to_numeric(df.get("Decent"), errors="coerce")
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
        out["club_raw"] = df.get("Club", df.get("Club Name", np.nan))
        out["club_speed_mph"] = pd.to_numeric(df.get("ClubSpeed", df.get("Club Speed (mph)")), errors="coerce")
        out["ball_speed_mph"] = pd.to_numeric(df.get("BallSpeed", df.get("Ball Speed (mph)")), errors="coerce")
        out["carry_yd"] = pd.to_numeric(df.get("Carry", df.get("Carry Dist (yd)")), errors="coerce")
        out["total_yd"] = pd.to_numeric(df.get("TotalDistance", df.get("Total Dist (yd)")), errors="coerce")

        offline_raw = df.get("Offline", df.get("Offline (yd)", np.nan))
        out["offline_yd"] = offline_raw.apply(parse_dir_value) if isinstance(offline_raw, pd.Series) else np.nan

        out["peak_height_yd"] = pd.to_numeric(df.get("PeakHeight", df.get("Peak Height (yd)")), errors="coerce")

        desc_raw = df.get("Decent", df.get("Desc Angle", np.nan))
        out["descent_deg"] = desc_raw.apply(parse_dir_value) if isinstance(desc_raw, pd.Series) else np.nan

        hla_raw = df.get("HLA", np.nan)
        out["hla_deg"] = hla_raw.apply(parse_dir_value) if isinstance(hla_raw, pd.Series) else np.nan

        vla_raw = df.get("VLA", np.nan)
        out["vla_deg"] = vla_raw.apply(parse_dir_value) if isinstance(vla_raw, pd.Series) else np.nan

        out["backspin_rpm"] = pd.to_numeric(df.get("BackSpin", df.get("Back Spin")), errors="coerce")

        spin_raw = df.get("Spin Axis", df.get("rawSpinAxis", df.get("SideSpin", np.nan)))
        out["spin_axis_deg"] = spin_raw.apply(parse_dir_value) if isinstance(spin_raw, pd.Series) else np.nan

        aoa_raw = df.get("AoA", df.get("Club AoA", np.nan))
        out["aoa_deg"] = aoa_raw.apply(parse_dir_value) if isinstance(aoa_raw, pd.Series) else np.nan

        path_raw = df.get("Path", df.get("Club Path", np.nan))
        out["club_path_deg"] = path_raw.apply(parse_dir_value) if isinstance(path_raw, pd.Series) else np.nan

        ftp_raw = df.get("FaceToPath", df.get("Face to Path", np.nan))
        out["face_to_path_deg"] = ftp_raw.apply(parse_dir_value) if isinstance(ftp_raw, pd.Series) else np.nan

        ftt_raw = df.get("FaceToTarget", df.get("Face to Target", np.nan))
        out["face_to_target_deg"] = ftt_raw.apply(parse_dir_value) if isinstance(ftt_raw, pd.Series) else np.nan

        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    out["club_id"] = out["club_raw"].apply(normalize_club_label)
    return out, fmt


# =========================================================
# Target model
# =========================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _interp(low_v: float, high_v: float, t: float) -> float:
    return low_v + (high_v - low_v) * t


PGA_BASELINES: Dict[str, Dict[str, float]] = {
    "DR": {"club_speed": 113, "ball_speed": 167, "launch": 10.9, "spin": 2686, "peak_height": 32.0, "descent": 38.0, "carry": 275},
    "3W": {"club_speed": 107, "ball_speed": 158, "launch": 9.2, "spin": 3655, "peak_height": 30.0, "descent": 43.0, "carry": 243},
    "5W": {"club_speed": 103, "ball_speed": 152, "launch": 9.4, "spin": 4350, "peak_height": 31.0, "descent": 47.0, "carry": 230},
    "7W": {"club_speed": 101, "ball_speed": 149, "launch": 11.0, "spin": 4550, "peak_height": 31.0, "descent": 48.0, "carry": 222},
    "2H": {"club_speed": 101, "ball_speed": 147, "launch": 9.8, "spin": 4300, "peak_height": 28.5, "descent": 46.0, "carry": 228},
    "3H": {"club_speed": 100, "ball_speed": 146, "launch": 10.2, "spin": 4437, "peak_height": 29.0, "descent": 47.0, "carry": 225},
    "4H": {"club_speed": 97, "ball_speed": 141, "launch": 11.5, "spin": 4700, "peak_height": 29.5, "descent": 48.0, "carry": 210},
    "5H": {"club_speed": 94, "ball_speed": 136, "launch": 12.8, "spin": 5000, "peak_height": 30.0, "descent": 49.0, "carry": 198},
    "3I": {"club_speed": 98, "ball_speed": 142, "launch": 10.4, "spin": 4630, "peak_height": 27.0, "descent": 46.0, "carry": 212},
    "4I": {"club_speed": 96, "ball_speed": 137, "launch": 11.0, "spin": 4836, "peak_height": 28.0, "descent": 48.0, "carry": 203},
    "5I": {"club_speed": 94, "ball_speed": 132, "launch": 12.1, "spin": 5361, "peak_height": 31.0, "descent": 49.0, "carry": 194},
    "6I": {"club_speed": 92, "ball_speed": 127, "launch": 14.1, "spin": 6231, "peak_height": 30.0, "descent": 50.0, "carry": 183},
    "7I": {"club_speed": 90, "ball_speed": 120, "launch": 16.3, "spin": 7097, "peak_height": 32.0, "descent": 50.0, "carry": 172},
    "8I": {"club_speed": 87, "ball_speed": 115, "launch": 18.1, "spin": 7998, "peak_height": 31.0, "descent": 50.0, "carry": 160},
    "9I": {"club_speed": 85, "ball_speed": 109, "launch": 20.4, "spin": 8647, "peak_height": 30.0, "descent": 51.0, "carry": 148},
    "PW": {"club_speed": 83, "ball_speed": 102, "launch": 24.2, "spin": 9304, "peak_height": 29.0, "descent": 52.0, "carry": 136},
    "GW": {"club_speed": 80, "ball_speed": 98, "launch": 26.5, "spin": 9800, "peak_height": 28.0, "descent": 53.0, "carry": 122},
    "SW": {"club_speed": 76, "ball_speed": 92, "launch": 29.0, "spin": 10200, "peak_height": 27.0, "descent": 55.0, "carry": 108},
    "LW": {"club_speed": 72, "ball_speed": 86, "launch": 31.5, "spin": 10600, "peak_height": 26.0, "descent": 57.0, "carry": 94},
}

LPGA_BASELINES: Dict[str, Dict[str, float]] = {
    "DR": {"club_speed": 94, "ball_speed": 140, "launch": 13.2, "spin": 2611, "peak_height": 25.0, "descent": 37.0, "carry": 218},
    "3W": {"club_speed": 90, "ball_speed": 132, "launch": 11.2, "spin": 2704, "peak_height": 23.0, "descent": 39.0, "carry": 195},
    "5W": {"club_speed": 88, "ball_speed": 128, "launch": 12.1, "spin": 4501, "peak_height": 26.0, "descent": 43.0, "carry": 185},
    "7W": {"club_speed": 85, "ball_speed": 123, "launch": 12.7, "spin": 4693, "peak_height": 25.0, "descent": 46.0, "carry": 174},
    "2H": {"club_speed": 83, "ball_speed": 121, "launch": 12.8, "spin": 4550, "peak_height": 24.5, "descent": 44.0, "carry": 178},
    "3H": {"club_speed": 82, "ball_speed": 119, "launch": 13.3, "spin": 4700, "peak_height": 24.5, "descent": 44.5, "carry": 174},
    "4H": {"club_speed": 80, "ball_speed": 116, "launch": 14.3, "spin": 4801, "peak_height": 24.0, "descent": 43.0, "carry": 169},
    "5H": {"club_speed": 79, "ball_speed": 112, "launch": 14.8, "spin": 5081, "peak_height": 23.0, "descent": 45.0, "carry": 161},
    "3I": {"club_speed": 81, "ball_speed": 118, "launch": 13.8, "spin": 4700, "peak_height": 24.0, "descent": 42.0, "carry": 174},
    "4I": {"club_speed": 80, "ball_speed": 116, "launch": 14.3, "spin": 4801, "peak_height": 24.0, "descent": 43.0, "carry": 169},
    "5I": {"club_speed": 79, "ball_speed": 112, "launch": 14.8, "spin": 5081, "peak_height": 23.0, "descent": 45.0, "carry": 161},
    "6I": {"club_speed": 78, "ball_speed": 109, "launch": 17.1, "spin": 5943, "peak_height": 25.0, "descent": 46.0, "carry": 152},
    "7I": {"club_speed": 76, "ball_speed": 104, "launch": 19.0, "spin": 6699, "peak_height": 26.0, "descent": 47.0, "carry": 141},
    "8I": {"club_speed": 74, "ball_speed": 100, "launch": 20.8, "spin": 7494, "peak_height": 25.0, "descent": 47.0, "carry": 130},
    "9I": {"club_speed": 72, "ball_speed": 93, "launch": 23.9, "spin": 7589, "peak_height": 26.0, "descent": 47.0, "carry": 119},
    "PW": {"club_speed": 70, "ball_speed": 86, "launch": 25.7, "spin": 8403, "peak_height": 23.0, "descent": 48.0, "carry": 107},
    "GW": {"club_speed": 67, "ball_speed": 82, "launch": 27.5, "spin": 9000, "peak_height": 23.0, "descent": 49.0, "carry": 95},
    "SW": {"club_speed": 63, "ball_speed": 77, "launch": 30.0, "spin": 9600, "peak_height": 22.0, "descent": 52.0, "carry": 82},
    "LW": {"club_speed": 59, "ball_speed": 71, "launch": 32.0, "spin": 10100, "peak_height": 21.0, "descent": 55.0, "carry": 70},
}


def _baseline_pair(club_id: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    if club_id in PGA_BASELINES and club_id in LPGA_BASELINES:
        return PGA_BASELINES[club_id], LPGA_BASELINES[club_id]

    fam = club_family(club_id)
    if fam == "Fairway Wood":
        return PGA_BASELINES["5W"], LPGA_BASELINES["5W"]
    if fam == "Hybrid":
        return PGA_BASELINES["4H"], LPGA_BASELINES["4H"]
    if fam == "Iron":
        return PGA_BASELINES["7I"], LPGA_BASELINES["7I"]
    if fam == "Wedge":
        return PGA_BASELINES["PW"], LPGA_BASELINES["PW"]
    return PGA_BASELINES["DR"], LPGA_BASELINES["DR"]


def _speed_factor(club_speed: float, lpga_speed: float, pga_speed: float) -> float:
    if _is_nan(club_speed):
        return 0.5
    if pga_speed <= lpga_speed:
        return 0.5
    raw = (club_speed - lpga_speed) / (pga_speed - lpga_speed)
    return _clamp(raw, -0.20, 1.20)


def interpolated_targets_for_club(club_id: str, club_speed_mph: float) -> Dict[str, float]:
    pga, lpga = _baseline_pair(club_id)
    t = _speed_factor(club_speed_mph, lpga["club_speed"], pga["club_speed"])

    out: Dict[str, float] = {}
    for k in ["ball_speed", "launch", "spin", "peak_height", "descent", "carry"]:
        out[k] = round(_interp(lpga[k], pga[k], t), 1)
    return out


def metric_window(club_id: str, metric: str, target_value: float) -> Tuple[float, float]:
    fam = club_family(club_id)

    if metric == "launch":
        tol = 1.5 if fam == "Wedge" else 1.8
        return (target_value - tol, target_value + tol)

    if metric == "spin":
        if fam == "Driver":
            tol = 400.0
        elif fam in {"Fairway Wood", "Hybrid"}:
            tol = 550.0
        elif fam == "Iron":
            tol = 800.0
        else:
            tol = 1000.0
        return (target_value - tol, target_value + tol)

    if metric == "peak_height":
        tol = 3.0 if fam in {"Driver", "Fairway Wood", "Hybrid"} else 2.5
        return (target_value - tol, target_value + tol)

    if metric == "descent":
        if fam == "Driver":
            return (35.0, 40.0)
        if fam in {"Fairway Wood", "Hybrid"}:
            return (43.0, 50.0)
        if fam == "Iron":
            return (45.0, 52.0)
        return (48.0, 58.0)

    if metric == "ball_speed":
        tol = 3.0 if fam == "Driver" else 2.5
        return (target_value - tol, target_value + tol)

    if metric == "carry":
        tol = 7.0 if fam == "Driver" else 6.0
        return (target_value - tol, target_value + tol)

    return (target_value, target_value)


def targets_for_club(club_id: str, club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
    tgt = interpolated_targets_for_club(club_id, club_speed_mph)
    launch_lo, launch_hi = metric_window(club_id, "launch", tgt["launch"])
    spin_lo, spin_hi = metric_window(club_id, "spin", tgt["spin"])
    return {
        "launch": (round(launch_lo, 1), round(launch_hi, 1)),
        "spin": (round(spin_lo, 0), round(spin_hi, 0)),
    }


# =========================================================
# Anti-overreaction tuning profiles + helpers
# =========================================================
@dataclass(frozen=True)
class ClubTuningProfile:
    launch_buffer_deg: float
    spin_buffer_rpm: float
    peak_buffer_yd: float
    descent_buffer_deg: float
    ball_speed_buffer_mph: float
    min_spec_signals: int
    setup_good_min_checks: int
    require_peak_descent_for_good_gate: bool


def tuning_profile_for_club(club_id: str) -> ClubTuningProfile:
    fam = club_family(club_id)
    if fam == "Hybrid":
        return ClubTuningProfile(1.2, 450.0, 3.0, 2.0, 2.0, 2, 4, True)
    if fam == "Fairway Wood":
        return ClubTuningProfile(1.0, 400.0, 2.5, 2.0, 2.0, 2, 4, True)
    if fam == "Driver":
        return ClubTuningProfile(0.8, 300.0, 2.5, 1.5, 2.0, 2, 3, False)
    return ClubTuningProfile(0.8, 350.0, 2.0, 1.5, 2.0, 2, 3, False)


def _buffered_ok(value: float, low: float, high: float, buffer_amt: float) -> bool:
    if np.isnan(value):
        return False
    return (low - buffer_amt) <= value <= (high + buffer_amt)


# =========================================================
# Hosel estimate with carry / height projection
# =========================================================
LAUNCH_FROM_DYNAMIC_WEIGHT = 0.85


@dataclass(frozen=True)
class LaunchSpinEstimate:
    launch_change_deg: float
    launch_range_deg: Tuple[float, float]
    spin_change_rpm: int
    spin_range_rpm: Tuple[int, int]
    carry_change_yd: float
    carry_range_yd: Tuple[float, float]
    peak_height_change_yd: float
    peak_height_range_yd: Tuple[float, float]
    notes: str


def _carry_center_per_loft(club_id: str) -> float:
    fam = club_family(club_id)
    if fam == "Driver":
        return 4.0
    if fam == "Fairway Wood":
        return 3.5
    if fam == "Hybrid":
        return 3.0
    return 2.5


def _peak_height_center_per_loft(club_id: str) -> float:
    fam = club_family(club_id)
    if fam == "Driver":
        return 2.5
    if fam == "Fairway Wood":
        return 3.0
    if fam == "Hybrid":
        return 3.0
    return 2.0


def estimate_launch_spin_change(
    delta_static_loft_deg: float,
    k_loft_to_dynamic: float,
    club_id: str,
) -> LaunchSpinEstimate:
    fam = club_family(club_id)

    launch_est = LAUNCH_FROM_DYNAMIC_WEIGHT * k_loft_to_dynamic * delta_static_loft_deg
    launch_unc = max(0.6, abs(launch_est) * 0.35)
    launch_low = launch_est - launch_unc
    launch_high = launch_est + launch_unc

    if fam == "Driver":
        center, lo, hi = 250, 150, 400
    elif fam == "Fairway Wood":
        center, lo, hi = 300, 180, 450
    elif fam == "Hybrid":
        center, lo, hi = 320, 200, 500
    else:
        center, lo, hi = 250, 150, 400

    spin_est = int(round(center * delta_static_loft_deg))
    spin_low = int(round(lo * delta_static_loft_deg))
    spin_high = int(round(hi * delta_static_loft_deg))
    spin_range = (min(spin_low, spin_high), max(spin_low, spin_high))

    carry_center = _carry_center_per_loft(club_id) * delta_static_loft_deg
    carry_unc = max(2.0, abs(carry_center) * 0.5)
    carry_range = (round(carry_center - carry_unc, 1), round(carry_center + carry_unc, 1))

    peak_center = _peak_height_center_per_loft(club_id) * delta_static_loft_deg
    peak_unc = max(1.5, abs(peak_center) * 0.5)
    peak_range = (round(peak_center - peak_unc, 1), round(peak_center + peak_unc, 1))

    return LaunchSpinEstimate(
        launch_change_deg=launch_est,
        launch_range_deg=(launch_low, launch_high),
        spin_change_rpm=spin_est,
        spin_range_rpm=spin_range,
        carry_change_yd=round(carry_center, 1),
        carry_range_yd=carry_range,
        peak_height_change_yd=round(peak_center, 1),
        peak_height_range_yd=peak_range,
        notes="These are directional estimates only. Actual changes still depend on strike quality, speed retention, and delivery.",
    )


# =========================================================
# Recommendation helpers
# =========================================================
def miss_tendency(offline_avg: float) -> str:
    if np.isnan(offline_avg):
        return "Unknown"
    if offline_avg > 5:
        return "Right miss tendency"
    if offline_avg < -5:
        return "Left miss tendency"
    return "Centered"


def _smash_floor_driver(club_speed_avg: float) -> float:
    if _is_nan(club_speed_avg):
        return 1.45
    if club_speed_avg < 90:
        return 1.44
    if club_speed_avg < 100:
        return 1.46
    if club_speed_avg < 110:
        return 1.48
    return 1.49


def smash_flag_driver(smash_avg: float, club_speed_avg: Optional[float] = None) -> Optional[str]:
    if np.isnan(smash_avg):
        return None

    floor = _smash_floor_driver(float(club_speed_avg) if club_speed_avg is not None else float("nan"))

    if smash_avg < floor - 0.03:
        return f"Smash factor is low ({smash_avg:.2f}). Strike efficiency looks like a clear limiter."
    if smash_avg < floor:
        return f"Smash factor is slightly low ({smash_avg:.2f}). There may still be efficiency left on the table."
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
    scored = []
    for s in settings:
        d = translate_fn(brand, system_name, s, handedness)
        loft = getattr(d, "loft_deg", None)
        lie = getattr(d, "lie_deg", None)
        if loft is None or lie is None:
            continue

        score = abs(loft - needed_loft_delta) * 1.5 + abs(lie - needed_lie_delta)
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
        return {
            "type": "guidance",
            "message": f"Exact chart not encoded for this hosel. Guidance: {', '.join(direction)}.",
        }

    scored.sort(key=lambda x: x[0])
    score, setting, loft, lie = scored[0]
    return {
        "type": "exact",
        "current": current_setting,
        "recommended": {"setting": setting, "loft_delta": loft, "lie_delta": lie, "score": score},
    }


# =========================================================
# Shot shape detection
# =========================================================
@dataclass(frozen=True)
class ShotShapeSummary:
    start_line: str
    curve: str
    shape_label: str
    shot_count_used: int


def classify_shot_shape_row(row: pd.Series) -> str:
    hla = pd.to_numeric(pd.Series([row.get("hla_deg")]), errors="coerce").iloc[0]
    ftp = pd.to_numeric(pd.Series([row.get("face_to_path_deg")]), errors="coerce").iloc[0]
    spin_axis = pd.to_numeric(pd.Series([row.get("spin_axis_deg")]), errors="coerce").iloc[0]

    start = "center"
    if not np.isnan(hla):
        if hla >= 1.5:
            start = "right"
        elif hla <= -1.5:
            start = "left"

    curve = "straight"
    if not np.isnan(ftp):
        if ftp >= 1.5:
            curve = "fade"
        elif ftp <= -1.5:
            curve = "draw"
    elif not np.isnan(spin_axis):
        if spin_axis >= 3.0:
            curve = "fade"
        elif spin_axis <= -3.0:
            curve = "draw"

    if start == "right" and curve == "fade":
        return "Push Fade"
    if start == "right" and curve == "draw":
        return "Push Draw"
    if start == "left" and curve == "fade":
        return "Pull Fade"
    if start == "left" and curve == "draw":
        return "Pull Draw"
    if start == "right" and curve == "straight":
        return "Straight Block"
    if start == "left" and curve == "straight":
        return "Straight Pull"
    if start == "center" and curve == "fade":
        return "Straight Fade"
    if start == "center" and curve == "draw":
        return "Straight Draw"
    return "Straight"


def shot_shape_summary(df: pd.DataFrame) -> ShotShapeSummary:
    if df.empty:
        return ShotShapeSummary("Unknown", "Unknown", "Unknown", 0)

    labels = df.apply(classify_shot_shape_row, axis=1)
    valid = labels.dropna()
    if valid.empty:
        return ShotShapeSummary("Unknown", "Unknown", "Unknown", 0)

    mode_label = valid.value_counts().idxmax()

    start_map = {
        "Push Fade": "Right",
        "Push Draw": "Right",
        "Straight Block": "Right",
        "Pull Fade": "Left",
        "Pull Draw": "Left",
        "Straight Pull": "Left",
        "Straight Fade": "On line",
        "Straight Draw": "On line",
        "Straight": "On line",
    }
    curve_map = {
        "Push Fade": "Fade",
        "Pull Fade": "Fade",
        "Straight Fade": "Fade",
        "Push Draw": "Draw",
        "Pull Draw": "Draw",
        "Straight Draw": "Draw",
        "Straight Block": "Straight",
        "Straight Pull": "Straight",
        "Straight": "Straight",
    }

    return ShotShapeSummary(
        start_line=start_map.get(mode_label, "Unknown"),
        curve=curve_map.get(mode_label, "Unknown"),
        shape_label=mode_label,
        shot_count_used=int(len(valid)),
    )


# =========================================================
# Distance potential
# =========================================================
@dataclass(frozen=True)
class DistancePotential:
    expected_carry_yd: float
    actual_carry_yd: float
    carry_gap_yd: float
    status: str
    message: str


def distance_potential_for_summary(summary: "ClubSummary") -> DistancePotential:
    tgt = interpolated_targets_for_club(summary.club_id, summary.club_speed_avg)
    expected = float(tgt["carry"])
    actual = float(summary.carry_avg) if not np.isnan(summary.carry_avg) else float("nan")

    if np.isnan(actual):
        return DistancePotential(round(expected, 1), float("nan"), float("nan"), "unknown", "Not enough carry data to estimate distance potential.")

    gap = round(expected - actual, 1)
    if gap <= 3:
        status = "optimized"
        message = "Carry looks close to optimized for this speed."
    elif gap <= 8:
        status = "small_gap"
        message = f"You may be leaving about {gap:.0f} yards on the table."
    else:
        status = "meaningful_gap"
        message = f"You may be leaving meaningful distance on the table ({gap:.0f}+ yards)."

    return DistancePotential(round(expected, 1), round(actual, 1), gap, status, message)


# =========================================================
# Summary
# =========================================================
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
    peak_height_avg: float
    peak_height_std: float
    descent_avg: float
    descent_std: float
    hla_avg: float
    spin_axis_avg: float
    face_to_path_avg: float


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
            peak_height_avg=safe_mean(g["peak_height_yd"]),
            peak_height_std=safe_std(g["peak_height_yd"]),
            descent_avg=safe_mean(g["descent_deg"]),
            descent_std=safe_std(g["descent_deg"]),
            hla_avg=safe_mean(g["hla_deg"]),
            spin_axis_avg=safe_mean(g["spin_axis_deg"]),
            face_to_path_avg=safe_mean(g["face_to_path_deg"]),
        )
    return summaries


# =========================================================
# Recommendation dataclasses
# =========================================================
@dataclass
class DriverUserSetup:
    brand: str = "Titleist"
    model: str = "TSR3"
    loft_deg: float = 10.0
    hosel_setting: str = "A1"
    shaft_model: str = "HZRDUS Black"
    shaft_weight_g: float = 60.0
    shaft_flex: str = "6.0"


@dataclass
class RecommendationBlock:
    title: str
    suggestion: str
    why: str
    tone: str


@dataclass
class RecommendationBundle:
    swing: RecommendationBlock
    driver_settings: RecommendationBlock
    equipment_adjustment: RecommendationBlock
    debug: Dict[str, float]


@dataclass
class DriverCompareMetrics:
    label: str
    shots: int
    club_speed: float
    ball_speed: float
    smash: float
    carry: float
    total: float
    launch: float
    spin: float
    aoa: float
    offline: float
    fairway_pct: float


# =========================================================
# Compare helpers
# =========================================================
def driver_metrics_from_df(driver_df: pd.DataFrame, label: str) -> DriverCompareMetrics:
    d = driver_df.copy()
    if d.empty:
        return DriverCompareMetrics(label, 0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    offline_valid = pd.to_numeric(d["offline_yd"], errors="coerce").dropna()
    fairway_pct = float((offline_valid.abs() <= 15).mean() * 100.0) if len(offline_valid) else math.nan

    return DriverCompareMetrics(
        label=label,
        shots=int(len(d)),
        club_speed=safe_mean(d["club_speed_mph"]),
        ball_speed=safe_mean(d["ball_speed_mph"]),
        smash=safe_mean(d["smash"]),
        carry=safe_mean(d["carry_yd"]),
        total=safe_mean(d["total_yd"]),
        launch=safe_mean(d["vla_deg"]),
        spin=safe_mean(d["backspin_rpm"]),
        aoa=safe_mean(d["aoa_deg"]),
        offline=safe_mean(d["offline_yd"].abs()),
        fairway_pct=fairway_pct,
    )


def compare_driver_setups(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    label_a: str = "Setup A",
    label_b: str = "Setup B",
) -> Dict[str, object]:
    a = driver_metrics_from_df(a_df, label_a)
    b = driver_metrics_from_df(b_df, label_b)

    def better_high(x: float, y: float) -> str:
        if np.isnan(x) and np.isnan(y):
            return "Tie"
        if np.isnan(y):
            return label_a
        if np.isnan(x):
            return label_b
        if abs(x - y) < 1e-9:
            return "Tie"
        return label_a if x > y else label_b

    def better_low(x: float, y: float) -> str:
        if np.isnan(x) and np.isnan(y):
            return "Tie"
        if np.isnan(y):
            return label_a
        if np.isnan(x):
            return label_b
        if abs(x - y) < 1e-9:
            return "Tie"
        return label_a if x < y else label_b

    score_a = 0
    score_b = 0
    if better_high(a.carry, b.carry) == label_a:
        score_a += 1
    elif better_high(a.carry, b.carry) == label_b:
        score_b += 1
    if better_high(a.ball_speed, b.ball_speed) == label_a:
        score_a += 1
    elif better_high(a.ball_speed, b.ball_speed) == label_b:
        score_b += 1
    if better_low(a.offline, b.offline) == label_a:
        score_a += 1
    elif better_low(a.offline, b.offline) == label_b:
        score_b += 1
    if better_high(a.fairway_pct, b.fairway_pct) == label_a:
        score_a += 1
    elif better_high(a.fairway_pct, b.fairway_pct) == label_b:
        score_b += 1

    overall = "Tie"
    if score_a > score_b:
        overall = label_a
    elif score_b > score_a:
        overall = label_b

    return {
        "a": a,
        "b": b,
        "winners": {
            "longest_carry": better_high(a.carry, b.carry),
            "fastest_ball_speed": better_high(a.ball_speed, b.ball_speed),
            "straightest": better_low(a.offline, b.offline),
            "most_fairways": better_high(a.fairway_pct, b.fairway_pct),
            "best_overall": overall,
        },
    }


# =========================================================
# Recommendation internals
# =========================================================
def _classify(value: float, low: float, high: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "ok"


def _driver_swing_note(summary: ClubSummary, ball_speed_low: bool, launch_off: bool) -> RecommendationBlock:
    if ball_speed_low:
        return RecommendationBlock(
            title="Swing",
            suggestion="Focus on strike consistency before making a major spec change.",
            why="The biggest limiter here looks more like strike efficiency than club spec.",
            tone="yellow",
        )
    if summary.vla_std >= 2.5 or summary.spin_std >= 900 or summary.peak_height_std >= 4.0:
        return RecommendationBlock(
            title="Swing",
            suggestion="Keep the motion simple and prioritize repeating strike and start line.",
            why="The variation here looks more delivery-driven than equipment-driven.",
            tone="yellow",
        )
    if launch_off:
        return RecommendationBlock(
            title="Swing",
            suggestion="Stay with a repeatable motion and avoid over-chasing a mechanical change.",
            why="This flight window often relates to delivered loft and strike location as much as equipment.",
            tone="green",
        )
    return RecommendationBlock(
        title="Swing",
        suggestion="Your current swing pattern looks playable enough to fit around.",
        why="FitCaddie should fit the swing you have, not assume you need a swing rebuild.",
        tone="green",
    )


def _non_driver_swing_note(summary: ClubSummary, family: str, ball_speed_low: bool) -> RecommendationBlock:
    if ball_speed_low:
        return RecommendationBlock(
            title="Swing",
            suggestion="Prioritize centered contact before making a major spec change.",
            why=f"This {family.lower()} looks more limited by strike quality than by equipment alone.",
            tone="yellow",
        )
    if summary.vla_std >= 2.5 or summary.spin_std >= 900 or summary.peak_height_std >= 4.0:
        return RecommendationBlock(
            title="Swing",
            suggestion="Stay with a stable motion and focus on repeatable delivery.",
            why="The pattern looks more delivery-driven than equipment-driven.",
            tone="yellow",
        )
    return RecommendationBlock(
        title="Swing",
        suggestion="No major swing note stands out here.",
        why="The current motion looks consistent enough to judge equipment around it.",
        tone="green",
    )


def _bias_from_face_delivery(summary: ClubSummary) -> str:
    if np.isnan(summary.hla_avg) or np.isnan(summary.spin_axis_avg):
        if np.isnan(summary.offline_avg):
            return "neutral"
        if summary.offline_avg >= 7:
            return "right"
        if summary.offline_avg <= -7:
            return "left"
        return "neutral"

    if summary.hla_avg > 1.0 and summary.spin_axis_avg > 2.0:
        return "right"
    if summary.hla_avg > 1.0 and summary.spin_axis_avg < -2.0:
        return "right"
    if summary.hla_avg < -1.0 and summary.spin_axis_avg < -2.0:
        return "left"
    if summary.hla_avg < -1.0 and summary.spin_axis_avg > 2.0:
        return "left"

    if summary.hla_avg > 1.5:
        return "right"
    if summary.hla_avg < -1.5:
        return "left"
    return "neutral"


def _spec_signal_counts(
    summary: ClubSummary,
    club_id: str,
    launch_lo: float,
    launch_hi: float,
    spin_lo: float,
    spin_hi: float,
    peak_lo: float,
    peak_hi: float,
    desc_lo: float,
    desc_hi: float,
) -> Dict[str, int]:
    fam = club_family(club_id)
    low_flight = 0
    high_flight = 0
    right_bias = 0
    left_bias = 0

    if not np.isnan(summary.vla_avg) and summary.vla_avg < launch_lo:
        low_flight += 1
    if not np.isnan(summary.spin_avg) and summary.spin_avg < spin_lo:
        low_flight += 1
    if fam in {"Fairway Wood", "Hybrid", "Iron", "Wedge"}:
        if not np.isnan(summary.peak_height_avg) and summary.peak_height_avg < peak_lo:
            low_flight += 1
        if not np.isnan(summary.descent_avg) and summary.descent_avg < desc_lo:
            low_flight += 1

    if not np.isnan(summary.vla_avg) and summary.vla_avg > launch_hi:
        high_flight += 1
    if not np.isnan(summary.spin_avg) and summary.spin_avg > spin_hi:
        high_flight += 1
    if fam in {"Fairway Wood", "Hybrid", "Iron", "Wedge"}:
        if not np.isnan(summary.peak_height_avg) and summary.peak_height_avg > peak_hi:
            high_flight += 1
        if not np.isnan(summary.descent_avg) and summary.descent_avg > desc_hi:
            high_flight += 1

    bias = _bias_from_face_delivery(summary)
    if bias == "right":
        right_bias += 2
    elif bias == "left":
        left_bias += 2

    return {
        "low_flight": low_flight,
        "high_flight": high_flight,
        "right_bias": right_bias,
        "left_bias": left_bias,
    }


def _current_setup_good_eval(
    summary: ClubSummary,
    club_id: str,
    launch_lo: float,
    launch_hi: float,
    spin_lo: float,
    spin_hi: float,
    peak_lo: float,
    peak_hi: float,
    desc_lo: float,
    desc_hi: float,
    bs_lo: float,
    bs_hi: float,
) -> Tuple[bool, List[str]]:
    profile = tuning_profile_for_club(club_id)
    fam = club_family(club_id)
    reasons: List[str] = []

    launch_ok = _buffered_ok(summary.vla_avg, launch_lo, launch_hi, profile.launch_buffer_deg)
    spin_ok = _buffered_ok(summary.spin_avg, spin_lo, spin_hi, profile.spin_buffer_rpm)
    peak_ok = _buffered_ok(summary.peak_height_avg, peak_lo, peak_hi, profile.peak_buffer_yd)
    descent_ok = _buffered_ok(summary.descent_avg, desc_lo, desc_hi, profile.descent_buffer_deg)
    ball_speed_ok = _buffered_ok(summary.ball_speed_avg, bs_lo, bs_hi, profile.ball_speed_buffer_mph)

    if launch_ok:
        reasons.append("launch is in a playable window")
    if spin_ok:
        reasons.append("spin is in a playable window")
    if ball_speed_ok or np.isnan(summary.ball_speed_avg):
        reasons.append("ball speed is reasonable for the current speed")

    if fam == "Driver":
        if not np.isnan(summary.offline_std) and summary.offline_std <= 16:
            reasons.append("dispersion is playable")
        if not np.isnan(summary.smash_avg) and summary.smash_avg >= _smash_floor_driver(summary.club_speed_avg) - 0.02:
            reasons.append("strike efficiency looks acceptable")
    else:
        if not np.isnan(summary.offline_std) and summary.offline_std <= 14:
            reasons.append("dispersion is playable")
        if not np.isnan(summary.smash_avg) and summary.smash_avg >= 1.30:
            reasons.append("strike efficiency looks acceptable")

    if profile.require_peak_descent_for_good_gate:
        if peak_ok:
            reasons.append("peak height is playable")
        if descent_ok:
            reasons.append("descent angle is playable")
    elif peak_ok:
        reasons.append("peak height is playable")

    return (len(reasons) >= profile.setup_good_min_checks), reasons


def _settings_block(
    club_id: str,
    hosel_setting: Optional[str],
    summary: ClubSummary,
    launch_lo: float,
    launch_hi: float,
    spin_lo: float,
    spin_hi: float,
    peak_lo: float,
    peak_hi: float,
    desc_lo: float,
    desc_hi: float,
    current_setup_good: bool,
) -> RecommendationBlock:
    family = club_family(club_id)
    adjustable = family in {"Driver", "Fairway Wood", "Hybrid"}
    current_txt = f" from your current {hosel_setting} setting" if hosel_setting else ""
    profile = tuning_profile_for_club(club_id)

    if not adjustable:
        return RecommendationBlock(
            title="Club Settings",
            suggestion="No hosel-first change applies here.",
            why=f"This {family.lower()} is less about adjustable hosel settings and more about loft, build, and flight window.",
            tone="green",
        )
    if current_setup_good:
        return RecommendationBlock(
            title="Club Settings",
            suggestion=f"Stay with the current setting{current_txt} for now.",
            why="The flight is already in a playable enough window that a hosel change is not strongly justified.",
            tone="green",
        )

    signal_counts = _spec_signal_counts(summary, club_id, launch_lo, launch_hi, spin_lo, spin_hi, peak_lo, peak_hi, desc_lo, desc_hi)

    if signal_counts["low_flight"] >= profile.min_spec_signals:
        return RecommendationBlock(
            title="Club Settings",
            suggestion=f"Test a slightly higher-loft or more upright hosel setting{current_txt} before changing shafts.",
            why="Multiple flight signals point to a flatter-than-ideal flight, so hosel is the easiest and most reversible first change.",
            tone="green",
        )
    if signal_counts["high_flight"] >= profile.min_spec_signals:
        return RecommendationBlock(
            title="Club Settings",
            suggestion=f"Test a slightly lower-loft or flatter hosel setting{current_txt} before changing shafts.",
            why="Multiple flight signals point to a higher-than-ideal flight, so a small hosel change is the safest first lever.",
            tone="yellow",
        )
    if signal_counts["right_bias"] >= profile.min_spec_signals:
        return RecommendationBlock(
            title="Club Settings",
            suggestion=f"If your hosel allows it, test a slightly more upright or draw-friendlier setting{current_txt}.",
            why="The miss pattern has enough right-bias signal to justify a small reversible setting test.",
            tone="green",
        )
    if signal_counts["left_bias"] >= profile.min_spec_signals:
        return RecommendationBlock(
            title="Club Settings",
            suggestion=f"If your hosel allows it, test a slightly flatter or fade-friendlier setting{current_txt}.",
            why="The miss pattern has enough left-bias signal to justify a small reversible setting test.",
            tone="yellow",
        )
    return RecommendationBlock(
        title="Club Settings",
        suggestion=f"Stay with the current setting{current_txt} for now.",
        why="A single number may be imperfect, but there is not enough agreement across the flight pattern to justify a hosel change.",
        tone="green",
    )



def _delivery_consistency_ok(summary: ClubSummary, family: str) -> bool:
    if family == "Driver":
        return (
            (np.isnan(summary.vla_std) or summary.vla_std <= 2.4)
            and (np.isnan(summary.spin_std) or summary.spin_std <= 850)
            and (np.isnan(summary.offline_std) or summary.offline_std <= 16)
        )
    return (
        (np.isnan(summary.vla_std) or summary.vla_std <= 2.2)
        and (np.isnan(summary.spin_std) or summary.spin_std <= 800)
        and (np.isnan(summary.offline_std) or summary.offline_std <= 14)
    )


def _strike_efficiency_ok(summary: ClubSummary, family: str) -> bool:
    if np.isnan(summary.smash_avg):
        return False
    if family == "Driver":
        return summary.smash_avg >= _smash_floor_driver(summary.club_speed_avg) - 0.02
    if family in {"Fairway Wood", "Hybrid"}:
        return summary.smash_avg >= 1.38
    if family == "Iron":
        return summary.smash_avg >= 1.30
    return summary.smash_avg >= 1.20


def _head_or_shaft_direction_driver(
    summary: ClubSummary,
    signal_counts: Dict[str, int],
    current_setup_good: bool,
    ball_speed_status: str,
    fairway_hit_pct: Optional[float],
) -> RecommendationBlock:
    family = "Driver"
    strike_ok = _strike_efficiency_ok(summary, family)
    consistency_ok = _delivery_consistency_ok(summary, family)

    if current_setup_good:
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Current driver setup looks good. No strong head or shaft change stands out from this sample.",
            why="The launch, spin, and playable flight window are close enough that FitCaddie would leave this alone for now.",
            tone="green",
        )

    if not strike_ok and ball_speed_status == "low":
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="No clear head-or-shaft recommendation yet. Check strike and settings first.",
            why="Strike efficiency is not strong enough yet to cleanly isolate whether the bigger issue is the head, the shaft, or simply impact quality.",
            tone="yellow",
        )

    if fairway_hit_pct is not None and not np.isnan(fairway_hit_pct) and fairway_hit_pct < 60:
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Head-first test: a more forgiving driver head or more stable total build may help more than a shaft-only change.",
            why="The bigger miss pattern is playable control, which usually points toward forgiveness and head/build stability before a profile-only shaft change.",
            tone="yellow",
        )

    if signal_counts["low_flight"] >= 3:
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Head-first test: look for more launch help from loft/head design before assuming the shaft is the main issue.",
            why="The flight is coming out too flat in multiple ways, and that usually points more toward head/loft help than a pure shaft answer.",
            tone="green",
        )

    if signal_counts["high_flight"] >= 2 and strike_ok and consistency_ok:
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Shaft-first test: after hosel testing, a firmer or lower-launch/lower-spin shaft profile could be worth testing.",
            why="Strike and delivery look stable enough that the remaining issue looks more like flight-profile tuning than a head forgiveness problem.",
            tone="yellow",
        )

    if signal_counts["low_flight"] >= 2 and strike_ok and consistency_ok:
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Shaft-second, head-check path: test hosel first, then compare a slightly higher-launch shaft only if the head/loft window still looks too flat.",
            why="There is a real low-flight pattern, but not enough evidence yet to blame the shaft before settings and head/loft are checked.",
            tone="green",
        )

    return RecommendationBlock(
        title="Equipment Direction",
        suggestion="No isolated head-or-shaft recommendation yet. Test settings first and collect a few more shots.",
        why="The pattern is not clean enough yet to say the head or shaft is the main lever.",
        tone="green",
    )


def _head_or_shaft_direction_non_driver(
    summary: ClubSummary,
    family: str,
    signal_counts: Dict[str, int],
    current_setup_good: bool,
    ball_speed_status: str,
    shaft_weight_g: Optional[float],
) -> RecommendationBlock:
    strike_ok = _strike_efficiency_ok(summary, family)
    consistency_ok = _delivery_consistency_ok(summary, family)

    if current_setup_good:
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Current setup looks good. No strong head or shaft change stands out from this data.",
            why="The club is already in a playable enough flight window for its speed and intended job.",
            tone="green",
        )

    if not strike_ok and ball_speed_status == "low":
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="No clear head-or-shaft recommendation yet. Check strike and settings first.",
            why="Strike efficiency is not strong enough yet to cleanly separate a head issue from a shaft issue.",
            tone="yellow",
        )

    if family in {"Fairway Wood", "Hybrid"}:
        if signal_counts["low_flight"] >= 3:
            return RecommendationBlock(
                title="Equipment Direction",
                suggestion="Head-first test: this looks more like a loft/head-launch issue than a shaft-first issue.",
                why="The flight is too flat in multiple ways, and for woods/hybrids that usually means more help from loft or head design before shaft fine-tuning.",
                tone="green",
            )

        if signal_counts["high_flight"] >= 2 and strike_ok and consistency_ok:
            return RecommendationBlock(
                title="Equipment Direction",
                suggestion="Shaft-first test: after hosel testing, a firmer or lower-launch profile could be worth testing.",
                why="The remaining issue looks more like profile control than a need for more head help.",
                tone="yellow",
            )

        if signal_counts["low_flight"] >= 2 and strike_ok and consistency_ok:
            return RecommendationBlock(
                title="Equipment Direction",
                suggestion="Head-first, then shaft path: test more loft or a higher-launching head first, then only test shaft if the club still flies too flat.",
                why="For fairway woods and hybrids, peak height and descent usually point to head/loft fit before shaft fit.",
                tone="green",
            )

        if shaft_weight_g is not None and not np.isnan(float(shaft_weight_g)) and summary.offline_avg > 5:
            return RecommendationBlock(
                title="Equipment Direction",
                suggestion="Shaft-first test: a slightly heavier or more stable shaft build may be worth testing.",
                why="The flight window is not the main problem here; playable control and timing look like the bigger issue.",
                tone="green",
            )

        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="No isolated head-or-shaft recommendation yet. Test settings first and collect a few more shots.",
            why="This sample does not cleanly separate a head issue from a shaft issue yet.",
            tone="green",
        )

    if family == "Iron":
        if signal_counts["low_flight"] >= 2:
            return RecommendationBlock(
                title="Equipment Direction",
                suggestion="Head-first test: this iron flight looks more like a loft/head-launch issue than a shaft-first issue.",
                why="When irons fly too flat, stopping power usually points more toward loft, head design, or launch help than a profile-only shaft answer.",
                tone="yellow",
            )
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="No strong head-or-shaft recommendation stands out yet.",
            why="The sample does not show a clean enough pattern to isolate the equipment answer.",
            tone="green",
        )

    if family == "Wedge":
        return RecommendationBlock(
            title="Equipment Direction",
            suggestion="Head/loft and strike matter more than shaft here.",
            why="For wedges, launch, spin, loft, groove condition, and strike usually explain more than shaft profile does.",
            tone="yellow",
        )

    return RecommendationBlock(
        title="Equipment Direction",
        suggestion="No strong head-or-shaft recommendation stands out yet.",
        why="The pattern is not clean enough yet to isolate the equipment answer.",
        tone="green",
    )

def _equipment_block_driver(
    summary: ClubSummary,
    user_setup: DriverUserSetup,
    current_setup_good: bool,
    launch_lo: float,
    launch_hi: float,
    spin_lo: float,
    spin_hi: float,
    peak_lo: float,
    peak_hi: float,
    desc_lo: float,
    desc_hi: float,
    ball_speed_status: str,
    fairway_hit_pct: Optional[float],
) -> RecommendationBlock:
    signal_counts = _spec_signal_counts(summary, "DR", launch_lo, launch_hi, spin_lo, spin_hi, peak_lo, peak_hi, desc_lo, desc_hi)
    return _head_or_shaft_direction_driver(
        summary=summary,
        signal_counts=signal_counts,
        current_setup_good=current_setup_good,
        ball_speed_status=ball_speed_status,
        fairway_hit_pct=fairway_hit_pct,
    )


def _equipment_block_non_driver(
    summary: ClubSummary,
    family: str,
    stated_loft_deg: Optional[float],
    shaft_weight_g: Optional[float],
    hosel_setting: Optional[str],
    current_setup_good: bool,
    launch_lo: float,
    launch_hi: float,
    spin_lo: float,
    spin_hi: float,
    peak_lo: float,
    peak_hi: float,
    desc_lo: float,
    desc_hi: float,
    ball_speed_status: str,
) -> RecommendationBlock:
    signal_counts = _spec_signal_counts(summary, summary.club_id, launch_lo, launch_hi, spin_lo, spin_hi, peak_lo, peak_hi, desc_lo, desc_hi)
    return _head_or_shaft_direction_non_driver(
        summary=summary,
        family=family,
        signal_counts=signal_counts,
        current_setup_good=current_setup_good,
        ball_speed_status=ball_speed_status,
        shaft_weight_g=shaft_weight_g,
    )


# =========================================================
# Driver recommendation engine
# =========================================================
def build_driver_recommendations(
    summary: ClubSummary,
    user_setup: DriverUserSetup,
    fairway_hit_pct: Optional[float] = None,
) -> RecommendationBundle:
    tgt = interpolated_targets_for_club("DR", summary.club_speed_avg)

    launch_lo, launch_hi = metric_window("DR", "launch", tgt["launch"])
    spin_lo, spin_hi = metric_window("DR", "spin", tgt["spin"])
    peak_lo, peak_hi = metric_window("DR", "peak_height", tgt["peak_height"])
    desc_lo, desc_hi = metric_window("DR", "descent", tgt["descent"])
    bs_lo, bs_hi = metric_window("DR", "ball_speed", tgt["ball_speed"])

    ball_speed_status = _classify(summary.ball_speed_avg, bs_lo, bs_hi)

    current_setup_good, good_reasons = _current_setup_good_eval(
        summary, "DR", launch_lo, launch_hi, spin_lo, spin_hi, peak_lo, peak_hi, desc_lo, desc_hi, bs_lo, bs_hi
    )

    swing = _driver_swing_note(summary=summary, ball_speed_low=(ball_speed_status == "low"), launch_off=not current_setup_good)

    driver_settings = _settings_block(
        club_id="DR",
        hosel_setting=user_setup.hosel_setting,
        summary=summary,
        launch_lo=launch_lo,
        launch_hi=launch_hi,
        spin_lo=spin_lo,
        spin_hi=spin_hi,
        peak_lo=peak_lo,
        peak_hi=peak_hi,
        desc_lo=desc_lo,
        desc_hi=desc_hi,
        current_setup_good=current_setup_good,
    )

    equipment_adjustment = _equipment_block_driver(
        summary=summary,
        user_setup=user_setup,
        current_setup_good=current_setup_good,
        launch_lo=launch_lo,
        launch_hi=launch_hi,
        spin_lo=spin_lo,
        spin_hi=spin_hi,
        peak_lo=peak_lo,
        peak_hi=peak_hi,
        desc_lo=desc_lo,
        desc_hi=desc_hi,
        ball_speed_status=ball_speed_status,
        fairway_hit_pct=fairway_hit_pct,
    )

    return RecommendationBundle(
        swing=swing,
        driver_settings=driver_settings,
        equipment_adjustment=equipment_adjustment,
        debug={
            "club_speed": summary.club_speed_avg,
            "ball_speed": summary.ball_speed_avg,
            "smash": summary.smash_avg,
            "carry": summary.carry_avg,
            "offline_avg": summary.offline_avg,
            "launch": summary.vla_avg,
            "spin": summary.spin_avg,
            "aoa": summary.aoa_avg,
            "peak_height": summary.peak_height_avg,
            "descent": summary.descent_avg,
            "target_launch": tgt["launch"],
            "target_spin": tgt["spin"],
            "target_peak_height": tgt["peak_height"],
            "target_descent": tgt["descent"],
            "fairway_hit_pct": fairway_hit_pct if fairway_hit_pct is not None else math.nan,
            "current_setup_good": float(current_setup_good),
            "good_reason_count": float(len(good_reasons)),
            "hla": summary.hla_avg,
            "spin_axis": summary.spin_axis_avg,
            "face_to_path": summary.face_to_path_avg,
        },
    )


# =========================================================
# Non-driver recommendation engine
# =========================================================
def build_non_driver_recommendations(
    summary: ClubSummary,
    stated_loft_deg: Optional[float] = None,
    brand: Optional[str] = None,
    model: Optional[str] = None,
    shaft_model: Optional[str] = None,
    shaft_weight_g: Optional[float] = None,
    shaft_flex: Optional[str] = None,
    hosel_setting: Optional[str] = None,
) -> RecommendationBundle:
    club_id = summary.club_id
    family = club_family(club_id)
    tgt = interpolated_targets_for_club(club_id, summary.club_speed_avg)

    launch_lo, launch_hi = metric_window(club_id, "launch", tgt["launch"])
    spin_lo, spin_hi = metric_window(club_id, "spin", tgt["spin"])
    peak_lo, peak_hi = metric_window(club_id, "peak_height", tgt["peak_height"])
    desc_lo, desc_hi = metric_window(club_id, "descent", tgt["descent"])
    bs_lo, bs_hi = metric_window(club_id, "ball_speed", tgt["ball_speed"])

    ball_speed_status = _classify(summary.ball_speed_avg, bs_lo, bs_hi)

    current_setup_good, good_reasons = _current_setup_good_eval(
        summary, club_id, launch_lo, launch_hi, spin_lo, spin_hi, peak_lo, peak_hi, desc_lo, desc_hi, bs_lo, bs_hi
    )

    swing = _non_driver_swing_note(summary=summary, family=family, ball_speed_low=(ball_speed_status == "low"))

    settings = _settings_block(
        club_id=club_id,
        hosel_setting=hosel_setting,
        summary=summary,
        launch_lo=launch_lo,
        launch_hi=launch_hi,
        spin_lo=spin_lo,
        spin_hi=spin_hi,
        peak_lo=peak_lo,
        peak_hi=peak_hi,
        desc_lo=desc_lo,
        desc_hi=desc_hi,
        current_setup_good=current_setup_good,
    )

    equipment = _equipment_block_non_driver(
        summary=summary,
        family=family,
        stated_loft_deg=stated_loft_deg,
        shaft_weight_g=shaft_weight_g,
        hosel_setting=hosel_setting,
        current_setup_good=current_setup_good,
        launch_lo=launch_lo,
        launch_hi=launch_hi,
        spin_lo=spin_lo,
        spin_hi=spin_hi,
        peak_lo=peak_lo,
        peak_hi=peak_hi,
        desc_lo=desc_lo,
        desc_hi=desc_hi,
        ball_speed_status=ball_speed_status,
    )

    return RecommendationBundle(
        swing=swing,
        driver_settings=settings,
        equipment_adjustment=equipment,
        debug={
            "club_speed": summary.club_speed_avg,
            "ball_speed": summary.ball_speed_avg,
            "smash": summary.smash_avg,
            "carry": summary.carry_avg,
            "offline_avg": summary.offline_avg,
            "launch": summary.vla_avg,
            "spin": summary.spin_avg,
            "aoa": summary.aoa_avg,
            "peak_height": summary.peak_height_avg,
            "descent": summary.descent_avg,
            "target_launch": tgt["launch"],
            "target_spin": tgt["spin"],
            "target_peak_height": tgt["peak_height"],
            "target_descent": tgt["descent"],
            "current_setup_good": float(current_setup_good),
            "good_reason_count": float(len(good_reasons)),
            "hla": summary.hla_avg,
            "spin_axis": summary.spin_axis_avg,
            "face_to_path": summary.face_to_path_avg,
        },
    )
