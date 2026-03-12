from __future__ import annotations

import math
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
# Club normalization
# -----------------------------
def normalize_club_label(label: str) -> str:
    if not isinstance(label, str):
        return "OTHER"

    s = label.strip().upper()

    if s == "DR":
        return "DR"

    if s.startswith("W") and len(s) == 2:
        return f"{s[1]}W"

    if s.startswith("H") and len(s) == 2:
        return f"{s[1]}H"

    if s.startswith("I") and len(s) == 2:
        return f"{s[1]}I"

    if s in ["PW", "GW", "SW", "LW"]:
        return s

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
        out["offline_yd"] = df.get("Offline", df.get("Offline (yd)")).apply(parse_dir_value)
        out["peak_height_yd"] = pd.to_numeric(df.get("PeakHeight", df.get("Peak Height (yd)")), errors="coerce")
        out["descent_deg"] = df.get("Decent", df.get("Desc Angle", np.nan)).apply(parse_dir_value)
        out["hla_deg"] = df.get("HLA", np.nan).apply(parse_dir_value) if "HLA" in df.columns else np.nan
        out["vla_deg"] = df.get("VLA", np.nan).apply(parse_dir_value) if "VLA" in df.columns else np.nan
        out["backspin_rpm"] = pd.to_numeric(df.get("BackSpin", df.get("Back Spin")), errors="coerce")
        out["spin_axis_deg"] = df.get("Spin Axis", df.get("rawSpinAxis", df.get("SideSpin", np.nan)))
        if isinstance(out["spin_axis_deg"], pd.Series):
            out["spin_axis_deg"] = out["spin_axis_deg"].apply(parse_dir_value)
        out["aoa_deg"] = df.get("AoA", df.get("Club AoA", np.nan))
        if isinstance(out["aoa_deg"], pd.Series):
            out["aoa_deg"] = out["aoa_deg"].apply(parse_dir_value)
        out["club_path_deg"] = df.get("Path", df.get("Club Path", np.nan))
        if isinstance(out["club_path_deg"], pd.Series):
            out["club_path_deg"] = out["club_path_deg"].apply(parse_dir_value)
        out["face_to_path_deg"] = df.get("FaceToPath", df.get("Face to Path", np.nan))
        if isinstance(out["face_to_path_deg"], pd.Series):
            out["face_to_path_deg"] = out["face_to_path_deg"].apply(parse_dir_value)
        out["face_to_target_deg"] = df.get("FaceToTarget", df.get("Face to Target", np.nan))
        if isinstance(out["face_to_target_deg"], pd.Series):
            out["face_to_target_deg"] = out["face_to_target_deg"].apply(parse_dir_value)
        out["smash"] = (out["ball_speed_mph"] / out["club_speed_mph"]).replace([np.inf, -np.inf], np.nan)

    out["club_id"] = out["club_raw"].apply(normalize_club_label)
    return out, fmt


# -----------------------------
# Targets
# -----------------------------
def targets_for_club(club_id: str, club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
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

    if fam == "FW":
        if club_id in {"2W", "3W"}:
            return {"launch": (11.0, 15.0), "spin": (2800.0, 3800.0)}
        if club_id in {"4W", "5W"}:
            return {"launch": (13.0, 17.0), "spin": (3200.0, 4500.0)}
        return {"launch": (14.0, 19.0), "spin": (3800.0, 5200.0)}

    if fam == "HY":
        if club_id in {"2H", "3H"}:
            return {"launch": (13.0, 18.0), "spin": (3500.0, 5200.0)}
        if club_id in {"4H", "5H"}:
            return {"launch": (15.0, 20.0), "spin": (3800.0, 5600.0)}
        return {"launch": (16.0, 22.0), "spin": (4200.0, 6200.0)}

    return {"launch": (0.0, 99.0), "spin": (0.0, 99999.0)}


# -----------------------------
# Hosel change estimate
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
# Recommendation helpers
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
    scored = []

    for s in settings:
        d = translate_fn(brand, system_name, s, handedness)
        loft = getattr(d, "loft_deg", None)
        lie = getattr(d, "lie_deg", None)
        if loft is None or lie is None:
            continue

        score = abs(loft - needed_loft_delta) * 1.5 + abs(lie - needed_lie_delta) * 1.0

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
# Aggregation
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


# -----------------------------
# Driver fitting / comparison
# -----------------------------
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


@dataclass
class DriverRecommendationBundle:
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


def _fmt_num(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{decimals}f}"


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


def compare_driver_setups(a_df: pd.DataFrame, b_df: pd.DataFrame, label_a: str = "Setup A", label_b: str = "Setup B") -> Dict[str, object]:
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

    # Carry
    if better_high(a.carry, b.carry) == label_a:
        score_a += 1
    elif better_high(a.carry, b.carry) == label_b:
        score_b += 1

    # Ball speed
    if better_high(a.ball_speed, b.ball_speed) == label_a:
        score_a += 1
    elif better_high(a.ball_speed, b.ball_speed) == label_b:
        score_b += 1

    # Dispersion
    if better_low(a.offline, b.offline) == label_a:
        score_a += 1
    elif better_low(a.offline, b.offline) == label_b:
        score_b += 1

    # Fairways
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


def build_driver_recommendations(
    summary: ClubSummary,
    user_setup: DriverUserSetup,
    fairway_hit_pct: Optional[float] = None,
) -> DriverRecommendationBundle:
    launch_lo, launch_hi = targets_for_club("DR", summary.club_speed_avg)["launch"]
    spin_lo, spin_hi = targets_for_club("DR", summary.club_speed_avg)["spin"]

    low_launch = not np.isnan(summary.vla_avg) and summary.vla_avg < launch_lo
    very_low_launch = not np.isnan(summary.vla_avg) and summary.vla_avg < 9.5
    high_launch = not np.isnan(summary.vla_avg) and summary.vla_avg > launch_hi
    playable_spin = not np.isnan(summary.spin_avg) and spin_lo <= summary.spin_avg <= spin_hi
    high_spin = not np.isnan(summary.spin_avg) and summary.spin_avg > spin_hi
    right_miss = not np.isnan(summary.offline_avg) and summary.offline_avg > 5
    left_miss = not np.isnan(summary.offline_avg) and summary.offline_avg < -5
    negative_aoa = not np.isnan(summary.aoa_avg) and summary.aoa_avg < 0
    wide_dispersion = not np.isnan(summary.offline_std) and summary.offline_std >= 18
    low_smash = not np.isnan(summary.smash_avg) and summary.smash_avg < 1.45
    light_shaft = user_setup.shaft_weight_g <= 55

    # Swing recommendation: just one change
    if negative_aoa and low_launch:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Move the ball a touch farther forward and feel one small upward strike through impact.",
            why="Your attack angle and launch window suggest you are giving away carry distance."
        )
    elif right_miss:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Use one simple feel only: square the face a little earlier through impact.",
            why="Your data trends right often enough that face control matters more than a major swing rebuild."
        )
    elif low_smash:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Prioritize centered strike over extra speed on your next session.",
            why="Better strike quality should improve both ball speed and dispersion."
        )
    else:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Keep the swing thought simple and repeatable: same tee height, same ball position, same tempo.",
            why="Your next gains are more likely to come from consistency than from a major motion change."
        )

    # Driver settings recommendation
    if low_launch:
        if user_setup.brand.lower() == "titleist":
            settings_text = f"Try adding loft in the SureFit sleeve before changing flex. Example: test {user_setup.hosel_setting} against a higher-loft setting such as A2."
        else:
            settings_text = f"Add loft slightly from your current setting ({user_setup.hosel_setting}) before testing a stiffer shaft."
        settings_why = "Your launch is below target, so a loft increase is usually safer than making the shaft play firmer."
    elif right_miss:
        settings_text = f"Test a slightly more upright / draw-help setting from {user_setup.hosel_setting} and compare dispersion."
        settings_why = "Your miss pattern trends right, so a hosel tweak may help start line and face presentation."
    elif high_launch and high_spin:
        settings_text = f"Your current hosel setting is worth testing against a slightly lower-loft option from {user_setup.hosel_setting}."
        settings_why = "Launch and spin are both elevated enough to justify a head-setting test."
    else:
        settings_text = f"Keep your current hosel setting ({user_setup.hosel_setting}) for now and confirm strike pattern first."
        settings_why = "The current data does not strongly demand a hosel move before strike consistency is confirmed."

    driver_settings = RecommendationBlock(
        title="Driver Settings",
        suggestion=settings_text,
        why=settings_why,
    )

    # Equipment recommendation
    equipment_bits: List[str] = []

    if wide_dispersion or (fairway_hit_pct is not None and fairway_hit_pct < 60):
        if user_setup.brand.lower() == "titleist" and user_setup.model.upper() == "TSR3":
            equipment_bits.append("Consider a more forgiving head such as TSR2 if you want help on mishits.")
        elif user_setup.brand.lower() == "ping" and "LST" in user_setup.model.upper():
            equipment_bits.append("Consider moving from an LST head to a MAX-style head for more forgiveness.")
        else:
            equipment_bits.append("A more forgiving, higher-MOI head may help more than a stiffer shaft.")

    if low_launch and right_miss and playable_spin:
        if light_shaft:
            equipment_bits.append("Test a heavier 60–65g shaft in the same flex before testing 6.5.")
        else:
            equipment_bits.append("Stay in the same flex first and test a different shaft profile before moving stiffer.")
    elif light_shaft and (right_miss or wide_dispersion):
        equipment_bits.append("A heavier shaft in the same flex may improve tempo and face control.")
    elif high_spin and not low_launch and not right_miss and not np.isnan(summary.club_speed_avg) and summary.club_speed_avg >= 103:
        equipment_bits.append("A stiffer or lower-spin shaft can be tested, but only after launch and strike stay stable.")
    else:
        equipment_bits.append("Current shaft category looks reasonable; profile testing matters more than jumping flex.")

    equipment_adjustment = RecommendationBlock(
        title="Equipment Adjustment",
        suggestion=" ".join(equipment_bits),
        why="This recommendation prioritizes launch, carry, forgiveness, and dispersion before stiffness.",
    )

    return DriverRecommendationBundle(
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
            "fairway_hit_pct": fairway_hit_pct if fairway_hit_pct is not None else math.nan,
        },
    )
