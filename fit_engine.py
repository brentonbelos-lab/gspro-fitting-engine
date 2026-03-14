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


def _rec(title: str, detail: str, why: str | None = None) -> str:
    """
    Cleaner, golfer-friendly recommendation tone.
    """
    text = f"**{title}** — {detail.strip()}"
    if why:
        text += f" This should help {why.strip()}."
    return text


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
        return "Driver"
    if club_id.endswith("W"):
        return "Fairway Wood"
    if club_id.endswith("H"):
        return "Hybrid"
    return "Other"


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


# -----------------------------
# Targets
# -----------------------------
def targets_for_club(club_id: str, club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
    fam = club_family(club_id)

    if fam == "Driver":
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

    if fam == "Fairway Wood":
        if club_id in {"2W", "3W"}:
            return {"launch": (11.0, 15.0), "spin": (2800.0, 3800.0)}
        if club_id in {"4W", "5W"}:
            return {"launch": (13.0, 17.0), "spin": (3200.0, 4500.0)}
        return {"launch": (14.0, 19.0), "spin": (3800.0, 5200.0)}

    if fam == "Hybrid":
        if club_id in {"2H", "3H"}:
            return {"launch": (13.0, 18.0), "spin": (3500.0, 5200.0)}
        if club_id in {"4H", "5H"}:
            return {"launch": (15.0, 20.0), "spin": (3800.0, 5600.0)}
        return {"launch": (16.0, 22.0), "spin": (4200.0, 6200.0)}

    return {"launch": (0.0, 99.0), "spin": (0.0, 99999.0)}


# -----------------------------
# Hosel estimate
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

    return LaunchSpinEstimate(
        launch_change_deg=launch_est,
        launch_range_deg=(launch_low, launch_high),
        spin_change_rpm=spin_est,
        spin_range_rpm=spin_range,
        notes="These are estimates only. Real ball-flight changes will vary with strike location, face angle, and delivery.",
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

        return {"type": "guidance", "message": f"Exact chart not encoded for this hosel. Guidance: {', '.join(direction)}."}

    scored.sort(key=lambda x: x[0])
    score, setting, loft, lie = scored[0]
    return {
        "type": "exact",
        "current": current_setting,
        "recommended": {"setting": setting, "loft_delta": loft, "lie_delta": lie, "score": score},
    }


# -----------------------------
# Summary
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
# Recommendation dataclasses
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


# -----------------------------
# Compare helpers
# -----------------------------
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


# -----------------------------
# Driver recommendation engine
# -----------------------------
def build_driver_recommendations(
    summary: ClubSummary,
    user_setup: DriverUserSetup,
    fairway_hit_pct: Optional[float] = None,
) -> RecommendationBundle:
    launch_lo, launch_hi = targets_for_club("DR", summary.club_speed_avg)["launch"]
    spin_lo, spin_hi = targets_for_club("DR", summary.club_speed_avg)["spin"]

    low_launch = not np.isnan(summary.vla_avg) and summary.vla_avg < launch_lo
    high_launch = not np.isnan(summary.vla_avg) and summary.vla_avg > launch_hi
    playable_spin = not np.isnan(summary.spin_avg) and spin_lo <= summary.spin_avg <= spin_hi
    high_spin = not np.isnan(summary.spin_avg) and summary.spin_avg > spin_hi
    right_miss = not np.isnan(summary.offline_avg) and summary.offline_avg > 5
    negative_aoa = not np.isnan(summary.aoa_avg) and summary.aoa_avg < 0
    wide_dispersion = not np.isnan(summary.offline_std) and summary.offline_std >= 18
    low_smash = not np.isnan(summary.smash_avg) and summary.smash_avg < 1.45
    light_shaft = user_setup.shaft_weight_g <= 55

    if negative_aoa and low_launch:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Move the ball slightly forward and feel one small upward strike through impact.",
            why="Low launch plus a downward attack angle usually means you are giving away carry distance.",
            tone="green",
        )
    elif right_miss:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Use one simple feel only: square the face a little earlier through impact.",
            why="Your right miss pattern suggests face control matters more than making a large swing change.",
            tone="yellow",
        )
    elif low_smash:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Prioritize centered strike over extra speed during your next session.",
            why="Strike quality should improve both ball speed and dispersion.",
            tone="yellow",
        )
    else:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Keep the current swing thought simple and repeatable.",
            why="Your next gains are more likely to come from consistency than a major swing rebuild.",
            tone="green",
        )

    if low_launch:
        if user_setup.brand.lower() == "titleist":
            settings_text = f"Try adding loft in the hosel before changing flex. Example: test {user_setup.hosel_setting} against A2."
        else:
            settings_text = f"Add loft slightly from your current setting ({user_setup.hosel_setting}) before testing a stiffer shaft."
        driver_settings = RecommendationBlock(
            title="Driver Settings",
            suggestion=settings_text,
            why="Your launch is below target, so loft is the safer first adjustment.",
            tone="green",
        )
    elif right_miss:
        driver_settings = RecommendationBlock(
            title="Driver Settings",
            suggestion=f"Test a slightly more upright or draw-help setting from {user_setup.hosel_setting}.",
            why="Your miss pattern trends right, so a hosel tweak may help face presentation and start line.",
            tone="yellow",
        )
    elif high_launch and high_spin:
        driver_settings = RecommendationBlock(
            title="Driver Settings",
            suggestion=f"Your current hosel setting is worth testing against a slightly lower-loft option from {user_setup.hosel_setting}.",
            why="Launch and spin are high enough to justify a settings test.",
            tone="yellow",
        )
    else:
        driver_settings = RecommendationBlock(
            title="Driver Settings",
            suggestion=f"Keep your current hosel setting ({user_setup.hosel_setting}) for now.",
            why="Your current data does not strongly demand a hosel change before strike consistency is confirmed.",
            tone="green",
        )

    equipment_lines: List[str] = []
    tone = "green"

    if wide_dispersion or (fairway_hit_pct is not None and fairway_hit_pct < 60):
        if user_setup.brand.lower() == "titleist" and user_setup.model.upper() == "TSR3":
            equipment_lines.append("Consider a more forgiving head such as TSR2 if you want help on mishits.")
            tone = "yellow"
        elif user_setup.brand.lower() == "ping" and "LST" in user_setup.model.upper():
            equipment_lines.append("Consider moving from an LST head to a MAX-style head for more forgiveness.")
            tone = "yellow"
        else:
            equipment_lines.append("A more forgiving, higher-MOI head may help more than a stiffer shaft.")
            tone = "yellow"

    if low_launch and right_miss and playable_spin:
        if light_shaft:
            equipment_lines.append("Test a heavier 60–65g shaft in the same flex before testing 6.5.")
            tone = "green"
        else:
            equipment_lines.append("Stay in the same flex first and test a different profile before moving stiffer.")
            tone = "green"
    elif high_spin and not low_launch and not right_miss and not np.isnan(summary.club_speed_avg) and summary.club_speed_avg >= 103:
        equipment_lines.append("A stiffer or lower-spin shaft can be tested, but only after launch and strike stay stable.")
        tone = "yellow"
    else:
        if not equipment_lines:
            equipment_lines.append("Current shaft category looks reasonable; profile testing matters more than jumping flex.")
            tone = "green"

    if low_launch and right_miss and playable_spin:
        equipment_lines.append("Avoid moving stiffer first if launch is already low and the miss is right.")

    equipment_adjustment = RecommendationBlock(
        title="Equipment Adjustment",
        suggestion=" ".join(equipment_lines),
        why="This recommendation prioritizes launch, forgiveness, and dispersion before stiffness.",
        tone=tone,
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
            "fairway_hit_pct": fairway_hit_pct if fairway_hit_pct is not None else math.nan,
        },
    )


# -----------------------------
# Fairway / Hybrid recommendation engine
# -----------------------------
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

    launch_lo, launch_hi = targets_for_club(club_id, summary.club_speed_avg)["launch"]
    spin_lo, spin_hi = targets_for_club(club_id, summary.club_speed_avg)["spin"]

    low_launch = not np.isnan(summary.vla_avg) and summary.vla_avg < launch_lo
    high_launch = not np.isnan(summary.vla_avg) and summary.vla_avg > launch_hi
    low_spin = not np.isnan(summary.spin_avg) and summary.spin_avg < spin_lo
    high_spin = not np.isnan(summary.spin_avg) and summary.spin_avg > spin_hi
    right_miss = not np.isnan(summary.offline_avg) and summary.offline_avg > 5
    left_miss = not np.isnan(summary.offline_avg) and summary.offline_avg < -5
    negative_aoa = not np.isnan(summary.aoa_avg) and summary.aoa_avg < 0
    wide_dispersion = not np.isnan(summary.offline_std) and summary.offline_std >= 16
    low_smash = not np.isnan(summary.smash_avg) and summary.smash_avg < 1.42

    if low_launch and negative_aoa:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Keep the ball slightly farther forward and make one shallow sweeping move through impact.",
            why=f"This {family.lower()} appears to need more launch, and a steeper hit usually works against that.",
            tone="green",
        )
    elif right_miss:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Focus on one feel only: let the face close a touch earlier through impact.",
            why=f"Your {family.lower()} pattern trends right, so face control is the simplest first change.",
            tone="yellow",
        )
    elif low_smash:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Prioritize centered strike over extra speed with this club.",
            why=f"This {family.lower()} is likely losing ball speed from strike quality more than from raw speed.",
            tone="yellow",
        )
    else:
        swing = RecommendationBlock(
            title="Swing",
            suggestion="Keep the motion stable and focus on repeating strike location.",
            why=f"This {family.lower()} looks more likely to improve through consistency than through a major motion change.",
            tone="green",
        )

    if low_launch:
        if hosel_setting:
            settings_text = f"Test a slightly higher-loft setting from {hosel_setting} before changing shaft stiffness."
        else:
            settings_text = "Test a slightly higher-loft setting before changing shaft stiffness."
        settings = RecommendationBlock(
            title="Club Settings",
            suggestion=settings_text,
            why=f"Launch is below the target window for this {family.lower()}, so loft is the cleanest first adjustment.",
            tone="green",
        )
    elif right_miss:
        if hosel_setting:
            settings_text = f"Test a slightly more upright or draw-help setting from {hosel_setting}."
        else:
            settings_text = "Test a slightly more upright or draw-help setting."
        settings = RecommendationBlock(
            title="Club Settings",
            suggestion=settings_text,
            why=f"Your miss pattern trends right, so start-line and face presentation are worth adjusting first.",
            tone="yellow",
        )
    elif high_launch and high_spin:
        if hosel_setting:
            settings_text = f"Test a slightly lower-loft setting from {hosel_setting}."
        else:
            settings_text = "Test a slightly lower-loft setting."
        settings = RecommendationBlock(
            title="Club Settings",
            suggestion=settings_text,
            why=f"Launch and spin are both above the target window for this {family.lower()}.",
            tone="yellow",
        )
    else:
        settings = RecommendationBlock(
            title="Club Settings",
            suggestion=f"Keep the current setting{f' ({hosel_setting})' if hosel_setting else ''} for now.",
            why=f"This {family.lower()} does not strongly demand a settings change before strike and gapping are confirmed.",
            tone="green",
        )

    equipment_lines: List[str] = []
    tone = "green"

    if family == "Fairway Wood":
        if low_launch and (low_spin or not high_spin):
            equipment_lines.append("Consider more loft or a more launch-friendly fairway wood setup before testing a stiffer shaft.")
            tone = "green"
        if wide_dispersion:
            equipment_lines.append("A more forgiving fairway wood head may help on slight strike misses.")
            tone = "yellow"
        if right_miss and shaft_weight_g is not None and shaft_weight_g < 70:
            equipment_lines.append("A slightly heavier fairway shaft in the same flex may improve tempo and face control.")
            tone = "green"
        elif right_miss:
            equipment_lines.append("Test a smoother or more neutral shaft profile before moving stiffer.")
            tone = "yellow"
        if high_spin and not low_launch:
            equipment_lines.append("If the ball flight climbs too much, test a slightly lower-spin profile only after launch stays playable.")
            tone = "yellow"

    elif family == "Hybrid":
        if low_launch:
            equipment_lines.append("Consider a little more loft or a more launch-friendly hybrid setup before changing flex.")
            tone = "green"
        if right_miss:
            equipment_lines.append("If this hybrid leaks right, test a slightly more upright setting or a profile that is easier to square.")
            tone = "yellow"
        if left_miss:
            equipment_lines.append("If this hybrid turns over too much, keep the shaft profile stable and test a more neutral setting first.")
            tone = "yellow"
        if wide_dispersion:
            equipment_lines.append("A more forgiving hybrid head or slightly heavier shaft may help control start line.")
            tone = "yellow"
        if high_spin and high_launch:
            equipment_lines.append("If trajectory is too floaty, test a slightly flatter or lower-spin setup after strike quality is confirmed.")
            tone = "yellow"

    if not equipment_lines:
        equipment_lines.append("Current equipment category looks reasonable; confirm gapping and strike consistency before making a bigger change.")
        tone = "green"

    equipment = RecommendationBlock(
        title="Equipment Adjustment",
        suggestion=" ".join(equipment_lines),
        why=f"This recommendation prioritizes launch window, miss tendency, and forgiveness for your {family.lower()}.",
        tone=tone,
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
        },
    )
