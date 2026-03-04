# hosel_db.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

Handedness = Literal["RH", "LH"]

@dataclass(frozen=True)
class HoselSettingDelta:
    loft_deg: Optional[float] = None   # relative to stated setting (effective loft)
    lie_deg: Optional[float] = None    # + upright, - flat
    face_deg: Optional[float] = None   # + closed, - open (rarely encoded)
    note: str = ""

@dataclass(frozen=True)
class HoselSystem:
    system_name: str
    family: str
    settings_rh: List[str]
    settings_lh: List[str]
    deltas_rh: Dict[str, HoselSettingDelta]
    deltas_lh: Dict[str, HoselSettingDelta]
    loft_range_deg: Optional[Tuple[float, float]] = None
    lie_range_deg: Optional[Tuple[float, float]] = None
    face_range_deg: Optional[Tuple[float, float]] = None
    notes: str = ""

def _matrix_16() -> List[str]:
    return [
        "A1","A2","A3","A4",
        "B1","B2","B3","B4",
        "C1","C2","C3","C4",
        "D1","D2","D3","D4",
    ]

# ---------------------------
# Titleist SureFit (Driver/FW)
# ---------------------------
# Titleist confirms driver/fairway SureFit increments are 0.75° loft/lie. :contentReference[oaicite:7]{index=7}
TITLEIST_SUREFIT_DF_RH_DELTAS: Dict[str, HoselSettingDelta] = {
    "A1": HoselSettingDelta(0.00, 0.00, None, "Standard"),
    "A2": HoselSettingDelta(0.00, +1.50, None, "Upright"),
    "A3": HoselSettingDelta(+1.50, +1.50, None, "Higher + upright"),
    "A4": HoselSettingDelta(+1.50, 0.00, None, "Higher"),

    "B1": HoselSettingDelta(0.00, -0.75, None, "Flatter"),
    "B2": HoselSettingDelta(0.00, +0.75, None, "Slight upright"),
    "B3": HoselSettingDelta(+1.50, +0.75, None, "Higher + slight upright"),
    "B4": HoselSettingDelta(+1.50, -0.75, None, "Higher + flatter"),

    "C1": HoselSettingDelta(-0.75, -0.75, None, "Lower + flatter"),
    "C2": HoselSettingDelta(-0.75, +0.75, None, "Lower + slight upright"),
    "C3": HoselSettingDelta(+0.75, +0.75, None, "Slight higher + slight upright"),
    "C4": HoselSettingDelta(+0.75, -0.75, None, "Slight higher + flatter"),

    "D1": HoselSettingDelta(-0.75, 0.00, None, "Lower"),
    "D2": HoselSettingDelta(-0.75, +1.50, None, "Lower + upright"),
    "D3": HoselSettingDelta(+0.75, +1.50, None, "Slight higher + upright"),
    "D4": HoselSettingDelta(+0.75, 0.00, None, "Slight higher"),
}

# For MVP we avoid encoding LH SureFit deltas unless you want the full LH chart.
TITLEIST_SUREFIT_DF_LH_DELTAS = {s: HoselSettingDelta(None, None, None, "LH mapping not encoded; treat as range-only.") for s in _matrix_16()}

TITLEIST_SUREFIT_DRIVER_FAIRWAY = HoselSystem(
    system_name="Titleist SureFit (Driver/Fairway)",
    family="matrix_16",
    settings_rh=_matrix_16(),
    settings_lh=_matrix_16(),
    deltas_rh=TITLEIST_SUREFIT_DF_RH_DELTAS,
    deltas_lh=TITLEIST_SUREFIT_DF_LH_DELTAS,
    loft_range_deg=(-0.75, +1.50),
    lie_range_deg=(-0.75, +1.50),
    notes="Driver/FW SureFit: 0.75° loft/lie increments. Exact LH mapping not encoded (safe MVP).",
)

# ---------------------------
# Titleist SureFit (Hybrid)
# ---------------------------
# Titleist confirms hybrid SureFit uses 1° increments. :contentReference[oaicite:8]{index=8}
# Exact A1..D4 delta map varies by chart; MVP keeps settings selectable but uses range-based translation.
TITLEIST_SUREFIT_HYBRID = HoselSystem(
    system_name="Titleist SureFit (Hybrid)",
    family="matrix_16_range_only",
    settings_rh=_matrix_16(),
    settings_lh=_matrix_16(),
    deltas_rh={s: HoselSettingDelta(None, None, None, "Hybrid chart not encoded; range-only.") for s in _matrix_16()},
    deltas_lh={s: HoselSettingDelta(None, None, None, "Hybrid chart not encoded; range-only.") for s in _matrix_16()},
    loft_range_deg=(-1.0, +2.0),
    lie_range_deg=(-1.0, +2.0),
    notes="Hybrid SureFit: 1° increments. Use range-only until chart is encoded.",
)

# ---------------------------
# Callaway OptiFit (8 combos)
# ---------------------------
# Callaway says OptiFit provides 8 loft/lie combinations. :contentReference[oaicite:9]{index=9}
CALLAWAY_OPTIFIT_SETTINGS = ["-1 N", "S N", "+1 N", "+2 N", "-1 D", "S D", "+1 D", "+2 D"]
CALLAWAY_OPTIFIT_DELTAS = {
    "-1 N": HoselSettingDelta(-1.0, 0.0, None, ""),
    "S N":  HoselSettingDelta(0.0,  0.0, None, "Stated loft"),
    "+1 N": HoselSettingDelta(+1.0, 0.0, None, ""),
    "+2 N": HoselSettingDelta(+2.0, 0.0, None, ""),
    "-1 D": HoselSettingDelta(-1.0, +1.0, None, "Draw/upright (approx)"),
    "S D":  HoselSettingDelta(0.0,  +1.0, None, "Draw/upright (approx)"),
    "+1 D": HoselSettingDelta(+1.0, +1.0, None, "Draw/upright (approx)"),
    "+2 D": HoselSettingDelta(+2.0, +1.0, None, "Draw/upright (approx)"),
}
CALLAWAY_OPTIFIT = HoselSystem(
    system_name="Callaway OptiFit",
    family="dual_cog_8",
    settings_rh=CALLAWAY_OPTIFIT_SETTINGS,
    settings_lh=CALLAWAY_OPTIFIT_SETTINGS,
    deltas_rh=CALLAWAY_OPTIFIT_DELTAS,
    deltas_lh=CALLAWAY_OPTIFIT_DELTAS,
    loft_range_deg=(-1.0, +2.0),
    lie_range_deg=(0.0, +1.0),
    notes="OptiFit: upper cog loft (-1/S/+1/+2), lower cog lie (N/D).",
)

# ---------------------------
# PING Trajectory Tuning 2.0 (8 pos)
# ---------------------------
# PING fairway woods: 8-position hosel; ±1.5° loft, lie up to 3° flatter. :contentReference[oaicite:10]{index=10}
PING_TT2 = HoselSystem(
    system_name="PING Trajectory Tuning 2.0 (8-position)",
    family="sleeve_8_range",
    settings_rh=["STD", "+0.5", "+1.0", "+1.5", "-0.5", "-1.0", "-1.5", "FLAT"],
    settings_lh=["STD", "+0.5", "+1.0", "+1.5", "-0.5", "-1.0", "-1.5", "FLAT"],
    deltas_rh={
        "STD": HoselSettingDelta(0.0, 0.0, None, ""),
        "+0.5": HoselSettingDelta(+0.5, None, None, "Lie varies by sleeve; range-only"),
        "+1.0": HoselSettingDelta(+1.0, None, None, "Lie varies by sleeve; range-only"),
        "+1.5": HoselSettingDelta(+1.5, None, None, "Lie varies by sleeve; range-only"),
        "-0.5": HoselSettingDelta(-0.5, None, None, "Lie varies by sleeve; range-only"),
        "-1.0": HoselSettingDelta(-1.0, None, None, "Lie varies by sleeve; range-only"),
        "-1.5": HoselSettingDelta(-1.5, None, None, "Lie varies by sleeve; range-only"),
        "FLAT": HoselSettingDelta(0.0, -3.0, None, "Up to 3° flatter than std (range-based)"),
    },
    deltas_lh={},
    loft_range_deg=(-1.5, +1.5),
    lie_range_deg=(-3.0, 0.0),
    notes="PING TT2: loft ±1.5°, lie includes flat option.",
)

# ---------------------------
# Cobra MyFly (8 settings)
# ---------------------------
# Cobra says MyFly: 8 loft settings; spin changes up to ±450 rpm. :contentReference[oaicite:11]{index=11}
COBRA_MYFLY = HoselSystem(
    system_name="Cobra MyFly (8 settings)",
    family="myfly_8_range",
    settings_rh=[f"Setting {i}" for i in range(1, 9)],
    settings_lh=[f"Setting {i}" for i in range(1, 9)],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-1.0, +1.0),
    notes="MyFly labels differ by head loft; MVP uses generic setting list.",
)

# ---------------------------
# TaylorMade Loft Sleeve (12 pos) — range-based
# ---------------------------
# TM manual: 12 positions; each click changes loft 0.5–0.75°, lie 0.5–0.75°, face 1–2°. :contentReference[oaicite:12]{index=12}
TM_LOFT_SLEEVE_12 = HoselSystem(
    system_name="TaylorMade Loft Sleeve (12-position)",
    family="sleeve_12_range",
    settings_rh=["STD"] + [f"POS{i}" for i in range(1, 12)],
    settings_lh=["STD"] + [f"POS{i}" for i in range(1, 12)],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-2.0, +2.0),
    lie_range_deg=(-1.0, +1.0),
    face_range_deg=(-2.0, +2.0),
    notes="TM: chart varies by model/year; use range-based until we encode model-specific chart.",
)

# Registry: brand -> systems
HOSEL_SYSTEMS_BY_BRAND: Dict[str, List[HoselSystem]] = {
    "Titleist": [TITLEIST_SUREFIT_DRIVER_FAIRWAY, TITLEIST_SUREFIT_HYBRID],
    "Callaway": [CALLAWAY_OPTIFIT],
    "PING": [PING_TT2],
    "Cobra": [COBRA_MYFLY],
    "TaylorMade": [TM_LOFT_SLEEVE_12],
}

def get_supported_brands() -> List[str]:
    return sorted(HOSEL_SYSTEMS_BY_BRAND.keys())

def get_brand_systems(brand: str) -> List[HoselSystem]:
    return HOSEL_SYSTEMS_BY_BRAND.get(brand, [])

def get_system(brand: str, system_name: str) -> Optional[HoselSystem]:
    for s in get_brand_systems(brand):
        if s.system_name == system_name:
            return s
    return None

def list_settings(brand: str, system_name: str, handedness: Handedness) -> List[str]:
    sys = get_system(brand, system_name)
    if not sys:
        return []
    return sys.settings_rh if handedness == "RH" else sys.settings_lh

def translate_setting(brand: str, system_name: str, setting: str, handedness: Handedness) -> HoselSettingDelta:
    sys = get_system(brand, system_name)
    if not sys:
        return HoselSettingDelta(None, None, None, note="Brand/system not found.")
    deltas = sys.deltas_rh if handedness == "RH" else sys.deltas_lh
    if setting in deltas and (deltas[setting].loft_deg is not None or deltas[setting].lie_deg is not None):
        return deltas[setting]
    # range-only fallback
    return HoselSettingDelta(
        None, None, None,
        note=f"Exact delta not encoded. System ranges: loft={sys.loft_range_deg}, lie={sys.lie_range_deg}."
    )

def system_ranges(brand: str, system_name: str) -> Dict[str, Optional[Tuple[float, float]]]:
    sys = get_system(brand, system_name)
    if not sys:
        return {"loft_range_deg": None, "lie_range_deg": None, "face_range_deg": None}
    return {
        "loft_range_deg": sys.loft_range_deg,
        "lie_range_deg": sys.lie_range_deg,
        "face_range_deg": sys.face_range_deg,
    }
