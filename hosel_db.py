# hosel_db.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple

Handedness = Literal["RH", "LH"]

@dataclass(frozen=True)
class HoselSettingDelta:
    loft_deg: Optional[float] = None      # + adds loft, - reduces loft
    lie_deg: Optional[float] = None       # + more upright, - flatter
    face_deg: Optional[float] = None      # + more closed, - more open
    note: str = ""                        # any human-readable caveats

@dataclass(frozen=True)
class HoselSystem:
    system_name: str
    family: str  # "matrix", "dual_cog", "sleeve_12", "sleeve_8", "futurefit33", etc.
    settings_rh: List[str]
    settings_lh: List[str]
    deltas_rh: Dict[str, HoselSettingDelta]
    deltas_lh: Dict[str, HoselSettingDelta]
    loft_range_deg: Optional[Tuple[float, float]] = None   # (min, max) relative adjust
    lie_range_deg: Optional[Tuple[float, float]] = None
    face_range_deg: Optional[Tuple[float, float]] = None
    notes: str = ""


# ---------------------------
# Titleist SureFit (Driver/FW)
# ---------------------------
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

TITLEIST_SUREFIT_DF_LH_LABELS = [
    "A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4","D1","D2","D3","D4"
]
TITLEIST_SUREFIT_DF_LH_DELTAS: Dict[str, HoselSettingDelta] = {
    s: HoselSettingDelta(None, None, None, "LH SureFit mapping not encoded; treat as range-only.")
    for s in TITLEIST_SUREFIT_DF_LH_LABELS
}

TITLEIST_SUREFIT_DRIVER_FAIRWAY = HoselSystem(
    system_name="Titleist SureFit (Driver/Fairway)",
    family="matrix_16",
    settings_rh=list(TITLEIST_SUREFIT_DF_RH_DELTAS.keys()),
    settings_lh=TITLEIST_SUREFIT_DF_LH_LABELS,
    deltas_rh=TITLEIST_SUREFIT_DF_RH_DELTAS,
    deltas_lh=TITLEIST_SUREFIT_DF_LH_DELTAS,
    loft_range_deg=(-0.75, +1.50),
    lie_range_deg=(-0.75, +1.50),
    face_range_deg=None,
    notes="Driver/FW uses 0.75° increments. RH chart encoded. LH treated as range-only.",
)

# ---------------------------
# Callaway OptiFit (8 combos)
# ---------------------------
CALLAWAY_OPTIFIT_RH_SETTINGS = [
    "-1 N", "S N", "+1 N", "+2 N",
    "-1 D", "S D", "+1 D", "+2 D"
]
CALLAWAY_OPTIFIT_DELTAS = {
    "-1 N": HoselSettingDelta(-1.0, 0.0, None, "Neutral lie"),
    "S N":  HoselSettingDelta(0.0,  0.0, None, "Stated loft, neutral lie"),
    "+1 N": HoselSettingDelta(+1.0, 0.0, None, "Neutral lie"),
    "+2 N": HoselSettingDelta(+2.0, 0.0, None, "Neutral lie"),
    "-1 D": HoselSettingDelta(-1.0, +1.0, None, "Draw/upright lie (approx)"),
    "S D":  HoselSettingDelta(0.0,  +1.0, None, "Draw/upright lie (approx)"),
    "+1 D": HoselSettingDelta(+1.0, +1.0, None, "Draw/upright lie (approx)"),
    "+2 D": HoselSettingDelta(+2.0, +1.0, None, "Draw/upright lie (approx)"),
}

CALLAWAY_OPTIFIT = HoselSystem(
    system_name="Callaway OptiFit",
    family="dual_cog_8",
    settings_rh=CALLAWAY_OPTIFIT_RH_SETTINGS,
    settings_lh=CALLAWAY_OPTIFIT_RH_SETTINGS,
    deltas_rh=CALLAWAY_OPTIFIT_DELTAS,
    deltas_lh=CALLAWAY_OPTIFIT_DELTAS,
    loft_range_deg=(-1.0, +2.0),
    lie_range_deg=(0.0, +1.0),
    notes="Upper cog controls loft; lower cog sets lie (Neutral vs Draw).",
)

# ---------------------------
# TaylorMade Loft Sleeve (12 pos) — range-only
# ---------------------------
TM_LOFT_SLEEVE_12 = HoselSystem(
    system_name="TaylorMade Loft Sleeve (12-position)",
    family="sleeve_12",
    settings_rh=["STD", "LOWER", "HIGHER", "UPRIGHT", "STD (alt)"] + [f"POS{i}" for i in range(1, 9)],
    settings_lh=["STD", "LOWER", "HIGHER", "UPRIGHT", "STD (alt)"] + [f"POS{i}" for i in range(1, 9)],
    deltas_rh={},  # range-only (varies by generation)
    deltas_lh={},
    loft_range_deg=(-2.0, +2.0),
    lie_range_deg=(0.0, +4.0),
    face_range_deg=None,
    notes="Range-only unless you encode a model/year-specific chart.",
)

# ---------------------------
# Ping Trajectory Tuning 2.0 (8 pos)
# ---------------------------
PING_TT2 = HoselSystem(
    system_name="PING Trajectory Tuning 2.0 (8-position)",
    family="sleeve_8",
    settings_rh=["STD", "+0.5", "+1.0", "+1.5", "-0.5", "-1.0", "-1.5", "FLAT"],
    settings_lh=["STD", "+0.5", "+1.0", "+1.5", "-0.5", "-1.0", "-1.5", "FLAT"],
    deltas_rh={
        "STD":  HoselSettingDelta(0.0, 0.0, None, ""),
        "+0.5": HoselSettingDelta(+0.5, None, None, "Lie varies; range-only"),
        "+1.0": HoselSettingDelta(+1.0, None, None, "Lie varies; range-only"),
        "+1.5": HoselSettingDelta(+1.5, None, None, "Lie varies; range-only"),
        "-0.5": HoselSettingDelta(-0.5, None, None, "Lie varies; range-only"),
        "-1.0": HoselSettingDelta(-1.0, None, None, "Lie varies; range-only"),
        "-1.5": HoselSettingDelta(-1.5, None, None, "Lie varies; range-only"),
        "FLAT": HoselSettingDelta(0.0, -3.0, None, "Up to 3° flatter than std (range-based)"),
    },
    deltas_lh={},
    loft_range_deg=(-1.5, +1.5),
    lie_range_deg=(-3.0, 0.0),
    notes="Ping labels differ slightly by gen; keep enums simple.",
)

# ---------------------------
# Cobra MyFly + FutureFit33 — range-only
# ---------------------------
COBRA_MYFLY = HoselSystem(
    system_name="Cobra MyFly (8 loft settings)",
    family="myfly_8",
    settings_rh=["Setting 1","Setting 2","Setting 3","Setting 4","Setting 5","Setting 6","Setting 7","Setting 8"],
    settings_lh=["Setting 1","Setting 2","Setting 3","Setting 4","Setting 5","Setting 6","Setting 7","Setting 8"],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-1.0, +1.0),
    notes="Labels differ by head loft; range-only.",
)

COBRA_FUTUREFIT33 = HoselSystem(
    system_name="Cobra FutureFit33 (33 unique settings)",
    family="futurefit33",
    settings_rh=[f"FF33-{i:02d}" for i in range(1, 34)],
    settings_lh=[f"FF33-{i:02d}" for i in range(1, 34)],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-2.0, +2.0),
    lie_range_deg=(-2.0, +2.0),
    notes="Range-only unless you encode the full 33-setting chart.",
)

# ---------------------------
# Registry: brand -> system(s)
# ---------------------------
HOSEL_SYSTEMS_BY_BRAND: Dict[str, List[HoselSystem]] = {
    "Titleist": [TITLEIST_SUREFIT_DRIVER_FAIRWAY],
    "Callaway": [CALLAWAY_OPTIFIT],
    "TaylorMade": [TM_LOFT_SLEEVE_12],
    "PING": [PING_TT2],
    "Cobra": [COBRA_MYFLY, COBRA_FUTUREFIT33],
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
    if setting in deltas:
        return deltas[setting]

    return HoselSettingDelta(
        None, None, None,
        note=f"No exact delta stored. System ranges: loft={sys.loft_range_deg}, lie={sys.lie_range_deg}, face={sys.face_range_deg}"
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
