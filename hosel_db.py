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
    # If exact per-setting deltas are known, they live here.
    deltas_rh: Dict[str, HoselSettingDelta]
    deltas_lh: Dict[str, HoselSettingDelta]
    # If OEM only provides ranges / no stable chart, store ranges here.
    loft_range_deg: Optional[Tuple[float, float]] = None   # (min, max) relative adjust
    lie_range_deg: Optional[Tuple[float, float]] = None
    face_range_deg: Optional[Tuple[float, float]] = None
    notes: str = ""


# ---------------------------
# Titleist SureFit (Driver/FW)
# ---------------------------
# Based on commonly published SureFit driver/fairway mapping:
# - Driver/FW increments are 0.75° loft/lie steps; 16 settings.
# OEM: Titleist confirms increments + standard positions. :contentReference[oaicite:10]{index=10}
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

# LH SureFit driver/fairway uses a mirrored baseline (Titleist says LH starts D•4). :contentReference[oaicite:11]{index=11}
# For MVP, we expose the LH labels and keep deltas "unknown" unless you want to encode the full LH chart.
# This avoids us being wrong.
TITLEIST_SUREFIT_DF_LH_LABELS = [
    "A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4","D1","D2","D3","D4"
]
TITLEIST_SUREFIT_DF_LH_DELTAS: Dict[str, HoselSettingDelta] = {
    s: HoselSettingDelta(None, None, None, "LH SureFit mapping varies by chart; treat as range-only unless encoded.")
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
    notes="Driver/FW uses 0.75° increments; standard RH=A1, LH=D4 per Titleist. Loft is 'effective loft' when squared. "
          "Face-angle effect depends on how the club is soled at address. "
)


# ---------------------------
# Callaway OptiFit (8 combos)
# ---------------------------
# OEM: two cogs; total 8 loft/lie combinations. :contentReference[oaicite:12]{index=12}
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
    settings_lh=CALLAWAY_OPTIFIT_RH_SETTINGS,  # same label format for LH; actual effect is symmetric
    deltas_rh=CALLAWAY_OPTIFIT_DELTAS,
    deltas_lh=CALLAWAY_OPTIFIT_DELTAS,
    loft_range_deg=(-1.0, +2.0),
    lie_range_deg=(0.0, +1.0),  # conservative: treat D as more upright; exact degree can vary by model
    notes="Upper cog controls loft; lower cog sets lie (Neutral vs Draw)."
)


# ---------------------------
# TaylorMade Loft Sleeve (12 pos)
# ---------------------------
# OEM: 12 positions; loft up to ±2° (Qi4D) and tuning manual confirms 12 pos affecting loft/lie/face. :contentReference[oaicite:13]{index=13}
TM_LOFT_SLEEVE_12 = HoselSystem(
    system_name="TaylorMade Loft Sleeve (12-position)",
    family="sleeve_12",
    settings_rh=["STD", "LOWER", "HIGHER", "UPRIGHT", "STD (alt)"] + [f"POS{i}" for i in range(1, 9)],
    settings_lh=["STD", "LOWER", "HIGHER", "UPRIGHT", "STD (alt)"] + [f"POS{i}" for i in range(1, 9)],
    deltas_rh={},  # chart varies by generation; don't fake exact per-position deltas
    deltas_lh={},
    loft_range_deg=(-2.0, +2.0),
    lie_range_deg=(0.0, +4.0),   # OEM language: can move toward upright; treat as range not exact per setting
    face_range_deg=None,
    notes="Exact per-notch loft/lie/face differs by sleeve version. Use range-based messaging unless you store a model-specific chart."
)


# ---------------------------
# Ping Trajectory Tuning 2.0 (8 pos)
# ---------------------------
# OEM: 8 positions; loft ±1.5°, lie up to 3° flatter than std. :contentReference[oaicite:14]{index=14}
PING_TT2 = HoselSystem(
    system_name="PING Trajectory Tuning 2.0 (8-position)",
    family="sleeve_8",
    settings_rh=["STD", "+0.5", "+1.0", "+1.5", "-0.5", "-1.0", "-1.5", "FLAT"],
    settings_lh=["STD", "+0.5", "+1.0", "+1.5", "-0.5", "-1.0", "-1.5", "FLAT"],
    deltas_rh={
        "STD":  HoselSettingDelta(0.0, 0.0, None, ""),
        "+0.5": HoselSettingDelta(+0.5, None, None, "Lie depends on sleeve; treat as range-only"),
        "+1.0": HoselSettingDelta(+1.0, None, None, "Lie depends on sleeve; treat as range-only"),
        "+1.5": HoselSettingDelta(+1.5, None, None, "Lie depends on sleeve; treat as range-only"),
        "-0.5": HoselSettingDelta(-0.5, None, None, "Lie depends on sleeve; treat as range-only"),
        "-1.0": HoselSettingDelta(-1.0, None, None, "Lie depends on sleeve; treat as range-only"),
        "-1.5": HoselSettingDelta(-1.5, None, None, "Lie depends on sleeve; treat as range-only"),
        "FLAT": HoselSettingDelta(0.0, -3.0, None, "OEM describes up to 3° flatter than std (range-based)"),
    },
    deltas_lh={},
    loft_range_deg=(-1.5, +1.5),
    lie_range_deg=(-3.0, 0.0),
    notes="Ping chart labels can differ slightly by generation; keep enums simple and treat lie as range-based except FLAT."
)


# ---------------------------
# Cobra MyFly (8 settings) + FutureFit33 (33 unique)
# ---------------------------
# OEM: MyFly has eight loft settings. :contentReference[oaicite:15]{index=15}
COBRA_MYFLY = HoselSystem(
    system_name="Cobra MyFly (8 loft settings)",
    family="myfly_8",
    settings_rh=["Setting 1","Setting 2","Setting 3","Setting 4","Setting 5","Setting 6","Setting 7","Setting 8"],
    settings_lh=["Setting 1","Setting 2","Setting 3","Setting 4","Setting 5","Setting 6","Setting 7","Setting 8"],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-1.0, +1.0),
    notes="MyFly labels are printed as loft numbers on the sleeve (varies by head loft). Store per-head later if you want exact labels."
)

# OEM: FutureFit33 offers 33 unique loft/lie settings, ±2° in every direction. :contentReference[oaicite:16]{index=16}
COBRA_FUTUREFIT33 = HoselSystem(
    system_name="Cobra FutureFit33 (33 unique settings)",
    family="futurefit33",
    settings_rh=[f"FF33-{i:02d}" for i in range(1, 34)],
    settings_lh=[f"FF33-{i:02d}" for i in range(1, 34)],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-2.0, +2.0),
    lie_range_deg=(-2.0, +2.0),
    notes="FF33 uses 33 unique loft/lie combos. For MVP: ask user for their chart row/col or FF33 code; later we can encode the full chart."
)


# ---------------------------
# Srixon/Cleveland-style 12-position sleeve (STD / -1.5 / +1.5 / STD FL)
# ---------------------------
# Srixon hosel sleeve guide shows 12 positions with primary marks STD, -1.5, +1.5, STD FL. :contentReference[oaicite:17]{index=17}
# Cleveland Launcher XL manual shows same 12-position concept. :contentReference[oaicite:18]{index=18}
SLEEVE_12_STD_15 = HoselSystem(
    system_name="12-position sleeve (STD / -1.5 / +1.5 / STD FL)",
    family="sleeve_12_std_15",
    settings_rh=["STD", "-1.5", "+1.5", "STD FL"] + [f"POS{i}" for i in range(1, 9)],
    settings_lh=["STD", "-1.5", "+1.5", "STD FL"] + [f"POS{i}" for i in range(1, 9)],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-1.5, +1.5),
    notes="Exact intermediate POS deltas vary; keep range-based until we load per-model chart."
)


# ---------------------------
# Mizuno Quick Switch (common 8-setting ecosystem)
# ---------------------------
# Mizuno product pages describe Quick Switch providing 4 degrees of loft adjustability. :contentReference[oaicite:19]{index=19}
MIZUNO_QUICK_SWITCH = HoselSystem(
    system_name="Mizuno Quick Switch",
    family="sleeve_8",
    settings_rh=["STD", "+1", "+2", "-1", "-2", "UPRIGHT", "FLAT", "ALT"],
    settings_lh=["STD", "+1", "+2", "-1", "-2", "UPRIGHT", "FLAT", "ALT"],
    deltas_rh={},
    deltas_lh={},
    loft_range_deg=(-2.0, +2.0),
    notes="Mizuno sleeve charts vary by generation. Treat as range-based unless you store exact chart per model."
)


# ---------------------------
# PXG (8 settings)
# ---------------------------
# OEM PXG describes an adapter with eight settings. :contentReference[oaicite:20]{index=20}
PXG_ADAPTER_8 = HoselSystem(
    system_name="PXG Adapter (8 settings)",
    family="sleeve_8",
    settings_rh=["STD", "+", "++", "-", "--", "UPRIGHT", "FLAT", "ALT"],
    settings_lh=["STD", "+", "++", "-", "--", "UPRIGHT", "FLAT", "ALT"],
    deltas_rh={},
    deltas_lh={},
    notes="PXG publishes an 8-setting adapter chart; encode exact deltas later if you want strict translation."
)


# ---------------------------
# Wilson Fast Fit (6 settings)
# ---------------------------
# Fast Fit adjustable hosels feature six settings; hybrids include lie adjust too. :contentReference[oaicite:21]{index=21}
WILSON_FAST_FIT_6 = HoselSystem(
    system_name="Wilson Fast Fit (6 settings)",
    family="fastfit_6",
    settings_rh=["STD", "-1", "-0.5", "+1", "+1.5", "+2"],
    settings_lh=["STD", "-1", "-0.5", "+1", "+1.5", "+2"],
    deltas_rh={
        "STD":  HoselSettingDelta(0.0, 0.0, None, ""),
        "-1":   HoselSettingDelta(-1.0, None, None, "Lie effect varies; treat as range-based"),
        "-0.5": HoselSettingDelta(-0.5, None, None, "Lie effect varies; treat as range-based"),
        "+1":   HoselSettingDelta(+1.0, None, None, "Lie effect varies; treat as range-based"),
        "+1.5": HoselSettingDelta(+1.5, None, None, "Lie effect varies; treat as range-based"),
        "+2":   HoselSettingDelta(+2.0, None, None, "Lie effect varies; treat as range-based"),
    },
    deltas_lh={},
    loft_range_deg=(-1.0, +2.0),
    notes="Wilson Fast Fit is less common now, but still shows up in older bags."
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
    "Srixon": [SLEEVE_12_STD_15],
    "Cleveland": [SLEEVE_12_STD_15],
    "Mizuno": [MIZUNO_QUICK_SWITCH],
    "PXG": [PXG_ADAPTER_8],
    "Wilson": [WILSON_FAST_FIT_6],
}


def get_supported_brands() -> List[str]:
    return sorted(HOSEL_SYSTEMS_BY_BRAND.keys())


def get_brand_systems(brand: str) -> List[HoselSystem]:
    return HOSEL_SYSTEMS_BY_BRAND.get(brand, [])


def list_settings(brand: str, system_name: str, handedness: Handedness) -> List[str]:
    systems = get_brand_systems(brand)
    for s in systems:
        if s.system_name == system_name:
            return s.settings_rh if handedness == "RH" else s.settings_lh
    return []


def translate_setting(
    brand: str,
    system_name: str,
    setting: str,
    handedness: Handedness
) -> HoselSettingDelta:
    systems = get_brand_systems(brand)
    for s in systems:
        if s.system_name != system_name:
            continue
        deltas = s.deltas_rh if handedness == "RH" else s.deltas_lh
        if setting in deltas:
            return deltas[setting]
        # fallback = range-only / unknown
        return HoselSettingDelta(
            None, None, None,
            note=f"No exact delta stored for {brand} / {system_name} / {setting}. Use system ranges: "
                 f"loft{ s.loft_range_deg }, lie{ s.lie_range_deg }, face{ s.face_range_deg }"
        )
    return HoselSettingDelta(None, None, None, note="Brand/system not found.")
