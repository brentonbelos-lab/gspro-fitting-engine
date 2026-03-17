from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from viz import render_dispersion, render_compare_dispersion

from hosel_db import (
    get_supported_brands,
    get_brand_systems,
    list_settings,
    translate_setting,
    system_ranges,
)

from fit_engine import (
    DriverUserSetup,
    build_driver_recommendations,
    build_non_driver_recommendations,
    canonicalize,
    compare_driver_setups,
    distance_potential_for_summary,
    estimate_launch_spin_change,
    miss_tendency,
    pick_one_hosel_setting,
    shot_shape_summary,
    smash_flag_driver,
    summarize_by_club,
    targets_for_club,
)


# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="FitCaddie", layout="wide")


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1500px;
    }

    section.main > div {
        gap: 0.55rem;
    }

    :root {
        --fc-navy: #17324d;
        --fc-blue: #2f80ed;
        --fc-blue-soft: #eef5ff;
        --fc-line: #dbe8f5;
        --fc-card: #ffffff;
        --fc-text-soft: #577089;
        --fc-green-bg: #edf9f1;
        --fc-green-bd: #8ed1a8;
        --fc-yellow-bg: #fff7e5;
        --fc-yellow-bd: #e7c76a;
        --fc-red-bg: #fff0f0;
        --fc-red-bd: #e1a1a1;
    }

    html, body, [class*="css"] {
        font-size: 15px;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.01em;
    }

    .fc-shell {
        margin-bottom: 0.6rem;
    }

    .fc-hero {
        background: linear-gradient(135deg, #17324d 0%, #245f9c 100%);
        color: white;
        border-radius: 20px;
        padding: 20px 22px;
        margin-bottom: 10px;
        box-shadow: 0 10px 28px rgba(18, 49, 77, 0.16);
    }

    .fc-hero h1 {
        margin: 0 0 6px 0;
        font-size: 1.9rem;
    }

    .fc-hero p {
        margin: 0;
        color: rgba(255,255,255,0.92);
    }

    .fc-card {
        background: var(--fc-card);
        border: 1px solid var(--fc-line);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(28, 78, 128, 0.06);
        margin-bottom: 12px;
    }

    .fc-card h3 {
        margin-top: 0;
        margin-bottom: 0.65rem;
        color: var(--fc-navy);
    }

    .fc-subtle {
        color: var(--fc-text-soft);
        font-size: 0.92rem;
    }

    .fc-rec {
        border-radius: 16px;
        padding: 14px 15px;
        margin-bottom: 10px;
        border: 2px solid;
    }

    .fc-rec-green {
        background: var(--fc-green-bg);
        border-color: var(--fc-green-bd);
    }

    .fc-rec-yellow {
        background: var(--fc-yellow-bg);
        border-color: var(--fc-yellow-bd);
    }

    .fc-rec-red {
        background: var(--fc-red-bg);
        border-color: var(--fc-red-bd);
    }

    .fc-status {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .fc-status-green {
        background: #d9f3e2;
        color: #17653a;
    }

    .fc-status-yellow {
        background: #fff0c8;
        color: #7a5b00;
    }

    .fc-status-red {
        background: #ffd7d7;
        color: #8a1f1f;
    }

    .fc-verdict {
        background: var(--fc-blue-soft);
        border: 2px solid var(--fc-blue);
        border-radius: 18px;
        padding: 15px 16px;
        margin-bottom: 12px;
    }

    .fc-stat {
        border: 1px solid #e6eef7;
        border-radius: 14px;
        padding: 12px 13px;
        background: #fbfdff;
        height: 100%;
    }

    .fc-stat-label {
        color: var(--fc-text-soft);
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .fc-stat-value {
        color: var(--fc-navy);
        font-size: 1.28rem;
        font-weight: 800;
        line-height: 1.2;
    }

    div[data-testid="metric-container"] {
        background: #fbfdff;
        border: 1px solid #e6eef7;
        padding: 0.7rem 0.8rem;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 8px 14px;
    }

    .stExpander {
        border-radius: 14px;
        border: 1px solid #e6eef7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# APP TITLE
# =========================================================
st.markdown(
    """
    <div class="fc-shell">
        <div class="fc-hero">
            <h1>FitCaddie</h1>
            <p>Upload GSPro data, pick a club, and get cleaner fitting guidance without the clutter.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
def _init_state():
    defaults = {
        "analysis_mode": "Single Club Analysis",
        "k_loft_to_dynamic": 1.0,
        "min_shots": 5,
        "show_raw": False,
        "selected_focus_family": "Driver",
        "selected_focus_club": "DR",

        "single_driver_brand": "Titleist",
        "single_driver_model": "TSR3",
        "single_driver_loft": 10.0,
        "single_driver_hosel": "A1",
        "single_driver_shaft_model": "HZRDUS Black",
        "single_driver_shaft_weight": 60.0,
        "single_driver_shaft_flex": "6.0",

        "single_nd_brand": "Titleist",
        "single_nd_model": "GT2",
        "single_nd_loft": 15.0,
        "single_nd_hosel": "A1",
        "single_nd_shaft_model": "HZRDUS Black",
        "single_nd_shaft_weight": 80.0,
        "single_nd_shaft_flex": "6.0",

        "cmpA_driver_brand": "Titleist",
        "cmpA_driver_model": "TSR3",
        "cmpA_driver_loft": 10.0,
        "cmpA_driver_hosel": "A1",
        "cmpA_driver_shaft_model": "HZRDUS Black",
        "cmpA_driver_shaft_weight": 60.0,
        "cmpA_driver_shaft_flex": "6.0",

        "cmpB_driver_brand": "Titleist",
        "cmpB_driver_model": "TSR3",
        "cmpB_driver_loft": 10.0,
        "cmpB_driver_hosel": "A2",
        "cmpB_driver_shaft_model": "HZRDUS Smoke Red RDX",
        "cmpB_driver_shaft_weight": 60.0,
        "cmpB_driver_shaft_flex": "6.0",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# =========================================================
# CONSTANTS
# =========================================================
WEDGE_CODES = {"PW", "AW", "GW", "SW", "LW"}
FAIRWAY_CODES = {"2W", "3W", "4W", "5W", "7W", "9W", "11W"}
HYBRID_CODES = {"2H", "3H", "4H", "5H", "6H", "7H"}

TOP_BRANDS = [
    "Titleist",
    "TaylorMade",
    "Callaway",
    "PING",
    "Mizuno",
    "Srixon",
    "Cleveland",
    "Cobra",
    "PXG",
    "Wilson Staff",
    "Other",
]

DRIVER_MODEL_OPTIONS = {
    "Titleist": ["GT2", "GT3", "GT4", "TSR2", "TSR3", "TSR4", "Other"],
    "TaylorMade": ["Qi35", "Qi35 Max", "Qi35 LS", "Qi35 Max Lite", "Qi10", "Qi10 LS", "Other"],
    "Callaway": ["Elyte", "Elyte X", "Elyte Triple Diamond", "Elyte Triple Diamond Max", "Paradym Ai Smoke Max", "Paradym Ai Smoke Triple Diamond", "Other"],
    "PING": ["G440 MAX", "G440 LST", "G440 SFT", "G440 K", "G430 MAX 10K", "G430 LST", "Other"],
    "Mizuno": ["ST-G 440", "ST-MAX 230", "ST-X 230", "ST-Z 230", "Other"],
    "Srixon": ["ZXi", "ZXi LS", "ZXi MAX", "ZX5 LS Mk II", "ZX5", "ZX7 Mk II", "Other"],
    "Cleveland": ["Other"],
    "Cobra": ["DS-ADAPT X", "DS-ADAPT LS", "DS-ADAPT MAX-K", "DS-ADAPT MAX-D", "DARKSPEED X", "DARKSPEED LS", "Other"],
    "PXG": ["Black Ops", "Black Ops Tour-1", "Black Ops Ultra-Lite", "0311 GEN6", "0311 XF GEN6", "Other"],
    "Wilson Staff": ["DYNAPWR LS", "DYNAPWR Carbon", "DYNAPWR Max", "Other"],
    "Other": ["Other"],
}

FAIRWAY_MODEL_OPTIONS = {
    "Titleist": ["GT2", "GT3", "TSR2", "TSR3", "TSi2", "TSi3", "Other"],
    "TaylorMade": ["Qi35", "Qi35 Max", "Qi10", "Qi10 Tour", "Stealth 2", "Other"],
    "Callaway": ["Elyte", "Elyte X", "Elyte Triple Diamond", "Paradym Ai Smoke Max", "Other"],
    "PING": ["G440 MAX", "G440 LST", "G440 SFT", "G430 MAX", "G430 LST", "Other"],
    "Mizuno": ["ST-MAX 230", "ST-X 230", "ST-Z 230", "Other"],
    "Srixon": ["ZXi", "ZX Mk II", "ZX", "Other"],
    "Cleveland": ["Launcher XL Halo", "Other"],
    "Cobra": ["DS-ADAPT X", "DS-ADAPT LS", "DS-ADAPT MAX", "DARKSPEED X", "Other"],
    "PXG": ["Black Ops", "0311 Black Ops", "0211", "Other"],
    "Wilson Staff": ["DYNAPWR Carbon", "DYNAPWR Max", "Other"],
    "Other": ["Other"],
}

HYBRID_MODEL_OPTIONS = {
    "Titleist": ["GT2", "GT3", "TSR2", "TSR3", "TSi2", "TSi3", "Other"],
    "TaylorMade": ["Qi35 Rescue", "Qi10 Rescue", "Stealth 2 Rescue", "Other"],
    "Callaway": ["Elyte", "Elyte X", "Paradym Ai Smoke", "Apex UW", "Other"],
    "PING": ["G440", "G430", "G425", "iCrossover", "Other"],
    "Mizuno": ["CLK", "ST-MAX 230", "JPX Fli-Hi", "Other"],
    "Srixon": ["ZXi", "ZX Mk II", "ZX", "Other"],
    "Cleveland": ["Halo XL", "Launcher XL Halo", "Other"],
    "Cobra": ["DS-ADAPT", "DARKSPEED", "AEROJET", "KING TEC", "Other"],
    "PXG": ["Black Ops", "0311 Black Ops", "0211", "Other"],
    "Wilson Staff": ["DYNAPWR", "Launch Pad 2", "Other"],
    "Other": ["Other"],
}

IRON_MODEL_OPTIONS = {
    "Titleist": ["T100", "T150", "T200", "T350", "U505", "Other"],
    "TaylorMade": ["P7CB", "P7MC", "P770", "P790", "Qi", "Other"],
    "Callaway": ["Apex Pro", "Apex CB", "Apex Ai200", "Apex Ai300", "Paradym Ai Smoke", "Other"],
    "PING": ["Blueprint T", "Blueprint S", "i530", "i230", "G430", "Other"],
    "Mizuno": ["JPX 923 Tour", "JPX 923 Forged", "JPX 925 Hot Metal", "Pro 243", "Pro 245", "Other"],
    "Srixon": ["ZX7", "ZX5", "ZX4", "Z-Forged", "Other"],
    "Cleveland": ["Launcher XL Halo Irons", "Other"],
    "Cobra": ["King Tour", "Forged Tec", "Darkspeed", "Aerojet", "Other"],
    "PXG": ["0317 T", "0317 CB", "0311 P", "0311 XP", "Other"],
    "Wilson Staff": ["Staff Model CB", "Staff Model Blade", "Dynapwr Forged", "Dynapwr", "Other"],
    "Other": ["Other"],
}

WEDGE_MODEL_OPTIONS = {
    "Titleist": ["Vokey SM10", "Vokey SM9", "Vokey SM8", "Other"],
    "TaylorMade": ["MG4", "Hi-Toe 3", "Other"],
    "Callaway": ["Opus", "Jaws Raw", "Other"],
    "PING": ["s159", "Glide 4.0", "Other"],
    "Mizuno": ["T24", "S23", "Other"],
    "Srixon": ["Cleveland RTX 6 ZipCore", "Cleveland CBX 4 ZipCore", "Other"],
    "Cleveland": ["RTX 6 ZipCore", "CBX 4 ZipCore", "Smart Sole", "Other"],
    "Cobra": ["Snakebite", "King Wedge", "Other"],
    "PXG": ["Sugar Daddy III", "Other"],
    "Wilson Staff": ["Staff Model", "Harmonized", "Other"],
    "Other": ["Other"],
}


# =========================================================
# HELPERS
# =========================================================
def _fmt(value, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        if np.isnan(value):
            return "—"
    except Exception:
        pass
    return f"{value:.{decimals}f}{suffix}"


def _fmt_diff(value, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        if np.isnan(value):
            return "—"
    except Exception:
        pass
    return f"{value:+.{decimals}f}{suffix}"


def _normalize_club_id(club_id: str) -> str:
    if club_id is None:
        return ""

    c = str(club_id).strip().upper()

    # Convert prefix iron/hybrid/wood styles to suffix styles
    # I4 -> 4I, H3 -> 3H, W3 -> 3W
    if len(c) >= 2 and c[0] in {"I", "H", "W"} and c[1:].isdigit():
        c = f"{c[1:]}{c[0]}"

    return c


def _is_iron_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return len(c) >= 2 and c[:-1].isdigit() and c.endswith("I")


def _is_wood_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return len(c) >= 2 and c[:-1].isdigit() and c.endswith("W")


def _is_hybrid_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return len(c) >= 2 and c[:-1].isdigit() and c.endswith("H")


def _is_wedge_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return c in {"PW", "GW", "AW", "UW", "SW", "LW"}


def _normalize_club_id(club_id: str) -> str:
    if club_id is None:
        return ""

    c = str(club_id).strip().upper()

    # I4 -> 4I, H3 -> 3H, W3 -> 3W
    if len(c) >= 2 and c[0] in {"I", "H", "W"} and c[1:].isdigit():
        c = f"{c[1:]}{c[0]}"

    return c


def _is_iron_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return len(c) >= 2 and c[:-1].isdigit() and c.endswith("I")


def _is_wood_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return len(c) >= 2 and c[:-1].isdigit() and c.endswith("W")


def _is_hybrid_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return len(c) >= 2 and c[:-1].isdigit() and c.endswith("H")


def _is_wedge_id(club_id: str) -> bool:
    c = _normalize_club_id(club_id)
    return c in {"PW", "GW", "AW", "UW", "SW", "LW"}


def _club_family_from_id(club_id: str) -> str:
    c = _normalize_club_id(club_id)

    if c == "DR":
        return "Driver"
    if _is_wood_id(c):
        return "Fairway Wood"
    if _is_hybrid_id(c):
        return "Hybrid"
    if _is_iron_id(c):
        return "Iron"
    if _is_wedge_id(c):
        return "Wedge"
    return "Other"

def _club_sort_key(club_id: str):
    order = {
        "DR": 0,
        "2W": 1, "3W": 2, "4W": 3, "5W": 4, "7W": 5, "9W": 6, "11W": 7,
        "2H": 10, "3H": 11, "4H": 12, "5H": 13, "6H": 14, "7H": 15,
        "3I": 20, "4I": 21, "5I": 22, "6I": 23, "7I": 24, "8I": 25, "9I": 26,
        "PW": 30, "AW": 31, "GW": 32, "SW": 33, "LW": 34,
        "PT": 99,
    }
    return order.get(str(club_id).upper().strip(), 999)


def _default_loft_for_club(club_id: str) -> float:
    c = _normalize_club_id(club_id)

    mapping = {
        "DR": 10.0,
        "2W": 13.0, "3W": 15.0, "4W": 16.5, "5W": 18.0, "7W": 21.0,
        "2H": 17.0, "3H": 19.0, "4H": 21.0, "5H": 24.0,
        "3I": 21.0, "4I": 24.0, "5I": 27.0, "6I": 30.0,
        "7I": 34.0, "8I": 38.0, "9I": 42.0,
        "PW": 46.0, "GW": 50.0, "AW": 50.0, "UW": 52.0, "SW": 54.0, "LW": 58.0,
    }
    return mapping.get(c, 20.0)


def _model_options_for_family(brand: str, family: str) -> List[str]:
    if family == "Fairway Wood":
        return FAIRWAY_MODEL_OPTIONS.get(brand, ["Other"])
    if family == "Hybrid":
        return HYBRID_MODEL_OPTIONS.get(brand, ["Other"])
    if family == "Iron":
        return IRON_MODEL_OPTIONS.get(brand, ["Other"])
    if family == "Wedge":
        return WEDGE_MODEL_OPTIONS.get(brand, ["Other"])
    return ["Other"]


def _available_families_from_clubs(selected_clubs: List[str]) -> List[str]:
    clubs = [_normalize_club_id(c) for c in selected_clubs]

    families_present = []
    if any(c == "DR" for c in clubs):
        families_present.append("Driver")
    if any(_is_wood_id(c) for c in clubs):
        families_present.append("Fairway Wood")
    if any(_is_hybrid_id(c) for c in clubs):
        families_present.append("Hybrid")
    if any(_is_iron_id(c) for c in clubs):
        families_present.append("Iron")
    if any(_is_wedge_id(c) for c in clubs):
        families_present.append("Wedge")

    return families_present


def _clubs_for_family(selected_clubs: List[str], family: str) -> List[str]:
    clubs = [str(c).upper().strip() for c in selected_clubs]

    if family == "Driver":
        available = [c for c in clubs if c == "DR"]
    elif family == "Fairway Wood":
        available = [c for c in clubs if c in FAIRWAY_CODES]
    elif family == "Hybrid":
        available = [c for c in clubs if c in HYBRID_CODES]
    elif family == "Iron":
        available = [c for c in clubs if c.endswith("I")]
    elif family == "Wedge":
        available = [c for c in clubs if c in WEDGE_CODES]
    else:
        available = []

    return sorted(set(available), key=_club_sort_key)


def _driver_setup_from_prefix(prefix: str) -> DriverUserSetup:
    return DriverUserSetup(
        brand=st.session_state[f"{prefix}_driver_brand"],
        model=st.session_state[f"{prefix}_driver_model"],
        loft_deg=float(st.session_state[f"{prefix}_driver_loft"]),
        hosel_setting=st.session_state[f"{prefix}_driver_hosel"],
        shaft_model=st.session_state[f"{prefix}_driver_shaft_model"],
        shaft_weight_g=float(st.session_state[f"{prefix}_driver_shaft_weight"]),
        shaft_flex=st.session_state[f"{prefix}_driver_shaft_flex"],
    )


def _club_build_from_prefix(prefix: str) -> Dict[str, object]:
    return {
        "brand": st.session_state[f"{prefix}_brand"],
        "model": st.session_state[f"{prefix}_model"],
        "loft_deg": float(st.session_state[f"{prefix}_loft"]),
        "hosel_setting": st.session_state[f"{prefix}_hosel"],
        "shaft_model": st.session_state[f"{prefix}_shaft_model"],
        "shaft_weight_g": float(st.session_state[f"{prefix}_shaft_weight"]),
        "shaft_flex": st.session_state[f"{prefix}_shaft_flex"],
    }


def _status_html(tone: str) -> str:
    if tone == "green":
        return '<span class="fc-status fc-status-green">Positive change</span>'
    if tone == "yellow":
        return '<span class="fc-status fc-status-yellow">Test carefully</span>'
    return '<span class="fc-status fc-status-red">Avoid first</span>'


def _render_stat_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="fc-stat">
            <div class="fc-stat-label">{label}</div>
            <div class="fc-stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOCUS PICKER
# =========================================================
def _club_sort_key(club_id: str):
    c = _normalize_club_id(club_id)

    if c == "DR":
        return (0, 0)

    if _is_wood_id(c):
        return (1, int(c[:-1]))

    if _is_hybrid_id(c):
        return (2, int(c[:-1]))

    if _is_iron_id(c):
        return (3, int(c[:-1]))

    wedge_order = {
        "PW": 0,
        "GW": 1,
        "AW": 2,
        "UW": 3,
        "SW": 4,
        "LW": 5,
    }
    if _is_wedge_id(c):
        return (4, wedge_order.get(c, 99))

    return (9, 999)


def _render_focus_picker(selected_clubs: List[str], club_counts: Dict[str, int]) -> str:
    normalized_map = {_normalize_club_id(c): c for c in selected_clubs}
    normalized_clubs = list(normalized_map.keys())

    families_present = _available_families_from_clubs(normalized_clubs)

    if not families_present:
        st.warning("No supported club data found.")
        st.stop()

    if st.session_state["selected_focus_family"] not in families_present:
        st.session_state["selected_focus_family"] = families_present[0]

    st.markdown('<div class="fc-card">', unsafe_allow_html=True)
    st.subheader("Choose Fitting Focus")

    if len(families_present) > 1:
        st.session_state["selected_focus_family"] = st.radio(
            "What are we fitting today?",
            families_present,
            horizontal=True,
            index=families_present.index(st.session_state["selected_focus_family"]),
        )
    else:
        st.session_state["selected_focus_family"] = families_present[0]
        st.caption(f"Detected focus: {families_present[0]}")

    family = st.session_state["selected_focus_family"]

    if family == "Driver":
        available = [c for c in normalized_clubs if c == "DR"]
    elif family == "Fairway Wood":
        available = [c for c in normalized_clubs if _is_wood_id(c)]
    elif family == "Hybrid":
        available = [c for c in normalized_clubs if _is_hybrid_id(c)]
    elif family == "Iron":
        available = [c for c in normalized_clubs if _is_iron_id(c)]
    else:
        available = [c for c in normalized_clubs if _is_wedge_id(c)]

    available = sorted(available, key=_club_sort_key)

    if not available:
        st.warning("No clubs found in that family.")
        st.stop()

    if st.session_state["selected_focus_club"] not in available:
        st.session_state["selected_focus_club"] = available[0]

    option_labels = [f"{c} ({club_counts.get(normalized_map[c], club_counts.get(c, 0))} shots)" for c in available]
    label_to_club = dict(zip(option_labels, available))

    current_label = next(
        (lbl for lbl, cid in label_to_club.items() if cid == st.session_state["selected_focus_club"]),
        option_labels[0],
    )

    chosen_label = st.selectbox(
        "Choose club",
        option_labels,
        index=option_labels.index(current_label),
    )

    st.session_state["selected_focus_club"] = label_to_club[chosen_label]

    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state["selected_focus_club"]

# =========================================================
# SETUP / BUILD FORMS
# =========================================================
def _render_driver_setup(prefix: str, title: str):
    st.markdown(f'<div class="fc-card"><h3>{title}</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        current_brand = st.session_state[f"{prefix}_driver_brand"]
        if current_brand not in TOP_BRANDS:
            current_brand = "Other"
        st.session_state[f"{prefix}_driver_brand"] = st.selectbox(
            "Driver Brand",
            TOP_BRANDS,
            index=TOP_BRANDS.index(current_brand),
            key=f"{prefix}_driver_brand_select",
        )

    with c2:
        brand = st.session_state[f"{prefix}_driver_brand"]
        model_options = DRIVER_MODEL_OPTIONS.get(brand, ["Other"])
        current_model = st.session_state[f"{prefix}_driver_model"]
        if current_model not in model_options:
            current_model = model_options[0]
        st.session_state[f"{prefix}_driver_model"] = st.selectbox(
            "Head Model",
            model_options,
            index=model_options.index(current_model),
            key=f"{prefix}_driver_model_select",
        )

    with c3:
        st.session_state[f"{prefix}_driver_loft"] = st.number_input(
            "Loft (°)",
            min_value=7.0,
            max_value=14.0,
            value=float(st.session_state[f"{prefix}_driver_loft"]),
            step=0.5,
            key=f"{prefix}_driver_loft_input",
        )

    with c4:
        st.session_state[f"{prefix}_driver_hosel"] = st.text_input(
            "Hosel Setting",
            value=st.session_state[f"{prefix}_driver_hosel"],
            key=f"{prefix}_driver_hosel_input",
        )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.session_state[f"{prefix}_driver_shaft_model"] = st.text_input(
            "Shaft Model",
            value=st.session_state[f"{prefix}_driver_shaft_model"],
            key=f"{prefix}_driver_shaft_model_input",
        )

    with s2:
        st.session_state[f"{prefix}_driver_shaft_weight"] = st.number_input(
            "Shaft Weight (g)",
            min_value=40.0,
            max_value=90.0,
            value=float(st.session_state[f"{prefix}_driver_shaft_weight"]),
            step=1.0,
            key=f"{prefix}_driver_shaft_weight_input",
        )

    with s3:
        st.session_state[f"{prefix}_driver_shaft_flex"] = st.text_input(
            "Shaft Flex",
            value=st.session_state[f"{prefix}_driver_shaft_flex"],
            key=f"{prefix}_driver_shaft_flex_input",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_non_driver_build(prefix: str, title: str, club_id: str):
    family = _club_family_from_id(club_id)
    adjustable = family in {"Fairway Wood", "Hybrid"}

    if family == "Other":
        return

    default_loft = _default_loft_for_club(club_id)
    st.session_state.setdefault(f"{prefix}_loft", default_loft)
    st.session_state.setdefault(f"{prefix}_hosel", "")
    st.session_state.setdefault(f"{prefix}_brand", "Titleist")
    st.session_state.setdefault(f"{prefix}_model", "Other")
    st.session_state.setdefault(f"{prefix}_shaft_model", "")
    st.session_state.setdefault(f"{prefix}_shaft_weight", 80.0 if family in {"Fairway Wood", "Hybrid"} else 110.0)
    st.session_state.setdefault(f"{prefix}_shaft_flex", "6.0")

    st.markdown(f'<div class="fc-card"><h3>{title}</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        current_brand = st.session_state[f"{prefix}_brand"]
        if current_brand not in TOP_BRANDS:
            current_brand = "Other"
        st.session_state[f"{prefix}_brand"] = st.selectbox(
            f"{family} Brand",
            TOP_BRANDS,
            index=TOP_BRANDS.index(current_brand),
            key=f"{prefix}_brand_select",
        )

    with c2:
        brand = st.session_state[f"{prefix}_brand"]
        model_options = _model_options_for_family(brand, family)
        current_model = st.session_state[f"{prefix}_model"]
        if current_model not in model_options:
            current_model = model_options[0]
        st.session_state[f"{prefix}_model"] = st.selectbox(
            f"{family} Model",
            model_options,
            index=model_options.index(current_model),
            key=f"{prefix}_model_select",
        )

    with c3:
        min_loft = 10.0 if family in {"Fairway Wood", "Hybrid"} else 15.0
        max_loft = 30.0 if family in {"Fairway Wood", "Hybrid"} else 65.0
        loft_val = float(st.session_state.get(f"{prefix}_loft", default_loft))
        if loft_val < min_loft or loft_val > max_loft:
            loft_val = default_loft

        st.session_state[f"{prefix}_loft"] = st.number_input(
            "Loft (°)",
            min_value=float(min_loft),
            max_value=float(max_loft),
            value=float(loft_val),
            step=0.5,
            key=f"{prefix}_loft_input",
        )

    with c4:
        if adjustable:
            st.session_state[f"{prefix}_hosel"] = st.text_input(
                "Hosel Setting",
                value=st.session_state[f"{prefix}_hosel"],
                key=f"{prefix}_hosel_input",
            )
        else:
            st.session_state[f"{prefix}_hosel"] = ""
            st.caption("No adjustable hosel input needed for this club.")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.session_state[f"{prefix}_shaft_model"] = st.text_input(
            "Shaft Model",
            value=st.session_state[f"{prefix}_shaft_model"],
            key=f"{prefix}_shaft_model_input",
        )

    with s2:
        max_weight = 140.0 if family in {"Iron", "Wedge"} else 120.0
        st.session_state[f"{prefix}_shaft_weight"] = st.number_input(
            "Shaft Weight (g)",
            min_value=40.0,
            max_value=float(max_weight),
            value=float(st.session_state[f"{prefix}_shaft_weight"]),
            step=1.0,
            key=f"{prefix}_shaft_weight_input",
        )

    with s3:
        st.session_state[f"{prefix}_shaft_flex"] = st.text_input(
            "Shaft Flex",
            value=st.session_state[f"{prefix}_shaft_flex"],
            key=f"{prefix}_shaft_flex_input",
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# HOSEL HELPERS
# =========================================================
def _reset_system_and_settings_for_club(club_id: str):
    brand_key = f"{club_id}_brand"
    sys_key = f"{club_id}_sys"
    cur_key = f"{club_id}_cur"
    new_key = f"{club_id}_new"
    hand_key = f"{club_id}_hand"

    brand = st.session_state.get(brand_key)
    handedness = st.session_state.get(hand_key, "RH")

    systems = get_brand_systems(brand) if brand else []
    system_names = [s.system_name for s in systems]

    if not system_names:
        st.session_state[sys_key] = "(no systems found)"
        st.session_state[cur_key] = "STD"
        st.session_state[new_key] = "STD"
        return

    st.session_state[sys_key] = system_names[0]
    settings = list_settings(brand, st.session_state[sys_key], handedness)
    first_setting = settings[0] if settings else "STD"
    st.session_state[cur_key] = first_setting
    st.session_state[new_key] = first_setting


def _reset_settings_for_club(club_id: str):
    brand_key = f"{club_id}_brand"
    sys_key = f"{club_id}_sys"
    cur_key = f"{club_id}_cur"
    new_key = f"{club_id}_new"
    hand_key = f"{club_id}_hand"

    brand = st.session_state.get(brand_key)
    system_name = st.session_state.get(sys_key)
    handedness = st.session_state.get(hand_key, "RH")

    settings = list_settings(brand, system_name, handedness) if (brand and system_name) else []
    first_setting = settings[0] if settings else "STD"
    st.session_state[cur_key] = first_setting
    st.session_state[new_key] = first_setting


def _render_hosel_block(club_id: str, title: str, k_loft_to_dynamic: float) -> Dict[str, Dict]:
    hosel_configs: Dict[str, Dict] = {}
    family = _club_family_from_id(club_id)

    if family not in {"Driver", "Fairway Wood", "Hybrid"}:
        st.markdown(
            f"""
            <div class="fc-card">
                <h3>{title}</h3>
                <p class="fc-subtle">Hosel setting recommendations apply only to adjustable driver, fairway, and hybrid heads.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return hosel_configs

    st.markdown(f'<div class="fc-card"><h3>{title}</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.0, 1.3, 1.2, 1.1])

    with c1:
        handedness = st.selectbox(
            f"{club_id} Hand",
            ["RH", "LH"],
            index=0,
            key=f"{club_id}_hand",
            on_change=_reset_settings_for_club,
            args=(club_id,),
        )

    with c2:
        supported_brands = get_supported_brands()
        brand = st.selectbox(
            f"{club_id} Brand",
            supported_brands,
            index=0,
            key=f"{club_id}_brand",
            on_change=_reset_system_and_settings_for_club,
            args=(club_id,),
        )

    systems = get_brand_systems(brand)
    system_names = [s.system_name for s in systems] if systems else ["(no systems found)"]

    with c3:
        system_name = st.selectbox(
            f"{club_id} Hosel System",
            system_names,
            index=0,
            key=f"{club_id}_sys",
            on_change=_reset_settings_for_club,
            args=(club_id,),
        )

    with c4:
        default_loft = float(_default_loft_for_club(club_id))

        if family == "Driver":
            min_loft, max_loft = 6.0, 15.0
        elif family == "Fairway Wood":
            min_loft, max_loft = 12.0, 24.0
        else:
            min_loft, max_loft = 16.0, 30.0

        default_loft = max(min_loft, min(default_loft, max_loft))

        stated_loft = st.number_input(
            f"{club_id} Stated Loft (°)",
            min_value=float(min_loft),
            max_value=float(max_loft),
            value=float(default_loft),
            step=0.5,
            key=f"{club_id}_loft",
        )

    settings = list_settings(brand, system_name, handedness)

    s1, s2 = st.columns(2)
    with s1:
        current_setting = st.selectbox(
            f"{club_id} Current Setting",
            settings if settings else ["STD"],
            key=f"{club_id}_cur",
        )
    with s2:
        proposed_setting = st.selectbox(
            f"{club_id} Proposed Setting",
            settings if settings else ["STD"],
            key=f"{club_id}_new",
        )

    cur_delta = translate_setting(brand, system_name, current_setting, handedness)
    new_delta = translate_setting(brand, system_name, proposed_setting, handedness)

    cur_loft = getattr(cur_delta, "loft_deg", None)
    new_loft = getattr(new_delta, "loft_deg", None)

    st.markdown("#### Projected Change")

    if (cur_loft is not None) and (new_loft is not None) and (proposed_setting != current_setting):
        delta_static_loft = new_loft - cur_loft
        est = estimate_launch_spin_change(delta_static_loft, k_loft_to_dynamic, club_id)

        st.info(
            f"Estimated launch change: **{est.launch_change_deg:+.1f}°** "
            f"(range {est.launch_range_deg[0]:+.1f}° to {est.launch_range_deg[1]:+.1f}°)\n\n"
            f"Estimated spin change: **{est.spin_change_rpm:+d} rpm** "
            f"(range {est.spin_range_rpm[0]:+d} to {est.spin_range_rpm[1]:+d})\n\n"
            f"Estimated carry effect: **{est.carry_change_yd:+.1f} yds** "
            f"(range {est.carry_range_yd[0]:+.1f} to {est.carry_range_yd[1]:+.1f})\n\n"
            f"Estimated peak height effect: **{est.peak_height_change_yd:+.1f} yds** "
            f"(range {est.peak_height_range_yd[0]:+.1f} to {est.peak_height_range_yd[1]:+.1f})"
        )
        st.caption(est.notes)
    elif proposed_setting != current_setting:
        ranges = system_ranges(brand, system_name)
        st.warning(
            "Exact deltas for these settings are not encoded yet.\n\n"
            f"System ranges: loft={ranges.get('loft_range_deg')}, lie={ranges.get('lie_range_deg')}."
        )
    else:
        st.caption("Choose a different proposed setting to see projected launch, spin, carry, and height changes.")

    st.markdown("</div>", unsafe_allow_html=True)

    hosel_configs[club_id] = {
        "club_id": club_id,
        "handedness": handedness,
        "brand": brand,
        "system_name": system_name,
        "stated_loft": stated_loft,
        "current_setting": current_setting,
        "proposed_setting": proposed_setting,
        "cur_delta": asdict(cur_delta),
        "new_delta": asdict(new_delta),
    }

    return hosel_configs


# =========================================================
# RENDER HELPERS
# =========================================================
def _render_recommendation_cards(bundle):
    for block in [bundle.swing, bundle.driver_settings, bundle.equipment_adjustment]:
        css_class = {
            "green": "fc-rec fc-rec-green",
            "yellow": "fc-rec fc-rec-yellow",
            "red": "fc-rec fc-rec-red",
        }.get(block.tone, "fc-rec fc-rec-yellow")

        st.markdown(
            f"""
            <div class="{css_class}">
                {_status_html(block.tone)}
                <h4 style="margin:0 0 8px 0;">{block.title}</h4>
                <p style="margin:0 0 6px 0;"><strong>Suggestion:</strong> {block.suggestion}</p>
                <p style="margin:0;"><strong>Why:</strong> {block.why}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_summary_cards(summary, focus_df: pd.DataFrame):
    r1 = st.columns(4)
    with r1[0]:
        _render_stat_card("Club Speed", _fmt(summary.club_speed_avg, 1))
    with r1[1]:
        _render_stat_card("Ball Speed", _fmt(summary.ball_speed_avg, 1))
    with r1[2]:
        _render_stat_card("Smash", _fmt(summary.smash_avg, 2))
    with r1[3]:
        _render_stat_card("Carry", _fmt(summary.carry_avg, 1))

    r2 = st.columns(4)
    with r2[0]:
        _render_stat_card("Offline", _fmt(summary.offline_avg, 1))
    with r2[1]:
        _render_stat_card("Launch", _fmt(summary.vla_avg, 1))
    with r2[2]:
        _render_stat_card("Spin", _fmt(summary.spin_avg, 0))
    with r2[3]:
        _render_stat_card("AoA", _fmt(summary.aoa_avg, 1))

    r3 = st.columns(4)
    with r3[0]:
        _render_stat_card("Peak Height", _fmt(getattr(summary, "peak_height_avg", np.nan), 1))
    with r3[1]:
        _render_stat_card("Descent", _fmt(getattr(summary, "descent_avg", np.nan), 1))
    with r3[2]:
        _render_stat_card("Club Speed SD", _fmt(summary.club_speed_std, 1))
    with r3[3]:
        _render_stat_card("Offline SD", _fmt(summary.offline_std, 1))

    shape = shot_shape_summary(focus_df)
    dp = distance_potential_for_summary(summary)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Shot Shape")
        st.write(f"**Typical shape:** {shape.shape_label}")
        st.write(f"**Start line:** {shape.start_line}")
        st.write(f"**Curve:** {shape.curve}")

    with c2:
        st.markdown("#### Distance Potential")
        st.write(f"**Expected carry:** {_fmt(dp.expected_carry_yd, 1)} yds")
        st.write(f"**Actual carry:** {_fmt(dp.actual_carry_yd, 1)} yds")
        st.write(f"**Carry gap:** {_fmt(dp.carry_gap_yd, 1)} yds")
        st.caption(dp.message)


def _render_driver_recommendations(driver_df: pd.DataFrame, driver_setup: DriverUserSetup):
    summaries = summarize_by_club(driver_df)
    if "DR" not in summaries:
        return None

    offline_valid = driver_df["offline_yd"].dropna()
    fairway_pct = float((offline_valid.abs() <= 15).mean() * 100.0) if len(offline_valid) else None

    return build_driver_recommendations(
        summary=summaries["DR"],
        user_setup=driver_setup,
        fairway_hit_pct=fairway_pct,
    )


def _render_non_driver_recommendations(
    focus_summary,
    hosel_configs: Dict[str, Dict],
    build_cfg: Optional[Dict[str, object]] = None,
):
    club_id = focus_summary.club_id
    cfg = hosel_configs.get(club_id, {})
    build_cfg = build_cfg or {}

    return build_non_driver_recommendations(
        summary=focus_summary,
        stated_loft_deg=build_cfg.get("loft_deg", cfg.get("stated_loft")),
        brand=build_cfg.get("brand", cfg.get("brand")),
        model=build_cfg.get("model"),
        shaft_model=build_cfg.get("shaft_model"),
        shaft_weight_g=build_cfg.get("shaft_weight_g"),
        shaft_flex=build_cfg.get("shaft_flex"),
        hosel_setting=build_cfg.get("hosel_setting", cfg.get("current_setting")),
    )


def _render_advanced_analysis(
    club_id: str,
    canon_df: pd.DataFrame,
    hosel_configs: Dict[str, Dict],
    k_loft_to_dynamic: float,
):
    summary = summarize_by_club(canon_df)[club_id]
    t = targets_for_club(club_id, summary.club_speed_avg)
    launch_lo, launch_hi = t["launch"]
    spin_lo, spin_hi = t["spin"]

    with st.expander("Advanced Analysis", expanded=False):
        st.markdown("### Limiting Factors")
        lim: List[str] = []

        miss = miss_tendency(summary.offline_avg)
        lim.append(f"Miss tendency: **{miss}**" if miss != "Unknown" else "Miss tendency: Unknown")

        shape = shot_shape_summary(canon_df[canon_df["club_id"] == club_id])
        lim.append(f"Typical shot shape: **{shape.shape_label}**")

        if club_id == "DR":
            smash_msg = smash_flag_driver(summary.smash_avg, summary.club_speed_avg)
            if smash_msg:
                lim.append(smash_msg)

        if not np.isnan(summary.vla_avg) and (summary.vla_avg < launch_lo or summary.vla_avg > launch_hi):
            lim.append(f"Launch window miss: {summary.vla_avg:.1f}° vs target {launch_lo:.1f}–{launch_hi:.1f}°.")

        if not np.isnan(summary.spin_avg) and (summary.spin_avg < spin_lo or summary.spin_avg > spin_hi):
            lim.append(f"Spin window miss: {summary.spin_avg:.0f} rpm vs target {spin_lo:.0f}–{spin_hi:.0f} rpm.")

        dp = distance_potential_for_summary(summary)
        lim.append(
            f"Distance potential: expected carry **{dp.expected_carry_yd:.1f} yd**, actual **{dp.actual_carry_yd:.1f} yd**."
        )

        for item in lim:
            st.write("•", item)

        family = _club_family_from_id(club_id)
        if family in {"Driver", "Fairway Wood", "Hybrid"}:
            st.markdown("### Settings Recommendation")
            recos: List[str] = []

            if club_id in hosel_configs:
                h = hosel_configs[club_id]
                brand = h["brand"]
                system_name = h["system_name"]
                handedness = h["handedness"]
                current_setting = h["current_setting"]

                needed_loft = 0.0
                needed_lie = 0.0

                if not np.isnan(summary.vla_avg):
                    launch_per_static_loft = 0.85 * k_loft_to_dynamic
                    if summary.vla_avg < launch_lo:
                        needed_loft = min(2.0, (launch_lo - summary.vla_avg) / max(0.05, launch_per_static_loft))
                    elif summary.vla_avg > launch_hi:
                        needed_loft = max(-2.0, -(summary.vla_avg - launch_hi) / max(0.05, launch_per_static_loft))

                miss = miss_tendency(summary.offline_avg)
                if miss == "Right miss tendency":
                    needed_lie = +0.75
                elif miss == "Left miss tendency":
                    needed_lie = -0.75

                if not np.isnan(summary.spin_avg):
                    if summary.spin_avg < spin_lo:
                        needed_loft = max(needed_loft, +0.75)
                    elif summary.spin_avg > spin_hi:
                        needed_loft = min(needed_loft, -0.75)

                settings = list_settings(brand, system_name, handedness)
                reco = pick_one_hosel_setting(
                    settings=settings,
                    translate_fn=translate_setting,
                    brand=brand,
                    system_name=system_name,
                    handedness=handedness,
                    current_setting=current_setting,
                    needed_loft_delta=needed_loft,
                    needed_lie_delta=needed_lie,
                )

                if abs(needed_loft) < 0.25 and abs(needed_lie) < 0.25:
                    recos.append("Current setting looks reasonable for launch, spin, and start-line tendencies.")
                else:
                    recos.append(f"Hosel goal: loft Δ **{needed_loft:+.2f}°**, lie Δ **{needed_lie:+.2f}°**.")
                    if reco["type"] == "exact":
                        r = reco["recommended"]
                        recos.append(
                            f"Suggested change: **{reco['current']} → {r['setting']}** "
                            f"(loft {r['loft_delta']:+.2f}°, lie {r['lie_delta']:+.2f}°)."
                        )
                    else:
                        recos.append(reco["message"])
            else:
                recos.append("Configure this club in Club Settings to enable a setting recommendation.")

            for r in recos:
                st.write("•", r)

        st.markdown("### Shot Table")
        show_cols = [
            "club_raw", "club_id",
            "club_speed_mph", "ball_speed_mph", "smash",
            "carry_yd", "total_yd", "offline_yd",
            "peak_height_yd", "descent_deg",
            "vla_deg", "backspin_rpm", "aoa_deg",
            "club_path_deg", "face_to_path_deg", "face_to_target_deg",
        ]
        cols_present = [c for c in show_cols if c in canon_df.columns]

        st.dataframe(
            canon_df[canon_df["club_id"] == club_id][cols_present].reset_index(drop=True),
            use_container_width=True,
        )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("FitCaddie")

    st.session_state["analysis_mode"] = st.radio(
        "Mode",
        ["Single Club Analysis", "Compare Driver Setups"],
        index=0 if st.session_state["analysis_mode"] == "Single Club Analysis" else 1,
    )

    st.divider()

    st.subheader("Model")
    st.session_state["k_loft_to_dynamic"] = st.slider(
        "Loft → delivered loft multiplier (k)",
        min_value=0.6,
        max_value=1.6,
        value=float(st.session_state["k_loft_to_dynamic"]),
        step=0.05,
    )

    st.divider()

    st.subheader("Filters")
    st.session_state["min_shots"] = st.slider(
        "Min shots per club",
        min_value=3,
        max_value=50,
        value=int(st.session_state["min_shots"]),
        step=1,
    )

    st.divider()

    st.subheader("Debug")
    st.session_state["show_raw"] = st.checkbox(
        "Show raw tables",
        value=bool(st.session_state["show_raw"]),
    )

    st.divider()

    if st.session_state["analysis_mode"] == "Single Club Analysis":
        uploaded = st.file_uploader("Upload GSPro CSV", type=["csv"], key="single_upload")
    else:
        uploaded_a = st.file_uploader("Upload Setup A CSV", type=["csv"], key="compare_upload_a")
        uploaded_b = st.file_uploader("Upload Setup B CSV", type=["csv"], key="compare_upload_b")


analysis_mode = st.session_state["analysis_mode"]
k_loft_to_dynamic = float(st.session_state["k_loft_to_dynamic"])
min_shots = int(st.session_state["min_shots"])
show_raw = bool(st.session_state["show_raw"])


# =========================================================
# SINGLE CLUB ANALYSIS
# =========================================================
if analysis_mode == "Single Club Analysis":
    if not uploaded:
        st.info("Upload a GSPro CSV to begin.")
        st.stop()

    raw_df = pd.read_csv(uploaded)
    canon_df, fmt = canonicalize(raw_df)

    st.success(f"Loaded {len(canon_df)} shots. Detected export format: **{fmt}**")

    if show_raw:
        raw_c1, raw_c2 = st.columns(2)
        with raw_c1:
            st.subheader("Raw CSV")
            st.dataframe(raw_df.head(200), use_container_width=True)
        with raw_c2:
            st.subheader("Canonicalized")
            st.dataframe(canon_df.head(200), use_container_width=True)

    club_counts = canon_df["club_id"].value_counts().to_dict()

    club_ids_all = [
        c for c in club_counts.keys()
        if c not in {"OTHER", "PT", "PUTTER"}
    ]
    club_ids = [c for c in club_ids_all if club_counts.get(c, 0) >= min_shots]
    club_ids = sorted(club_ids, key=_club_sort_key)

    if not club_ids:
        st.warning(f"No clubs have at least {min_shots} shots. Try lowering the filter or collect more data.")
        st.stop()

    focus_club = _render_focus_picker(club_ids, club_counts)
    focus_df = canon_df[canon_df["club_id"] == focus_club].copy()

    summaries = summarize_by_club(focus_df)
    if focus_club not in summaries:
        st.warning("No valid data for the selected club.")
        st.stop()

    focus_summary = summaries[focus_club]
    focus_family = _club_family_from_id(focus_club)

    hdr1, hdr2, hdr3 = st.columns([1.1, 1.1, 1.4])
    with hdr1:
        st.markdown(
            f"""
            <div class="fc-card">
                <h3 style="margin-bottom:4px;">Focus Club</h3>
                <div class="fc-subtle">Current analysis target</div>
                <div style="font-size:1.8rem;font-weight:800;color:#17324d;margin-top:6px;">{focus_club}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hdr2:
        st.markdown(
            f"""
            <div class="fc-card">
                <h3 style="margin-bottom:4px;">Club Family</h3>
                <div class="fc-subtle">Detected from uploaded file</div>
                <div style="font-size:1.35rem;font-weight:800;color:#17324d;margin-top:10px;">{focus_family}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hdr3:
        st.markdown(
            f"""
            <div class="fc-card">
                <h3 style="margin-bottom:4px;">Sample Size</h3>
                <div class="fc-subtle">Shots available for this club after filtering</div>
                <div style="font-size:1.8rem;font-weight:800;color:#17324d;margin-top:6px;">{len(focus_df)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    main_left, main_right = st.columns([1.45, 1.0], gap="large")

    with main_left:
        st.markdown('<div class="fc-card"><h3>Dispersion</h3><p class="fc-subtle">Locked to your selected focus club so the chart and picker stay in sync.</p>', unsafe_allow_html=True)
        render_dispersion(focus_df, key_prefix="single_focus", lock_club=focus_club)
        st.markdown("</div>", unsafe_allow_html=True)

    with main_right:
        st.markdown(f'<div class="fc-card"><h3>{focus_club} Overview</h3>', unsafe_allow_html=True)
        _render_summary_cards(focus_summary, focus_df)
        st.markdown("</div>", unsafe_allow_html=True)

    tab_overview, tab_build, tab_hosel, tab_reco = st.tabs(
        ["Overview", "Club Build", "Hosel", "Recommendations"]
    )

    with tab_overview:
        _render_advanced_analysis(
            club_id=focus_club,
            canon_df=focus_df,
            hosel_configs={},
            k_loft_to_dynamic=k_loft_to_dynamic,
        )

    with tab_build:
        if focus_club == "DR":
            _render_driver_setup("single", "Driver Build")
        else:
            _render_non_driver_build("single_nd", f"{focus_family} Build", focus_club)

    with tab_hosel:
        _render_hosel_block(
            club_id=focus_club,
            title=f"Hosel Settings — {focus_club}",
            k_loft_to_dynamic=k_loft_to_dynamic,
        )

    with tab_reco:
        recommendation_bundle = None

        if focus_club == "DR":
            recommendation_bundle = _render_driver_recommendations(
                focus_df,
                _driver_setup_from_prefix("single"),
            )
        else:
            build_cfg = _club_build_from_prefix("single_nd")
            hosel_configs = _render_hosel_block(
                club_id=focus_club,
                title=f"Hosel Settings — {focus_club}",
                k_loft_to_dynamic=k_loft_to_dynamic,
            )
            recommendation_bundle = _render_non_driver_recommendations(
                focus_summary,
                hosel_configs,
                build_cfg,
            )

        if recommendation_bundle is not None:
            st.markdown('<div class="fc-card"><h3>Fitter Recommendations</h3>', unsafe_allow_html=True)
            _render_recommendation_cards(recommendation_bundle)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No recommendation bundle available for this club yet.")


# =========================================================
# COMPARE DRIVER SETUPS
# =========================================================
else:
    st.markdown(
        """
        <div class="fc-card">
            <h3>Compare Two Driver Setups</h3>
            <p class="fc-subtle">Best for shaft vs shaft, hosel vs hosel, or head vs head testing with separate uploads.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        _render_driver_setup("cmpA", "Setup A")
    with c2:
        _render_driver_setup("cmpB", "Setup B")

    if not uploaded_a or not uploaded_b:
        st.info("Upload both Setup A and Setup B CSV files to compare them.")
        st.stop()

    raw_a = pd.read_csv(uploaded_a)
    raw_b = pd.read_csv(uploaded_b)

    canon_a, fmt_a = canonicalize(raw_a)
    canon_b, fmt_b = canonicalize(raw_b)

    st.success(
        f"Loaded Setup A: {len(canon_a)} shots (**{fmt_a}**) | "
        f"Loaded Setup B: {len(canon_b)} shots (**{fmt_b}**)"
    )

    if show_raw:
        ra, rb = st.columns(2)
        with ra:
            st.subheader("Setup A — Canonicalized")
            st.dataframe(canon_a.head(200), use_container_width=True)
        with rb:
            st.subheader("Setup B — Canonicalized")
            st.dataframe(canon_b.head(200), use_container_width=True)

    dr_a = canon_a[canon_a["club_id"] == "DR"].copy()
    dr_b = canon_b[canon_b["club_id"] == "DR"].copy()

    if dr_a.empty or dr_b.empty:
        st.warning("Both uploaded files need driver shots (DR) for compare mode.")
        st.stop()

    compare = compare_driver_setups(dr_a, dr_b, "Setup A", "Setup B")

    st.markdown('<div class="fc-card"><h3>Compare Dispersion</h3>', unsafe_allow_html=True)
    render_compare_dispersion(
        dr_a,
        dr_b,
        key_prefix="driver_compare",
        label_a="Setup A",
        label_b="Setup B",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    a = compare["a"]
    b = compare["b"]

    metric_df = pd.DataFrame(
        {
            "Metric": [
                "Shots", "Club Speed", "Ball Speed", "Smash",
                "Carry", "Total", "Launch", "Spin", "AoA",
                "Avg Abs Offline", "Fairway %"
            ],
            "Setup A": [
                a.shots,
                _fmt(a.club_speed, 1),
                _fmt(a.ball_speed, 1),
                _fmt(a.smash, 2),
                _fmt(a.carry, 1),
                _fmt(a.total, 1),
                _fmt(a.launch, 1),
                _fmt(a.spin, 0),
                _fmt(a.aoa, 1),
                _fmt(a.offline, 1),
                _fmt(a.fairway_pct, 0, "%"),
            ],
            "Setup B": [
                b.shots,
                _fmt(b.club_speed, 1),
                _fmt(b.ball_speed, 1),
                _fmt(b.smash, 2),
                _fmt(b.carry, 1),
                _fmt(b.total, 1),
                _fmt(b.launch, 1),
                _fmt(b.spin, 0),
                _fmt(b.aoa, 1),
                _fmt(b.offline, 1),
                _fmt(b.fairway_pct, 0, "%"),
            ],
            "Difference (B - A)": [
                _fmt_diff(b.shots - a.shots, 0),
                _fmt_diff(b.club_speed - a.club_speed, 1),
                _fmt_diff(b.ball_speed - a.ball_speed, 1),
                _fmt_diff(b.smash - a.smash, 2),
                _fmt_diff(b.carry - a.carry, 1),
                _fmt_diff(b.total - a.total, 1),
                _fmt_diff(b.launch - a.launch, 1),
                _fmt_diff(b.spin - a.spin, 0),
                _fmt_diff(b.aoa - a.aoa, 1),
                _fmt_diff(b.offline - a.offline, 1),
                _fmt_diff(b.fairway_pct - a.fairway_pct, 0, "%"),
            ],
        }
    )

    st.markdown('<div class="fc-card"><h3>Side-by-Side Metrics</h3>', unsafe_allow_html=True)
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    winners = compare["winners"]

    st.markdown(
        f"""
        <div class="fc-verdict">
            <h3 style="margin-top:0;">Comparison Verdict</h3>
            <p><strong>Longest carry:</strong> {winners['longest_carry']}</p>
            <p><strong>Fastest ball speed:</strong> {winners['fastest_ball_speed']}</p>
            <p><strong>Straightest:</strong> {winners['straightest']}</p>
            <p><strong>Most fairways:</strong> {winners['most_fairways']}</p>
            <p><strong>Best overall gamer:</strong> {winners['best_overall']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_a = _driver_setup_from_prefix("cmpA")
    setup_b = _driver_setup_from_prefix("cmpB")

    summaries_a = summarize_by_club(dr_a)
    summaries_b = summarize_by_club(dr_b)

    rec_a = build_driver_recommendations(
        summary=summaries_a["DR"],
        user_setup=setup_a,
        fairway_hit_pct=float((dr_a["offline_yd"].dropna().abs() <= 15).mean() * 100.0) if len(dr_a["offline_yd"].dropna()) else None,
    )
    rec_b = build_driver_recommendations(
        summary=summaries_b["DR"],
        user_setup=setup_b,
        fairway_hit_pct=float((dr_b["offline_yd"].dropna().abs() <= 15).mean() * 100.0) if len(dr_b["offline_yd"].dropna()) else None,
    )

    rc1, rc2 = st.columns(2, gap="large")
    with rc1:
        st.markdown('<div class="fc-card"><h3>Setup A Recommendation</h3>', unsafe_allow_html=True)
        _render_recommendation_cards(rec_a)
        st.markdown("</div>", unsafe_allow_html=True)
    with rc2:
        st.markdown('<div class="fc-card"><h3>Setup B Recommendation</h3>', unsafe_allow_html=True)
        _render_recommendation_cards(rec_b)
        st.markdown("</div>", unsafe_allow_html=True)

    interpretation = []
    best_overall = winners["best_overall"]

    if best_overall == "Setup A":
        interpretation.append("Setup A looks like the stronger overall gamer from this test.")
    elif best_overall == "Setup B":
        interpretation.append("Setup B looks like the stronger overall gamer from this test.")
    else:
        interpretation.append("This comparison is close enough that neither setup clearly dominates overall.")

    if winners["longest_carry"] != winners["straightest"]:
        interpretation.append("One setup appears better for distance while the other appears better for control.")
    else:
        interpretation.append("The same setup appears to be winning both carry and dispersion, which is a strong sign.")

    interpretation.append("Repeat the test on another day with 8–10 fresh shots per setup to confirm the winner.")

    st.markdown(
        f"""
        <div class="fc-card">
            <h3>Best Overall Interpretation</h3>
            <p>{" ".join(interpretation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Next smart upgrade: add a confidence label like high / moderate / low confidence to each recommendation set.")
