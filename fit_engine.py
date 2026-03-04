# fit_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np


# Launch estimate model:
# Launch is strongly related to delivered (dynamic) loft; we approximate:
#   ΔLaunch ≈ 0.85 * (k * ΔStaticLoft)
# where k is a user-tunable multiplier (delivery/strike changes).
LAUNCH_FROM_DYNAMIC_WEIGHT = 0.85


@dataclass(frozen=True)
class LaunchSpinEstimate:
    launch_change_deg: float
    launch_range_deg: Tuple[float, float]  # (low, high)
    spin_change_rpm: int
    spin_range_rpm: Tuple[int, int]        # (low, high)
    notes: str


def driver_targets(club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
    """Simple speed-adjusted driver windows (MVP heuristic)."""
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


def estimate_launch_spin_change(
    delta_static_loft_deg: float,
    k_loft_to_dynamic: float,
    club_type: str,  # "DR", "FW", "HY"
) -> LaunchSpinEstimate:
    """
    Show the user what *might* happen if they change hosel setting (loft).
    Spin is highly variable, so we return a conservative band.
    """
    # Launch
    launch_est = LAUNCH_FROM_DYNAMIC_WEIGHT * k_loft_to_dynamic * delta_static_loft_deg
    launch_unc = max(0.6, abs(launch_est) * 0.35)  # conservative uncertainty band
    launch_low = launch_est - launch_unc
    launch_high = launch_est + launch_unc

    # Spin bands (conservative, varies by club type)
    if club_type == "DR":
        center, lo, hi = 250, 150, 400   # rpm per degree
    elif club_type == "FW":
        center, lo, hi = 300, 180, 450
    else:  # HY
        center, lo, hi = 320, 200, 500

    spin_est = int(round(center * delta_static_loft_deg))
    spin_low = int(round(lo * delta_static_loft_deg))
    spin_high = int(round(hi * delta_static_loft_deg))
    spin_range = (min(spin_low, spin_high), max(spin_low, spin_high))

    return LaunchSpinEstimate(
        launch_change_deg=launch_est,
        launch_range_deg=(launch_low, launch_high),
        spin_change_rpm=spin_est,
        spin_range_rpm=spin_range,
        notes="Estimates assume delivery is similar; real outcomes can vary due to strike location, face angle at address, and shaft lean.",
    )


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
    Otherwise returns guidance text.
    """
    scored = []
    for s in settings:
        d = translate_fn(brand, system_name, s, handedness)
        loft = getattr(d, "loft_deg", None)
        lie = getattr(d, "lie_deg", None)
        if loft is None or lie is None:
            continue
        score = abs(loft - needed_loft_delta) * 1.5 + abs(lie - needed_lie_delta) * 1.0
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
