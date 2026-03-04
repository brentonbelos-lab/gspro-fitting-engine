# fit_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

# Research-backed model:
# Launch angle is driven by delivered dynamic loft + AoA. TrackMan & fitting education commonly cite
# a simple split: launch ≈ 0.85*DynamicLoft + 0.15*AoA. :contentReference[oaicite:13]{index=13}
#
# We don't have DynamicLoft in GSPro exports, so we treat a sleeve loft change as shifting dynamic loft by:
#   ΔDynamicLoft ≈ k * ΔStaticLoft
# and assume AoA stays ~constant for the estimate.
LAUNCH_FROM_DYNAMIC_WEIGHT = 0.85


@dataclass(frozen=True)
class LaunchSpinEstimate:
    launch_change_deg: float
    launch_range_deg: Tuple[float, float]   # (low, high)
    spin_change_rpm: int
    spin_range_rpm: Tuple[int, int]         # (low, high)
    notes: str


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_launch_spin_change(
    delta_static_loft_deg: float,
    k_loft_to_dynamic: float,
    club_type: str,
) -> LaunchSpinEstimate:
    """
    Returns estimated launch + spin change when the hosel loft changes by delta_static_loft_deg.

    club_type: "DR", "FW", "HY"
    """

    # Launch estimate
    launch_est = LAUNCH_FROM_DYNAMIC_WEIGHT * k_loft_to_dynamic * delta_static_loft_deg

    # Uncertainty:
    # - real-world delivery/strike changes can dominate; we present a conservative band that scales with change
    # - baseline ±0.6° even for small changes, wider for bigger changes
    launch_unc = max(0.6, abs(launch_est) * 0.35)
    launch_low = launch_est - launch_unc
    launch_high = launch_est + launch_unc

    # Spin estimate band (very variable):
    # - EngineeredGolf suggests ~200–300 rpm per 1° loft depending on speed :contentReference[oaicite:14]{index=14}
    # - Cobra MyFly notes changes up to ±450 rpm with loft adjustments :contentReference[oaicite:15]{index=15}
    # We'll use a conservative "typical" center + range:
    #   Driver: 250 rpm/deg (range 150–400)
    #   Fairway: 300 rpm/deg (range 180–450)
    #   Hybrid: 320 rpm/deg (range 200–500)
    if club_type == "DR":
        center = 250
        lo, hi = 150, 400
    elif club_type == "FW":
        center = 300
        lo, hi = 180, 450
    else:
        center = 320
        lo, hi = 200, 500

    spin_est = int(round(center * delta_static_loft_deg))
    # Range scales with magnitude
    spin_low = int(round(lo * delta_static_loft_deg))
    spin_high = int(round(hi * delta_static_loft_deg))

    # Ensure ordering for negative deltas
    spin_range = (min(spin_low, spin_high), max(spin_low, spin_high))

    note = (
        "Estimates assume swing delivery is similar. Real changes can be larger/smaller due to strike location, "
        "face angle at address, and shaft lean."
    )

    return LaunchSpinEstimate(
        launch_change_deg=launch_est,
        launch_range_deg=(launch_low, launch_high),
        spin_change_rpm=spin_est,
        spin_range_rpm=spin_range,
        notes=note,
    )


def driver_targets(club_speed_mph: float) -> Dict[str, Tuple[float, float]]:
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
    Returns ONE recommended hosel setting (best match) if exact deltas exist.
    Otherwise returns guidance.
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
