# ---------------------------
# Ping Trajectory Tuning 2.0 (8 pos) — EXACT deltas
# ---------------------------
# Chart shows 8 settings with explicit Loft Adj and Avg Lie:
# O (neutral), Big+/Big-, Small+/Small-, and Flat options (F, F-, F+). :contentReference[oaicite:2]{index=2}
PING_TT2 = HoselSystem(
    system_name="PING Trajectory Tuning 2.0 (8-position)",
    family="sleeve_8",
    settings_rh=["O", "Big +", "Small +", "Small -", "Big -", "F", "F-", "F+"],
    settings_lh=["O", "Big +", "Small +", "Small -", "Big -", "F", "F-", "F+"],
    deltas_rh={
        # Baseline: O = stated loft, neutral lie
        "O":       HoselSettingDelta(0.0,  0.0,  None, "Neutral"),

        # Neutral lie zone (but lie is slightly flatter vs O per chart)
        "Big +":   HoselSettingDelta(+1.5, -1.5, None, "Higher loft; lie flatter vs O"),
        "Small +": HoselSettingDelta(+1.0, -1.0, None, "Higher loft; lie slightly flatter vs O"),
        "Small -": HoselSettingDelta(-1.0, -1.0, None, "Lower loft; lie slightly flatter vs O"),
        "Big -":   HoselSettingDelta(-1.5, -1.5, None, "Lower loft; lie flatter vs O"),

        # Flat lie zone
        "F":       HoselSettingDelta(0.0,  -3.0, None, "Flat setting (max flat)"),
        "F-":      HoselSettingDelta(-1.0, -2.0, None, "Flat + lower loft"),
        "F+":      HoselSettingDelta(+1.0, -2.0, None, "Flat + higher loft"),
    },
    deltas_lh={
        # Keep the same deltas for LH in MVP. If you want later, we can add a LH-specific baseline.
        "O":       HoselSettingDelta(0.0,  0.0,  None, "Neutral"),
        "Big +":   HoselSettingDelta(+1.5, -1.5, None, "Higher loft; lie flatter vs O"),
        "Small +": HoselSettingDelta(+1.0, -1.0, None, "Higher loft; lie slightly flatter vs O"),
        "Small -": HoselSettingDelta(-1.0, -1.0, None, "Lower loft; lie slightly flatter vs O"),
        "Big -":   HoselSettingDelta(-1.5, -1.5, None, "Lower loft; lie flatter vs O"),
        "F":       HoselSettingDelta(0.0,  -3.0, None, "Flat setting (max flat)"),
        "F-":      HoselSettingDelta(-1.0, -2.0, None, "Flat + lower loft"),
        "F+":      HoselSettingDelta(+1.0, -2.0, None, "Flat + higher loft"),
    },
    loft_range_deg=(-1.5, +1.5),
    lie_range_deg=(-3.0, 0.0),
    face_range_deg=None,
    notes="Exact loft/lie deltas encoded from an 8-setting TT2 chart. Flat options reduce lie substantially. :contentReference[oaicite:3]{index=3}"
)
