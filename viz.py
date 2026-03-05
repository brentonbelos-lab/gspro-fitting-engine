# viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Config
# -----------------------------
@dataclass
class DispersionConfig:
    distance_mode: str = "carry"          # "carry" or "total"
    fairway_width_yd: float = 70.0        # default wider
    fairway_end_mode: str = "p95"         # "max" or "p95"
    right_miss_down: bool = True

    # Padding
    x_pad_pct: float = 0.08
    y_pad_pct: float = 0.18

    # Visual toggles
    show_centerline: bool = True
    show_target_marker: bool = True
    keep_proportions: bool = True

    # Dispersion circle settings
    circle_mode: str = "p90"              # "p80" | "p90" | "p95" | "1sigma" | "2sigma"
    circle_min_radius_yd: float = 5.0     # prevent tiny circles
    circle_opacity: float = 0.18


# -----------------------------
# Dispersion circle helpers
# -----------------------------
def _club_circle_radius(dx: np.ndarray, dy: np.ndarray, mode: str) -> float:
    """
    dx, dy are centered coordinates for a club.
    Returns a radius in yards.
    - pXX: radius that contains XX% of points (based on radial distance)
    - 1sigma/2sigma: based on combined std in x/y (rough but intuitive)
    """
    r = np.sqrt(dx * dx + dy * dy)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0

    m = mode.lower().strip()
    if m.startswith("p") and len(m) >= 2:
        try:
            p = float(m[1:])
            return float(np.nanpercentile(r, p))
        except Exception:
            pass

    if m == "1sigma":
        sx = float(np.nanstd(dx, ddof=1)) if dx.size > 1 else float(np.nanstd(dx))
        sy = float(np.nanstd(dy, ddof=1)) if dy.size > 1 else float(np.nanstd(dy))
        return float(np.sqrt(sx * sx + sy * sy))

    if m == "2sigma":
        sx = float(np.nanstd(dx, ddof=1)) if dx.size > 1 else float(np.nanstd(dx))
        sy = float(np.nanstd(dy, ddof=1)) if dy.size > 1 else float(np.nanstd(dy))
        return 2.0 * float(np.sqrt(sx * sx + sy * sy))

    # fallback
    return float(np.nanpercentile(r, 90))


def _circle_trace(cx: float, cy: float, radius: float, color: str, name: str, opacity: float) -> go.Scatter:
    """
    Draw a circle as a parametric line trace (works with scaleanchor properly).
    """
    t = np.linspace(0, 2 * np.pi, 181)
    x = cx + radius * np.cos(t)
    y = cy + radius * np.sin(t)

    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(width=2, color=color),
        name=f"{name} circle",
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,  # keeps legend clean; shots already appear in legend
    )


# -----------------------------
# Core builder
# -----------------------------
def _build_dispersion_figure(
    df: pd.DataFrame,
    cfg: DispersionConfig,
    club_filter: Optional[str] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Uses FitCaddie canonical columns:
      club_id, carry_yd, total_yd, offline_yd, vla_deg, backspin_rpm, etc.
    Returns (fig, df_plot)
    """
    d = df.copy()

    if club_filter:
        d = d[d["club_id"] == club_filter].copy()

    # Choose X distance
    if cfg.distance_mode.lower() == "total" and "total_yd" in d.columns:
        d["_x"] = pd.to_numeric(d["total_yd"], errors="coerce")
    else:
        d["_x"] = pd.to_numeric(d["carry_yd"], errors="coerce")

    # Offline to Y (flip so Right plots DOWN)
    off = pd.to_numeric(d["offline_yd"], errors="coerce")
    d["_y"] = (-off) if cfg.right_miss_down else off

    # Basic cleaning
    d = d.dropna(subset=["_x", "_y"]).copy()
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No plottable shots for dispersion map.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig.update_layout(height=680, margin=dict(l=20, r=20, t=45, b=20))
        return fig, d

    # X range
    x_min = 0.0
    x_max_data = float(d["_x"].max())

    if cfg.fairway_end_mode == "p95":
        fw_x1 = float(np.nanpercentile(d["_x"].values, 95))
        fw_x1 = max(fw_x1, float(np.nanpercentile(d["_x"].values, 75)))
    else:
        fw_x1 = x_max_data

    x_pad = max(5.0, fw_x1 * cfg.x_pad_pct)
    x_max = fw_x1 + x_pad

    # Y range
    y_abs_max = float(np.nanmax(np.abs(d["_y"].values)))
    y_pad = max(6.0, y_abs_max * cfg.y_pad_pct)
    y_lim = max(y_abs_max + y_pad, cfg.fairway_width_yd * 0.9)

    # Build plot
    fig = go.Figure()

    # -----------------------------
    # Layered course look
    # -----------------------------
    rough_half = cfg.fairway_width_yd * 1.8
    firstcut_half = cfg.fairway_width_yd * 1.25
    fw_half = cfg.fairway_width_yd / 2.0

    # ROUGH (lightest)
    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-rough_half, y1=rough_half,
        line=dict(width=0),
        fillcolor="rgba(34, 139, 34, 0.06)",
        layer="below",
    )

    # FIRST CUT (medium)
    fig.add_shape(
        type="rect",
        x0=0, x1=fw_x1,
        y0=-firstcut_half, y1=firstcut_half,
        line=dict(width=0),
        fillcolor="rgba(34, 139, 34, 0.12)",
        layer="below",
    )

    # FAIRWAY (darkest)
    fig.add_shape(
        type="rect",
        x0=0, x1=fw_x1,
        y0=-fw_half, y1=fw_half,
        line=dict(color="rgba(20,80,30,0.30)", width=1),
        fillcolor="rgba(30,150,70,0.22)",
        layer="below",
    )

    # Centerline
    if cfg.show_centerline:
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[0, 0],
            mode="lines",
            line=dict(dash="dash", width=1),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.45,
            name="Centerline",
        ))

    # Target marker (mean downrange)
    if cfg.show_target_marker:
        tx = float(d["_x"].mean())
        fig.add_trace(go.Scatter(
            x=[tx], y=[0],
            mode="markers",
            marker=dict(size=10, symbol="x"),
            hovertemplate="Target<br>Downrange: %{x:.1f} yd<extra></extra>",
            showlegend=False,
            opacity=0.7,
        ))

    # Add stable shot index for selection
    d = d.reset_index(drop=True)
    d["_row_i"] = np.arange(len(d))
    
    # Hover payload (only if columns exist)
    hover_bits: List[str] = []
    cd_cols: List[str] = []
    
    def _add(col: str, label: str, fmt: str):
        """
        Adds a column to customdata and a matching hover line.
        fmt should be like '.1f', '.2f', '.0f'
        """
        if col in d.columns:
            idx = len(cd_cols)
            cd_cols.append(col)
            hover_bits.append(f"{label}: %{{customdata[{idx}]:{fmt}}}")
    
    _add("club_speed_mph", "Club Speed (mph)", ".1f")
    _add("ball_speed_mph", "Ball Speed (mph)", ".1f")
    _add("smash", "Smash", ".2f")
    _add("vla_deg", "Launch (°)", ".1f")
    _add("backspin_rpm", "Spin (rpm)", ".0f")

    # Color palette
    club_palette = [
        "#2563eb",  # blue
        "#dc2626",  # red
        "#16a34a",  # green
        "#ea580c",  # orange
        "#7c3aed",  # purple
        "#0891b2",  # teal
        "#be185d",  # pink
        "#0f766e",  # deep teal
    ]

    clubs_present = sorted(d["club_id"].dropna().unique())

    # --- Per-club traces + dispersion circles ---
    for i, club in enumerate(clubs_present):
        color = club_palette[i % len(club_palette)]
        dc = d[d["club_id"] == club].copy()
        if dc.empty:
            continue

        customdata = dc[cd_cols].to_numpy() if cd_cols else None

        # Points
        fig.add_trace(go.Scatter(
            x=dc["_x"],
            y=dc["_y"],
            mode="markers",
            marker=dict(size=10, opacity=0.85, color=color),
            name=str(club),
            customdata=customdata,
            hovertemplate=(
                f"<b>{club}</b><br>"
                "Downrange: %{x:.1f} yd<br>"
                "Lateral: %{y:.1f} yd<br>"
                + ("<br>".join(hover_bits) + "<br>" if hover_bits else "")
                + "<extra></extra>"
            ),
        ))

        # Circle center (mean)
        cx = float(dc["_x"].mean())
        cy = float(dc["_y"].mean())

        dx = (dc["_x"].values - cx).astype(float)
        dy = (dc["_y"].values - cy).astype(float)

        rad = _club_circle_radius(dx, dy, cfg.circle_mode)
        rad = max(rad, cfg.circle_min_radius_yd)

        fig.add_trace(_circle_trace(cx, cy, rad, color=color, name=str(club), opacity=cfg.circle_opacity))

        # Small center dot for the club mean
        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers",
            marker=dict(size=6, color=color, symbol="circle"),
            showlegend=False,
            hovertemplate=(
                f"<b>{club} mean</b><br>"
                "Downrange: %{x:.1f} yd<br>"
                "Lateral: %{y:.1f} yd<br>"
                f"Circle radius ({cfg.circle_mode}): {rad:.1f} yd"
                "<extra></extra>"
            ),
        ))

    title = "Shot Dispersion Map"
    if club_filter:
        title += f" — {club_filter}"

    fig.update_layout(
        title=title,
        height=680,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(
            title="Downrange (yd)",
            range=[x_min, x_max],
            zeroline=False,
            showgrid=True,
        ),
        yaxis=dict(
            title="Lateral (yd) — right miss down",
            range=[-y_lim, y_lim],
            zeroline=False,
            showgrid=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if cfg.keep_proportions:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, d


def _extract_selected_point(plot_state: Optional[dict]) -> Optional[int]:
    """
    Streamlit plotly selection structure varies slightly by version.
    Returns the selected point index if possible.
    """
    if not plot_state or not isinstance(plot_state, dict):
        return None

    sel = plot_state.get("selection")
    if isinstance(sel, dict):
        pts = sel.get("points")
        if isinstance(pts, list) and len(pts) > 0:
            p0 = pts[0]
            pi = p0.get("pointIndex")
            if pi is not None:
                return int(pi)

    sel2 = plot_state.get("selected_points")
    if isinstance(sel2, dict):
        pts = sel2.get("points")
        if isinstance(pts, list) and len(pts) > 0:
            pi = pts[0].get("pointIndex")
            if pi is not None:
                return int(pi)

    return None


# -----------------------------
# Public function used by app.py
# -----------------------------
def render_dispersion(canon_df: pd.DataFrame):
    """
    Called by app.py as: render_dispersion(canon_df)
    """
    if canon_df is None or canon_df.empty:
        st.info("No shot data available.")
        return

    c0, c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2])

    clubs = sorted([c for c in canon_df["club_id"].dropna().unique().tolist()])
    with c0:
        club_filter = st.selectbox("Club", ["ALL"] + clubs, index=0)

    with c1:
        distance_mode = st.selectbox("Distance", ["carry", "total"], index=0)

    with c2:
        fairway_width = st.slider("Fairway width (yd)", 30, 140, 70, 1)

    with c3:
        end_mode = st.selectbox("Fairway length", ["p95", "max"], index=0)

    with c4:
        circle_mode = st.selectbox("Dispersion circle", ["p80", "p90", "p95", "1sigma", "2sigma"], index=1)

    cfg = DispersionConfig(
        distance_mode=distance_mode,
        fairway_width_yd=float(fairway_width),
        fairway_end_mode=end_mode,
        right_miss_down=True,
        circle_mode=circle_mode,
    )

    club = None if club_filter == "ALL" else club_filter
    fig, df_plot = _build_dispersion_figure(canon_df, cfg=cfg, club_filter=club)

    plot_key = "dispersion_plot"

    try:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=plot_key,
            on_select="rerun",
            selection_mode=("points",),
        )
        plot_state = st.session_state.get(plot_key)
        selected_i = _extract_selected_point(plot_state)
    except Exception:
        st.plotly_chart(fig, use_container_width=True)
        selected_i = None

    st.subheader("Selected Shot")

    if selected_i is None:
        st.caption("Tap a point (or box-select) to see shot details.")
        return

    if 0 <= selected_i < len(df_plot):
        row = df_plot.iloc[selected_i]
        show_cols = [
            "club_raw", "club_id",
            "club_speed_mph", "ball_speed_mph", "smash",
            "carry_yd", "total_yd", "offline_yd",
            "vla_deg", "backspin_rpm", "aoa_deg",
            "club_path_deg", "face_to_path_deg", "face_to_target_deg",
        ]
        payload = {c: row[c] for c in show_cols if c in df_plot.columns and pd.notna(row.get(c))}
        st.json(payload)
    else:
        st.caption("Selection index out of range (unexpected).")
