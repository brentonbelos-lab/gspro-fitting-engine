# viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
    fairway_width_yd: float = 45.0        # make wider if you want (e.g. 55)
    fairway_end_mode: str = "p95"         # "max" or "p95" (p95 looks nicer)
    right_miss_down: bool = True

    x_pad_pct: float = 0.08               # padding on right side
    y_pad_pct: float = 0.18               # padding on top/bottom

    show_centerline: bool = True
    show_target_marker: bool = True
    keep_proportions: bool = True         # locks x/y scale ratio


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

    fw_half = cfg.fairway_width_yd / 2.0

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
        fillcolor="rgba(34, 139, 34, 0.08)",
        layer="below",
    )
    
    # FIRST CUT
    fig.add_shape(
        type="rect",
        x0=0, x1=fw_x1,
        y0=-firstcut_half, y1=firstcut_half,
        line=dict(width=0),
        fillcolor="rgba(34, 139, 34, 0.14)",
        layer="below",
    )
    
    # FAIRWAY (clean center)
    fig.add_shape(
        type="rect",
        x0=0, x1=fw_x1,
        y0=-fw_half, y1=fw_half,
        line=dict(color="rgba(20,80,30,0.35)", width=1),
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

    # Make a nicer hover payload (works even if some cols missing)
    hover_bits = []
    if "club_speed_mph" in d.columns: hover_bits.append("Club Speed: %{customdata[0]:.1f} mph")
    if "ball_speed_mph" in d.columns: hover_bits.append("Ball Speed: %{customdata[1]:.1f} mph")
    if "smash" in d.columns: hover_bits.append("Smash: %{customdata[2]:.2f}")
    if "vla_deg" in d.columns: hover_bits.append("Launch: %{customdata[3]:.1f}°")
    if "backspin_rpm" in d.columns: hover_bits.append("Spin: %{customdata[4]:.0f} rpm")

    # Customdata order must match above
    cd_cols = [
        "club_speed_mph" if "club_speed_mph" in d.columns else None,
        "ball_speed_mph" if "ball_speed_mph" in d.columns else None,
        "smash" if "smash" in d.columns else None,
        "vla_deg" if "vla_deg" in d.columns else None,
        "backspin_rpm" if "backspin_rpm" in d.columns else None,
    ]
    cd_cols = [c for c in cd_cols if c is not None]
    customdata = d[cd_cols].to_numpy() if cd_cols else None

    # Color palette (clean + readable)
    club_palette = [
        "#2563eb",  # blue
        "#dc2626",  # red
        "#16a34a",  # green
        "#ea580c",  # orange
        "#7c3aed",  # purple
        "#0891b2",  # teal
        "#be185d",  # pink
    ]
    
    clubs_present = sorted(d["club_id"].dropna().unique())
    
    for i, club in enumerate(clubs_present):
        dc = d[d["club_id"] == club]
    
        fig.add_trace(go.Scatter(
            x=dc["_x"],
            y=dc["_y"],
            mode="markers",
            marker=dict(
                size=10,
                opacity=0.85,
                color=club_palette[i % len(club_palette)],
            ),
            name=club,
            customdata=dc[cd_cols].to_numpy() if cd_cols else None,
            hovertemplate=(
                f"<b>{club}</b><br>"
                "Downrange: %{x:.1f} yd<br>"
                "Lateral: %{y:.1f} yd<br>"
                + ("<br>".join(hover_bits) + "<br>" if hover_bits else "")
                + "<extra></extra>"
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

    # Keep fairway proportions true (no stretching)
    if cfg.keep_proportions:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, d


def _extract_selected_point(plot_state: Optional[dict]) -> Optional[int]:
    """
    Streamlit plotly selection structure varies slightly by version.
    Tries a few common shapes and returns the selected point index if possible.
    """
    if not plot_state or not isinstance(plot_state, dict):
        return None

    # Newer Streamlit: state["selection"]["points"]
    sel = plot_state.get("selection")
    if isinstance(sel, dict):
        pts = sel.get("points")
        if isinstance(pts, list) and len(pts) > 0:
            p0 = pts[0]
            # Plotly point index is often present
            pi = p0.get("pointIndex")
            if pi is not None:
                return int(pi)

    # Sometimes: state["selected_points"]
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
    Renders:
      - club filter dropdown
      - distance mode toggle
      - fairway width slider
      - plot
      - selected shot details panel
    """

    if canon_df is None or canon_df.empty:
        st.info("No shot data available.")
        return

    # Controls row
    c0, c1, c2, c3 = st.columns([1.4, 1.2, 1.2, 1.2])

    clubs = sorted([c for c in canon_df["club_id"].dropna().unique().tolist()])
    with c0:
        club_filter = st.selectbox("Club", ["ALL"] + clubs, index=0)

    with c1:
        distance_mode = st.selectbox("Distance", ["carry", "total"], index=0)

    with c2:
        fairway_width = st.slider("Fairway width (yd)", 30, 120, 70, 1)

    with c3:
        end_mode = st.selectbox("Fairway length", ["p95", "max"], index=0)

    cfg = DispersionConfig(
        distance_mode=distance_mode,
        fairway_width_yd=float(fairway_width),
        fairway_end_mode=end_mode,
        right_miss_down=True,
    )

    club = None if club_filter == "ALL" else club_filter
    fig, df_plot = _build_dispersion_figure(canon_df, cfg=cfg, club_filter=club)

    plot_key = "dispersion_plot"

    # Try to enable selection (works on Streamlit versions that support it)
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
        # Fallback: still show chart, but no selection
        st.plotly_chart(fig, use_container_width=True)
        selected_i = None

    st.subheader("Selected Shot")

    if selected_i is None:
        st.caption("Tap a point (or box-select) to see shot details.")
        return

    # Because pointIndex corresponds to df_plot row order
    if 0 <= selected_i < len(df_plot):
        row = df_plot.iloc[selected_i]
        # Show a clean shot detail set (only columns that exist)
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
