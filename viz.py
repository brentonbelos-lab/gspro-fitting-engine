# viz.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# -----------------------------
# Helpers
# -----------------------------
def _to_float(series: pd.Series) -> pd.Series:
    """
    Robust numeric parser for GSPro exports.
    Handles strings like "181.28 yds", "40.1°", "11.4 R", "-7.2 L".
    """
    if series is None:
        return series
    s = series.astype(str).str.strip()

    # Convert "11.4 R" / "11.4 L" into signed values (R positive, L negative)
    # Also handles "11.4R" / "11.4L"
    def parse_one(x: str) -> float:
        if x is None:
            return np.nan
        x = str(x).strip()
        if x == "" or x.lower() in {"nan", "none"}:
            return np.nan

        # Directional suffix
        m = re.match(r"^\s*([-+]?\d*\.?\d+)\s*([RL])\s*$", x, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            dirc = m.group(2).upper()
            return val if dirc == "R" else -val

        # Strip everything except digits, sign, decimal
        m2 = re.search(r"[-+]?\d*\.?\d+", x)
        return float(m2.group(0)) if m2 else np.nan

    return s.map(parse_one)


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@dataclass
class DispersionConfig:
    # Which distance to plot on X
    distance_mode: str = "carry"  # "carry" or "total"

    # Fairway geometry
    fairway_width_yd: float = 35.0  # you can widen this (e.g. 45)
    fairway_start_x_yd: float = 0.0
    fairway_end_x_yd: Optional[float] = None  # auto from data if None

    # Plot padding
    x_pad_pct: float = 0.08  # 8% padding
    y_pad_pct: float = 0.18  # 18% padding

    # Right-miss-down behavior
    right_miss_down: bool = True  # your preference

    # Visual options
    show_centerline: bool = True
    show_target_marker: bool = True


def build_dispersion_figure(
    df_raw: pd.DataFrame,
    cfg: DispersionConfig = DispersionConfig(),
    club_filter: Optional[str] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Returns:
      - Plotly Figure
      - Cleaned dataframe used for plotting (includes columns: x_downrange, y_lateral_plot)
    """

    df = df_raw.copy()

    # --- Find likely GSPro columns (support both "portal" + "software" exports) ---
    club_col = _first_existing(df, ["Club Name", "Club", "club", "ClubName"])
    carry_col = _first_existing(df, ["Carry Dist (yd)", "Carry Distance", "Carry", "CarryDist"])
    total_col = _first_existing(df, ["Total Dist (yd)", "Total Distance", "Total", "TotalDist"])
    offline_col = _first_existing(df, ["Offline (yd)", "Offline", "Side", "Lateral (yd)", "Lateral"])
    shot_col = _first_existing(df, ["Shot Number", "Global Shot Number", "Shot", "ShotNum", "shot_id"])

    # If the key columns don't exist, fail gently with a helpful message in the chart
    if offline_col is None or (carry_col is None and total_col is None):
        fig = go.Figure()
        fig.add_annotation(
            text="Missing required columns for dispersion plot. Need Offline + (Carry or Total distance).",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig.update_layout(height=650, margin=dict(l=20, r=20, t=30, b=20))
        return fig, df

    # --- Clean numeric fields ---
    df["_offline_yd"] = _to_float(df[offline_col])

    if cfg.distance_mode.lower() == "total" and total_col is not None:
        df["_downrange_yd"] = _to_float(df[total_col])
    else:
        # default to carry if available
        if carry_col is not None:
            df["_downrange_yd"] = _to_float(df[carry_col])
        else:
            df["_downrange_yd"] = _to_float(df[total_col])

    # Optional filter by club
    if club_filter and club_col:
        df = df[df[club_col].astype(str) == str(club_filter)].copy()

    # Drop rows that can't plot
    df = df.dropna(subset=["_offline_yd", "_downrange_yd"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No plottable shots after cleaning / filtering.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig.update_layout(height=650, margin=dict(l=20, r=20, t=30, b=20))
        return fig, df

    # --- Apply your axis rule: right miss should appear DOWN ---
    # Assume Offline + = Right, - = Left (common). If your data is opposite, flip sign here.
    lateral = df["_offline_yd"].astype(float)
    y_plot = -lateral if cfg.right_miss_down else lateral

    df["x_downrange"] = df["_downrange_yd"].astype(float)
    df["y_lateral_plot"] = y_plot

    # --- Compute bounds + padding ---
    x_min = 0.0
    x_max_data = float(df["x_downrange"].max())
    x_pad = max(5.0, x_max_data * cfg.x_pad_pct)
    x_max = x_max_data + x_pad

    y_abs_max = float(np.nanmax(np.abs(df["y_lateral_plot"])))
    y_pad = max(6.0, y_abs_max * cfg.y_pad_pct)
    y_lim = y_abs_max + y_pad
    # Ensure a reasonable minimum span so the fairway doesn't look squeezed
    y_lim = max(y_lim, cfg.fairway_width_yd * 0.9)

    # --- Fairway geometry (centered on 0 lateral) ---
    fw_half = cfg.fairway_width_yd / 2.0
    fw_x0 = cfg.fairway_start_x_yd
    fw_x1 = cfg.fairway_end_x_yd if cfg.fairway_end_x_yd is not None else x_max_data

    # --- Build figure ---
    fig = go.Figure()

    # Background "course" layer (subtle)
    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-y_lim, y1=y_lim,
        line=dict(width=0),
        fillcolor="rgba(25, 90, 35, 0.10)",
        layer="below",
    )

    # Fairway rectangle
    fig.add_shape(
        type="rect",
        x0=fw_x0, x1=fw_x1,
        y0=-fw_half, y1=fw_half,
        line=dict(color="rgba(20, 80, 30, 0.35)", width=1),
        fillcolor="rgba(30, 150, 70, 0.14)",
        layer="below",
    )

    # Fairway edges (for crispness)
    fig.add_trace(go.Scatter(
        x=[fw_x0, fw_x1, fw_x1, fw_x0, fw_x0],
        y=[-fw_half, -fw_half, fw_half, fw_half, -fw_half],
        mode="lines",
        line=dict(width=1),
        hoverinfo="skip",
        showlegend=False,
        name="Fairway",
        opacity=0.55,
    ))

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

    # Target marker (at average downrange, centerline)
    if cfg.show_target_marker:
        target_x = float(df["x_downrange"].mean())
        fig.add_trace(go.Scatter(
            x=[target_x],
            y=[0],
            mode="markers",
            marker=dict(size=10, symbol="x"),
            hovertemplate="Target<br>Downrange: %{x:.1f} yd<br>Lateral: %{y:.1f} yd<extra></extra>",
            showlegend=False,
            name="Target",
            opacity=0.7,
        ))

    # Shot markers (interactive)
    # Use shot id if present, else index
    if shot_col is not None:
        df["_shot_id"] = df[shot_col].astype(str)
    else:
        df["_shot_id"] = df.index.astype(str)

    # Hover payload
    hover_cols = []
    if club_col: hover_cols.append(club_col)
    if carry_col: hover_cols.append(carry_col)
    if total_col: hover_cols.append(total_col)
    if offline_col: hover_cols.append(offline_col)

    # customdata includes row index (for selection)
    df["_row_i"] = np.arange(len(df))

    fig.add_trace(go.Scatter(
        x=df["x_downrange"],
        y=df["y_lateral_plot"],
        mode="markers",
        marker=dict(size=10, opacity=0.85),
        name="Shots",
        customdata=np.stack([df["_row_i"].values, df["_shot_id"].values], axis=1),
        hovertemplate=(
            "<b>Shot %{customdata[1]}</b><br>"
            "Downrange: %{x:.1f} yd<br>"
            "Lateral: %{y:.1f} yd<br>"
            "<extra></extra>"
        ),
    ))

    # Layout polish
    title = "Shot Dispersion"
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
            scaleanchor="x",  # keeps fairway proportions correct
            scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig, df


def pick_selected_shot(
    plot_state: Optional[dict],
    df_plot: pd.DataFrame
) -> Optional[pd.Series]:
    """
    Streamlit Plotly selection info usually lands in st.session_state[key].
    We try to extract the selected point and return the corresponding df row.
    """
    if not plot_state or not isinstance(plot_state, dict):
        return None

    # Streamlit versions differ; handle a couple common shapes
    # Expected: plot_state["selection"]["points"] -> list with point data
    selection = plot_state.get("selection") or plot_state.get("selected_points") or {}
    points = selection.get("points") if isinstance(selection, dict) else None
    if not points:
        return None

    # First selected point
    p0 = points[0]

    # Plotly points often contain "customdata" with our [row_i, shot_id]
    cd = p0.get("customdata")
    if cd and len(cd) >= 1:
        row_i = int(cd[0])
        if 0 <= row_i < len(df_plot):
            return df_plot.iloc[row_i]

    return None
