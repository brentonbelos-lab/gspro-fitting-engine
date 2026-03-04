# viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


@dataclass(frozen=True)
class DispersionConfig:
    default_distance: str = "carry"  # "carry" or "total"
    sigma: float = 1.5               # default cloud size
    show_ellipses: bool = True
    default_fairway_width: int = 30  # yards


def _ellipse_points(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
    n: int = 80,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Covariance ellipse around mean. Returns (ex, ey) or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return None

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 1e-9)
    vecs = vecs[:, order]

    rx = sigma * np.sqrt(vals[0])
    ry = sigma * np.sqrt(vals[1])

    theta = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([rx * np.cos(theta), ry * np.sin(theta)])  # 2 x n
    ellipse = vecs @ circle

    ex = ellipse[0] + float(np.mean(x))
    ey = ellipse[1] + float(np.mean(y))
    return ex, ey


def _pretty_hover_lines(r: pd.Series, dist_col: str) -> str:
    """Robust hover tooltip."""
    lines: List[str] = [f"<b>{r.get('club_id','')}</b>"]
    dname = "Carry" if dist_col == "carry_yd" else "Total"
    lines.append(f"{dname}: {float(r[dist_col]):.1f} yd")
    lines.append(f"Offline: {float(r['offline_yd']):+.1f} yd")

    def add_if(col: str, fmt: str):
        v = r.get(col, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return
        try:
            lines.append(f"{col}: {fmt.format(float(v))}")
        except Exception:
            lines.append(f"{col}: {v}")

    add_if("club_speed_mph", "{:.1f} mph")
    add_if("ball_speed_mph", "{:.1f} mph")
    add_if("smash", "{:.2f}")
    add_if("vla_deg", "{:.1f}°")
    add_if("backspin_rpm", "{:.0f} rpm")
    add_if("spin_axis", "{:.1f}")
    add_if("aoa_deg", "{:.1f}°")
    add_if("club_path_deg", "{:.1f}°")
    add_if("face_to_path_deg", "{:.1f}°")
    add_if("face_to_target_deg", "{:.1f}°")

    # show original row index if present
    if "index" in r.index and pd.notna(r["index"]):
        lines.append(f"Row: {int(r['index'])}")

    return "<br>".join(lines)


def render_dispersion(
    df: pd.DataFrame,
    *,
    config: Optional[DispersionConfig] = None,
    title: str = "Shot Dispersion Map",
) -> None:
    """
    Beautiful GSPro-style dispersion chart:
      - Carry/Total vs Offline
      - Fairway band shading centered at offline=0
      - Right misses displayed BELOW centerline (Y axis flipped)
      - Per-club covariance ellipse (shot cloud)
      - Hover tooltips + shot dropdown details (works everywhere)

    Required columns:
      - club_id
      - offline_yd
      - carry_yd and/or total_yd
    """
    if config is None:
        config = DispersionConfig()

    required = {"club_id", "offline_yd"}
    if not required.issubset(df.columns):
        st.warning(f"Dispersion chart needs columns: {sorted(required)}")
        return

    # Controls
    st.subheader(title)

    distance_choice = st.radio(
        "Distance axis",
        options=["carry", "total"],
        index=0 if config.default_distance == "carry" else 1,
        horizontal=True,
        key="disp_distance_axis",
    )
    dist_col = "carry_yd" if distance_choice == "carry" else "total_yd"
    if dist_col not in df.columns:
        st.warning(f"Missing '{dist_col}' in data.")
        return

    c1, c2, c3 = st.columns([1.2, 1.4, 1.6])
    with c1:
        sigma = st.select_slider(
            "Cloud size (σ)",
            options=[0.5, 1.0, 1.5, 2.0],
            value=float(config.sigma),
            key="disp_sigma",
        )
    with c2:
        show_ellipses = st.checkbox("Show shot clouds", value=config.show_ellipses, key="disp_show_ellipses")
    with c3:
        fairway_width = st.slider(
            "Fairway width (yd)",
            min_value=10,
            max_value=60,
            value=int(config.default_fairway_width),
            step=2,
            key="disp_fairway_width",
        )

    # Clean plot dataframe
    plot_df = df.copy()
    plot_df = plot_df[
        np.isfinite(plot_df[dist_col].astype(float))
        & np.isfinite(plot_df["offline_yd"].astype(float))
    ].copy()

    if len(plot_df) == 0:
        st.info("No valid shots to plot after filtering missing values.")
        return

    # Dynamic y-range so fairway/rough shading looks correct
    y_vals = plot_df["offline_yd"].astype(float).to_numpy()
    y_abs_max = float(np.nanmax(np.abs(y_vals))) if len(y_vals) else 40.0
    y_pad = 18.0
    y_lim = max(45.0, y_abs_max + y_pad)  # ensures nice framing

    # Optional dynamic x padding too (feels more "portal")
    x_vals = plot_df[dist_col].astype(float).to_numpy()
    x_min = float(np.nanmin(x_vals)) if len(x_vals) else 0.0
    x_max = float(np.nanmax(x_vals)) if len(x_vals) else 250.0
    x_pad = max(8.0, (x_max - x_min) * 0.06)
    x_rng = [x_min - x_pad, x_max + x_pad]

    # Build figure
    fig = go.Figure()

    # Background shading (below traces)
    half_fw = fairway_width / 2.0

    # Rough (top and bottom of visible area)
    fig.add_hrect(
        y0=half_fw, y1=y_lim,
        fillcolor="rgba(170,170,170,0.18)",
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=-y_lim, y1=-half_fw,
        fillcolor="rgba(170,170,170,0.18)",
        line_width=0,
        layer="below",
    )

    # Fairway band
    fig.add_hrect(
        y0=-half_fw, y1=half_fw,
        fillcolor="rgba(0,180,90,0.25)",
        line_width=0,
        layer="below",
    )

    # Target corridor inside fairway (±10 yd)
    corridor = min(10.0, half_fw)
    fig.add_hrect(
        y0=-corridor, y1=corridor,
        fillcolor="rgba(0,180,90,0.38)",
        line_width=0,
        layer="below",
    )

    # Centerline
    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="solid",
        opacity=0.55,
        layer="above",
    )

    clubs = sorted(plot_df["club_id"].unique().tolist())

    # Per-club traces
    for club in clubs:
        sub = plot_df[plot_df["club_id"] == club].reset_index(drop=False)  # keep original index in 'index'
        x = sub[dist_col].astype(float).to_numpy()
        y = sub["offline_yd"].astype(float).to_numpy()

        # Hover text
        hover = [_pretty_hover_lines(sub.iloc[i], dist_col) for i in range(len(sub))]

        # Points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=club,
                hovertext=hover,
                hoverinfo="text",
                customdata=sub["index"].to_numpy(),  # original df index
                marker=dict(size=9),
            )
        )

        # Mean marker
        fig.add_trace(
            go.Scatter(
                x=[float(np.mean(x))],
                y=[float(np.mean(y))],
                mode="markers",
                showlegend=False,
                marker=dict(size=14, symbol="x"),
                hoverinfo="skip",
            )
        )

        # Cloud ellipse
        if show_ellipses:
            ell = _ellipse_points(x, y, sigma=float(sigma))
            if ell is not None:
                ex, ey = ell
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        showlegend=False,
                        fill="toself",
                        opacity=0.18,
                        hoverinfo="skip",
                        line=dict(width=2),
                    )
                )

    # Layout styling (portal-like)
    fig.update_layout(
        title=title,
        xaxis_title="Carry (yd)" if dist_col == "carry_yd" else "Total (yd)",
        yaxis_title="Offline (yd)  (Left ↑ / Right ↓)",
        height=540,
        margin=dict(l=32, r=18, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(245,245,245,1)",
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        showgrid=True,
        zeroline=False,
        range=x_rng,
    )

    # IMPORTANT: y-axis flipped so right misses appear LOWER on chart
    fig.update_yaxes(
        showgrid=True,
        zeroline=False,
        autorange=False,
        range=[y_lim, -y_lim],  # reversed
    )

    st.plotly_chart(fig, use_container_width=True)

    # Shot inspection (works everywhere via dropdown)
    st.subheader("Selected Shot")
    tmp = plot_df.reset_index(drop=False)
    dname = "Carry" if dist_col == "carry_yd" else "Total"
    tmp["label"] = tmp.apply(
        lambda r: f"[{r['club_id']}] {dname} {float(r[dist_col]):.1f} yd, Offline {float(r['offline_yd']):+.1f} yd  (row {int(r['index'])})",
        axis=1,
    )

    chosen = st.selectbox("Pick a shot", options=tmp["label"].tolist(), index=0, key="disp_manual_pick")
    selected_idx = int(tmp.loc[tmp["label"] == chosen, "index"].iloc[0])

    if selected_idx in df.index:
        row = df.loc[selected_idx]
        kcols = st.columns(6)

        def _metric(i: int, label: str, col: str, fmt):
            if col in row.index and pd.notna(row[col]):
                kcols[i].metric(label, fmt(row[col]))
            else:
                kcols[i].metric(label, "—")

        _metric(0, "Club", "club_id", lambda v: str(v))
        _metric(1, "Club Speed", "club_speed_mph", lambda v: f"{float(v):.1f} mph")
        _metric(2, "Ball Speed", "ball_speed_mph", lambda v: f"{float(v):.1f} mph")
        _metric(3, "Carry", "carry_yd", lambda v: f"{float(v):.1f} yd")
        _metric(4, "Offline", "offline_yd", lambda v: f"{float(v):+.1f} yd")
        _metric(5, "Launch", "vla_deg", lambda v: f"{float(v):.1f}°")

        st.dataframe(pd.DataFrame(row).T, use_container_width=True)
    else:
        st.info("No shot selected.")
