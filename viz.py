# viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go


@dataclass(frozen=True)
class DispersionConfig:
    default_distance: str = "carry"  # "carry" or "total"
    sigma: float = 1.0               # 1.0 ~ 68%, 2.0 ~ 95%
    show_ellipses: bool = True


def _ellipse_points(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
    n: int = 60,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(x) < 4:
        return None

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
    circle = np.vstack([rx * np.cos(theta), ry * np.sin(theta)])

    ellipse = (vecs @ circle)
    ex = ellipse[0, :] + np.mean(x)
    ey = ellipse[1, :] + np.mean(y)
    return ex, ey


def render_dispersion(
    df: pd.DataFrame,
    *,
    config: Optional[DispersionConfig] = None,
    title: str = "Shot Dispersion Map",
) -> None:
    if config is None:
        config = DispersionConfig()

    required = {"club_id", "offline_yd"}
    if not required.issubset(df.columns):
        st.warning(f"Dispersion chart needs columns: {sorted(required)}")
        return

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

    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        sigma = st.select_slider(
            "Cloud size (σ)",
            options=[0.5, 1.0, 1.5, 2.0],
            value=float(config.sigma),
            key="disp_sigma",
        )

    with c2:
        show_ellipses = st.checkbox(
            "Show shot clouds",
            value=config.show_ellipses,
            key="disp_show_ellipses"
        )
    
    with c3:
        fairway_width = st.slider(
            "Fairway width (yd)",
            min_value=10,
            max_value=60,
            value=30,
            step=2,
            help="Visual fairway band centered at 0 offline."
        )

    plot_df = df.copy()
    plot_df = plot_df[np.isfinite(plot_df[dist_col].astype(float)) & np.isfinite(plot_df["offline_yd"].astype(float))].copy()
    if len(plot_df) == 0:
        st.info("No valid shots to plot after filtering missing values.")
        return

    fig = go.Figure()
    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="solid",
        opacity=0.5
    )
    # --- Fairway band shading (stronger + visible) ---
    half_fw = fairway_width / 2
    
    # Fairway (green)
    fig.add_hrect(
        y0=-half_fw,
        y1=half_fw,
        fillcolor="rgba(0,180,90,0.22)",
        line_width=0,
        layer="below",
    )
    
    # Target corridor (lighter stripe inside fairway) ±10 yd
    fig.add_hrect(
        y0=-10,
        y1=10,
        fillcolor="rgba(0,180,90,0.32)",
        line_width=0,
        layer="below",
    )
    
    # Rough (top and bottom bands)
    fig.add_hrect(
        y0=half_fw,
        y1=half_fw + 120,
        fillcolor="rgba(160,160,160,0.14)",
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=-half_fw - 120,
        y1=-half_fw,
        fillcolor="rgba(160,160,160,0.14)",
        line_width=0,
        layer="below",
    )
    clubs = sorted(plot_df["club_id"].unique().tolist())

    for club in clubs:
        sub = plot_df[plot_df["club_id"] == club].reset_index(drop=False)  # keep original index
        x = sub[dist_col].astype(float).to_numpy()
        y = sub["offline_yd"].astype(float).to_numpy()

        # Hover text (clean, robust)
        hovertext = []
        for _, r in sub.iterrows():
            hovertext.append(
                "<br>".join([
                    f"<b>{r['club_id']}</b>",
                    f"{'Carry' if dist_col=='carry_yd' else 'Total'}: {float(r[dist_col]):.1f} yd",
                    f"Offline: {float(r['offline_yd']):+.1f} yd",
                    f"Club Speed: {float(r['club_speed_mph']):.1f} mph" if "club_speed_mph" in sub.columns and pd.notna(r.get("club_speed_mph")) else "",
                    f"Ball Speed: {float(r['ball_speed_mph']):.1f} mph" if "ball_speed_mph" in sub.columns and pd.notna(r.get("ball_speed_mph")) else "",
                    f"Launch: {float(r['vla_deg']):.1f}°" if "vla_deg" in sub.columns and pd.notna(r.get("vla_deg")) else "",
                    f"Spin: {float(r['backspin_rpm']):.0f} rpm" if "backspin_rpm" in sub.columns and pd.notna(r.get("backspin_rpm")) else "",
                    f"Row: {int(r['index'])}",
                ]).replace("<br><br>", "<br>")
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=club,
                hovertext=hovertext,
                hoverinfo="text",
                customdata=sub["index"].to_numpy(),  # store original df index
                marker=dict(size=9),
            )
        )

        # mean marker
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
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Carry (yd)" if dist_col == "carry_yd" else "Total (yd)",
        yaxis_title="Offline (yd)  (Left ↑ / Right ↓)",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(245,245,245,1)",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        zeroline=False,
        autorange="reversed"
    )

    # Render chart
    st.plotly_chart(fig, use_container_width=True)

    # Click-to-inspect fallback (works everywhere)
    st.subheader("Selected Shot")
    tmp = plot_df.reset_index(drop=False)
    tmp["label"] = tmp.apply(
        lambda r: f"[{r['club_id']}] "
                  f"{('Carry' if dist_col=='carry_yd' else 'Total')} {float(r[dist_col]):.1f} yd, "
                  f"Offline {float(r['offline_yd']):+.1f} yd  (row {int(r['index'])})",
        axis=1,
    )
    chosen = st.selectbox("Pick a shot", options=tmp["label"].tolist(), index=0, key="disp_manual_pick")
    selected_idx = int(tmp.loc[tmp["label"] == chosen, "index"].iloc[0])

    if selected_idx in df.index:
        row = df.loc[selected_idx]
        kcols = st.columns(6)

        def _metric(i, label, col, fmt):
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
