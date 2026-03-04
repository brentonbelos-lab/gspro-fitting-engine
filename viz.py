# viz.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    go = None


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
    """
    Returns (ex, ey) ellipse points for x/y using covariance, centered at mean.
    sigma=1 gives 1 std ellipse; sigma=2 gives ~95% contour if data is roughly normal.
    """
    if len(x) < 4:
        return None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[m], y[m]
        if len(x) < 4:
            return None

    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return None

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Guard
    vals = np.maximum(vals, 1e-9)

    # radii
    rx = sigma * np.sqrt(vals[0])
    ry = sigma * np.sqrt(vals[1])

    # angle
    theta = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([rx * np.cos(theta), ry * np.sin(theta)])  # 2 x n

    # rotate
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
    """
    Renders a Plotly dispersion scatter (distance vs offline) with optional covariance ellipse(s).
    Click a shot to show that shot's row.

    Requirements in df:
      - club_id
      - offline_yd
      - carry_yd and/or total_yd
    """
    if config is None:
        config = DispersionConfig()

    if go is None:
        st.warning("Plotly is not available in this environment. Install 'plotly' to enable dispersion charts.")
        return

    required = {"club_id", "offline_yd"}
    if not required.issubset(df.columns):
        st.warning(f"Dispersion chart needs columns: {sorted(required)}")
        return

    # Choose distance axis
    distance_choice = st.radio(
        "Distance axis",
        options=["carry", "total"],
        index=0 if config.default_distance == "carry" else 1,
        horizontal=True,
        key="disp_distance_axis",
    )
    dist_col = "carry_yd" if distance_choice == "carry" else "total_yd"
    if dist_col not in df.columns:
        st.warning(f"Missing '{dist_col}' in data. Available columns: {list(df.columns)}")
        return

    # Controls
    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
    with c1:
        sigma = st.select_slider(
            "Cloud size (σ)",
            options=[0.5, 1.0, 1.5, 2.0],
            value=float(config.sigma),
            key="disp_sigma",
            help="1σ ~ 68% ellipse, 2σ ~ ~95% (if roughly normal).",
        )
    with c2:
        show_ellipses = st.checkbox("Show shot clouds", value=config.show_ellipses, key="disp_show_ellipses")
    with c3:
        st.caption("Tip: hover for tooltips. If your Streamlit supports selection, click a shot to inspect it.")

    plot_df = df.copy()
    plot_df = plot_df[np.isfinite(plot_df[dist_col].astype(float)) & np.isfinite(plot_df["offline_yd"].astype(float))].copy()
    if len(plot_df) == 0:
        st.info("No valid shots to plot after filtering missing values.")
        return

    # Build figure
    fig = go.Figure()
    clubs = sorted(plot_df["club_id"].unique().tolist())

    # Add per-club scatter + ellipse + mean marker
    for club in clubs:
        sub = plot_df[plot_df["club_id"] == club].reset_index(drop=False)  # keep original index
        x = sub[dist_col].astype(float).to_numpy()
        y = sub["offline_yd"].astype(float).to_numpy()

        # Points
        hover_cols = []
        for col in ["club_speed_mph", "ball_speed_mph", "smash", "vla_deg", "backspin_rpm", "aoa_deg"]:
            if col in sub.columns:
                hover_cols.append(col)

        customdata = np.vstack([sub["index"].to_numpy()]).T  # original index for selection lookup

        hovertemplate = (
            f"<b>{club}</b><br>"
            + f"{'Carry' if dist_col=='carry_yd' else 'Total'}: %{x:.1f} yd<br>"
            + "Offline: %{y:.1f} yd<br>"
        )
        # Add optional hover fields
        for col in hover_cols:
            hovertemplate += f"{col}: %{{customdata[{len(customdata[0])}]}}
"  # placeholder won't work with varying customdata
        # We'll instead use hovertext
        hovertext = []
        for i in range(len(sub)):
            bits = [
                f"{'Carry' if dist_col=='carry_yd' else 'Total'}: {x[i]:.1f} yd",
                f"Offline: {y[i]:.1f} yd",
            ]
            for col in hover_cols:
                v = sub.loc[i, col]
                if pd.isna(v):
                    continue
                if isinstance(v, (int, float, np.floating)):
                    if "rpm" in col:
                        bits.append(f"{col}: {float(v):.0f}")
                    elif "deg" in col:
                        bits.append(f"{col}: {float(v):.1f}")
                    else:
                        bits.append(f"{col}: {float(v):.2f}" if col == "smash" else f"{col}: {float(v):.1f}")
                else:
                    bits.append(f"{col}: {v}")
            hovertext.append("<br>".join(bits))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=club,
                customdata=customdata,
                hovertext=hovertext,
                hoverinfo="text",
                marker=dict(size=9, line=dict(width=0)),
            )
        )

        # Mean marker
        fig.add_trace(
            go.Scatter(
                x=[float(np.mean(x))],
                y=[float(np.mean(y))],
                mode="markers",
                name=f"{club} avg",
                showlegend=False,
                marker=dict(size=14, symbol="x"),
                hoverinfo="skip",
            )
        )

        # Ellipse
        if show_ellipses:
            ell = _ellipse_points(x, y, sigma=float(sigma))
            if ell is not None:
                ex, ey = ell
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        name=f"{club} cloud",
                        showlegend=False,
                        line=dict(width=2),
                        fill="toself",
                        opacity=0.18,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Carry (yd)" if dist_col == "carry_yd" else "Total (yd)",
        yaxis_title="Offline (yd)  (Left - / Right +)",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=1)

    # ---------
    # Selection / click-to-inspect
    # ---------
    selected_idx: Optional[int] = None

    # Newer Streamlit supports on_select + return selection state.
    # If not supported, we fall back to a dropdown selector.
    try:
        sel = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("points",),
            key="disp_plot",
        )
        # sel can be a dict-like with selection info depending on Streamlit version
        # We handle multiple shapes of this object safely.
        if sel and isinstance(sel, dict):
            points = sel.get("selection", {}).get("points", [])
            if points:
                # customdata[0] contains original df index
                cd = points[0].get("customdata", None)
                if isinstance(cd, (list, tuple)) and len(cd) >= 1:
                    selected_idx = int(cd[0])
                elif isinstance(cd, (int, float)):
                    selected_idx = int(cd)
    except TypeError:
        # Older Streamlit
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Shot")

    if selected_idx is None:
        # fallback: allow manual selection
        st.caption("Click selection not available in your current Streamlit version — use this selector:")
        # Build a readable label
        tmp = plot_df.reset_index(drop=False)
        tmp["label"] = tmp.apply(
            lambda r: f"[{r['club_id']}] "
                      f"{('Carry' if dist_col=='carry_yd' else 'Total')} {r[dist_col]:.1f} yd, "
                      f"Offline {r['offline_yd']:+.1f} yd  (row {int(r['index'])})",
            axis=1,
        )
        chosen = st.selectbox("Pick a shot", options=tmp["label"].tolist(), index=0, key="disp_manual_pick")
        selected_idx = int(tmp.loc[tmp["label"] == chosen, "index"].iloc[0])

    # Render selected row details
    if selected_idx is not None and selected_idx in df.index:
        row = df.loc[selected_idx]
        # key metrics card
        kcols = st.columns(6)
        def _m(i, name, col, fmt):
            if col in row.index and pd.notna(row[col]):
                kcols[i].metric(name, fmt(row[col]))
            else:
                kcols[i].metric(name, "—")

        _m(0, "Club", "club_id", lambda v: str(v))
        _m(1, "Club Speed", "club_speed_mph", lambda v: f"{float(v):.1f} mph")
        _m(2, "Ball Speed", "ball_speed_mph", lambda v: f"{float(v):.1f} mph")
        _m(3, "Carry", "carry_yd", lambda v: f"{float(v):.1f} yd")
        _m(4, "Offline", "offline_yd", lambda v: f"{float(v):+.1f} yd")
        _m(5, "Launch", "vla_deg", lambda v: f"{float(v):.1f}°")

        # Full row table
        st.dataframe(pd.DataFrame(row).T, use_container_width=True)
    else:
        st.info("No shot selected.")
