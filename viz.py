from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Config
# -----------------------------
@dataclass
class DispersionConfig:
    distance_mode: str = "carry"
    fairway_width_yd: float = 70.0
    fairway_end_mode: str = "p95"
    right_miss_down: bool = True

    x_pad_pct: float = 0.08
    y_pad_pct: float = 0.18

    show_centerline: bool = True
    show_target_marker: bool = True
    keep_proportions: bool = False

    circle_mode: str = "1sigma"
    circle_min_radius_yd: float = 3.5
    circle_opacity: float = 0.18


# -----------------------------
# Helpers
# -----------------------------
def _dispersion_ellipse_radii(dx: np.ndarray, dy: np.ndarray, mode: str) -> Tuple[float, float]:
    dx = dx[np.isfinite(dx)]
    dy = dy[np.isfinite(dy)]

    if dx.size == 0 or dy.size == 0:
        return 0.0, 0.0

    m = mode.lower().strip()

    if m.startswith("p") and len(m) >= 2:
        try:
            p = float(m[1:])
            rx = float(np.nanpercentile(np.abs(dx), p))
            ry = float(np.nanpercentile(np.abs(dy), p))
            return rx, ry
        except Exception:
            pass

    sx = float(np.nanstd(dx, ddof=1)) if dx.size > 1 else float(np.nanstd(dx))
    sy = float(np.nanstd(dy, ddof=1)) if dy.size > 1 else float(np.nanstd(dy))

    if m == "2sigma":
        return 2.0 * sx, 2.0 * sy

    return sx, sy


def _ellipse_trace(cx: float, cy: float, rx: float, ry: float, color: str, name: str, opacity: float) -> go.Scatter:
    t = np.linspace(0, 2 * np.pi, 181)
    x = cx + rx * np.cos(t)
    y = cy + ry * np.sin(t)

    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(width=2, color=color),
        name=f"{name} ellipse",
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


def _empty_figure(message: str = "No plottable shots for dispersion map.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=680, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def _prepare_plot_df(
    df: pd.DataFrame,
    cfg: DispersionConfig,
    club_filter: Optional[str] = None,
) -> pd.DataFrame:
    d = df.copy()

    if club_filter:
        d = d[d["club_id"] == club_filter].copy()

    if cfg.distance_mode.lower() == "total" and "total_yd" in d.columns:
        d["_x"] = pd.to_numeric(d["total_yd"], errors="coerce")
    else:
        d["_x"] = pd.to_numeric(d["carry_yd"], errors="coerce")

    off = pd.to_numeric(d["offline_yd"], errors="coerce")
    d["_y"] = (-off) if cfg.right_miss_down else off

    d = d.dropna(subset=["_x", "_y"]).copy()
    if d.empty:
        return d

    d = d.reset_index(drop=True)
    d["_row_i"] = np.arange(len(d))
    return d


def _compute_bounds(d: pd.DataFrame, cfg: DispersionConfig) -> Dict[str, float]:
    x_min = 0.0
    x_max_data = float(d["_x"].max())

    # Chart ends 20 yards after the furthest shot
    x_max = x_max_data + 20.0

    y_abs_max = float(np.nanmax(np.abs(d["_y"].values)))
    y_pad = max(5.0, y_abs_max * 0.12)
    y_lim = y_abs_max + y_pad

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_lim": y_lim,
    }


def _add_course_layers(fig: go.Figure, bounds: Dict[str, float], cfg: DispersionConfig):
    x_min = bounds["x_min"]
    x_max = bounds["x_max"]

    fw_half = cfg.fairway_width_yd / 2.0
    rough_half = fw_half * 1.9
    deep_rough_half = fw_half * 2.8

    # Deep rough
    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-deep_rough_half, y1=deep_rough_half,
        line=dict(width=0),
        fillcolor="rgba(34, 139, 34, 0.05)",
        layer="below",
    )

    # Rough
    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-rough_half, y1=rough_half,
        line=dict(width=0),
        fillcolor="rgba(34, 139, 34, 0.10)",
        layer="below",
    )

    # Fairway
    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-fw_half, y1=fw_half,
        line=dict(color="rgba(20,80,30,0.20)", width=1),
        fillcolor="rgba(30,150,70,0.18)",
        layer="below",
    )

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


def _build_hover_payload(d: pd.DataFrame):
    hover_bits: List[str] = []
    cd_cols: List[str] = []

    def _add(col: str, label: str, fmt: str):
        if col in d.columns:
            idx = len(cd_cols)
            cd_cols.append(col)
            hover_bits.append(f"{label}: %{{customdata[{idx}]:{fmt}}}")

    _add("club_speed_mph", "Club Speed (mph)", ".1f")
    _add("ball_speed_mph", "Ball Speed (mph)", ".1f")
    _add("smash", "Smash", ".2f")
    _add("vla_deg", "Launch (°)", ".1f")
    _add("backspin_rpm", "Spin (rpm)", ".0f")

    return cd_cols, hover_bits


def _extract_selected_point(plot_state: Optional[dict]) -> Optional[int]:
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
# Single-dataset figure
# -----------------------------
def _build_dispersion_figure(
    df: pd.DataFrame,
    cfg: DispersionConfig,
    club_filter: Optional[str] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    d = _prepare_plot_df(df, cfg, club_filter=club_filter)

    if d.empty:
        return _empty_figure(), d

    bounds = _compute_bounds(d, cfg)
    fig = go.Figure()

    _add_course_layers(fig, bounds, cfg)

    if cfg.show_target_marker:
        tx = float(d["_x"].mean())
        fig.add_trace(go.Scatter(
            x=[tx], y=[0],
            mode="markers",
            marker=dict(size=8, symbol="x"),
            hovertemplate="Target<br>Downrange: %{x:.1f} yd<extra></extra>",
            showlegend=False,
            opacity=0.7,
        ))

    cd_cols, hover_bits = _build_hover_payload(d)

    club_palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#ea580c",
        "#7c3aed",
        "#0891b2",
        "#be185d",
        "#0f766e",
    ]

    clubs_present = sorted(d["club_id"].dropna().unique())

    for i, club in enumerate(clubs_present):
        color = club_palette[i % len(club_palette)]
        dc = d[d["club_id"] == club].copy()
        if dc.empty:
            continue

        customdata = dc[cd_cols].to_numpy() if cd_cols else None

        fig.add_trace(go.Scatter(
            x=dc["_x"],
            y=dc["_y"],
            mode="markers",
            marker=dict(size=8, opacity=0.82, color=color),
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

        cx = float(dc["_x"].mean())
        cy = float(dc["_y"].mean())

        dx = (dc["_x"].values - cx).astype(float)
        dy = (dc["_y"].values - cy).astype(float)

        rx, ry = _dispersion_ellipse_radii(dx, dy, cfg.circle_mode)
        rx = max(rx, cfg.circle_min_radius_yd)
        ry = max(ry, cfg.circle_min_radius_yd)

        fig.add_trace(_ellipse_trace(cx, cy, rx, ry, color=color, name=str(club), opacity=cfg.circle_opacity))

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
                f"Dispersion ellipse ({cfg.circle_mode}): {rx:.1f} yd x {ry:.1f} yd"
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
            range=[bounds["x_min"], bounds["x_max"]],
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(180, 190, 205, 0.35)",
        ),
        yaxis=dict(
            title="Lateral (yd) — right miss down",
            range=[-bounds["y_lim"], bounds["y_lim"]],
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(180, 190, 205, 0.35)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if cfg.keep_proportions:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_yaxes(constrain="domain")

    return fig, d


# -----------------------------
# Compare figure
# -----------------------------
def _build_compare_dispersion_figure(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    cfg: DispersionConfig,
    club_filter: Optional[str] = None,
    label_a: str = "Setup A",
    label_b: str = "Setup B",
) -> go.Figure:
    a = _prepare_plot_df(df_a, cfg, club_filter=club_filter)
    b = _prepare_plot_df(df_b, cfg, club_filter=club_filter)

    both = pd.concat([a, b], ignore_index=True)
    if both.empty:
        return _empty_figure("No plottable shots for compare dispersion.")

    bounds = _compute_bounds(both, cfg)
    fig = go.Figure()

    _add_course_layers(fig, bounds, cfg)

    color_a = "#2563eb"
    color_b = "#f97316"

    def add_setup_trace(d: pd.DataFrame, label: str, color: str):
        if d.empty:
            return

        fig.add_trace(go.Scatter(
            x=d["_x"],
            y=d["_y"],
            mode="markers",
            marker=dict(size=8, opacity=0.80, color=color),
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Downrange: %{x:.1f} yd<br>"
                "Lateral: %{y:.1f} yd<br>"
                "<extra></extra>"
            ),
        ))

        cx = float(d["_x"].mean())
        cy = float(d["_y"].mean())
        dx = (d["_x"].values - cx).astype(float)
        dy = (d["_y"].values - cy).astype(float)

        rx, ry = _dispersion_ellipse_radii(dx, dy, cfg.circle_mode)
        rx = max(rx, cfg.circle_min_radius_yd)
        ry = max(ry, cfg.circle_min_radius_yd)

        fig.add_trace(_ellipse_trace(cx, cy, rx, ry, color=color, name=label, opacity=cfg.circle_opacity))

        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers",
            marker=dict(size=6, color=color, symbol="diamond"),
            name=f"{label} mean",
            hovertemplate=(
                f"<b>{label} mean</b><br>"
                "Downrange: %{x:.1f} yd<br>"
                "Lateral: %{y:.1f} yd<br>"
                f"Dispersion ellipse ({cfg.circle_mode}): {rx:.1f} yd x {ry:.1f} yd"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    add_setup_trace(a, label_a, color_a)
    add_setup_trace(b, label_b, color_b)

    title = "Compare Dispersion"
    if club_filter:
        title += f" — {club_filter}"

    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(
            title="Downrange (yd)",
            range=[bounds["x_min"], bounds["x_max"]],
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(180, 190, 205, 0.35)",
        ),
        yaxis=dict(
            title="Lateral (yd) — right miss down",
            range=[-bounds["y_lim"], bounds["y_lim"]],
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(180, 190, 205, 0.35)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if cfg.keep_proportions:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_yaxes(constrain="domain")

    return fig


# -----------------------------
# Public functions
# -----------------------------
def render_dispersion(
    canon_df: pd.DataFrame,
    key_prefix: str = "dispersion",
    lock_club: Optional[str] = None,
):
    """
    Standard single-dataset dispersion chart with unique widget keys.
    If lock_club is provided, the club selector is removed.
    """
    if canon_df is None or canon_df.empty:
        st.info("No shot data available.")
        return

    if lock_club is None:
        c0, c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2])

        clubs = sorted([c for c in canon_df["club_id"].dropna().unique().tolist()])

        with c0:
            club_filter = st.selectbox(
                "Club",
                ["ALL"] + clubs,
                index=0,
                key=f"{key_prefix}_club_filter",
            )

        with c1:
            distance_mode = st.selectbox(
                "Distance",
                ["carry", "total"],
                index=0,
                key=f"{key_prefix}_distance_mode",
            )

        with c2:
            fairway_width = st.slider(
                "Fairway width (yd)",
                30, 140, 70, 1,
                key=f"{key_prefix}_fairway_width",
            )

        with c3:
            end_mode = st.selectbox(
                "Fairway length",
                ["p95", "max"],
                index=1,
                key=f"{key_prefix}_end_mode",
            )

        with c4:
            circle_mode = st.selectbox(
                "Dispersion ellipse",
                ["p80", "p90", "p95", "1sigma", "2sigma"],
                index=3,
                key=f"{key_prefix}_circle_mode",
            )

        club = None if club_filter == "ALL" else club_filter

    else:
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])

        with c1:
            distance_mode = st.selectbox(
                "Distance",
                ["carry", "total"],
                index=0,
                key=f"{key_prefix}_distance_mode",
            )

        with c2:
            fairway_width = st.slider(
                "Fairway width (yd)",
                30, 140, 70, 1,
                key=f"{key_prefix}_fairway_width",
            )

        with c3:
            end_mode = st.selectbox(
                "Fairway length",
                ["p95", "max"],
                index=1,
                key=f"{key_prefix}_end_mode",
            )

        with c4:
            circle_mode = st.selectbox(
                "Dispersion ellipse",
                ["p80", "p90", "p95", "1sigma", "2sigma"],
                index=3,
                key=f"{key_prefix}_circle_mode",
            )

        club = lock_club
        st.caption(f"Locked to club: {lock_club}")

    cfg = DispersionConfig(
        distance_mode=distance_mode,
        fairway_width_yd=float(fairway_width),
        fairway_end_mode=end_mode,
        right_miss_down=True,
        circle_mode=circle_mode,
        keep_proportions=False,
        circle_min_radius_yd=3.5,
    )

    fig, df_plot = _build_dispersion_figure(canon_df, cfg=cfg, club_filter=club)
    plot_key = f"{key_prefix}_plot"

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


def render_compare_dispersion(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_prefix: str = "compare_dispersion",
    label_a: str = "Setup A",
    label_b: str = "Setup B",
):
    """
    One combined compare chart for two uploads.
    """
    if df_a is None or df_a.empty or df_b is None or df_b.empty:
        st.info("Both datasets need shot data for compare dispersion.")
        return

    c0, c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2])

    clubs_a = set(df_a["club_id"].dropna().unique().tolist())
    clubs_b = set(df_b["club_id"].dropna().unique().tolist())
    clubs = sorted(list(clubs_a.intersection(clubs_b))) if (clubs_a and clubs_b) else []

    with c0:
        club_filter = st.selectbox(
            "Club",
            ["ALL"] + clubs,
            index=0,
            key=f"{key_prefix}_club_filter",
        )

    with c1:
        distance_mode = st.selectbox(
            "Distance",
            ["carry", "total"],
            index=0,
            key=f"{key_prefix}_distance_mode",
        )

    with c2:
        fairway_width = st.slider(
            "Fairway width (yd)",
            30, 140, 70, 1,
            key=f"{key_prefix}_fairway_width",
        )

    with c3:
        end_mode = st.selectbox(
            "Fairway length",
            ["p95", "max"],
            index=1,
            key=f"{key_prefix}_end_mode",
        )

    with c4:
        circle_mode = st.selectbox(
            "Dispersion ellipse",
            ["p80", "p90", "p95", "1sigma", "2sigma"],
            index=3,
            key=f"{key_prefix}_circle_mode",
        )

    cfg = DispersionConfig(
        distance_mode=distance_mode,
        fairway_width_yd=float(fairway_width),
        fairway_end_mode=end_mode,
        right_miss_down=True,
        circle_mode=circle_mode,
        keep_proportions=False,
        circle_min_radius_yd=3.5,
    )

    club = None if club_filter == "ALL" else club_filter
    fig = _build_compare_dispersion_figure(
        df_a=df_a,
        df_b=df_b,
        cfg=cfg,
        club_filter=club,
        label_a=label_a,
        label_b=label_b,
    )

    st.plotly_chart(fig, use_container_width=True)
