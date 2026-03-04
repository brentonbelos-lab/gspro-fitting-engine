import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ClubSummary:
    club: str
    n_total: int
    n_used: int
    confidence: str
    metrics: Dict
    variability: Dict


@dataclass
class ClubAnalysis:
    summary: ClubSummary
    limiting_factors: List[str]
    recommendations: List[Dict]


@dataclass
class SessionResult:
    club_results: Dict[str, ClubAnalysis]


# ----------------------------
# Helpers
# ----------------------------

def _to_numeric_lr(series: pd.Series) -> pd.Series:
    """
    Extract numeric values from GSPro fields like:
    '11.4 R', '2.6° R', '0.8° I-O'
    """
    s = series.astype(str)

    is_left = s.str.contains(" L")

    num = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
    num = pd.to_numeric(num, errors="coerce")

    num = np.where(is_left, -num, num)

    return pd.Series(num, index=series.index)


def _confidence_label(n):
    if n >= 10:
        return "HIGH"
    if n >= 6:
        return "MED"
    return "LOW"


# ----------------------------
# Core analysis
# ----------------------------

def summarize_club(df: pd.DataFrame, club: str) -> ClubSummary:

    # parse problematic fields
    for col in ["offline", "vla", "hla", "spin_axis"]:
        if col in df.columns:
            df[col] = _to_numeric_lr(df[col])

    def mean(col):
        if col not in df.columns:
            return float("nan")
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    def std(col):
        if col not in df.columns:
            return float("nan")
        return float(pd.to_numeric(df[col], errors="coerce").std())

    metrics = {
        "carry": mean("carry"),
        "ball_speed": mean("ball_speed"),
        "club_speed": mean("club_speed"),
        "launch": mean("vla"),
        "spin": mean("spin"),
        "offline_mean": mean("offline"),
    }

    variability = {
        "offline_std": std("offline"),
        "launch_std": std("vla"),
        "spin_std": std("spin"),
    }

    return ClubSummary(
        club=club,
        n_total=len(df),
        n_used=len(df),
        confidence=_confidence_label(len(df)),
        metrics=metrics,
        variability=variability,
    )


def analyze_dataframe(df: pd.DataFrame) -> SessionResult:

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    club_map = {
        "dr": "DR",
        "driver": "DR",
        "h3": "HY",
        "hy": "HY"
    }

    df["club"] = df["club"].astype(str).str.lower()

    club_results = {}

    for club_raw, group in df.groupby("club"):

        bucket = club_map.get(club_raw)

        if bucket is None:
            continue

        summary = summarize_club(group.copy(), bucket)

        club_results[bucket] = ClubAnalysis(
            summary=summary,
            limiting_factors=[],
            recommendations=[]
        )

    return SessionResult(club_results)


def session_to_dict(result: SessionResult):

    out = {"clubs": {}}

    for club, analysis in result.club_results.items():

        out["clubs"][club] = {
            "summary": {
                "club": analysis.summary.club,
                "n_total": analysis.summary.n_total,
                "n_used": analysis.summary.n_used,
                "confidence": analysis.summary.confidence,
                "metrics": analysis.summary.metrics,
                "variability": analysis.summary.variability
            },
            "limiting_factors": analysis.limiting_factors,
            "recommendations": analysis.recommendations
        }

    return out
