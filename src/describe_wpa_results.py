#!/usr/bin/env python3
"""
Descriptive summary of the WPA methodology's output (methodology section E -
Analise Estatistica: EDA + observed behavior of WPA and RCP), run STRICTLY PER
SEASON (2019, 2020, 2021 - the only years with real ESPN WPA data). No pooled/
aggregated-across-seasons panel - each season is its own independent application
of the formulas, presented separately, per the decision to not blend seasons
together into a single number.

No inferential statistics here (no significance tests, no p-values) - only
descriptive summaries of what the methodology produces: counts, percentages,
medians, and a simple association measure (Spearman rho, reported as an
observed association strength, not a hypothesis test outcome).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR  # noqa: E402

IMPACT_PATH = PROJECT_DIR / "data" / "wpa_trade_impact_analysis.csv"
CLASS_PATH = PROJECT_DIR / "data" / "wpa_trade_classification.csv"
OUT_PATH = PROJECT_DIR / "data" / "wpa_descriptive_summary.csv"
OUT_POSITION_PATH = PROJECT_DIR / "data" / "wpa_positional_analysis_summary.csv"


def describe_h1(df, scope, rows):
    for cls in ["Preenche Lacuna", "Redundante"]:
        sub = df[df["trade_classification"] == cls]["delta_WPA_time"].dropna()
        rows.append({
            "escopo": scope, "eixo": "Preenche Lacuna vs Redundante", "grupo": cls,
            "n": len(sub), "mediana_delta_WPA_time": sub.median() if len(sub) else np.nan,
            "media_delta_WPA_time": sub.mean() if len(sub) else np.nan,
            "pct_positivo": (sub > 0).mean() * 100 if len(sub) else np.nan,
        })

    gap_sub = df[df["trade_classification"] == "Preenche Lacuna"].dropna(
        subset=["min_deltaP_acquired", "delta_WPA_time"])
    if len(gap_sub) >= 4:
        rho, _ = spearmanr(gap_sub["min_deltaP_acquired"].abs(), gap_sub["delta_WPA_time"])
        rows.append({
            "escopo": scope, "eixo": "Magnitude da lacuna (|deltaP|) vs impacto",
            "grupo": "Preenche Lacuna", "n": len(gap_sub),
            "mediana_delta_WPA_time": np.nan, "media_delta_WPA_time": np.nan,
            "pct_positivo": np.nan, "spearman_rho": rho,
        })


def describe_h2(df, scope, rows):
    for obj in ["Contender", "Intermediario", "Tanking"]:
        sub = df[df["objetivo_sazonal"] == obj]["delta_WPA_time"].dropna()
        rows.append({
            "escopo": scope, "eixo": "Objetivo Sazonal", "grupo": obj,
            "n": len(sub), "mediana_delta_WPA_time": sub.median() if len(sub) else np.nan,
            "media_delta_WPA_time": sub.mean() if len(sub) else np.nan,
            "pct_positivo": (sub > 0).mean() * 100 if len(sub) else np.nan,
        })


def describe_by_position(df_class, scope, rows):
    """Per-position breakdown: how many acquisitions filled a gap vs were
    redundant at each position group, that season."""
    for pos in ["Guards", "Forwards", "Centers"]:
        sub = df_class[df_class["posicao"] == pos]
        for cls in ["Preenche Lacuna", "Redundante"]:
            n = int((sub["preenche_lacuna"] == cls).sum())
            rows.append({"escopo": scope, "posicao": pos, "classificacao": cls, "n_jogadores": n})


def main():
    df_all = pd.read_csv(IMPACT_PATH)
    df_class_all = pd.read_csv(CLASS_PATH)

    print("=" * 70)
    print("  Resumo Descritivo da Metodologia WPA (2019-2021, dados reais ESPN)")
    print("  (por temporada, sem agregacao entre temporadas; sem teste de hipotese)")
    print("=" * 70)

    rows = []
    position_rows = []
    for season in sorted(df_all["season_end_year"].unique()):
        df = df_all[df_all["season_end_year"] == season]
        df_class = df_class_all[df_class_all["season_end_year"] == season]
        print(f"\n--- Temporada {season} (n={len(df)} trocas, {len(df_class)} jogadores adquiridos) ---")
        describe_h1(df, str(season), rows)
        describe_h2(df, str(season), rows)
        describe_by_position(df_class, str(season), position_rows)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved descriptive summary to {OUT_PATH}")
    print(df_out.to_string())

    df_pos_out = pd.DataFrame(position_rows)
    df_pos_out.to_csv(OUT_POSITION_PATH, index=False)
    print(f"\nSaved per-position breakdown to {OUT_POSITION_PATH}")
    print(df_pos_out.to_string())


if __name__ == "__main__":
    main()
