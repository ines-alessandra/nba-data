#!/usr/bin/env python3
"""
Runs the actual inferential significance tests for WPA H1 (Preenche Lacuna vs
Redundante) and H2 (Objetivo Sazonal) that describe_wpa_results.py deliberately
does NOT run, per its own docstring: "with only 3 real ESPN seasons... there is
no statistical power for hypothesis tests." Now that
build_wpa_trade_analysis_extended.py provides 370 trade events across 13
seasons (2013-2025) instead of 90 across 3, that limitation no longer holds --
this script is the direct answer to it.

H1 - Preenche Lacuna vs Redundante (2 independent groups, non-normal outcome
     var -> Mann-Whitney U, two-sided). Effect size: rank-biserial correlation
     r = 1 - 2U/(n1*n2), same -1..+1 correlation scale as Spearman's rho, so
     it's read with the SAME Cohen (1988) magnitude bands generate_h1_verdict.py
     already uses for the salary-inequality H1 (desprezivel/fraco/moderado/forte).

H2 - Objetivo Sazonal, 3 independent groups -> Kruskal-Wallis H-test (the
     omnibus test), with Bonferroni-corrected pairwise Mann-Whitney post-hoc
     tests if the omnibus is significant -- same pattern already used for the
     cluster comparison in pooled_clustering_analysis.py. Effect size: eta-
     squared (H - k + 1) / (n - k), which lives on a DIFFERENT scale than
     rank-biserial r (0..~1, not -1..+1), so it's read against the conventional
     ANOVA eta-squared bands (pequeno/medio/grande) instead of Cohen's
     correlation bands.

Both tests are run twice: POOLED across 2013-2025 (the headline result -- the
whole point of extending the season range), and restricted to 2019-2021 only
(the original ESPN-native sample), side by side, so the "N was too small
before" claim is directly falsifiable from the same script's own output rather
than asserted.

FOLLOW-UP INVESTIGATION (H1b/c/d below): the first pooled run of H1a showed a
genuine, non-obvious reversal -- significant in the original 2019-2021 sample
(p=0.0375) but NOT significant pooled across 2013-2025 (p=0.233). Before
reporting that at face value, four things were checked to rule out an
artifact:
  H1b - the SAME data treated as a continuous predictor (min_deltaP_acquired,
        Spearman rho vs delta_WPA_time) instead of thresholded at deltaP=0.
        This IS significant pooled (rho=-0.124, p=0.017) -- so collapsing a
        continuous gap severity into a binary label at exactly zero is
        discarding real information. But the continuous effect itself is NOT
        stable across eras: strong in 2019-2021 (rho=-0.313, p=0.0027), weak
        in 2013-2018 (ns), and essentially ABSENT in 2022-2025
        (rho=+0.025, p=0.776, wrong sign) -- an open question, not resolved
        here (real market-behavior shift? artifact of the inpredictable-
        sourced WPA in that era? insufficient N within just that 4-year
        window?).
  H1c - excluding 2020 (the pandemic-shortened season, already flagged as
        atypical in docs/METODOLOGIA_E_RESULTADOS.md) barely moves either
        result -- 2020 is not the driver of the reversal.
  H1d - splitting the binary test by position (Guards/Forwards/Centers)
        separately: none of the 3 is individually significant either, and
        Guards even points the wrong direction (median) -- no hidden
        significant subgroup was masked by pooling positions together.
Conclusion: the reversal is real, not a bug, not a 2020 artifact, and not
explained by position. The binary H1 classification specifically does not
replicate at scale; a weaker, era-unstable continuous signal survives.

Output: data/wpa_hypothesis_tests_extended.csv (one row per test/post-hoc pair)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpa_common import PROJECT_DIR  # noqa: E402

IMPACT_PATH = PROJECT_DIR / "data" / "wpa_trade_impact_analysis_extended.csv"
CLASS_PATH = PROJECT_DIR / "data" / "wpa_trade_classification_extended.csv"
OUT_PATH = PROJECT_DIR / "data" / "wpa_hypothesis_tests_extended.csv"

ALPHA = 0.05
ORIGINAL_SEASONS = (2019, 2021)  # inclusive; the pre-extension ESPN-only window


def rank_biserial_label(abs_r: float) -> str:
    """Cohen (1988) correlation-magnitude bands -- same ones generate_h1_verdict.py
    already uses for the salary-inequality Spearman rho, since rank-biserial r
    lives on the same -1..+1 correlation scale."""
    if abs_r < 0.10:
        return "desprezível"
    if abs_r < 0.30:
        return "fraco"
    if abs_r < 0.50:
        return "moderado"
    return "forte"


def eta_squared_label(eta2: float) -> str:
    """Conventional ANOVA eta-squared bands (Cohen 1988) -- NOT the same scale as
    rank-biserial r, so a separate labeling function on purpose."""
    if eta2 < 0.01:
        return "desprezível"
    if eta2 < 0.06:
        return "pequeno"
    if eta2 < 0.14:
        return "médio"
    return "grande"


def mannwhitney_test(df: pd.DataFrame, group_col: str, groups: tuple[str, str],
                      value_col: str, scope: str, hypothesis: str, rows: list) -> None:
    g1 = df.loc[df[group_col] == groups[0], value_col].dropna()
    g2 = df.loc[df[group_col] == groups[1], value_col].dropna()
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        rows.append({"hipotese": hypothesis, "escopo": scope, "teste": "Mann-Whitney U",
                      "grupos": f"{groups[0]} (n={n1}) vs {groups[1]} (n={n2})",
                      "estatistica": np.nan, "p_valor": np.nan,
                      "effect_size": np.nan, "effect_label": "N insuficiente",
                      "significativo": False})
        return

    u_stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    r = 1 - (2 * u_stat) / (n1 * n2)  # rank-biserial correlation
    sig = p < ALPHA
    rows.append({
        "hipotese": hypothesis, "escopo": scope, "teste": "Mann-Whitney U",
        "grupos": f"{groups[0]} (n={n1}) vs {groups[1]} (n={n2})",
        "estatistica": float(u_stat), "p_valor": float(p),
        "effect_size": float(r), "effect_label": rank_biserial_label(abs(r)),
        "significativo": bool(sig),
    })
    print(f"  [{scope}] {groups[0]} (n={n1}, mediana={g1.median():.4f}) vs "
          f"{groups[1]} (n={n2}, mediana={g2.median():.4f}): "
          f"U={u_stat:.1f}, p={p:.4f}, r={r:+.3f} ({rank_biserial_label(abs(r))}) "
          f"{'*** significativo' if sig else '(não significativo)'}")


def kruskal_test(df: pd.DataFrame, group_col: str, groups: tuple[str, ...],
                  value_col: str, scope: str, hypothesis: str, rows: list) -> None:
    group_series = {g: df.loc[df[group_col] == g, value_col].dropna() for g in groups}
    present = [(g, s) for g, s in group_series.items() if len(s) >= 2]
    if len(present) < 2:
        rows.append({"hipotese": hypothesis, "escopo": scope, "teste": "Kruskal-Wallis H",
                      "grupos": ", ".join(f"{g} (n={len(s)})" for g, s in group_series.items()),
                      "estatistica": np.nan, "p_valor": np.nan,
                      "effect_size": np.nan, "effect_label": "N insuficiente",
                      "significativo": False})
        return

    values = [s.values for _, s in present]
    n_total = sum(len(v) for v in values)
    k = len(values)
    h_stat, p = stats.kruskal(*values)
    eta2 = max((h_stat - k + 1) / (n_total - k), 0.0)  # clipped at 0: H can dip just below k-1 by chance
    sig = p < ALPHA
    rows.append({
        "hipotese": hypothesis, "escopo": scope, "teste": "Kruskal-Wallis H",
        "grupos": ", ".join(f"{g} (n={len(s)})" for g, s in group_series.items()),
        "estatistica": float(h_stat), "p_valor": float(p),
        "effect_size": float(eta2), "effect_label": eta_squared_label(eta2),
        "significativo": bool(sig),
    })
    print(f"\n  [{scope}] Kruskal-Wallis across {[g for g, _ in present]}: "
          f"H={h_stat:.3f}, p={p:.4f}, eta²={eta2:.4f} ({eta_squared_label(eta2)}) "
          f"{'*** significativo' if sig else '(não significativo)'}")
    for g, s in present:
        print(f"      {g}: n={len(s)}, mediana={s.median():.4f}, %positivo={100*(s > 0).mean():.1f}%")

    if sig and len(present) > 2:
        pairs = [(present[i][0], present[j][0]) for i in range(len(present)) for j in range(i + 1, len(present))]
        alpha_adj = ALPHA / len(pairs)
        print(f"    Post-hoc Mann-Whitney (Bonferroni alpha_adj={alpha_adj:.4f}):")
        for g1_name, g2_name in pairs:
            v1, v2 = group_series[g1_name].values, group_series[g2_name].values
            u_stat, p_pair = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            r_pair = 1 - (2 * u_stat) / (len(v1) * len(v2))
            sig_pair = p_pair < alpha_adj
            print(f"      {g1_name} vs {g2_name}: U={u_stat:.1f}, p={p_pair:.4f}, r={r_pair:+.3f} "
                  f"{'*** significativo (pós-Bonferroni)' if sig_pair else ''}")
            rows.append({
                "hipotese": f"{hypothesis} (post-hoc)", "escopo": scope, "teste": "Mann-Whitney U (Bonferroni)",
                "grupos": f"{g1_name} (n={len(v1)}) vs {g2_name} (n={len(v2)})",
                "estatistica": float(u_stat), "p_valor": float(p_pair),
                "effect_size": float(r_pair), "effect_label": rank_biserial_label(abs(r_pair)),
                "significativo": bool(sig_pair),
            })


def spearman_row(sub: pd.DataFrame, scope: str, hypothesis: str, rows: list) -> None:
    s = sub.dropna(subset=["min_deltaP_acquired", "delta_WPA_time"])
    if len(s) < 4:
        rows.append({"hipotese": hypothesis, "escopo": scope, "teste": "Spearman rho",
                      "grupos": f"n={len(s)}", "estatistica": np.nan, "p_valor": np.nan,
                      "effect_size": np.nan, "effect_label": "N insuficiente", "significativo": False})
        return
    rho, p = stats.spearmanr(s["min_deltaP_acquired"], s["delta_WPA_time"])
    sig = p < ALPHA
    rows.append({
        "hipotese": hypothesis, "escopo": scope, "teste": "Spearman rho",
        "grupos": f"n={len(s)}", "estatistica": float(rho), "p_valor": float(p),
        "effect_size": float(rho), "effect_label": rank_biserial_label(abs(rho)),
        "significativo": bool(sig),
    })
    print(f"  [{scope}] n={len(s)}, rho={rho:+.4f}, p={p:.4f} ({rank_biserial_label(abs(rho))}) "
          f"{'*** significativo' if sig else ''}")


def main():
    df = pd.read_csv(IMPACT_PATH)
    df_class = pd.read_csv(CLASS_PATH, parse_dates=["trade_date"])
    df_dated = pd.read_csv(IMPACT_PATH, parse_dates=["trade_date"])
    rows: list = []

    print("=" * 70)
    print("  Testes de significância WPA -- H1 e H2 (2013-2025 vs 2019-2021)")
    print("=" * 70)

    print("\nH1a: Preenche Lacuna vs Redundante -- classificação BINÁRIA (delta_WPA_time)")
    print("-" * 70)
    mannwhitney_test(df, "trade_classification", ("Preenche Lacuna", "Redundante"),
                      "delta_WPA_time", "2013-2025 (estendido, pooled)", "H1a-binario", rows)
    df_orig = df[df["season_end_year"].between(*ORIGINAL_SEASONS)]
    mannwhitney_test(df_orig, "trade_classification", ("Preenche Lacuna", "Redundante"),
                      "delta_WPA_time", f"{ORIGINAL_SEASONS[0]}-{ORIGINAL_SEASONS[1]} (amostra original)",
                      "H1a-binario", rows)

    print("\nH1b: min_deltaP_acquired (CONTÍNUO, sem threshold em zero) x delta_WPA_time")
    print("  (testa se a MAGNITUDE da lacuna, não só seu sinal, prediz o impacto -- "
          "o corte binário em deltaP=0 pode estar descartando informação)")
    print("-" * 70)
    spearman_row(df, "2013-2025 (estendido, pooled)", "H1b-continuo", rows)
    for lo, hi, label in [(2013, 2018, "2013-2018 (posição estática)"),
                           (2019, 2021, "2019-2021 (ESPN nativo)"),
                           (2022, 2025, "2022-2025 (híbrido)")]:
        spearman_row(df[df["season_end_year"].between(lo, hi)], label, "H1b-continuo", rows)

    print("\nH1c: Robustez -- excluindo 2020 (temporada COVID, já sinalizada como atípica)")
    print("-" * 70)
    df_no2020 = df[df["season_end_year"] != 2020]
    mannwhitney_test(df_no2020, "trade_classification", ("Preenche Lacuna", "Redundante"),
                      "delta_WPA_time", "2013-2025 sem 2020 (binário)", "H1c-robustez", rows)
    spearman_row(df_no2020, "2013-2025 sem 2020 (contínuo)", "H1c-robustez", rows)

    print("\nH1d: Por posição (nível jogador -- delta_WPA_time repete quando >1 jogador/evento,"
          " ver ressalva no docstring)")
    print("-" * 70)
    merged = df_class.merge(
        df_dated[["team_id", "season_end_year", "trade_date", "delta_WPA_time"]],
        on=["team_id", "season_end_year", "trade_date"], how="left",
    )
    for pos in ["Guards", "Forwards", "Centers"]:
        mannwhitney_test(merged[merged["posicao"] == pos], "preenche_lacuna",
                          ("Preenche Lacuna", "Redundante"), "delta_WPA_time",
                          f"posição={pos} (pooled, pseudo-replicado)", "H1d-por_posicao", rows)

    print("\nH2: Objetivo Sazonal (delta_WPA_time) -- teste omnibus (3 grupos)")
    print("-" * 70)
    kruskal_test(df, "objetivo_sazonal", ("Contender", "Intermediario", "Tanking"),
                 "delta_WPA_time", "2013-2025 (estendido, pooled)", "H2-omnibus", rows)
    kruskal_test(df_orig, "objetivo_sazonal", ("Contender", "Intermediario", "Tanking"),
                 "delta_WPA_time", f"{ORIGINAL_SEASONS[0]}-{ORIGINAL_SEASONS[1]} (amostra original)",
                 "H2-omnibus", rows)

    # --- Exact wording of the two hypotheses, as stated in the thesis, is directional,
    # not just "some difference exists". These tests answer the literal claims:
    #   H1: "trocas que preenchem lacuna produzem impacto MAIS POSITIVO que redundantes"
    #       -> one-sided Mann-Whitney (Lacuna > Redundante), not the two-sided H1a above.
    #   H2: "Contenders melhoram no curto prazo" (delta_WPA_time > 0) AND
    #       "Tanking tem impacto neutro OU negativo" (delta_WPA_time <= 0, i.e. NOT >0)
    #       -> one-sample Wilcoxon signed-rank per group (H0: mediana=0) for each
    #          group's own claim, PLUS a direct one-sided Contender > Tanking test,
    #          which is the hypothesis's core implicit contrast.
    print("\nH1-exata: teste DIRECIONAL (one-sided) -- 'Lacuna produz impacto MAIS POSITIVO' literalmente")
    print("-" * 70)
    g_lac = df["delta_WPA_time"].where(df["trade_classification"] == "Preenche Lacuna").dropna()
    g_red = df["delta_WPA_time"].where(df["trade_classification"] == "Redundante").dropna()
    u, p = stats.mannwhitneyu(g_lac, g_red, alternative="greater")
    r = 1 - (2 * u) / (len(g_lac) * len(g_red))
    sig = p < ALPHA
    print(f"  Lacuna (n={len(g_lac)}, mediana={g_lac.median():+.4f}) > Redundante "
          f"(n={len(g_red)}, mediana={g_red.median():+.4f})? U={u:.1f}, p={p:.4f}, r={r:+.3f} "
          f"{'*** significativo' if sig else '(não significativo)'}")
    rows.append({"hipotese": "H1-exata (one-sided)", "escopo": "2013-2025 (estendido, pooled)",
                  "teste": "Mann-Whitney U (one-sided, Lacuna>Redundante)",
                  "grupos": f"Preenche Lacuna (n={len(g_lac)}) > Redundante (n={len(g_red)})",
                  "estatistica": float(u), "p_valor": float(p), "effect_size": float(r),
                  "effect_label": rank_biserial_label(abs(r)), "significativo": bool(sig)})

    print("\nH2-exata: cada afirmação da hipótese testada literalmente")
    print("-" * 70)
    for obj, expectativa in [("Contender", "delta_WPA_time > 0 (melhora)"),
                              ("Tanking", "delta_WPA_time <= 0 (neutro ou negativo)")]:
        s = df.loc[df["objetivo_sazonal"] == obj, "delta_WPA_time"].dropna()
        w_stat, p_w = stats.wilcoxon(s)
        sig_w = p_w < ALPHA
        print(f"  {obj} (n={len(s)}, mediana={s.median():+.4f}, %positivo={100*(s>0).mean():.1f}%) "
              f"-- previsão: {expectativa}")
        print(f"    Wilcoxon (H0: mediana=0): W={w_stat:.1f}, p={p_w:.4f} -> "
              f"{'diferente de zero' if sig_w else 'NÃO distinguível de zero'} "
              f"({'positivo' if s.median() > 0 else 'negativo'})")
        rows.append({"hipotese": "H2-exata (one-sample)", "escopo": f"{obj} (2013-2025)",
                      "teste": "Wilcoxon signed-rank (H0: mediana=0)",
                      "grupos": f"n={len(s)}", "estatistica": float(w_stat), "p_valor": float(p_w),
                      "effect_size": float(s.median()), "effect_label": expectativa,
                      "significativo": bool(sig_w)})

    g_c = df.loc[df["objetivo_sazonal"] == "Contender", "delta_WPA_time"].dropna()
    g_t = df.loc[df["objetivo_sazonal"] == "Tanking", "delta_WPA_time"].dropna()
    u2, p2 = stats.mannwhitneyu(g_c, g_t, alternative="greater")
    r2 = 1 - (2 * u2) / (len(g_c) * len(g_t))
    sig2 = p2 < ALPHA
    print(f"\n  Contraste direto (one-sided): Contender (mediana={g_c.median():+.4f}) > "
          f"Tanking (mediana={g_t.median():+.4f})? U={u2:.1f}, p={p2:.4f}, r={r2:+.3f} "
          f"{'*** significativo' if sig2 else '(não significativo)'}")
    rows.append({"hipotese": "H2-exata (Contender>Tanking)", "escopo": "2013-2025 (estendido, pooled)",
                  "teste": "Mann-Whitney U (one-sided)",
                  "grupos": f"Contender (n={len(g_c)}) > Tanking (n={len(g_t)})",
                  "estatistica": float(u2), "p_valor": float(p2), "effect_size": float(r2),
                  "effect_label": rank_biserial_label(abs(r2)), "significativo": bool(sig2)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\n\nSaved {len(out)} test rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
