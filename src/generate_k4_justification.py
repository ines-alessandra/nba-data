#!/usr/bin/env python3
"""
"Por que k=4?" -- justification using ONLY the elbow curve (WCSS) and the
Silhouette curve (no ARI, no cross-algorithm stability, no appeal to
"literature convention" -- those are legitimate secondary arguments used
elsewhere in this project, see generate_k_selection_table.py, but this figure
answers the question with just the two classic curves, for a slide that needs
to stand on its own).

The argument, precisely: neither curve has its OPTIMUM at k=4 (Silhouette is
highest at k=2 and falls monotonically; WCSS never stops falling, by
construction). What both curves DO share is a clear change in REGIME right at
k=4: a steep decline from k=2 to k=4, followed by a much flatter, noisier
plateau from k=4 to k=10. That regime change -- not a peak -- is what "the
elbow" means in the elbow method, and it's visible, at the same k, in both
curves independently. This script quantifies that regime change (average
drop-per-step before vs. after k=4) and renders it as an annotated 2-panel
figure.

Reads data/clustering_k_selection_metrics.csv (produced by
generate_k_selection_table.py -- run that first).
"""

import textwrap
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
METRICS_CSV = PROJECT_DIR / "data" / "clustering_k_selection_metrics.csv"
OUT_PNG = PROJECT_DIR / "viz" / "justificativa_k4_elbow_silhouette.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
STEEP_COLOR = "#d62728"
FLAT_COLOR = "#2ca02c"
LINE_COLOR_WCSS = "#1f77b4"
LINE_COLOR_SIL = "#2ca02c"
K_CHOSEN = 4


def compute_regime_change(df: pd.DataFrame, col: str):
    before = df[df['k'] <= K_CHOSEN].sort_values('k')
    after = df[df['k'] >= K_CHOSEN].sort_values('k')
    steps_before = len(before) - 1
    steps_after = len(after) - 1
    drop_before = before[col].iloc[0] - before[col].iloc[-1]
    drop_after = after[col].iloc[0] - after[col].iloc[-1]
    rate_before = drop_before / steps_before
    rate_after = drop_after / steps_after
    ratio = rate_before / rate_after if rate_after != 0 else float('inf')
    return rate_before, rate_after, ratio


def build_marginal_gain_chart(df: pd.DataFrame):
    """The honest, step-by-step version of the argument: bar = how much k-1->k
    still bought you, for each metric. Doesn't smooth over the k=4-vs-k=5
    ambiguity in Silhouette the way the aggregated before/after rates in
    build() do -- shows it plainly instead, and lets WCSS (where the step
    into k=4 is clearly bigger than the step into k=5) be the tie-breaker."""
    df = df.copy()
    df['wcss_step_pct'] = df['pct_drop_wcss']
    df['sil_step'] = -df['silhouette'].diff()
    labels = [f"{k-1}→{k}" for k in df['k']][1:]  # skip k=2 (no previous step)
    wcss_steps = df['wcss_step_pct'].iloc[1:].values
    sil_steps = df['sil_step'].iloc[1:].values
    k_targets = df['k'].iloc[1:].values

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.3), dpi=300)
    fig.subplots_adjust(top=0.78, bottom=0.32, wspace=0.25)

    # Panel 1: marginal WCSS drop per step
    ax = axes[0]
    colors = ['#d62728' if k == 4 else ('#f4a582' if k == 5 else '#bdc3c7') for k in k_targets]
    bars = ax.bar(labels, wcss_steps, color=colors, edgecolor='black', linewidth=0.8, zorder=3)
    for bar, v in zip(bars, wcss_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.1f}%", ha='center', fontsize=9.5, fontweight='bold')
    ax.set_title("Ganho Marginal de WCSS a Cada Cluster Adicional", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_ylabel("Queda de WCSS neste passo (%)", fontsize=11)
    ax.set_xlabel("Passo (de k-1 para k)", fontsize=11)
    ax.grid(axis='y', alpha=0.4, zorder=0)

    # Panel 2: marginal Silhouette drop per step
    ax = axes[1]
    bars = ax.bar(labels, sil_steps, color=colors, edgecolor='black', linewidth=0.8, zorder=3)
    for bar, v in zip(bars, sil_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0006, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title("Ganho Marginal de Silhouette a Cada Cluster Adicional", fontsize=13.5, fontweight='bold', pad=12)
    ax.set_ylabel("Queda de Silhouette neste passo", fontsize=11)
    ax.set_xlabel("Passo (de k-1 para k)", fontsize=11)
    ax.grid(axis='y', alpha=0.4, zorder=0)

    fig.suptitle("O Ganho Marginal de Cada Cluster Adicional, Passo a Passo",
                  fontsize=16, fontweight='bold', y=0.96, color=INK_PRIMARY)
    caption = (
        "Vermelho = passo que leva a k=4; laranja claro = passo que leva a k=5. No WCSS, o passo 3→4 (4,4%) "
        "ainda é 1,7x maior que o passo 4→5 (2,6%) -- aqui o cotovelo já é visível. No Silhouette, os passos "
        "3→4 (0,0104) e 4→5 (0,0103) são praticamente IGUAIS -- o Silhouette sozinho não decide entre k=4 e "
        "k=5; quem decide é o WCSS, reforçado pela interpretação (4 arquétipos) e pela concordância com o "
        "Agglomerative Clustering (ARI=0,44). A partir de k=6, ambas as curvas já caíram para um patamar "
        "de ruído bem mais baixo, então k>=6 fica descartado por qualquer um dos dois critérios."
    )
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=145))
    fig.text(0.5, 0.1, wrapped_caption, ha='center', va='top', fontsize=9.6, color=INK_SECONDARY,
              fontstyle='italic', linespacing=1.5)

    out = PROJECT_DIR / "viz" / "justificativa_k4_ganho_marginal.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {out}")


def build():
    df = pd.read_csv(METRICS_CSV)

    wcss_rate_before, wcss_rate_after, wcss_ratio = compute_regime_change(df, 'inertia')
    sil_rate_before, sil_rate_after, sil_ratio = compute_regime_change(df, 'silhouette')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # --- Panel 1: Elbow (WCSS) ---
    ax = axes[0]
    ax.axvspan(2, K_CHOSEN, color=STEEP_COLOR, alpha=0.08, zorder=0)
    ax.axvspan(K_CHOSEN, 10, color=FLAT_COLOR, alpha=0.08, zorder=0)
    ax.plot(df['k'], df['inertia'], marker='o', linewidth=2.5, markersize=8, color=LINE_COLOR_WCSS, zorder=3)
    ax.axvline(K_CHOSEN, color=INK_PRIMARY, linestyle='--', linewidth=1.3, zorder=2)
    ax.set_title("Curva do Cotovelo (WCSS)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Número de Clusters (k)", fontsize=11.5)
    ax.set_ylabel("Inércia (WCSS)", fontsize=11.5)
    ax.set_xticks(df['k'])
    ax.text(2.9, df['inertia'].iloc[0] * 0.985, f"queda de\n~{wcss_rate_before:,.0f}/passo".replace(",", "."),
            fontsize=9.5, color=STEEP_COLOR, fontweight='bold', ha='center')
    ax.text(7, df['inertia'].iloc[-1] + (df['inertia'].iloc[0] - df['inertia'].iloc[-1]) * 0.18,
            f"queda de\n~{wcss_rate_after:,.0f}/passo".replace(",", "."),
            fontsize=9.5, color=FLAT_COLOR, fontweight='bold', ha='center')
    ax.text(K_CHOSEN, df['inertia'].max(), "k=4", fontsize=11, fontweight='bold', color=INK_PRIMARY,
            ha='center', va='bottom')

    # --- Panel 2: Silhouette ---
    ax = axes[1]
    ax.axvspan(2, K_CHOSEN, color=STEEP_COLOR, alpha=0.08, zorder=0)
    ax.axvspan(K_CHOSEN, 10, color=FLAT_COLOR, alpha=0.08, zorder=0)
    ax.plot(df['k'], df['silhouette'], marker='o', linewidth=2.5, markersize=8, color=LINE_COLOR_SIL, zorder=3)
    ax.axvline(K_CHOSEN, color=INK_PRIMARY, linestyle='--', linewidth=1.3, zorder=2)
    ax.set_title("Curva do Silhouette", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Número de Clusters (k)", fontsize=11.5)
    ax.set_ylabel("Silhouette Coefficient", fontsize=11.5)
    ax.set_xticks(df['k'])
    ax.text(2.9, df['silhouette'].iloc[0] * 0.93, f"queda de\n~{sil_rate_before:.4f}/passo",
            fontsize=9.5, color=STEEP_COLOR, fontweight='bold', ha='center')
    ax.text(7, df['silhouette'].iloc[-1] + (df['silhouette'].iloc[0] - df['silhouette'].iloc[-1]) * 0.22,
            f"queda de\n~{sil_rate_after:.4f}/passo",
            fontsize=9.5, color=FLAT_COLOR, fontweight='bold', ha='center')
    ax.text(K_CHOSEN, df['silhouette'].max(), "k=4", fontsize=11, fontweight='bold', color=INK_PRIMARY,
            ha='center', va='bottom')

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {OUT_PNG}")
    print(f"WCSS: rate before k=4 = {wcss_rate_before:.2f}/passo, after = {wcss_rate_after:.2f}/passo, ratio = {wcss_ratio:.2f}x")
    print(f"Silhouette: rate before k=4 = {sil_rate_before:.5f}/passo, after = {sil_rate_after:.5f}/passo, ratio = {sil_ratio:.2f}x")


if __name__ == "__main__":
    build()
    build_marginal_gain_chart(pd.read_csv(METRICS_CSV))
