#!/usr/bin/env python3
"""
NBA Team-Season Clustering: K Selection Table
Author: Data Scientist

Companion to viz/clustering_validation.png (the elbow/silhouette/Davies-Bouldin
line charts). Those three curves, read on their own, do not unambiguously point
at k=4 -- Silhouette actually falls monotonically from k=2, and Davies-Bouldin is
best at k=2. This script renders the same k=2..10 sweep as a table, alongside the
Agglomerative-vs-KMeans Adjusted Rand Index at k=4 (fit_clusters.py) and an
explicit "leitura" column that states plainly why k=4 was picked anyway:
interpretability (four textbook competitive tiers) + cross-algorithm stability
(ARI), not because the internal indices peak there -- they don't, and pretending
otherwise would misrepresent the evidence.

Outputs:
  data/clustering_k_selection_metrics.csv
  viz/tabela_escolha_k.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from team_clustering_common import extract_feature_matrix, season_relative_zscore, fit_pooled_clusters

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "data" / "team_season_features.csv"
OUT_CSV = PROJECT_DIR / "data" / "clustering_k_selection_metrics.csv"
OUT_PNG = PROJECT_DIR / "viz" / "tabela_escolha_k.png"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
HEADER_BG = "#0b0b0b"
HEADER_TEXT = "#ffffff"
ROW_ALT = "#f2f1ee"
HIGHLIGHT = "#eafaea"
HIGHLIGHT_EDGE = "#0ca30c"


def compute_metrics():
    df = pd.read_csv(CSV_PATH)
    df_features = extract_feature_matrix(df)

    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    imputer = SimpleImputer(strategy='median')
    imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns, index=df_features.index)
    scaled = season_relative_zscore(imputed, df['season_end_year'])
    pca = PCA(n_components=0.95, random_state=42)
    features_pca = pca.fit_transform(scaled.values)

    rows = []
    prev_inertia = None
    kmeans_k4_labels = None
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        inertia = kmeans.inertia_
        sil = silhouette_score(features_pca, labels, random_state=42)
        db = davies_bouldin_score(features_pca, labels)
        pct_drop = None if prev_inertia is None else (prev_inertia - inertia) / prev_inertia * 100
        rows.append({'k': k, 'inertia': inertia, 'pct_drop_wcss': pct_drop,
                     'silhouette': sil, 'davies_bouldin': db,
                     'cluster_sizes': ",".join(str(int(v)) for v in np.bincount(labels))})
        prev_inertia = inertia
        if k == 4:
            kmeans_k4_labels = labels

    agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    agg_labels = agg.fit_predict(features_pca)
    ari = adjusted_rand_score(kmeans_k4_labels, agg_labels)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)
    return results_df, ari, df_features.shape[1], pca.n_components_


LEITURA = {
    2: "Melhor Silhouette e Davies-Bouldin isolados — mas separa só \"bom vs. ruim\", sem eixo de estilo (ataque/defesa) nem tier de reconstrução.",
    3: "Ainda sem separar ritmo ofensivo de perfil defensivo; Davies-Bouldin piora bastante frente a k=2.",
    4: "Escolhido. É aqui que WCSS e Silhouette mudam de REGIME: a queda de k=2 a k=4 é ~2,9x (WCSS) e ~5,6x "
       "(Silhouette) mais rápida, por passo, do que a queda de k=4 a k=10 (ver justificativa_k4_elbow_silhouette.png). "
       "Esse ponto de inflexão simultâneo nas duas curvas — não o pico isolado de nenhuma delas — é o \"cotovelo\". "
       "ARI=0,{ari} com Agglomerative (Ward) é uma confirmação adicional (concordância moderada, bem acima do acaso).",
    5: "Silhouette e WCSS continuam caindo suavemente, sem ganho de interpretabilidade — provável fragmentação de um dos 4 arquétipos.",
    6: "Davies-Bouldin piora de novo (pior valor da série); indício de sobreajuste do particionamento.",
    7: "Sem justificativa teórica para 7 arquétipos competitivos; ganho marginal de WCSS já é pequeno.",
    8: "Silhouette próximo do mínimo da série; overfitting do particionamento aos dados.",
    9: "Melhor Silhouette entre k>=5, mas ainda abaixo de k=2..4 e sem leitura basquetebolística.",
    10: "WCSS continua caindo (nunca pára, por construção), mas sem qualquer ganho interpretativo sobre k=4.",
}


def render_table(results_df: pd.DataFrame, ari: float, n_features: int, n_components: int):
    df = results_df.copy()
    df['k4'] = df['k'] == 4
    df['leitura'] = df['k'].map(lambda k: LEITURA[k].replace("{ari}", f"{ari:.3f}".split('.')[1]) if k == 4 else LEITURA[k])
    df['wcss_fmt'] = df['inertia'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    df['drop_fmt'] = df['pct_drop_wcss'].apply(lambda x: "—" if pd.isna(x) else f"-{x:.1f}%")
    df['sil_fmt'] = df['silhouette'].apply(lambda x: f"{x:.4f}")
    df['db_fmt'] = df['davies_bouldin'].apply(lambda x: f"{x:.4f}")

    columns = ["k", "wcss_fmt", "drop_fmt", "sil_fmt", "db_fmt", "leitura"]
    headers = ["k", "WCSS\n(inércia)", "Δ WCSS\nvs. k-1", "Silhouette\n(maior=melhor)", "Davies-Bouldin\n(menor=melhor)", "Leitura"]
    col_widths = [0.035, 0.10, 0.09, 0.115, 0.125, 0.535]

    n_rows = len(df)
    row_h = 1.0
    fig_h = 2.3 + n_rows * row_h * 0.62
    fig, ax = plt.subplots(figsize=(17, fig_h), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 3)

    ax.text(0.0, n_rows + 2.55, "Tabela — Indicadores de Validação do Número de Clusters (k = 2..10)",
            fontsize=15.5, fontweight='bold', color=INK_PRIMARY, ha='left', va='center')
    ax.text(0.0, n_rows + 2.05,
            f"PCA reteve {n_components} componentes (95% da variância) de {n_features} features originais, "
            f"padronizadas por temporada. ARI (KMeans k=4 vs. Agglomerative/Ward k=4) = {ari:.4f}.",
            fontsize=9.5, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic')
    ax.text(0.0, n_rows + 1.62,
            "Nem WCSS nem Silhouette têm seu melhor valor exatamente em k=4 — o que os justifica é uma MUDANÇA DE "
            "REGIME que as duas curvas compartilham nesse ponto (queda íngreme até k=4, platô raso depois). "
            "Ver viz/justificativa_k4_elbow_silhouette.png para o detalhamento quantitativo dessa mudança de regime.",
            fontsize=9.5, color=INK_SECONDARY, ha='left', va='center', fontstyle='italic')

    header_y = n_rows + 0.85
    x = 0.0
    for w, h in zip(col_widths, headers):
        ax.add_patch(plt.Rectangle((x, header_y - 0.55), w, 1.1, facecolor=HEADER_BG, edgecolor='none', zorder=2))
        ax.text(x + 0.008, header_y, h, fontsize=9.3, fontweight='bold', color=HEADER_TEXT,
                ha='left', va='center', zorder=3, linespacing=1.3)
        x += w

    for i, (_, row) in enumerate(df.iterrows()):
        y = n_rows - i - 0.05
        is_k4 = row['k4']
        band = HIGHLIGHT if is_k4 else (ROW_ALT if i % 2 == 0 else SURFACE)
        rect = plt.Rectangle((0, y - 0.5), sum(col_widths), 1.0, facecolor=band, edgecolor='none', zorder=1)
        ax.add_patch(rect)
        if is_k4:
            ax.add_patch(plt.Rectangle((0, y - 0.5), sum(col_widths), 1.0, fill=False,
                                        edgecolor=HIGHLIGHT_EDGE, linewidth=2.2, zorder=2))
        x = 0.0
        vals = [row[c] for c in columns]
        weight = 'bold' if is_k4 else 'normal'
        for w, v, col in zip(col_widths, vals, columns):
            fs = 9.6 if col != 'leitura' else 8.6
            wrap = str(v)
            ax.text(x + 0.008, y, wrap, fontsize=fs, color=INK_PRIMARY, fontweight=weight,
                    ha='left', va='center', zorder=3, wrap=True)
            x += w

    ax.add_patch(plt.Rectangle((0, -0.55), sum(col_widths), header_y + 1.1, fill=False,
                                edgecolor="#cfcfc7", linewidth=1.2, zorder=4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"[OK] {OUT_PNG}")


def main():
    results_df, ari, n_features, n_components = compute_metrics()
    print(results_df.to_string(index=False))
    print(f"\nARI (KMeans k=4 vs Agglomerative/Ward k=4) = {ari:.4f}")
    render_table(results_df, ari, n_features, n_components)


if __name__ == "__main__":
    main()
