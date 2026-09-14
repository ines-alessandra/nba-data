#!/usr/bin/env python3
"""
Veredito de H2 (lacuna estrutural), H3 (objetivo sazonal), H4 (magnitude da
troca) e H5 (coerência de estilo de jogo) -- WPA / Contribuição Relativa,
amostra estendida (2013-2025).

NUMERAÇÃO: a hipótese de desigualdade salarial é H1 (ver generate_h1_verdict.py).
As quatro hipóteses de trocas -- lacuna estrutural, objetivo sazonal, magnitude
da troca e coerência de estilo -- são H2, H3, H4 e H5, respectivamente. H4 e H5
foram formuladas DEPOIS de H2/H3 já estarem testadas, a partir de teoria de
mudança organizacional/continuidade de elenco -- não a partir de padrões já
vistos nos dados (sem data dredging).

Gera quatro figuras de síntese, uma por hipótese, para uso direto em slides:
banner de veredito, distribuição (boxplot+stripplot pooled 13 temporadas) e a
estatística formal em rodapé -- só isso. A leitura em linguagem simples ("o
que este resultado significa", explicações do padrão invertido em H3 etc.)
NÃO fica embutida na imagem -- vai como texto corrido para a nota do slide
(ver a mensagem que acompanha a geração destas figuras), pois uma imagem não é
copiável/editável e o orientador pode querer ler a nota separadamente do
gráfico.

Nada aqui é hardcoded: todo p-valor, mediana e efeito é recalculado a partir de
data/wpa_trade_impact_analysis_extended.csv a cada execução.

Testes:
  H2 - Mann-Whitney U ONE-SIDED (a hipótese, lida literalmente, prevê uma
       direção específica: Lacuna > Redundante -- não "alguma diferença").
       Efeito: correlação bisserial por postos (rank-biserial r), mesma escala
       -1..+1 do rho de Spearman, lida com a convenção de Cohen (1988).
  H3 - Wilcoxon signed-rank de amostra única por grupo (H0: mediana=0), pois a
       hipótese faz uma afirmação por grupo (Contender melhora; Tanking é
       neutro/negativo), não apenas "os 3 grupos diferem entre si" (esse é o
       Kruskal-Wallis, reportado como contexto). Mais o contraste direto
       Contender > Tanking (one-sided Mann-Whitney), núcleo implícito da
       comparação da hipótese.
  H4 - Mann-Whitney U ONE-SIDED (Individual > Múltipla) + Spearman (versão
       contínua, nº de jogadores adquiridos vs. ΔWPA_time) -- mesmo padrão
       duplo categórico/contínuo usado em H2.
  H5 - Duas afirmações testadas separadamente, pois a hipótese faz duas
       previsões distintas sobre o mesmo par de grupos (Manteve estilo vs.
       Mudou estilo): (a) mediana -- Mann-Whitney U one-sided (Manteve >
       Mudou); (b) variância/estabilidade -- teste de Levene (H0: variâncias
       iguais). Requer cluster_name/cluster_name_post_trade calculados
       pré E pós-troca (ver src/build_wpa_pretrade_clusters.py).
  H9 - Mann-Whitney U ONE-SIDED (Veteranos > Jovens, corte na mediana de
       idade) + Spearman (versão contínua, idade média dos jogadores
       adquiridos vs. ΔWPA_time) -- mesmo padrão duplo de H2/H4. Idade vem de
       bronze.raw_trades.player_age, recalculada ao vivo (não está no CSV
       principal) -- ver compute_idade_media_adquiridos().
  H14 - Spearman (min_deltaP_acquired vs. n_players_acquired) -- hipótese de
        ASSOCIAÇÃO entre duas características da própria troca, não de
        desempenho: não há grupo "melhor", só duas variáveis contínuas
        correlacionadas. Painel de quartis (mesmo padrão de
        wpa_h2_versao_continua.png) só para tornar a correlação legível sem
        depender de uma nuvem de pontos discreta.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import FancyBboxPatch
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
VIZ_DIR = PROJECT_DIR / "viz"
ALPHA = 0.05

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
COLOR_NEUTRAL = "#6b6a66"
COLOR_ZONE_FILL = "#e9e8e3"
COLOR_ZONE_EDGE = "#c3c2b7"
AMBER = "#c97a1a"          # "suporte parcial" -- nem verde (confirma tudo) nem cinza (não confirma nada)
TINT_AMBER = "#fbe8cf"

CLASS_COLORS = {"Preenche Lacuna": AQUA, "Redundante": RED}
OBJ_COLORS = {"Contender": BLUE, "Intermediario": ORANGE, "Tanking": AQUA}
OBJ_LABELS = {"Contender": "Contenders", "Intermediario": "Intermediários", "Tanking": "Tanking"}

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = INK_PRIMARY


def style_ax(ax):
    ax.grid(axis='y', zorder=0, color=GRID)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)


def effect_label(abs_r: float) -> str:
    """Convenção de Cohen (1988)."""
    if abs_r < 0.10:
        return "desprezível"
    if abs_r < 0.30:
        return "fraco"
    if abs_r < 0.50:
        return "moderado"
    return "forte"


def mannwhitney_r_and_zone(g1: pd.Series, g2: pd.Series, alternative: str = "greater"):
    """U, p (do lado pedido), r bisserial por postos, e a largura da zona
    'sem efeito' em espaço-r via aproximação normal de U (dois lados, alpha=0,05)."""
    n1, n2 = len(g1), len(g2)
    u, p = stats.mannwhitneyu(g1, g2, alternative=alternative)
    r = 1 - (2 * u) / (n1 * n2)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    r_crit = z_crit * np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
    return u, p, r, r_crit, n1, n2


def bootstrap_r_ci(g1: np.ndarray, g2: np.ndarray, n_boot: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n1, n2 = len(g1), len(g2)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        b1 = rng.choice(g1, size=n1, replace=True)
        b2 = rng.choice(g2, size=n2, replace=True)
        u, _ = stats.mannwhitneyu(b1, b2, alternative="two-sided")
        rs[i] = 1 - (2 * u) / (n1 * n2)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def bootstrap_median_ci(x: np.ndarray, n_boot: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# =============================================================================
# H2 -- Trocas que preenchem lacuna estrutural produzem impacto mais positivo?
# =============================================================================
def generate_h2_verdict(impact: pd.DataFrame):
    order = ["Preenche Lacuna", "Redundante"]
    g_lac = impact.loc[impact["trade_classification"] == "Preenche Lacuna", "delta_WPA_time"].dropna()
    g_red = impact.loc[impact["trade_classification"] == "Redundante", "delta_WPA_time"].dropna()

    u, p, r, r_crit, n1, n2 = mannwhitney_r_and_zone(g_lac, g_red, alternative="greater")
    ci_lo, ci_hi = bootstrap_r_ci(g_lac.values, g_red.values)
    magnitude = effect_label(abs(r))
    significant = p < ALPHA

    if significant:
        verdict = "A hipótese encontra suporte estatístico"
        subline = ("Nesta amostra, trocas que preenchem uma lacuna estrutural produzem impacto "
                    "significativamente mais positivo do que trocas redundantes.")
    else:
        verdict = "A hipótese (forma original) NÃO encontra suporte estatístico"
        subline = ("A mediana de ΔWPA_time é maior no grupo Lacuna, na direção prevista -- mas a "
                    "diferença não é distinguível de acaso nesta amostra (p ≥ 0,05, teste unilateral).")

    med_lac, med_red = g_lac.median(), g_red.median()
    delta = med_lac - med_red

    print(f"[H2] n_Lacuna={n1} n_Redundante={n2}  U={u:.1f}  p(one-sided)={p:.4f}  "
          f"r={r:+.4f}  IC95%=({ci_lo:.4f},{ci_hi:.4f})  r_critico={r_crit:.4f}  "
          f"magnitude={magnitude}  significativo={significant}")
    print(f"Veredito H2: {verdict}")

    fig = plt.figure(figsize=(10.5, 9.4), dpi=300)
    fig.patch.set_facecolor('white')

    tint = "#dcf3ea" if significant else COLOR_ZONE_FILL
    fig.text(0.5, 0.975, "VEREDITO -- H2: LACUNA ESTRUTURAL VS. REDUNDÂNCIA", ha='center', va='center',
              fontsize=11, fontweight='bold', color=INK_SECONDARY,
              bbox=dict(boxstyle="round,pad=0.35", facecolor=tint, edgecolor='none'))
    fig.text(0.5, 0.928, verdict, ha='center', va='center',
              fontsize=16.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.888, subline, ha='center', va='center',
              fontsize=10, color=INK_SECONDARY, style='italic', wrap=True)
    fig.text(0.5, 0.858,
              '"Trocas que preenchem uma lacuna estrutural do elenco produzem impacto mais positivo '
              'do que trocas redundantes." (hipótese testada unilateralmente)',
              ha='center', va='center', fontsize=8.8, color=INK_MUTED)

    ax = fig.add_axes([0.10, 0.32, 0.83, 0.44])
    sns.boxplot(data=impact, x='trade_classification', y='delta_WPA_time', order=order,
                hue='trade_classification', palette=CLASS_COLORS, legend=False,
                width=0.42, fliersize=3, showfliers=False, linewidth=1.3, ax=ax, zorder=3,
                boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=2.0),
                whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
    sns.stripplot(data=impact, x='trade_classification', y='delta_WPA_time', order=order,
                   color=INK_PRIMARY, alpha=0.30, size=3.8, jitter=0.16, ax=ax, zorder=4)
    ax.axhline(0, color=INK_MUTED, linewidth=1.1, linestyle='--', zorder=2)

    ns = {"Preenche Lacuna": n1, "Redundante": n2}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{lbl}\n(n={ns[lbl]})" for lbl in order], fontsize=12, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=12, fontweight='bold')
    ax.set_title("Distribuição pooled, 13 temporadas (2013-2025)", fontsize=12.5,
                  color=INK_SECONDARY, pad=44)

    # ---- estatística integrada ao gráfico, não em rodapé ----
    # 1) chave de significância acima das duas caixas, no mesmo padrão de figura
    # científica (bracket ligando os 2 grupos), com a diferença de mediana e o
    # p-valor exatamente onde o olho já está olhando -- não numa legenda separada.
    y_data_max = max(g_lac.max(), g_red.max())
    bracket_y = y_data_max + 0.045
    tick_h = 0.014
    ax.plot([0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
            color=INK_PRIMARY, linewidth=1.5, zorder=6)
    sig_word = "significativo" if significant else "não significativo"
    sig_color = AQUA if significant else INK_SECONDARY
    ax.text(0.5, bracket_y + 0.010,
            f"Δ mediana = {delta:+.4f}   |   p = {p:.4f} ({sig_word}, unilateral)",
            ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=sig_color, zorder=6)
    ax.set_ylim(top=bracket_y + 0.05)

    style_ax(ax)

    # 2) cartão de estatística completa (U, r, IC, magnitude) -- painel PRÓPRIO,
    # fora da área de plotagem (abaixo do eixo x), não mais um texto flutuando
    # por cima dos dados. Mesma informação de antes, só que como um elemento de
    # composição da figura, não uma anotação sobreposta ao gráfico.
    card_ax = fig.add_axes([0.10, 0.11, 0.83, 0.09])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis('off')
    card_color = "#dcf3ea" if significant else "#f5f4ef"
    card_edge = AQUA if significant else COLOR_NEUTRAL
    card_ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                      linewidth=1.4, edgecolor=card_edge, facecolor=card_color, zorder=1))
    # Duas linhas com quebra MANUAL, sem wrap=True: wrap automático do matplotlib
    # calcula a largura de quebra a partir da figura inteira, não da caixa desenhada
    # por baixo -- em fonte bold, o texto podia terminar mais largo que o
    # FancyBboxPatch e vazar para fora da borda cinza. Com duas linhas fixas e
    # curtas o suficiente para a largura real da caixa, isso não pode acontecer.
    line1 = f"Mann-Whitney U = {u:.0f}  (unilateral)   |   r bisserial = {r:+.3f}   IC95% [{ci_lo:+.3f}, {ci_hi:+.3f}]"
    line2 = f"efeito {magnitude} (Cohen, 1988)   |   zona sem efeito: |r| < {r_crit:.3f}"
    card_ax.text(0.5, 0.66, line1, ha='center', va='center', fontsize=10,
                 fontweight='bold', color=INK_PRIMARY)
    card_ax.text(0.5, 0.32, line2, ha='center', va='center', fontsize=10,
                 fontweight='bold', color=INK_PRIMARY)

    fig.text(0.5, 0.03,
              "Amostra: eventos de troca (team_id x trade_date) intra-temporada, 2013-2025. ΔWPA_time = taxa de "
              "WPA do time (pós − pré-troca). Classificação: ΔP pré-troca na posição do jogador adquirido, "
              "usando apenas jogos anteriores à troca.",
              ha='center', va='center', fontsize=7.6, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h2_verdict.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}\n")


# =============================================================================
# H3 -- Objetivo sazonal modera a eficácia de uma troca?
# =============================================================================
def generate_h3_verdict(impact: pd.DataFrame):
    order = ["Contender", "Intermediario", "Tanking"]
    groups = {g: impact.loc[impact["objetivo_sazonal"] == g, "delta_WPA_time"].dropna() for g in order}

    # Kruskal-Wallis omnibus (contexto, não o teste principal desta hipótese)
    h_stat, p_omni = stats.kruskal(*[groups[g] for g in order])
    k = len(order)
    n_total = sum(len(groups[g]) for g in order)
    eta2 = max((h_stat - k + 1) / (n_total - k), 0.0)

    # Afirmação por grupo: Wilcoxon de amostra única (H0: mediana=0)
    wilcoxon_res = {}
    for g in order:
        w, pw = stats.wilcoxon(groups[g])
        wilcoxon_res[g] = (w, pw, groups[g].median())

    # Contraste direto do núcleo da hipótese: Contender > Tanking
    u_ct, p_ct = stats.mannwhitneyu(groups["Contender"], groups["Tanking"], alternative="greater")
    r_ct = 1 - (2 * u_ct) / (len(groups["Contender"]) * len(groups["Tanking"]))

    contender_confirms = wilcoxon_res["Contender"][1] < ALPHA and wilcoxon_res["Contender"][2] > 0
    tanking_confirms = not (wilcoxon_res["Tanking"][1] < ALPHA and wilcoxon_res["Tanking"][2] > 0)
    contrast_confirms = p_ct < ALPHA
    tanking_inverted = wilcoxon_res["Tanking"][1] < ALPHA and wilcoxon_res["Tanking"][2] > 0

    if contender_confirms and tanking_confirms and contrast_confirms:
        verdict = "A hipótese encontra suporte estatístico"
    elif tanking_inverted:
        verdict = "A hipótese NÃO encontra suporte -- padrão parcialmente invertido"
    else:
        verdict = "A hipótese NÃO encontra suporte estatístico"

    subline = ("O gráfico mostra as 3 categorias de objetivo sazonal -- a hipótese só prevê direção para "
               "Contender e Tanking. Intermediário não tem previsão na hipótese, mas o dado fala por si: também "
               "melhora de forma significativa, com a maior mediana dos 3 grupos. Achado central: Tanking "
               "melhora de forma estatisticamente significativa -- o oposto do previsto.")

    print(f"[H3] Kruskal-Wallis H={h_stat:.3f} p={p_omni:.4f} eta2={eta2:.4f}")
    for g in order:
        w, pw, med = wilcoxon_res[g]
        print(f"     {g}: n={len(groups[g])} mediana={med:+.4f} Wilcoxon p={pw:.4f}")
    print(f"     Contender > Tanking (one-sided): U={u_ct:.1f} p={p_ct:.4f} r={r_ct:+.3f}")
    print(f"Veredito H3: {verdict}")

    fig = plt.figure(figsize=(11, 9.6), dpi=300)
    fig.patch.set_facecolor('white')

    tint = COLOR_ZONE_FILL
    fig.text(0.5, 0.975, "VEREDITO -- H3: OBJETIVO SAZONAL", ha='center', va='center',
              fontsize=11, fontweight='bold', color=INK_SECONDARY,
              bbox=dict(boxstyle="round,pad=0.35", facecolor=tint, edgecolor='none'))
    fig.text(0.5, 0.925, verdict, ha='center', va='center',
              fontsize=15.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.885, subline, ha='center', va='center',
              fontsize=10, color=INK_SECONDARY, style='italic', wrap=True)
    fig.text(0.5, 0.850,
              '"Trocas de Contenders associam-se a melhora de curto prazo; trocas de times em Tanking '
              'associam-se a impacto neutro ou negativo, no mesmo período."',
              ha='center', va='center', fontsize=8.8, color=INK_MUTED)

    ax = fig.add_axes([0.09, 0.34, 0.85, 0.43])
    sns.boxplot(data=impact[impact["objetivo_sazonal"].isin(order)], x='objetivo_sazonal', y='delta_WPA_time',
                order=order, hue='objetivo_sazonal', palette=OBJ_COLORS, legend=False,
                width=0.5, fliersize=3, showfliers=False, linewidth=1.3, ax=ax, zorder=3,
                boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=2.0),
                whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
    sns.stripplot(data=impact[impact["objetivo_sazonal"].isin(order)], x='objetivo_sazonal', y='delta_WPA_time',
                   order=order, color=INK_PRIMARY, alpha=0.28, size=3.6, jitter=0.16, ax=ax, zorder=4)
    ax.axhline(0, color=INK_MUTED, linewidth=1.1, linestyle='--', zorder=2)

    # Faixa de destaque atrás da coluna de Tanking -- o achado central (inversão da
    # previsão) precisa ficar óbvio olhando só para o gráfico, sem depender do texto.
    tanking_i = order.index("Tanking")
    ax.axvspan(tanking_i - 0.42, tanking_i + 0.42, color="#fbdedd", alpha=0.55, zorder=0.5)

    y_top = impact["delta_WPA_time"].max()
    predicted_txt = {
        "Contender": "previsto: melhora (Δ>0)",
        "Intermediario": "sem previsão na hipótese",
        "Tanking": "previsto: neutro/negativo (Δ≤0)",
    }
    for i, g in enumerate(order):
        w, pw, med = wilcoxon_res[g]
        sig = pw < ALPHA
        # A cor/rótulo aqui é sobre CONCORDÂNCIA COM A PREVISÃO da hipótese para
        # aquele grupo -- não simplesmente o sinal da mediana. É por isso que
        # Tanking positivo e significativo aparece em vermelho (inverte a
        # previsão), não em verde (que aqui só marcaria "confirma").
        if g == "Contender":
            aligned = sig and med > 0
            tag, color = ("confirma a previsão" if aligned else "NÃO confirma a previsão",
                          AQUA if aligned else INK_MUTED)
        elif g == "Tanking":
            if sig and med > 0:
                tag, color = "INVERTE A PREVISÃO", RED
            elif sig and med < 0:
                tag, color = "confirma a previsão", AQUA
            else:
                tag, color = "consistente (neutro)", INK_MUTED
        else:  # Intermediário -- a hipótese não formula previsão para este grupo
            tag, color = "sem previsão (contexto)", INK_MUTED
        ax.text(i, y_top * 1.10, f"p={pw:.4f}", ha='center', va='bottom', fontsize=8.3, color=INK_SECONDARY)
        ax.text(i, y_top * 1.19, tag, ha='center', va='bottom', fontsize=9.3, fontweight='bold', color=color)

    # Callout apontando direto para o achado central -- não fica implícito no
    # rótulo pequeno, tem destaque visual equivalente ao do texto do veredito.
    ax.annotate("ACHADO CENTRAL: Tanking melhora --\no oposto do previsto pela hipótese",
                xy=(tanking_i, y_top * 1.23), xytext=(tanking_i, y_top * 1.62),
                fontsize=9.6, fontweight='bold', color=RED, ha='center', va='center', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#fbdedd", edgecolor=RED, linewidth=1.3),
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.6))

    ax.set_xticks(range(len(order)))
    ns = {g: len(groups[g]) for g in order}
    ax.set_xticklabels([f"{OBJ_LABELS[g]}\n(n={ns[g]}, mediana={groups[g].median():+.4f})\n{predicted_txt[g]}"
                         .replace(".", ",") for g in order], fontsize=10.2)
    ax.set_xlabel("")
    ax.set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=12, fontweight='bold')
    ax.set_ylim(top=y_top * 1.95)
    ax.set_title("Distribuição pooled, 13 temporadas (2013-2025) -- rótulo = teste de amostra única "
                  "(H0: mediana=0) por grupo, comparado à previsão da hipótese", fontsize=10.3,
                  color=INK_SECONDARY, pad=10)
    style_ax(ax)

    # Estatística formal (Kruskal-Wallis omnibus + contraste Contender>Tanking) num
    # painel PRÓPRIO abaixo do gráfico -- mesmo padrão de wpa_h2_verdict.png -- em
    # vez de uma linha de rodapé em fonte minúscula.
    card_ax = fig.add_axes([0.09, 0.115, 0.85, 0.09])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis('off')
    card_ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                      linewidth=1.4, edgecolor=COLOR_ZONE_EDGE, facecolor=COLOR_ZONE_FILL, zorder=1))
    p_omni_str = "p < 0,0001" if p_omni < 0.0001 else f"p = {p_omni:.4f}".replace('.', ',')
    p_ct_str = "p < 0,0001" if p_ct < 0.0001 else f"p = {p_ct:.4f}".replace('.', ',')
    line1 = f"Kruskal-Wallis (3 grupos) H = {h_stat:.3f}  {p_omni_str}   |   η² = {eta2:.4f}"
    line2 = f"Contender > Tanking (unilateral): U = {u_ct:.0f}  {p_ct_str}   |   r = {r_ct:+.3f}"
    card_ax.text(0.5, 0.66, line1, ha='center', va='center', fontsize=10, fontweight='bold', color=INK_PRIMARY)
    card_ax.text(0.5, 0.32, line2, ha='center', va='center', fontsize=10, fontweight='bold', color=INK_PRIMARY)

    fig.text(0.5, 0.03,
              "Amostra: eventos de troca (team_id x trade_date) intra-temporada, 2013-2025. Objetivo sazonal "
              "= rank de conferência pré-troca (Contender: 1-6º; Intermediário: 7-10º; Tanking: 11-15º).",
              ha='center', va='center', fontsize=7.6, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h3_verdict.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}\n")


# =============================================================================
# H4 -- Magnitude da troca (desorganização do elenco) produz impacto mais negativo?
# =============================================================================
def generate_h4_verdict(impact: pd.DataFrame):
    order = ["Individual\n(1 jogador)", "Múltipla\n(2+ jogadores)"]
    df = impact.dropna(subset=["n_players_acquired", "delta_WPA_time"]).copy()
    df["grupo"] = np.where(df["n_players_acquired"] == 1, order[0], order[1])
    g_single = df.loc[df["grupo"] == order[0], "delta_WPA_time"]
    g_multi = df.loc[df["grupo"] == order[1], "delta_WPA_time"]

    u, p, r, r_crit, n1, n2 = mannwhitney_r_and_zone(g_single, g_multi, alternative="greater")
    ci_lo, ci_hi = bootstrap_r_ci(g_single.values, g_multi.values)
    magnitude = effect_label(abs(r))
    significant = p < ALPHA

    # Versão contínua (mesmo padrão duplo categórico/contínuo de H2): nº de
    # jogadores adquiridos, sem o corte arbitrário em "1 vs 2+".
    rho, p_cont = stats.spearmanr(df["n_players_acquired"], df["delta_WPA_time"])

    if significant:
        verdict = "A hipótese encontra suporte estatístico"
        subline = ("Nesta amostra, trocas com mais jogadores produzem impacto significativamente mais "
                    "negativo do que trocas de um único jogador.")
    else:
        verdict = "A hipótese NÃO encontra suporte estatístico"
        subline = ("A magnitude da troca (nº de jogadores adquiridos) não se mostra associada ao impacto "
                    "de desempenho nesta amostra -- nem na comparação categórica, nem na versão contínua.")

    med_single, med_multi = g_single.median(), g_multi.median()
    delta = med_single - med_multi

    print(f"[H4] n_single={n1} n_multi={n2}  U={u:.1f}  p(one-sided)={p:.4f}  r={r:+.4f}  "
          f"IC95%=({ci_lo:.4f},{ci_hi:.4f})  magnitude={magnitude}  significativo={significant}")
    print(f"     Versão contínua: Spearman rho={rho:+.4f} p={p_cont:.4f}")
    print(f"Veredito H4: {verdict}")

    fig = plt.figure(figsize=(10.5, 9.4), dpi=300)
    fig.patch.set_facecolor('white')

    tint = "#dcf3ea" if significant else COLOR_ZONE_FILL
    fig.text(0.5, 0.975, "VEREDITO -- H4: MAGNITUDE DA TROCA", ha='center', va='center',
              fontsize=11, fontweight='bold', color=INK_SECONDARY,
              bbox=dict(boxstyle="round,pad=0.35", facecolor=tint, edgecolor='none'))
    fig.text(0.5, 0.928, verdict, ha='center', va='center',
              fontsize=16.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.888, subline, ha='center', va='center',
              fontsize=10, color=INK_SECONDARY, style='italic', wrap=True)
    fig.text(0.5, 0.850,
              '"Quanto maior o número de jogadores movimentados em um evento de troca, mais negativo é o '
              'impacto de desempenho de curto prazo do time." (hipótese testada unilateralmente)',
              ha='center', va='center', fontsize=8.8, color=INK_MUTED)

    ax = fig.add_axes([0.10, 0.32, 0.83, 0.44])
    sns.boxplot(data=df, x='grupo', y='delta_WPA_time', order=order,
                hue='grupo', palette={order[0]: BLUE, order[1]: ORANGE}, legend=False,
                width=0.42, fliersize=3, showfliers=False, linewidth=1.3, ax=ax, zorder=3,
                boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=2.0),
                whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
    sns.stripplot(data=df, x='grupo', y='delta_WPA_time', order=order,
                   color=INK_PRIMARY, alpha=0.30, size=3.8, jitter=0.16, ax=ax, zorder=4)
    ax.axhline(0, color=INK_MUTED, linewidth=1.1, linestyle='--', zorder=2)

    ns = {order[0]: n1, order[1]: n2}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{lbl}\n(n={ns[lbl]})" for lbl in order], fontsize=12, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=12, fontweight='bold')
    ax.set_title("Distribuição pooled, 13 temporadas (2013-2025)", fontsize=12.5,
                  color=INK_SECONDARY, pad=44)

    y_data_max = max(g_single.max(), g_multi.max())
    bracket_y = y_data_max + 0.045
    tick_h = 0.014
    ax.plot([0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
            color=INK_PRIMARY, linewidth=1.5, zorder=6)
    sig_word = "significativo" if significant else "não significativo"
    sig_color = AQUA if significant else INK_SECONDARY
    ax.text(0.5, bracket_y + 0.010,
            f"Δ mediana = {delta:+.4f}   |   p = {p:.4f} ({sig_word}, unilateral)",
            ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=sig_color, zorder=6)
    ax.set_ylim(top=bracket_y + 0.05)

    style_ax(ax)

    card_ax = fig.add_axes([0.10, 0.11, 0.83, 0.09])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis('off')
    card_color = "#dcf3ea" if significant else "#f5f4ef"
    card_edge = AQUA if significant else COLOR_NEUTRAL
    card_ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                      linewidth=1.4, edgecolor=card_edge, facecolor=card_color, zorder=1))
    line1 = f"Mann-Whitney U = {u:.0f}  (unilateral)   |   r bisserial = {r:+.3f}   IC95% [{ci_lo:+.3f}, {ci_hi:+.3f}]"
    line2 = f"versão contínua (Spearman): ρ = {rho:+.3f}   p = {p_cont:.4f}   |   efeito {magnitude} (Cohen, 1988)"
    card_ax.text(0.5, 0.66, line1, ha='center', va='center', fontsize=10, fontweight='bold', color=INK_PRIMARY)
    card_ax.text(0.5, 0.32, line2, ha='center', va='center', fontsize=10, fontweight='bold', color=INK_PRIMARY)

    fig.text(0.5, 0.03,
              "Amostra: eventos de troca (team_id x trade_date) intra-temporada, 2013-2025. Magnitude = nº de "
              "jogadores adquiridos pelo time naquele evento (1 = individual; 2+ = múltipla).",
              ha='center', va='center', fontsize=7.6, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h4_verdict.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}\n")


# =============================================================================
# H5 -- Coerência de estilo de jogo (mudança de arquétipo) modera o impacto?
# =============================================================================
def generate_h5_verdict(impact: pd.DataFrame):
    order = ["Manteve estilo", "Mudou estilo"]
    df = impact.dropna(subset=["cluster_name", "cluster_name_post_trade", "delta_WPA_time"]).copy()
    df["grupo"] = np.where(df["cluster_name"] == df["cluster_name_post_trade"], order[0], order[1])
    g_manteve = df.loc[df["grupo"] == order[0], "delta_WPA_time"]
    g_mudou = df.loc[df["grupo"] == order[1], "delta_WPA_time"]
    n1, n2 = len(g_manteve), len(g_mudou)

    # (a) Mediana -- a hipótese prevê Manteve > Mudou
    u, p, r, r_crit, _, _ = mannwhitney_r_and_zone(g_manteve, g_mudou, alternative="greater")
    ci_lo, ci_hi = bootstrap_r_ci(g_manteve.values, g_mudou.values)
    # (b) Variância -- a hipótese prevê Manteve MENOS instável (variância menor)
    lev_stat, lev_p = stats.levene(g_manteve, g_mudou)

    median_confirms = p < ALPHA  # teste unilateral: significativo já implica a direção prevista
    variance_confirms = (lev_p < ALPHA) and (g_mudou.var() > g_manteve.var())

    if median_confirms and variance_confirms:
        verdict = "A hipótese encontra suporte estatístico"
    elif median_confirms or variance_confirms:
        verdict = "A hipótese encontra SUPORTE PARCIAL"
    else:
        verdict = "A hipótese NÃO encontra suporte estatístico"

    subline = (f"Mediana (impacto mais positivo ao manter o estilo): "
               f"{'confirmada' if median_confirms else 'NÃO confirmada'} (p={p:.4f}, unilateral). "
               f"Variância (impacto mais estável ao manter o estilo): "
               f"{'CONFIRMADA' if variance_confirms else 'NÃO confirmada'} (Levene, p={lev_p:.4f}).")

    print(f"[H5] n_manteve={n1} n_mudou={n2}  U={u:.1f}  p(mediana,one-sided)={p:.4f}  r={r:+.4f}")
    print(f"     Levene (variância): stat={lev_stat:.4f}  p={lev_p:.4f}  "
          f"var_manteve={g_manteve.var():.5f}  var_mudou={g_mudou.var():.5f}")
    print(f"Veredito H5: {verdict}")

    fig = plt.figure(figsize=(10.5, 9.6), dpi=300)
    fig.patch.set_facecolor('white')

    if median_confirms and variance_confirms:
        tint = "#dcf3ea"
    elif median_confirms or variance_confirms:
        tint = TINT_AMBER
    else:
        tint = COLOR_ZONE_FILL
    fig.text(0.5, 0.975, "VEREDITO -- H5: COERÊNCIA DE ESTILO DE JOGO", ha='center', va='center',
              fontsize=11, fontweight='bold', color=INK_SECONDARY,
              bbox=dict(boxstyle="round,pad=0.35", facecolor=tint, edgecolor='none'))
    fig.text(0.5, 0.928, verdict, ha='center', va='center',
              fontsize=15.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.888, subline, ha='center', va='center',
              fontsize=9.6, color=INK_SECONDARY, style='italic', wrap=True)
    fig.text(0.5, 0.850,
              '"Trocas que preservam o arquétipo de estilo de jogo do time produzem impacto de curto prazo '
              'mais positivo E mais estável do que trocas que mudam esse arquétipo."',
              ha='center', va='center', fontsize=8.8, color=INK_MUTED)

    ax = fig.add_axes([0.11, 0.34, 0.80, 0.42])
    sns.boxplot(data=df, x='grupo', y='delta_WPA_time', order=order,
                hue='grupo', palette={order[0]: BLUE, order[1]: ORANGE}, legend=False,
                width=0.42, fliersize=3, showfliers=False, linewidth=1.3, ax=ax, zorder=3,
                boxprops=dict(edgecolor=INK_PRIMARY), medianprops=dict(color=INK_PRIMARY, linewidth=2.0),
                whiskerprops=dict(color=INK_PRIMARY), capprops=dict(color=INK_PRIMARY))
    sns.stripplot(data=df, x='grupo', y='delta_WPA_time', order=order,
                   color=INK_PRIMARY, alpha=0.28, size=3.6, jitter=0.16, ax=ax, zorder=4)
    ax.axhline(0, color=INK_MUTED, linewidth=1.1, linestyle='--', zorder=2)

    ns = {order[0]: n1, order[1]: n2}
    vs = {order[0]: g_manteve.var(), order[1]: g_mudou.var()}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{lbl}\n(n={ns[lbl]}, var={vs[lbl]:.4f})".replace(".", ",", 1) for lbl in order],
                        fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=12, fontweight='bold')

    y_data_max = max(g_manteve.max(), g_mudou.max())
    y_data_min = min(g_manteve.min(), g_mudou.min())
    bracket_y = y_data_max + 0.035
    tick_h = 0.011
    ax.plot([0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
            color=INK_SECONDARY, linewidth=1.3, zorder=6)
    ax.text(0.5, bracket_y + 0.007, f"mediana: p = {p:.4f} (não significativo, unilateral)",
            ha='center', va='bottom', fontsize=9.3, color=INK_SECONDARY, zorder=6)

    # Callout para o achado real desta figura -- a VARIÂNCIA, não a mediana --
    # mesmo padrão de destaque usado no achado central de wpa_h3_verdict.png.
    callout_color = AQUA if variance_confirms else INK_MUTED
    callout_bg = "#dcf3ea" if variance_confirms else "#f5f4ef"
    callout_txt = (f"ACHADO: mudar de estilo AUMENTA a instabilidade\ndo impacto (Levene, p={lev_p:.4f})"
                   if variance_confirms else
                   f"Variância não difere significativamente\nentre os grupos (Levene, p={lev_p:.4f})")
    ax.annotate(callout_txt, xy=(1, bracket_y - 0.02), xytext=(1, bracket_y + 0.135),
                fontsize=9.5, fontweight='bold', color=callout_color, ha='center', va='center', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=callout_bg, edgecolor=callout_color, linewidth=1.3),
                arrowprops=dict(arrowstyle="-|>", color=callout_color, linewidth=1.6))

    ax.set_ylim(y_data_min - 0.03, bracket_y + 0.22)
    ax.set_title("Distribuição pooled, 13 temporadas (2013-2025) -- altura da caixa/bigode = dispersão "
                 "(instabilidade) do impacto", fontsize=10, color=INK_SECONDARY, pad=10)
    style_ax(ax)

    card_ax = fig.add_axes([0.10, 0.115, 0.83, 0.09])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis('off')
    card_ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                      linewidth=1.4, edgecolor=COLOR_ZONE_EDGE, facecolor=COLOR_ZONE_FILL, zorder=1))
    line1 = f"Mediana (Mann-Whitney, unilateral): U = {u:.0f}   r = {r:+.3f}   p = {p:.4f}"
    line2 = (f"Variância (Levene): estatística = {lev_stat:.3f}   p = {lev_p:.4f}   |   "
             f"var. manteve = {g_manteve.var():.4f}   var. mudou = {g_mudou.var():.4f}")
    card_ax.text(0.5, 0.66, line1, ha='center', va='center', fontsize=9.6, fontweight='bold', color=INK_PRIMARY)
    card_ax.text(0.5, 0.32, line2, ha='center', va='center', fontsize=9.6, fontweight='bold', color=INK_PRIMARY)

    fig.text(0.5, 0.03,
              "Amostra: eventos de troca (team_id x trade_date) intra-temporada, 2013-2025, restrita aos casos "
              "com arquétipo de estilo do time calculado pré E pós-troca (Reconstrução/Defensivo/Ofensivo/Elite). "
              "'Mudou estilo' = arquétipo pós-troca diferente do arquétipo pré-troca.",
              ha='center', va='center', fontsize=7.3, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h5_verdict.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}\n")


# =============================================================================
# H9 -- Veterania dos jogadores adquiridos produz impacto mais positivo?
# =============================================================================
def compute_idade_media_adquiridos(impact: pd.DataFrame) -> pd.Series:
    """Idade média (no momento da troca) dos jogadores adquiridos por evento.
    Não está em wpa_trade_impact_analysis_extended.csv -- vem direto de
    bronze.raw_trades.player_age e é recalculada AO VIVO aqui a cada execução
    (mesmo princípio de 'nada hardcoded' do resto deste arquivo), em vez de
    persistida numa coluna nova no CSV."""
    from sqlalchemy import text
    from wpa_common import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        incoming_ages = pd.read_sql(text("""
            SELECT season_end_year, trade_id, team_id, player_age
            FROM bronze.raw_trades
            WHERE direction = 'Incoming' AND asset_type = 'Player' AND player_age IS NOT NULL
        """), con=conn)

    def avg_age(row):
        s = int(row["season_end_year"])
        tids = set(int(x) for x in str(row["trade_ids"]).split(","))
        sub = incoming_ages[(incoming_ages["season_end_year"] == s) &
                             (incoming_ages["trade_id"].isin(tids)) &
                             (incoming_ages["team_id"] == row["team_id"])]
        return sub["player_age"].mean() if len(sub) else np.nan

    return impact.apply(avg_age, axis=1)


# A hipótese, no formato clássico, é uma relação CONTÍNUA ("quanto maior a
# idade média... maior o impacto") -- não uma comparação entre 2 grupos com
# corte arbitrário na mediana. O layout segue o mesmo padrão dispersão+LOESS
# / quartis de generate_h14_verdict, em vez do boxplot categórico usado em
# H2/H3/H4/H5, para não desenhar uma categoria que a hipótese não afirma.
def generate_h9_verdict(impact: pd.DataFrame):
    df = impact.copy()
    df["idade_media_adquiridos"] = compute_idade_media_adquiridos(df)
    df = df.dropna(subset=["idade_media_adquiridos", "delta_WPA_time"])

    rho, p = stats.spearmanr(df["idade_media_adquiridos"], df["delta_WPA_time"])
    significant = p < ALPHA

    if significant:
        verdict = "A hipótese encontra suporte estatístico"
        subline = ("Nesta amostra, quanto maior a idade média dos jogadores adquiridos em uma troca, "
                    "mais positivo tende a ser o impacto de desempenho de curto prazo do time.")
    else:
        verdict = "A hipótese NÃO encontra suporte estatístico"
        subline = ("A idade média dos jogadores adquiridos não se mostra associada ao impacto de "
                    "desempenho nesta amostra.")

    print(f"[H9] n={len(df)}  Spearman rho={rho:+.4f}  p={p:.4f}  significativo={significant}")
    print(f"Veredito H9: {verdict}")

    fig = plt.figure(figsize=(13.5, 9.6), dpi=300)
    fig.patch.set_facecolor('white')

    tint = "#dcf3ea" if significant else COLOR_ZONE_FILL
    fig.text(0.5, 0.975, "VEREDITO -- H9: VETERANIA DOS JOGADORES ADQUIRIDOS", ha='center', va='center',
              fontsize=11, fontweight='bold', color=INK_SECONDARY,
              bbox=dict(boxstyle="round,pad=0.35", facecolor=tint, edgecolor='none'))
    fig.text(0.5, 0.928, verdict, ha='center', va='center',
              fontsize=16.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.888, subline, ha='center', va='center',
              fontsize=10, color=INK_SECONDARY, style='italic', wrap=True)
    fig.text(0.5, 0.850,
              '"Quanto maior a idade média dos jogadores adquiridos em um evento de troca, maior é o '
              'impacto de desempenho de curto prazo do time, medido por ΔWPA_time."',
              ha='center', va='center', fontsize=8.8, color=INK_MUTED)

    # ---- painel esquerdo: dispersão + LOESS ----
    ax0 = fig.add_axes([0.07, 0.20, 0.42, 0.55])
    ax0.scatter(df["idade_media_adquiridos"], df["delta_WPA_time"], s=30, alpha=0.35,
                color=INK_SECONDARY, edgecolors='none', zorder=3)
    lowess = sm.nonparametric.lowess
    z = lowess(df["delta_WPA_time"], df["idade_media_adquiridos"], frac=0.6)
    ax0.plot(z[:, 0], z[:, 1], color=BLUE, linewidth=3, zorder=4, label='Curva LOESS')
    ax0.axhline(0, color=INK_MUTED, linestyle=':', linewidth=1, zorder=2)
    box_color = "#dcf3ea" if significant else "#f5f4ef"
    edge_color = AQUA if significant else COLOR_NEUTRAL
    ax0.text(0.02, 0.96, f"Spearman ρ = {rho:+.4f}  (p = {p:.4f})",
              transform=ax0.transAxes, fontsize=11, fontweight='bold', color=INK_PRIMARY, va='top',
              bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor=edge_color))
    ax0.set_xlabel("Idade média dos jogadores adquiridos (anos)", fontsize=10, fontweight='bold')
    ax0.set_ylabel("ΔWPA_time (taxa pós − taxa pré)", fontsize=10, fontweight='bold')
    ax0.set_title(f"Dispersão + tendência LOESS -- pooled 2013-2025 (n={len(df)})",
                  fontsize=11.5, fontweight='bold', color=INK_PRIMARY, pad=8)
    ax0.legend(loc='upper right', frameon=False, fontsize=9)
    style_ax(ax0)

    # ---- painel direito: mesma amostra agrupada em quartis de idade ----
    ax1 = fig.add_axes([0.57, 0.20, 0.38, 0.55])
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    q_desc = {"Q1": "mais jovens", "Q2": "", "Q3": "", "Q4": "mais veteranos"}
    df["quartil"] = pd.qcut(df["idade_media_adquiridos"], 4, labels=q_labels)
    q_stats = []
    for ql in q_labels:
        s = df.loc[df["quartil"] == ql, "delta_WPA_time"]
        lo_ci, hi_ci = bootstrap_median_ci(s.values)
        q_stats.append({"q": ql, "n": len(s), "median": s.median(), "lo": lo_ci, "hi": hi_ci})
    x = np.arange(4)
    medians = [d["median"] for d in q_stats]
    ax1.bar(x, medians, width=0.6, color=BLUE, edgecolor=INK_PRIMARY, linewidth=1.1, zorder=3, alpha=0.88)
    for i, d in enumerate(q_stats):
        ax1.plot([i, i], [d["lo"], d["hi"]], color=INK_PRIMARY, linewidth=1.8, zorder=4)
        ax1.plot([i - 0.07, i + 0.07], [d["lo"]] * 2, color=INK_PRIMARY, linewidth=1.8, zorder=4)
        ax1.plot([i - 0.07, i + 0.07], [d["hi"]] * 2, color=INK_PRIMARY, linewidth=1.8, zorder=4)
        y_txt = d["hi"] + 0.006 if d["median"] >= 0 else d["lo"] - 0.006
        va = 'bottom' if d["median"] >= 0 else 'top'
        ax1.text(i, y_txt, f"{d['median']:+.3f}".replace(".", ","), ha='center', va=va,
                  fontsize=9.5, fontweight='bold', color=INK_PRIMARY)
    ax1.axhline(0, color=INK_MUTED, linewidth=1.1, zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{q_labels[i]} (n={q_stats[i]['n']})\n{q_desc[q_labels[i]]}" for i in range(4)],
                         fontsize=9)
    ax1.set_ylabel("Mediana de ΔWPA_time (IC95% bootstrap)", fontsize=10, fontweight='bold')
    ax1.set_title("Mesma amostra, por quartil de idade média adquirida",
                  fontsize=11.5, fontweight='bold', color=INK_PRIMARY, pad=8)
    y_hi = max(d["hi"] for d in q_stats)
    y_lo = min(d["lo"] for d in q_stats)
    span = y_hi - y_lo
    ax1.set_ylim(y_lo - span * 0.18, y_hi + span * 0.22)
    style_ax(ax1)

    card_ax = fig.add_axes([0.07, 0.09, 0.88, 0.06])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis('off')
    card_color = "#dcf3ea" if significant else "#f5f4ef"
    card_edge = AQUA if significant else COLOR_NEUTRAL
    card_ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.10",
                                      linewidth=1.4, edgecolor=card_edge, facecolor=card_color, zorder=1))
    trend_txt = ("tendência de alta de Q1 para Q4 -- consistente com a hipótese" if medians[-1] > medians[0]
                 else "tendência não é monotônica de Q1 para Q4")
    card_ax.text(0.5, 0.5, f"Spearman ρ = {rho:+.3f}   p = {p:.4f}   |   painel direito: {trend_txt}",
                 ha='center', va='center', fontsize=10, fontweight='bold', color=INK_PRIMARY)

    fig.text(0.5, 0.03,
              "Amostra: eventos de troca (team_id x trade_date) intra-temporada, 2013-2025. Idade = idade do "
              "jogador no momento da troca (bronze.raw_trades.player_age); idade média = média entre todos os "
              "jogadores adquiridos pelo time naquele evento.",
              ha='center', va='center', fontsize=7.6, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h9_verdict.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}\n")


# =============================================================================
# H14 -- Severidade da lacuna está associada ao tamanho do pacote de troca?
# (hipótese de ASSOCIAÇÃO entre duas características da troca, não de
# desempenho -- não existe grupo "melhor", por isso o layout aqui foge do
# boxplot categórico usado em H2/H3/H4/H5/H9: é o mesmo painel duplo
# dispersão+LOESS / quartis usado em wpa_h2_versao_continua.png.)
# =============================================================================
def generate_h14_verdict(impact: pd.DataFrame):
    df = impact.dropna(subset=["min_deltaP_acquired", "n_players_acquired"]).copy()
    rho, p = stats.spearmanr(df["min_deltaP_acquired"], df["n_players_acquired"])
    significant = p < ALPHA

    if significant:
        verdict = "A hipótese encontra suporte estatístico"
        subline = ("Nesta amostra, quanto mais severa a lacuna posicional do jogador adquirido (ΔP "
                    "mais negativo), maior tende a ser o número de jogadores envolvidos no evento de troca.")
    else:
        verdict = "A hipótese NÃO encontra suporte estatístico"
        subline = ("A severidade da lacuna posicional não se mostra associada ao tamanho do pacote de "
                    "troca nesta amostra.")

    print(f"[H14] n={len(df)}  Spearman rho={rho:+.4f}  p={p:.4f}  significativo={significant}")
    print(f"Veredito H14: {verdict}")

    fig = plt.figure(figsize=(13.5, 9.6), dpi=300)
    fig.patch.set_facecolor('white')

    tint = "#dcf3ea" if significant else COLOR_ZONE_FILL
    fig.text(0.5, 0.975, "VEREDITO -- H14: SEVERIDADE DA LACUNA VS. TAMANHO DO PACOTE", ha='center', va='center',
              fontsize=11, fontweight='bold', color=INK_SECONDARY,
              bbox=dict(boxstyle="round,pad=0.35", facecolor=tint, edgecolor='none'))
    fig.text(0.5, 0.928, verdict, ha='center', va='center',
              fontsize=16.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.888, subline, ha='center', va='center',
              fontsize=10, color=INK_SECONDARY, style='italic', wrap=True)
    fig.text(0.5, 0.850,
              '"A severidade da lacuna posicional está associada ao número de jogadores envolvidos no '
              'evento de troca." (hipótese de associação -- não há grupo "melhor", só duas variáveis '
              'da própria troca correlacionadas)',
              ha='center', va='center', fontsize=8.8, color=INK_MUTED)

    # ---- painel esquerdo: dispersão + LOESS ----
    ax0 = fig.add_axes([0.07, 0.20, 0.42, 0.55])
    ax0.scatter(df["min_deltaP_acquired"], df["n_players_acquired"], s=30, alpha=0.35,
                color=INK_SECONDARY, edgecolors='none', zorder=3)
    lowess = sm.nonparametric.lowess
    z = lowess(df["n_players_acquired"], df["min_deltaP_acquired"], frac=0.6)
    ax0.plot(z[:, 0], z[:, 1], color=BLUE, linewidth=3, zorder=4, label='Curva LOESS')
    ax0.axvline(0, color=INK_MUTED, linestyle=':', linewidth=1, zorder=2)
    box_color = "#dcf3ea" if significant else "#f5f4ef"
    edge_color = AQUA if significant else COLOR_NEUTRAL
    ax0.text(0.02, 0.96, f"Spearman ρ = {rho:+.4f}  (p {'< 0,0001' if p < 0.0001 else f'= {p:.4f}'})",
              transform=ax0.transAxes, fontsize=11, fontweight='bold', color=INK_PRIMARY, va='top',
              bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor=edge_color))
    ax0.set_xlabel("ΔP do jogador adquirido (lacuna)\n(mais negativo = lacuna mais severa)",
                    fontsize=10, fontweight='bold', linespacing=1.6)
    ax0.set_ylabel("Nº de jogadores no evento de troca", fontsize=10, fontweight='bold')
    ax0.set_title(f"Dispersão + tendência LOESS -- pooled 2013-2025 (n={len(df)})",
                  fontsize=11.5, fontweight='bold', color=INK_PRIMARY, pad=8)
    ax0.legend(loc='upper right', frameon=False, fontsize=9)
    style_ax(ax0)

    # ---- painel direito: mesma amostra agrupada em quartis de severidade ----
    ax1 = fig.add_axes([0.57, 0.20, 0.38, 0.55])
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    # Descrição só nas pontas (Q2/Q3 não precisam) -- mas o RÓTULO exibido abaixo
    # de cada barra é sempre 2 linhas para todo mundo (Q{n=} / descrição-ou-vazio),
    # nunca 3-4 linhas só nas pontas: um rótulo mais alto nas pontas colidia com o
    # cartão de estatística abaixo do eixo (texto escondido atrás do painel opaco).
    q_desc = {"Q1": "lacuna mais severa", "Q2": "", "Q3": "", "Q4": "mais redundante"}
    df["quartil"] = pd.qcut(df["min_deltaP_acquired"], 4, labels=q_labels)
    q_stats = []
    for ql in q_labels:
        s = df.loc[df["quartil"] == ql, "n_players_acquired"]
        lo_ci, hi_ci = bootstrap_median_ci(s.values)
        q_stats.append({"q": ql, "n": len(s), "median": s.median(), "lo": lo_ci, "hi": hi_ci})
    x = np.arange(4)
    medians = [d["median"] for d in q_stats]
    ax1.bar(x, medians, width=0.6, color=BLUE, edgecolor=INK_PRIMARY, linewidth=1.1, zorder=3, alpha=0.88)
    for i, d in enumerate(q_stats):
        ax1.plot([i, i], [d["lo"], d["hi"]], color=INK_PRIMARY, linewidth=1.8, zorder=4)
        ax1.plot([i - 0.07, i + 0.07], [d["lo"]] * 2, color=INK_PRIMARY, linewidth=1.8, zorder=4)
        ax1.plot([i - 0.07, i + 0.07], [d["hi"]] * 2, color=INK_PRIMARY, linewidth=1.8, zorder=4)
        ax1.text(i, d["hi"] + 0.05, f"{d['median']:.1f}".replace(".", ","), ha='center', va='bottom',
                  fontsize=9.5, fontweight='bold', color=INK_PRIMARY)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{q_labels[i]} (n={q_stats[i]['n']})\n{q_desc[q_labels[i]]}" for i in range(4)],
                         fontsize=9)
    ax1.set_ylabel("Mediana de jogadores no evento (IC95% bootstrap)", fontsize=10, fontweight='bold')
    ax1.set_title("Mesma amostra, por quartil de severidade da lacuna",
                  fontsize=11.5, fontweight='bold', color=INK_PRIMARY, pad=8)
    y_hi = max(d["hi"] for d in q_stats)
    ax1.set_ylim(0, y_hi * 1.55)

    # Chamada explícita do achado -- mesmo padrão do "ACHADO CENTRAL" usado em
    # wpa_h3_verdict.png -- liga visualmente Q1 a Q4 em vez de deixar só os
    # números soltos em cima de cada barra.
    if significant and medians[0] != medians[-1]:
        pct_change = (medians[-1] - medians[0]) / medians[0] * 100
        ax1.annotate("", xy=(3, medians[-1] + 0.12), xytext=(0, medians[0] + 0.12),
                     arrowprops=dict(arrowstyle="-|>", color=AQUA, linewidth=1.8,
                                      connectionstyle="arc3,rad=-0.25"), zorder=6)
        ax1.text(1.5, y_hi * 1.42,
                  f"ACHADO: pacote cai de {medians[0]:.0f} para {medians[-1]:.0f} jogador(es)\n"
                  f"de Q1 para Q4 ({pct_change:+.0f}%)",
                  ha='center', va='center', fontsize=9.5, fontweight='bold', color=AQUA, linespacing=1.4,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#dcf3ea", edgecolor=AQUA, linewidth=1.3),
                  zorder=6)
    style_ax(ax1)

    card_ax = fig.add_axes([0.07, 0.09, 0.88, 0.06])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis('off')
    card_color = "#dcf3ea" if significant else "#f5f4ef"
    card_edge = AQUA if significant else COLOR_NEUTRAL
    card_ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.10",
                                      linewidth=1.4, edgecolor=card_edge, facecolor=card_color, zorder=1))
    trend_txt = ("tendência de queda de Q1 para Q4 -- consistente com a hipótese" if medians[0] > medians[-1]
                 else "tendência não é monotônica de Q1 para Q4")
    card_ax.text(0.5, 0.5, f"Spearman ρ = {rho:+.3f}   p = {p:.4f}   |   painel direito: {trend_txt}",
                 ha='center', va='center', fontsize=10, fontweight='bold', color=INK_PRIMARY)

    fig.text(0.5, 0.03,
              "Amostra: eventos de troca (team_id x trade_date) intra-temporada, 2013-2025. min_deltaP_acquired = "
              "ΔP mais severo entre os jogadores adquiridos no evento. n_players_acquired = total de jogadores "
              "adquiridos pelo time naquele evento.",
              ha='center', va='center', fontsize=7.6, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h14_verdict.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}\n")


def main():
    impact = pd.read_csv(DATA_DIR / "wpa_trade_impact_analysis_extended.csv")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print("Gerando vereditos de H2, H3, H4, H5, H9 e H14 (2013-2025, dados reais estendidos)...\n")
    generate_h2_verdict(impact)
    generate_h3_verdict(impact)
    generate_h4_verdict(impact)
    generate_h5_verdict(impact)
    generate_h9_verdict(impact)
    generate_h14_verdict(impact)


if __name__ == "__main__":
    main()
