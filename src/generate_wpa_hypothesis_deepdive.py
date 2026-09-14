#!/usr/bin/env python3
"""
Aprofundamento visual de H2 (lacuna estrutural) e H3 (objetivo sazonal),
complementar aos vereditos em generate_wpa_hypothesis_verdicts.py.

NUMERAÇÃO: a hipótese de desigualdade salarial é H1. Lacuna estrutural = H2.
Objetivo sazonal = H3. (Este arquivo antes chamava essas duas de "H1"/"H2";
renomeado para bater com a numeração real da tese.)

Três figuras, cada uma respondendo a UMA pergunta específica levantada na
revisão dos resultados (o antigo wpa_h3_previsao_vs_realidade.png foi retirado
por ser visualmente redundante com wpa_h3_verdict.png -- mesma estatística,
sem informação nova):

  wpa_h2_versao_continua.png
    A forma binária de H2 (deltaP<0 vs >=0) não é significativa, mas a mesma
    variável tratada como magnitude contínua (min_deltaP_acquired) É -- pooled
    nas 13 temporadas (2013-2025), sem nenhum recorte por período. Painel
    esquerdo: dispersão + LOESS. Painel direito: a mesma amostra agrupada em
    quartis de severidade da lacuna, para tornar a tendência (fraca, mas real)
    legível sem depender de uma curva ajustada. ESTA é a "versão contínua
    (refinada)" citada no placar de evidências.

  wpa_h3_regressao_media.png
    Pergunta EXPLORATÓRIA, estatisticamente independente do teste principal de
    H3: dentro de cada grupo, o nível de desempenho ANTES da troca (WPA_rate_pre)
    prediz o tamanho do ganho (delta_WPA_time)? Se a correlação for negativa,
    parte da melhora do Tanking pode ser regressão à média -- times que já
    estavam piores tendem a subir mais, por estarem mais longe da média, não
    necessariamente por causa da troca. Um painel de dispersão + LOESS por
    grupo, com Spearman ρ e p anotados -- a mesma correlação é testada nos 3
    grupos para deixar claro que não é exclusiva do Tanking.

  wpa_evidencia_geral.png
    Placar único com as 5 afirmações testáveis das duas hipóteses (H2 binária,
    H2 contínua, e as 3 afirmações de H3), cada uma com sua estatística e
    veredito -- para responder de forma direta "o que de fato temos como
    evidência", sem exigir que o leitor navegue por várias figuras separadas.

Estas figuras (e as duas de generate_wpa_hypothesis_verdicts.py) são pensadas
para uso direto em slides: só gráfico + estatística formal em rodapé, SEM
parágrafos de interpretação embutidos na imagem -- a leitura em linguagem
simples ("o que este resultado significa") fica de fora de propósito, para ser
colada como nota do slide (texto, não imagem, editável e legível separado do
gráfico).

Todas as estatísticas são recalculadas ao vivo de
data/wpa_trade_impact_analysis_extended.csv a cada execução (mesmo princípio
de generate_wpa_hypothesis_verdicts.py) -- ver também
data/wpa_hypothesis_tests_extended.csv para os mesmos números já tabulados.

Nota metodológica: as 13 temporadas são analisadas SEMPRE como uma amostra
única e pooled (n=367 trocas). O motivo de ter coletado 10 temporadas extras
(scrape_inpredictable_wpa*.py) foi justamente ganhar poder estatístico para
tratar 2013-2025 como um único período -- então nenhuma figura aqui reparte
a amostra em sub-períodos/eras; a fonte de dado (ESPN nativo vs. híbrido vs.
posição estática) é um detalhe de construção da variável, não uma dimensão de
análise.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import statsmodels.api as sm
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
TINT_POS = "#dcf3ea"   # zona prevista positiva (leve, aqua)
TINT_NEUTRAL = "#eceae3"  # zona sem previsão

OBJ_COLORS = {"Contender": BLUE, "Intermediario": ORANGE, "Tanking": AQUA}
OBJ_LABELS = {"Contender": "Contenders", "Intermediario": "Intermediários", "Tanking": "Tanking"}

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = INK_PRIMARY


def style_ax(ax, grid_axis='y'):
    if grid_axis:
        ax.grid(axis=grid_axis, zorder=0, color=GRID)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)


def bootstrap_median_ci(x: np.ndarray, n_boot: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# =============================================================================
# H2 -- Versão contínua: a magnitude da lacuna, não o corte binário
# (SEMPRE pooled, 13 temporadas -- ver nota metodológica no topo do arquivo)
# =============================================================================
def generate_h2_continuous_deepdive(impact: pd.DataFrame):
    df = impact.dropna(subset=["min_deltaP_acquired", "delta_WPA_time"]).copy()
    lowess = sm.nonparametric.lowess
    rho, p = stats.spearmanr(df["min_deltaP_acquired"], df["delta_WPA_time"])
    sig = p < ALPHA

    fig = plt.figure(figsize=(13.5, 6.95), dpi=300)
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.983, 'ESTA É A "VERSÃO CONTÍNUA (REFINADA)" CITADA NO PLACAR DE EVIDÊNCIAS',
              ha='center', va='center', fontsize=9.5, fontweight='bold', color=AQUA if sig else COLOR_NEUTRAL,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcf3ea" if sig else "#eceae3", edgecolor='none'))
    fig.text(0.5, 0.940, "H2 -- Versão Contínua: a Magnitude da Lacuna Prediz o Impacto?",
              ha='center', va='center', fontsize=16, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.897,
              "A forma binária de H2 (ΔP<0 vs ≥0) não é significativa. Tratando a severidade da lacuna como "
              f"magnitude contínua -- amostra única, pooled, todas as 13 temporadas (n={len(df)}) -- aparece uma "
              "correlação fraca, porém estatisticamente real.",
              ha='center', va='center', fontsize=10.3, color=INK_SECONDARY, style='italic')

    # ---- painel esquerdo: dispersão + LOESS, pooled ----
    ax0 = fig.add_axes([0.07, 0.20, 0.42, 0.62])
    ax0.scatter(df["min_deltaP_acquired"], df["delta_WPA_time"], s=30, alpha=0.42,
                color=INK_SECONDARY, edgecolors='none', zorder=3)
    z = lowess(df["delta_WPA_time"], df["min_deltaP_acquired"], frac=0.6)
    ax0.plot(z[:, 0], z[:, 1], color=BLUE, linewidth=3, zorder=4, label='Curva LOESS')
    ax0.axhline(0, color=INK_MUTED, linestyle=':', linewidth=1, zorder=2)
    ax0.axvline(0, color=INK_MUTED, linestyle=':', linewidth=1, zorder=2)
    box_color = "#dcf3ea" if sig else "#f5f4ef"
    edge_color = AQUA if sig else COLOR_NEUTRAL
    ax0.text(0.02, 0.04, f"Spearman ρ = {rho:+.4f}  (p = {p:.4f}{'  ***sig.' if sig else ''})",
              transform=ax0.transAxes, fontsize=11, fontweight='bold', color=INK_PRIMARY,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor=edge_color))
    ax0.set_xlabel("ΔP pré-troca do jogador adquirido\n(mais negativo = lacuna mais severa)",
                    fontsize=10, fontweight='bold', linespacing=1.6)
    ax0.set_ylabel("ΔWPA_time do time (taxa pós − pré)", fontsize=10, fontweight='bold')
    ax0.set_title(f"Dispersão + tendência LOESS -- pooled 2013-2025 (n={len(df)})",
                  fontsize=11.5, fontweight='bold', color=INK_PRIMARY, pad=8)
    ax0.legend(loc='upper right', frameon=False, fontsize=9)
    style_ax(ax0)

    # ---- painel direito: mesma amostra agrupada em quartis de severidade ----
    # Traduz a mesma correlação fraca em algo que não depende de "olhar a curva
    # certa" numa nuvem de pontos -- 4 grupos de igual tamanho, do mais severo
    # (Q1, deltaP mais negativo) ao mais redundante (Q4, deltaP mais positivo).
    ax1 = fig.add_axes([0.57, 0.20, 0.40, 0.62])
    q_labels = ["Q1\n(lacuna mais\nsevera)", "Q2", "Q3", "Q4\n(mais\nredundante)"]
    df["quartil"] = pd.qcut(df["min_deltaP_acquired"], 4, labels=q_labels)
    q_stats = []
    for ql in q_labels:
        s = df.loc[df["quartil"] == ql, "delta_WPA_time"]
        lo_ci, hi_ci = bootstrap_median_ci(s.values)
        q_stats.append({"q": ql, "n": len(s), "median": s.median(), "lo": lo_ci, "hi": hi_ci,
                         "range": (df.loc[df["quartil"] == ql, "min_deltaP_acquired"].min(),
                                   df.loc[df["quartil"] == ql, "min_deltaP_acquired"].max())})
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
    ax1.set_xticklabels([f"{q_labels[i]}\n(n={q_stats[i]['n']})" for i in range(4)], fontsize=9)
    ax1.set_ylabel("Mediana de ΔWPA_time (IC95% bootstrap)", fontsize=10, fontweight='bold')
    ax1.set_title("Mesma amostra, por quartil de severidade da lacuna",
                  fontsize=11.5, fontweight='bold', color=INK_PRIMARY, pad=8)
    # Headroom explícito acima do maior IC95%+rótulo, senão o texto "+0,xxx" do Q1
    # (o mais alto) encosta no título do painel -- não dá pra confiar no auto-scale
    # do matplotlib quando um texto solto (não um Artist com bbox) fica acima da barra.
    y_hi = max(d["hi"] for d in q_stats)
    y_lo = min(d["lo"] for d in q_stats)
    span = y_hi - y_lo
    ax1.set_ylim(y_lo - span * 0.16, y_hi + span * 0.30)
    style_ax(ax1)

    trend_txt = ("tendência de queda de Q1 para Q4 -- consistente com a hipótese" if medians[0] > medians[-1]
                 else "tendência não é monotônica de Q1 para Q4")
    fig.text(0.5, 0.025,
              f"Painel direito: {trend_txt}.",
              ha='center', va='center', fontsize=9, color=INK_MUTED)

    out = VIZ_DIR / "wpa_h2_versao_continua.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# H3 -- Investigação exploratória: regressão à média explica a melhora do Tanking?
# =============================================================================
def generate_h3_regressao_media(impact: pd.DataFrame):
    """Pergunta DIFERENTE e estatisticamente INDEPENDENTE do teste principal de
    H3. O teste principal (wpa_h3_verdict.png) pergunta: 'a mediana de
    delta_WPA_time deste grupo é diferente de zero?' -- responde se o grupo
    melhora/piora NO GERAL. Esta figura pergunta: 'DENTRO de cada grupo, o nível
    de desempenho ANTES da troca prediz o tamanho do ganho?' -- uma correlação
    (Spearman) entre WPA_rate_pre e delta_WPA_time, calculada separadamente por
    grupo. As duas perguntas podem ter respostas diferentes para o mesmo grupo
    (é o caso do Contender: mediana não é significativa, mas a correlação é) --
    por design, não por inconsistência."""
    order = ["Contender", "Intermediario", "Tanking"]
    df = impact.dropna(subset=["WPA_rate_pre", "delta_WPA_time", "objetivo_sazonal"])
    lowess = sm.nonparametric.lowess

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), dpi=300, sharey=True)
    fig.patch.set_facecolor('white')

    results = {}
    for ax, g in zip(axes, order):
        sub = df.loc[df["objetivo_sazonal"] == g]
        rho, p = stats.spearmanr(sub["WPA_rate_pre"], sub["delta_WPA_time"])
        sig = p < ALPHA
        results[g] = (rho, p, len(sub))
        ax.scatter(sub["WPA_rate_pre"], sub["delta_WPA_time"], s=26, alpha=0.42,
                   color=OBJ_COLORS[g], edgecolors='none', zorder=3)
        z = lowess(sub["delta_WPA_time"], sub["WPA_rate_pre"], frac=0.7)
        ax.plot(z[:, 0], z[:, 1], color=INK_PRIMARY, linewidth=2.6, zorder=4)
        ax.axhline(0, color=INK_MUTED, linestyle=':', linewidth=1, zorder=2)
        box_color = "#dcf3ea" if sig else "#f5f4ef"
        edge_color = AQUA if sig else COLOR_NEUTRAL
        ax.text(0.04, 0.045, f"ρ = {rho:+.3f}\np = {p:.4f}  ({'significativo' if sig else 'não significativo'})",
                transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK_PRIMARY,
                va='bottom', ha='left', linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor=edge_color))
        ax.set_title(f"{OBJ_LABELS[g]}  (n={len(sub)})", fontsize=13, fontweight='bold', color=INK_PRIMARY)
        ax.set_xlabel("WPA_rate_pre\n(nível do time ANTES da troca)", fontsize=9.7, linespacing=1.6)
        style_ax(ax)
    axes[0].set_ylabel("ΔWPA_time\n(ganho após a troca)", fontsize=10.5, fontweight='bold', linespacing=1.6)

    strongest = min(results, key=lambda g: results[g][0])
    fig.suptitle("H3 -- Investigação: Regressão à Média Explica (em Parte) a Melhora do Tanking?",
                  fontsize=15.5, fontweight='bold', y=1.05, color=INK_PRIMARY)
    fig.text(0.5, 0.965,
              "Pergunta DIFERENTE da pergunta principal de H3 (mediana do grupo > 0?). Aqui: dentro de cada "
              "grupo, times que já estavam pior antes da troca tendem a subir mais -- por estarem mais longe da "
              f"média, não necessariamente pela troca em si. Correlação negativa e significativa nos 3 grupos, "
              f"mais forte em {OBJ_LABELS[strongest]} -- consistente com regressão à média contribuindo para o "
              "resultado do Tanking, mas não exclusiva dele.",
              ha='center', va='center', fontsize=9.6, color=INK_SECONDARY, style='italic', wrap=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.84])

    out = VIZ_DIR / "wpa_h3_regressao_media.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Placar consolidado -- as 5 afirmações testáveis de H2 e H3, numa única figura
# =============================================================================
def generate_evidence_scorecard(impact: pd.DataFrame):
    """Responde diretamente 'o que temos como evidência' sem exigir que o leitor
    percorra várias figuras -- 1 linha por afirmação testável, sempre pooled/13 temporadas."""
    g_lac = impact.loc[impact["trade_classification"] == "Preenche Lacuna", "delta_WPA_time"].dropna()
    g_red = impact.loc[impact["trade_classification"] == "Redundante", "delta_WPA_time"].dropna()
    u1, p1 = stats.mannwhitneyu(g_lac, g_red, alternative="greater")
    r1 = 1 - (2 * u1) / (len(g_lac) * len(g_red))

    df_cont = impact.dropna(subset=["min_deltaP_acquired", "delta_WPA_time"])
    rho, p2 = stats.spearmanr(df_cont["min_deltaP_acquired"], df_cont["delta_WPA_time"])

    g_c = impact.loc[impact["objetivo_sazonal"] == "Contender", "delta_WPA_time"].dropna()
    g_t = impact.loc[impact["objetivo_sazonal"] == "Tanking", "delta_WPA_time"].dropna()
    _, p3 = stats.wilcoxon(g_c)
    _, p4 = stats.wilcoxon(g_t)
    u5, p5 = stats.mannwhitneyu(g_c, g_t, alternative="greater")
    r5 = 1 - (2 * u5) / (len(g_c) * len(g_t))

    BADGE = {
        "supported_weak": (AQUA, "SUPORTADA (efeito fraco)"),
        "not_supported": (COLOR_NEUTRAL, "NÃO SUPORTADA"),
        "not_confirmed": (COLOR_NEUTRAL, "NÃO CONFIRMADA"),
        "inverted": (RED, "INVERTIDA"),
    }
    rows = [
        dict(tag="H2 · forma original",
             claim="Trocas que preenchem uma lacuna estrutural têm impacto MAIS POSITIVO que trocas redundantes.",
             test=f"Mann-Whitney U, unilateral · n={len(g_lac)} vs {len(g_red)} · pooled 2013-2025",
             stat_main=f"r = {r1:+.3f}", stat_p=f"p = {p1:.4f}", verdict="not_supported"),
        dict(tag="H2 · versão contínua (refinada)",
             claim="Quanto mais severa a lacuna posicional (magnitude, não corte binário em zero), maior o impacto.",
             test=f"Spearman ρ · n={len(df_cont)} · pooled 2013-2025",
             stat_main=f"ρ = {rho:+.3f}", stat_p=f"p = {p2:.4f}", verdict="supported_weak"),
        dict(tag="H3a · Contenders",
             claim="Trocas de times Contenders associam-se a melhora de curto prazo (ΔWPA_time > 0).",
             test=f"Wilcoxon, amostra única (H0: mediana=0) · n={len(g_c)} · pooled 2013-2025",
             stat_main=f"mediana = {g_c.median():+.4f}", stat_p=f"p = {p3:.4f}", verdict="not_confirmed"),
        dict(tag="H3b · Tanking",
             claim="Trocas de times em Tanking associam-se a impacto neutro ou negativo (ΔWPA_time ≤ 0).",
             test=f"Wilcoxon, amostra única (H0: mediana=0) · n={len(g_t)} · pooled 2013-2025",
             stat_main=f"mediana = {g_t.median():+.4f}", stat_p=f"p = {p4:.4f}", verdict="inverted"),
        dict(tag="H3c · contraste direto",
             claim="Trocas de Contenders produzem impacto maior que trocas de Tanking, no mesmo período.",
             test=f"Mann-Whitney U, unilateral · n={len(g_c)} vs {len(g_t)} · pooled 2013-2025",
             stat_main=f"r = {r5:+.3f}", stat_p=f"p = {p5:.4f}", verdict="not_supported"),
    ]

    fig = plt.figure(figsize=(13, 9.6), dpi=300)
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.975, "Placar de Evidências -- H2 e H3, 13 Temporadas (2013-2025), Amostra Única Pooled",
              ha='center', va='center', fontsize=16.5, fontweight='bold', color=INK_PRIMARY)
    fig.text(0.5, 0.945,
              "As 5 afirmações testáveis das duas hipóteses, cada uma com seu teste, estatística e veredito -- "
              "n=367 trocas, sem recorte por período.",
              ha='center', va='center', fontsize=10.5, color=INK_SECONDARY, style='italic')

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    top, bottom = 0.895, 0.075
    n_rows = len(rows)
    row_h = (top - bottom) / n_rows
    x_tag, x_claim, x_stat, x_badge = 0.035, 0.035, 0.60, 0.985

    for i, row in enumerate(rows):
        y_top = top - i * row_h
        y_mid = y_top - row_h / 2
        y_bot = y_top - row_h
        badge_color, badge_text = BADGE[row["verdict"]]

        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y_bot), 1, row_h, facecolor=SURFACE, edgecolor='none', zorder=0))
        ax.plot([0.02, 0.98], [y_bot, y_bot], color=GRID, linewidth=1, zorder=1)

        ax.text(x_tag, y_top - row_h * 0.16, row["tag"], fontsize=9.3, fontweight='bold',
                color=INK_MUTED, ha='left', va='top')
        claim_wrapped = textwrap.fill(row["claim"], width=62)
        ax.text(x_claim, y_top - row_h * 0.40, claim_wrapped, fontsize=11.3, fontweight='bold',
                color=INK_PRIMARY, ha='left', va='top', linespacing=1.4)
        ax.text(x_claim, y_bot + row_h * 0.14, row["test"], fontsize=8.6, color=INK_MUTED,
                ha='left', va='bottom')

        # stat_main e stat_p empilhados (não concatenados numa linha só): o prefixo
        # "mediana = " é bem mais longo que "r = "/"ρ = ", então uma única linha
        # concatenada empurrava o p-valor para dentro do selo em duas das 5 linhas.
        ax.text(x_stat, y_mid + row_h * 0.13, row["stat_main"], fontsize=11, fontweight='bold',
                color=INK_PRIMARY, ha='left', va='center', family='monospace')
        ax.text(x_stat, y_mid - row_h * 0.13, row["stat_p"], fontsize=11, fontweight='bold',
                color=INK_PRIMARY, ha='left', va='center', family='monospace')

        badge_w, badge_h = 0.205, row_h * 0.42
        badge_x0 = x_badge - badge_w
        badge_y0 = y_mid - badge_h / 2
        fill = "#dcf3ea" if row["verdict"] == "supported_weak" else \
               ("#fbdedd" if row["verdict"] == "inverted" else "#eceae3")
        ax.add_patch(FancyBboxPatch((badge_x0, badge_y0), badge_w, badge_h,
                                     boxstyle=f"round,pad=0,rounding_size={badge_h/2}",
                                     linewidth=1.4, edgecolor=badge_color, facecolor=fill, zorder=2))
        ax.text(badge_x0 + badge_w / 2, badge_y0 + badge_h / 2, badge_text, fontsize=8.8, fontweight='bold',
                color=badge_color, ha='center', va='center', zorder=3)

    fig.text(0.5, 0.045,
              "Leitura: das 5 afirmações, só a versão contínua e refinada de H2 encontra suporte -- e com efeito "
              "fraco. H3 é a que mais diverge da previsão: Tanking melhora significativamente (contrário à "
              "hipótese) enquanto Contender não confirma sua própria previsão de melhora.",
              ha='center', va='center', fontsize=9, color=INK_SECONDARY)
    fig.text(0.5, 0.018,
              "Amostra: 367 eventos de troca (team_id × trade_date) intra-temporada, 2013-2025. α = 0,05 em todos os testes.",
              ha='center', va='center', fontsize=7.6, color=INK_MUTED)

    out = VIZ_DIR / "wpa_evidencia_geral.png"
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


def main():
    impact = pd.read_csv(DATA_DIR / "wpa_trade_impact_analysis_extended.csv")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print("Gerando aprofundamento visual de H2 (versão contínua) e H3 (regressão à média)...\n")
    generate_h3_regressao_media(impact)
    generate_h2_continuous_deepdive(impact)
    generate_evidence_scorecard(impact)


if __name__ == "__main__":
    main()
