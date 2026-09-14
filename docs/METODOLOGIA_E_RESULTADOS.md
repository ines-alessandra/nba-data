# Metodologia e Resultados — TCC (Clusterização, Desigualdade Salarial e WPA)

Este documento explica, de forma didática, as três frentes analíticas do projeto — **clusterização de times**,
**desigualdade salarial x trocas** e **WPA / Contribuição Relativa** —, o que cada gráfico em `viz/` mostra, e
qual é a conclusão de cada frente. Ele também documenta as figuras novas geradas para preencher as lacunas
apontadas (escolha de k, legenda de arquétipos, adequação de métricas, glossário de variáveis).

Convenção de temporada usada em todo o projeto: o ano identifica o **fim** da temporada (2025 = temporada 2024-25).

---

## 1. Clusterização de Times (Arquétipos Competitivos)

> **Nota de nomenclatura**: cada cluster tem hoje **um único nome em português** (Reconstrução, Defensivo,
> Ofensivo, Elite) — nunca um composto tipo "Rebuilding / Lottery". O nome é definido uma única vez em
> [team_clustering_common.py](../src/team_clustering_common.py) e reutilizado por todo gráfico deste
> projeto, então mudar o nome ali propaga automaticamente para todas as figuras.

> **Figuras de apoio para os slides 2 e 3 da apresentação**:
> [tabela_dados_clusterizacao.png](../viz/tabela_dados_clusterizacao.png) *(NOVO)* resume o universo de dados
> (390 times-temporada, 59 variáveis em 3 grupos) e
> [fluxograma_clusterizacao.png](../viz/fluxograma_clusterizacao.png) *(NOVO)* ilustra as 5 etapas do
> pipeline (dados brutos → imputação → padronização por temporada → PCA → KMeans) com números reais deste
> projeto em cada painel, não só caixas de texto.

### 1.1 O que foi feito

Cada linha da base é um **time-temporada** (ex.: "Boston Celtics, 2023-24"). Para cada um, calculamos ~59
variáveis a partir dos jogos da temporada regular: médias e desvios-padrão de estatísticas tradicionais
(pontos, rebotes, assistências, %FG, %3P...), ratings avançados (ofensivo/defensivo/net rating), ritmo
(posses por jogo), tendência dentro da temporada (net rating dos 10 primeiros jogos vs. 10 últimos),
desempenho em jogos "clutch" (decididos por ≤5 pontos) e o net rating ajustado pela força do adversário
(SRS do oponente). Isso está em [team_season_features.csv](../data/team_season_features.csv), gerado por
[build_team_features.py](../src/build_team_features.py). A base cobre as temporadas **2012-13 a 2024-25**
(2011-12 foi excluída por ser a temporada do lockout, atípica) — 390 times-temporada no total.

Pipeline de clusterização ([team_clustering_common.py](../src/team_clustering_common.py)):

1. **Imputação** de valores faltantes pela mediana.
2. **Padronização por temporada** (não pelo pool 2013-2025 inteiro): cada feature vira um z-score relativo à
   própria temporada. Isso importa porque o ritmo de jogo e o volume de arremessos de 3 mudaram muito de 2013
   para 2025 — sem essa correção, o KMeans separava "time rápido para os padrões de 2014" de "time rápido
   para os padrões de 2024" como se fossem estilos diferentes, quando são a mesma coisa em escalas de época
   diferentes.
3. **PCA** retendo 95% da variância (31 componentes a partir de 59 features originais).
4. **KMeans com k=4**, `random_state=42`.

O resultado fica em [team_season_clusters.csv](../data/team_season_clusters.csv) (uma coluna `cluster` de
0 a 3 por time-temporada), gerado por [fit_clusters.py](../src/fit_clusters.py).

**Um ponto de dados técnico importante para quem for ler o código**: o KMeans não tem uma ordem fixa — rodar
de novo pode trocar qual inteiro (0,1,2,3) corresponde a qual arquétipo. Por isso todo script deste projeto
deriva o nome/cor de cada cluster do **centróide** (net rating + ritmo médios daquele cluster), nunca de um
mapeamento fixo tipo `{0: "Reconstrução"}`. Essa lógica está centralizada em `team_clustering_common.py` e é
usada por todos os gráficos de cluster — é o que garante que "vermelho = Reconstrução" em todos os gráficos
deste projeto, mesmo que o KMeans rotule esse grupo como cluster 2 numa rodada e cluster 0 em outra.

### 1.2 Por que "arquétipo" e não apenas "cluster" ou "grupo"?

Usamos a palavra **arquétipo** de propósito, no sentido de **um padrão típico e recorrente**, não uma
categoria fixa e permanente de uma franquia específica. Três coisas sustentam esse uso:

1. **O mesmo padrão se repete em times e anos diferentes.** Um cluster não é "o grupo dos times ruins de
   2020" — é um perfil estatístico (Net Rating + ritmo) que reaparece em 30 franquias diferentes ao longo de
   13 temporadas. O Chicago Bulls 2022-23 e o Toronto Raptors 2022-23 caem no mesmo arquétipo (Defensivo)
   não porque são o "mesmo time", mas porque produziram o mesmo *padrão* de números.
2. **Não é uma identidade permanente da franquia.** O [team_trajectories.png](../viz/team_trajectories.png)
   mostra o mesmo time (ex.: Boston Celtics) passando por 3 arquétipos diferentes em 13 anos — o arquétipo
   descreve a *temporada*, não "quem o time é".
3. **É derivado do centróide, não de um rótulo fixo.** Cada arquétipo é definido pelo perfil médio (o
   "protótipo") das temporadas que caem nele — exatamente o sentido estatístico/coloquial de arquétipo como
   "modelo do qual outros são variações", e não uma categoria com fronteiras rígidas decididas a priori.

Essa explicação está reproduzida como um banner no topo de
[legenda_arquetipos_cluster.png](../viz/legenda_arquetipos_cluster.png) *(NOVO)*, pronta para slide.

### 1.3 Escolha de k=4 — baseada SOMENTE em elbow e Silhouette

**[tabela_escolha_k.png](../viz/tabela_escolha_k.png)** *(NOVO)* — tabela com os três indicadores de validação
(WCSS/inércia, Silhouette, Davies-Bouldin) para k=2 a 10. Complementa o
[clustering_validation.png](../viz/clustering_validation.png) original (que só mostra as três curvas) com os
valores exatos.

**A pergunta honesta primeiro**: nenhuma das duas curvas tem seu MELHOR valor em k=4.
- Silhouette **cai monotonicamente** a partir de k=2: 0,1138 (k=2) → 0,0838 (k=3) → 0,0734 (k=4) → ... →
  0,0510 (k=9). O valor mais alto de toda a série está em k=2, não em k=4.
- WCSS (curva do cotovelo) **nunca para de cair** — é da natureza da métrica (mais clusters sempre reduzem
  a soma de distâncias internas). Não existe um "mínimo" a ser encontrado nela.

Se a pergunta fosse "qual k maximiza Silhouette?" ou "qual k minimiza WCSS?", a resposta seria k=2 ou "o
maior k testado" — não k=4. **Por isso a lógica correta não é achar o pico de uma curva, é achar onde as
DUAS curvas mudam de comportamento (regime)** — que é literalmente o que o "método do cotovelo" significa:
o ponto onde a curva deixa de cair rápido e passa a cair devagar.

**[justificativa_k4_elbow_silhouette.png](../viz/justificativa_k4_elbow_silhouette.png)** *(NOVO)* — figura
dedicada a essa mudança de regime, calculada explicitamente:

| Trecho | Queda de WCSS (por passo de k) | Queda de Silhouette (por passo de k) |
|---|---|---|
| k=2 → k=4 (2 passos) | 887 / passo | 0,0202 / passo |
| k=4 → k=10 (6 passos) | 303 / passo | 0,0036 / passo |
| **Razão (antes ÷ depois)** | **2,9x mais rápido antes de k=4** | **5,6x mais rápido antes de k=4** |

Ou seja: entre k=2 e k=4, cada cluster adicional ainda "compra" uma redução grande e real tanto de WCSS
quanto de Silhouette. A partir de k=4, as duas curvas achatam num platô raso e ruidoso — cada cluster
adicional (k=5, 6, 7...) compra cada vez menos, e de forma inconsistente (o Davies-Bouldin, terceiro
índice, chega a piorar em k=6 e k=8). **k=4 é o último ponto antes das duas curvas, independentemente,
"desligarem" — o ponto clássico de cotovelo**, mesmo sem ser o pico isolado de nenhuma delas.

Duas confirmações adicionais, secundárias a esse argumento central: (1) um Agglomerative Clustering
(linkage de Ward) rodado em paralelo com k=4 concorda com o KMeans com **ARI = 0,4354** (concordância
moderada, bem acima do acaso) — a mudança de regime não é um artefato do KMeans especificamente; (2) os 4
clusters resultantes ficam com tamanhos saudáveis (86 a 117 times-temporada cada), nada degenerado.

**A versão passo a passo, sem suavizar (a mais honesta, e a que resolve "por que não k=5?")** — ver
[justificativa_k4_ganho_marginal.png](../viz/justificativa_k4_ganho_marginal.png) *(NOVO)*, que mostra a
queda marginal de CADA passo individual, em vez de uma média de "antes" e "depois" de k=4:

| Passo | Queda de WCSS | Queda de Silhouette |
|---|---|---|
| 2→3 | 5,6% | 0,0300 |
| **3→4** | **4,4%** | **0,0104** |
| **4→5** | **2,6%** | **0,0103** |
| 5→6 | 2,1% | 0,0028 |
| 6→7 | 1,9% | 0,0052 |

O WCSS sozinho já mostra o cotovelo exatamente em k=4: o passo que leva a k=4 (4,4%) ainda é **1,7x maior**
que o passo seguinte, que leva a k=5 (2,6%) — uma desaceleração nítida. **O Silhouette, honestamente, NÃO
decide entre k=4 e k=5** — os passos 3→4 (0,0104) e 4→5 (0,0103) são, na prática, idênticos; só a partir do
passo 5→6 o Silhouette desaba para um patamar de ruído bem mais baixo (0,0028 e menos). Ou seja: se alguém
perguntar "por que não k=5?", a resposta correta não é "porque o Silhouette manda parar em 4" — é: **o WCSS
já desacelerou visivelmente em k=4, e como o Silhouette é indiferente entre 4 e 5, o desempate vai para
quem tem justificativa teórica (4 arquétipos reconhecíveis, não 5) e para a concordância entre algoritmos
(ARI)**. k=6 em diante, aí sim, os dois índices concordam em descartar — ambos caem para um patamar de
ruído.

### 1.4 Classificação e caracterização dos arquétipos

**[legenda_arquetipos_cluster.png](../viz/legenda_arquetipos_cluster.png)** *(NOVO)* — cartão de referência
com os 4 arquétipos, para ser usado como "chave de leitura" de qualquer outro gráfico de cluster deste
projeto (todos já usam a mesma nomenclatura e cor, derivadas de `team_clustering_common.py`):

| Cluster | Arquétipo | n | Net Rating médio | Rating Of./Def. | Ritmo (posses/jogo) | Exemplos típicos |
|---|---|---|---|---|---|---|
| 🟥 | **Reconstrução** | 86 | −6,80 | 105,4 / 112,2 | 98,3 | MEM 2017-18, CHA 2019-20, LAL 2016-17 |
| 🟩 | **Defensivo** | 117 | +1,35 | 109,7 / 108,3 | 97,0 | CHI 2022-23, TOR 2022-23, IND 2017-18 |
| 🟧 | **Ofensivo** | 92 | −0,62 | 109,5 / 110,1 | 100,0 | POR 2016-17, DAL 2012-13, NOH 2021-22 |
| 🟦 | **Elite** | 95 | +5,05 | 112,6 / 107,6 | 99,2 | UTA 2018-19, GSW 2013-14, MEM 2021-22 |

("Exemplos típicos" = os 3 times-temporada de cada cluster cujo net rating fica mais próximo da média do
próprio cluster — ou seja, casos representativos, não extremos.)

As 3 features que mais diferenciam os 4 clusters (maior F-estatística da ANOVA, ver
[cluster_anova_results.csv](../data/cluster_anova_results.csv)) são todas variantes de qualidade competitiva:
`avg_plus_minus` (F=352,7), `avg_net_rating` (F=350,5) e `adjusted_net_rating` (F=307,7) — por isso o
primeiro eixo de leitura dos clusters é "quão bom o time é", e só depois "seu estilo" (ritmo, que aparece
mais abaixo no ranking, com `std_possessions` em 10º lugar). Isso é reconhecido explicitamente no código:
os autores chamam esses grupos de **"arquétipos competitivos"**, não "clusters de estilo" — para um
agrupamento que isola só o eixo de estilo (sem o componente de qualidade), existe uma clusterização
alternativa em [fit_style_archetypes.py](../src/fit_style_archetypes.py) (`team_style_archetypes.csv`),
fora do escopo deste resumo.

**[cluster_heatmap.png](../viz/cluster_heatmap.png)** — heatmap com as ~59 features originais (linhas),
ordenadas pela força de diferenciação (ANOVA F, maior no topo), x os 4 clusters (colunas), com o valor
sendo o z-score da média daquele cluster naquela feature relativo à média entre os 4 clusters. É a versão
"completa" da legenda acima: mostra não só as 3 features de topo, mas todas as ~59, permitindo ver, por
exemplo, que o Cluster 1 (Ofensivo) se destaca em `std_possessions`, `avg_minutes`, `std_fga` —
sinais de ritmo/volume — enquanto tem z-scores próximos de zero nas métricas de net rating.

### 1.5 Estabilidade e migração entre arquétipos ao longo do tempo

**[sankey_cluster_migration.png](../viz/sankey_cluster_migration.png)** / **[.html interativo](../viz/sankey_cluster_migration.html)**
— diagrama de fluxo mostrando, temporada a temporada (2013→2014→...→2025), para qual arquétipo cada uma das
30 franquias migrou. Cada "nó" é um (temporada, arquétipo); a espessura de cada fluxo é quantos times fizeram
aquela transição específica.

**[sankey_additional_plots.png](../viz/sankey_additional_plots.png)** — decompõe as mesmas transições em 4
painéis:
- **Persistência vs. Mudança**: de 360 transições ano-a-ano observadas, 47,8% dos times permaneceram no
  mesmo arquétipo competitivo de um ano para o outro, e 52,2% mudaram — ou seja, a composição competitiva da
  liga é mais fluida do que estática, mas não caótica (quase metade dos times repete o "rótulo" no ano
  seguinte).
- **Destino ao sair da Reconstrução**: dos times que deixaram esse cluster, 45,0% foram para
  Ofensivo, 37,5% para Defensivo e só 17,5% saltaram direto para Elite —
  a subida geralmente passa por um patamar intermediário, não é um salto direto do "tanque" para o topo.
- **Origem dos times que chegaram à Elite**: 46,7% vieram de Defensivo, 37,8% de Ofensivo
  e apenas 15,6% vieram diretamente da Reconstrução — reforça o padrão acima, visto do lado de chegada.
- **Persistência por cluster**: Defensivo é o arquétipo mais "grudento" (54 manutenções
  consecutivas observadas), seguido por Elite (42), Reconstrução (39) e Ofensivo (37, o menos estável).

**[temporal_evolution.png](../viz/temporal_evolution.png)** — quantos times (de 30) estavam em cada
arquétipo, temporada a temporada, 2013-2025 (barras empilhadas + linhas por cluster). Mostra picos e vales:
por exemplo, 2023 tem o maior número de times em Reconstrução (12) da série, e a quantidade de times em
Elite oscila entre 4 (2019) e 12 (2017/2023) por ano.

**[team_trajectories.png](../viz/team_trajectories.png)** — a mesma lógica, mas em vez de contar times por
arquétipo, segue 4 franquias específicas (BOS, GSW, CLE, LAL) ano a ano — dá para "ler" visualmente a
trajetória competitiva de cada uma (ex.: BOS sai de Reconstrução em 2013/2017-19, passa por Defensivo em
2015-16/2021, e vira Elite a partir de 2022-2025; GSW passa a maior parte da série em Elite, com uma queda
para Defensivo/Ofensivo em 2019-2021, a "reconstrução" pós-dinastia).

### 1.6 Análise final — Clusterização

- **k=4 é o ponto onde as curvas de WCSS e Silhouette mudam de regime** (queda ~2,9x e ~5,6x mais rápida
  por passo antes de k=4 do que depois), não o pico isolado de nenhuma das duas — que favoreceriam k=2.
  Confirmado por estabilidade entre algoritmos (ARI=0,44) e por tamanhos de cluster saudáveis (86-117
  cada). É uma limitação honesta a reconhecer: por índices puramente estatísticos isolados, a estrutura
  "mais óbvia" nos dados seria mais próxima de 2 grupos; k=4 é sustentado pela mudança de regime conjunta,
  não por um pico único.
- **O eixo dominante da clusterização é qualidade competitiva (net rating), não estilo de jogo** — as
  variáveis de ritmo/volume só aparecem como diferenciadoras a partir da 10ª posição no ranking de ANOVA F.
- **A composição da liga por arquétipo é dinâmica**: mais da metade dos times muda de arquétipo de um ano
  para o outro, mas a ascensão à elite tipicamente passa por um patamar intermediário (Defensivo ou
  Ofensivo), raramente saltando direto da Reconstrução.

---

## 2. Desigualdade Salarial e Trocas de Meio de Temporada

### 2.1 Pergunta de pesquisa e desenho

A pergunta central: **entre os times que de fato fizeram trocas reais perto do trade deadline, uma mudança
na desigualdade salarial em quadra está associada a uma mudança no desempenho?** E a *magnitude* da troca em
si se relaciona com alguma das duas?

**[salary_methodology_flowchart.png](../viz/salary_methodology_flowchart.png)** resume o pipeline em 5 etapas
(identificar trocas reais → escolher a métrica de Gini → medir Δ desigualdade → medir Δ performance →
correlacionar). As próximas seções detalham cada uma.

**Por que 30 dias antes do deadline** ([build_trade_activity.py](../src/build_trade_activity.py),
`WINDOW_DAYS_BEFORE_DEADLINE`): a janela precisa evitar dois erros opostos — curta demais e perde trocas
genuinamente motivadas pelo deadline (que se concentram nas semanas finais antes da data-limite); longa
demais e mistura com transações de offseason/free agency, presentes nos mesmos arquivos brutos
(`data/trades/trades_{ano}.csv` cobre, em alguns casos, quase dois anos — o de 2018 vai de 06/07/2017 a
26/06/2019). 30 dias foi a escolha original, mas **não** era, até esta seção, validada empiricamente contra
outras larguras — era só uma decisão de desenho razoável.

**Teste de robustez** ([validate_trade_window_robustness.py](../src/validate_trade_window_robustness.py)):
refeito o teste central (Spearman ρ entre Δ Gini e Δ Performance Ajustada) com janelas de 14, 21, 30, 45 e
60 dias, mantendo tudo o resto igual:

| Janela | N | Spearman ρ | p-valor | Significativo? |
|---|---|---|---|---|
| 14 dias | 272 | −0,1746 | 0,0039 | sim |
| 21 dias | 274 | −0,1815 | 0,0026 | sim |
| **30 dias (produção)** | **282** | **−0,1892** | **0,0014** | **sim** |
| 45 dias | 291 | −0,1817 | 0,0019 | sim |
| 60 dias | 292 | −0,1813 | 0,0019 | sim |

ρ varia só entre −0,175 e −0,189 — praticamente estável — e o sinal (negativo, a favor da Teoria da
Equidade) e a significância (p<0,05) se mantêm em todas as 5 larguras testadas, de 14 a 60 dias. **O
resultado da H1 não depende da escolha específica de 30 dias.**

**[trade_window_robustness.png](../viz/trade_window_robustness.png)** *(NOVO)* — visualização da tabela
acima, gerada por
[generate_trade_window_robustness_chart.py](../src/generate_trade_window_robustness_chart.py): ρ (eixo Y)
por largura de janela (eixo X), com a faixa cinza marcando a zona sem efeito estatístico para cada N (a
mesma lógica de `salary_h1_verdict.png`) e o ponto de 30 dias destacado. A linha fica achatada, bem abaixo
da faixa cinza, nas 5 larguras — o mesmo resultado da tabela, em forma de gráfico. Título e conclusão no
rodapé são recalculados a partir dos dados a cada execução (não afirmam "robusto" a menos que os dados
realmente sustentem isso).

### 2.2 Por que o Gini Ponderado por Minutos, entre 6 métricas candidatas

Foram calculadas 6 métricas de desigualdade salarial por elenco/período (ver
[salary_inequality_metrics.py](../src/salary_inequality_metrics.py)): Desvio Padrão, Coeficiente de
Variação, Razão Top1/Bottom1, Proporção do Top 3, Gini simples e **Gini ponderado por minutos jogados**
(constrói a curva de Lorenz pesando cada jogador pelos minutos que efetivamente jogou, não por "1 jogador =
1 unidade"). A avaliação de qual usar segue 3 critérios ([select_inequality_metric.py](../src/select_inequality_metric.py)):

1. **Teórico (decisivo)**: das 6, só o Gini Ponderado por Minutos mede a desigualdade que os jogadores
   efetivamente **vivem em quadra** — um reserva de US$ 2M que joga 200 minutos pesa igual, nas outras 5
   métricas, a um titular de US$ 2M que joga 2.000. Esse é o motivo pelo qual ela foi escolhida, e essa
   escolha foi feita **antes** de olhar qualquer correlação empírica.
2. **Diagnóstico distribucional (descritivo)**: teste de Shapiro-Wilk e contagem de outliers via IQR em
   cada métrica — usado só para justificar o uso uniforme de Spearman (não-paramétrico) na análise
   principal, não para escolher a métrica.
3. **Empírico (confirmatório, não decisivo)**: correlação de Spearman de cada métrica com a mudança de
   performance, e uma matriz de correlação entre as 6 métricas.

**[inequality_metric_correlation_heatmap.png](../viz/inequality_metric_correlation_heatmap.png)** *(NOVO —
antes era metade de `inequality_metric_adequacy.png`, agora separado)* — mostra que o Gini Ponderado por
Minutos correlaciona apenas moderadamente com as outras 5 (ρ entre 0,12 e 0,57): não é redundante com elas,
captura algo distinto — o que justifica a ponderação por minutos não ser só um "enfeite" cosmético na
fórmula.

**[inequality_metric_performance_bar.png](../viz/inequality_metric_performance_bar.png)** *(NOVO — a outra
metade)* — mostra a correlação empírica de cada métrica com Δ Performance Ajustada (N=282). **O que
"confirmatório, não é o critério de escolha" significa, por extenso**: a métrica primária (Gini Ponderado,
em vermelho) já tinha sido escolhida pelo critério teórico (1) *antes* de este gráfico ser produzido. Se a
escolha fosse feita *depois*, olhando qual das 6 correlaciona melhor com o resultado, isso seria um
"specification search" circular — escolher a métrica que corrobora o resultado que ela mesma vai testar.
Este gráfico existe só para reportar, com transparência, que a métrica escolhida por teoria *também* acabou
tendo a maior correlação (em módulo) com a performance entre as 6 (ρ = −0,189, a mais forte da lista) — uma
coincidência favorável, documentada, não a razão da escolha.

### 2.3 Glossário de variáveis

**[salary_inequality_variable_glossary.png](../viz/salary_inequality_variable_glossary.png)** *(NOVO)* —
tabela explicando, para cada uma das 5 variáveis usadas em
[salary_inequality_correlation_matrix.png](../viz/salary_inequality_correlation_matrix.png) e
[salary_inequality_core_scatter.png](../viz/salary_inequality_core_scatter.png), o que ela é, como é
calculada e como ler seu sinal:

| Variável | O que é | Sinal |
|---|---|---|
| Gini (nível pré) | Gini Ponderado do elenco, só na janela antes do deadline | Nível, não variação |
| Δ Gini | `gini_pos − gini_pre` | >0 = desigualdade aumentou após o deadline |
| Magnitude da troca | Nº de jogadores movimentados (entrada+saída) em 30 dias antes do deadline | Maior = reformulação mais intensa |
| Δ Performance (ajustada) | Net Rating ponderado pela força do adversário, pós−pré | >0 = time melhorou (métrica principal) |
| Δ Net Rating (bruto) | Igual, sem ajuste de adversário | Variável de robustez |

### 2.4 Amostra

**[salary_sample_funnel.png](../viz/salary_sample_funnel.png)** — de 390 times-temporada no dataset
(2012-13 a 2024-25), todos têm dados completos de elenco pré/pós, mas só **282** tiveram uma troca real
registrada na janela de 30 dias antes do deadline daquela temporada — é essa amostra final (N=282) que
alimenta todas as análises de correlação desta seção. O painel da direita mostra a composição por
temporada: de 16-19 times/temporada nos anos mais antigos (2012-13 a 2016-17) até 25-28 nos anos recentes —
o mercado de trocas ficou mais movimentado ao longo da série.

### 2.5 Resultados

**[salary_gini_pre_post_paired.png](../viz/salary_gini_pre_post_paired.png)** — o Gini Ponderado por
Minutos sobe, em mediana, de 0,42 (pré) para 0,45 (pós) — teste de Wilcoxon pareado, p<0,0001: o aumento
é estatisticamente significativo. 191 dos 282 times (68%) tiveram aumento de desigualdade após o deadline,
contra 91 (32%) com queda — trocas de deadline tendem a **concentrar** a folha salarial em quadra (times
geralmente trazem uma peça cara para minutos importantes, não distribuem minutos igualmente entre os
recém-chegados).

**[salary_inequality_correlation_matrix.png](../viz/salary_inequality_correlation_matrix.png)** e
**[salary_inequality_core_scatter.png](../viz/salary_inequality_core_scatter.png)** — o teste central:
Spearman ρ = **−0,1892** entre Δ Gini e Δ Performance Ajustada, **p = 0,0014** (N=282) — estatisticamente
significativo. A curva LOESS no scatter mostra o padrão por trás do número único: para Δ Gini negativo
(desigualdade caiu), a performance tende a melhorar; para Δ Gini fortemente positivo, a performance tende a
piorar — mas o efeito não é linear nem enorme (ρ é fraco-a-moderado). Note também: Magnitude da troca não
correlaciona nem com Δ Gini (ρ=0,08) nem com Δ Performance (ρ=−0,01) — o **tamanho** da reformulação de
elenco, isoladamente, não prediz nada; o que importa é a *direção* da mudança de desigualdade.

**[salary_quadrant_analysis.png](../viz/salary_quadrant_analysis.png)** — o mesmo resultado, lido pela
lente das duas teorias concorrentes da literatura de economia do esporte (percentuais com 1 casa decimal,
para ficar visível que os 4 quadrantes somam ~100% da amostra, não os 99% que o arredondamento para inteiro
dava antes):
- **Teoria do Torneio** (mais desigualdade → mais incentivo para o "prêmio" no topo → melhor performance):
  suportada por 86 times (30,5%), contrariada por 105 (37,2%).
- **Teoria da Equidade** (menos desigualdade → mais coesão de equipe → melhor performance): suportada por
  48 times (17,0%), contrariada por 43 (15,2%).

Como o ρ é negativo e significativo, a leitura agregada favorece — moderadamente — a **Teoria da Equidade**
sobre a do Torneio, embora nenhuma das duas explique a maioria dos casos isoladamente (a maior fatia,
37,2%, está no quadrante "contra a Teoria do Torneio": desigualdade subiu e a performance piorou). Este
gráfico é descritivo/ilustrativo — mostra a dispersão real time a time — e não deve ser lido sozinho como
prova da H1: a prova propriamente dita é o teste agregado (Spearman ρ, que pesa magnitude e direção juntas)
e o veredito formal em
[salary_h1_verdict.png](../viz/salary_h1_verdict.png). (Nota: uma versão anterior deste gráfico decompunha
os times em taxas de melhora por direção da mudança — mas nenhuma das duas maiorias resultantes, 52,7% e
55,0%, é estatisticamente diferente de 50/50 num teste binomial — então essa leitura foi descartada por
sugerir confiança que os dados não sustentam.)

### 2.6 Análise por arquétipo competitivo (abordagem "pooled")

> **Nota exploratória — não faz parte da prova direta da H1.** Esta seção responde a uma pergunta diferente
> (a relação Gini×performance é *moderada* pelo arquétipo do time?), não a pergunta da H1 (existe a relação,
> e a favor de qual teoria?). Essa segunda pergunta já está respondida pelo teste agregado da Seção 2.5 e
> pelo veredito da Seção 2.7 — o conteúdo abaixo é contexto adicional, não evidência necessária para
> concluir a H1.

Esta parte reusa a clusterização da Seção 1 para perguntar: a relação entre desigualdade e performance é
igual em todos os arquétipos, ou muda conforme o tipo de time? Rodada em
[pooled_clustering_analysis.py](../src/pooled_clustering_analysis.py), que recalcula o Gini Ponderado por
Minutos **direto do banco de dados a cada execução** (não lê de uma tabela pré-computada e desatualizada) —
ou seja, os gráficos abaixo refletem os dados mais recentes disponíveis no momento em que o script rodou
(nesta sessão, 378 times-temporada válidos, 2013-2025).

**[pooled_team_trajectories.png](../viz/pooled_team_trajectories.png)** — "dados atuais?" **Sim.** A função
`load_full_season_gini()` faz uma consulta SQL nova a cada execução (junta `fact_player_gamelogs` +
`fact_player_salaries` do banco), então o gráfico está sempre sincronizado com o estado mais recente do
banco, não com um snapshot antigo. Acompanha 4 franquias (BOS, GSW, LAL, OKC) e anota, em cada ponto, o
arquétipo (`C0`..`C3`) daquela temporada — dá para ver, por exemplo, o Gini do LAL saltar de 0,48 (2013,
Elite) para 0,65 (2014, Reconstrução) quando o time desmontou o elenco após a lesão de Kobe.

**[pooled_temporal_evolution.png](../viz/pooled_temporal_evolution.png)** — "faz sentido usar?" **Sim, como
gráfico complementar, mas não como evidência estatística principal.** Ele mostra a média do Gini Ponderado
por arquétipo fixo, ano a ano, junto com a média da liga — útil para ver que os 4 arquétipos não têm uma
hierarquia de desigualdade estável ao longo do tempo (a linha vermelha, Reconstrução, ora é a mais desigual —
2015, 2021 — ora uma das menos, sem um padrão monotônico claro). O ponto de atenção: com só ~20-30 times
divididos em 4 grupos por ano, cada célula (arquétipo × temporada) tem poucas observações, então a linha é
naturalmente ruidosa. A evidência estatisticamente mais sólida para "os arquétipos diferem em
desigualdade?" está no **teste de Kruskal-Wallis por temporada** (rodado no mesmo script): das 13
temporadas, a diferença entre arquétipos só é estatisticamente significativa em **2 delas** (2014, p=0,032;
2025, p=0,018) — na maioria dos anos, **não há evidência de que um arquétipo seja sistematicamente mais ou
menos desigual que outro**. Isso é reforçado pelo boxplot agregado
([pooled_boxplot_aggregate.png](../viz/pooled_boxplot_aggregate.png)), onde as 4 caixas se sobrepõem quase
totalmente. Recomendação: use `pooled_temporal_evolution.png` para contar a história de ano a ano (contexto
narrativo), mas ancore a conclusão "arquétipos têm desigualdade parecida" no teste de Kruskal-Wallis e no
boxplot agregado, não na volatilidade visual da linha.

**[pooled_cluster_scatter_loess.png](../viz/pooled_cluster_scatter_loess.png)** *(regenerado nesta sessão)*
— a correlação Gini × performance (SRS), dentro de cada arquétipo fixo separadamente. Achado interessante:
a correlação só é estatisticamente significativa dentro do arquétipo **Ofensivo** (ρ=−0,271,
p=0,013, N=84) — nos outros três (Defensivo, Reconstrução, Elite) não há associação significativa. Ou seja, o
efeito "mais desigualdade, pior performance" encontrado na amostra geral (Seção 2.5) parece concentrado
sobretudo nos times de perfil ofensivo/ritmo acelerado, não é uniforme entre arquétipos. **Ressalva
importante**: isso testa a correlação separadamente em 4 subgrupos; com 4 testes a α=0,05, a chance de achar
pelo menos 1 "significativo" só por acaso já é de ~18,5% (1−0,95⁴) — então "1 de 4 significativo" é uma
pista para investigar melhor (ex.: com correção para múltiplas comparações, ou mais temporadas), não uma
conclusão estabelecida. Por isso este achado fica de fora da cadeia de prova da H1.

### 2.7 Veredito da Hipótese 1

**[salary_h1_verdict.png](../viz/salary_h1_verdict.png)** *(NOVO)* — figura de síntese, gerada por
[generate_h1_verdict.py](../src/generate_h1_verdict.py), com o único objetivo de responder diretamente à
Hipótese 1 (ver `docs/` de apresentação): qual das duas teorias concorrentes — Equidade ou Torneio —
encontra suporte nos dados, ou se o resultado é estatisticamente indiferente. Nada no script é fixo a
dedo: o veredito (texto, cor do marcador, largura da faixa "sem efeito") é recalculado a partir da amostra
a cada execução, então a figura permanece correta caso o resultado mude de sinal ou deixe de ser
significativo no futuro (ex.: se a amostra crescer). Mostra o ρ de Spearman posicionado numa régua de -1
(Equidade) a +1 (Torneio), com IC 95% (transformação z de Fisher) e uma faixa cinza central cuja largura é
o |ρ| crítico para p<0,05 nesta N (aproximação t) — ou seja, a "zona de indiferença" é derivada da própria
amostra, não arbitrária. Com os dados atuais: ρ = -0,1892, p = 0,0014, N = 282 — o marcador cai fora da
faixa cinza, do lado da Equidade, e o veredito exibido é "A Teoria da Equidade encontra suporte
estatístico" (efeito fraco pela convenção de Cohen, 1988).

### 2.8 Análise final — Desigualdade Salarial

- A métrica escolhida (Gini Ponderado por Minutos) tem justificativa teórica clara e é empiricamente
  distinta das outras 5 candidatas, sem ser redundante.
- Trocas de deadline **aumentam** a desigualdade salarial em quadra na maioria dos times (68%), de forma
  estatisticamente significativa.
- Existe uma associação negativa e significativa (ρ=−0,19, p=0,0014) entre aumento de desigualdade e
  variação de performance — **evidência (moderada) a favor da Teoria da Equidade**, embora o efeito seja
  fraco e não universal: 37,2% dos times ficam no quadrante oposto ao previsto por qualquer uma das duas
  teorias. **Este é o resultado que responde à H1** (ver veredito completo na Seção 2.7).
- A magnitude da troca (quantos jogadores mudam de mão) não prediz nada sozinha — o que importa é a direção
  da mudança de desigualdade, não o tamanho da reformulação.
- *(Nota exploratória, fora da prova da H1 — Seção 2.6)* Há um indício de que o efeito não é uniforme entre
  arquétipos competitivos, concentrado nos times de perfil ofensivo/ritmo acelerado — mas é um achado
  isolado (1 de 4 subgrupos testados) sem correção para múltiplas comparações, então fica como pista para
  investigação futura, não como parte da conclusão da H1.

---

## 3. WPA — Contribuição Relativa (Detectando Buracos de Elenco)

> **Slides desta seção**: com só 3 temporadas reais disponíveis (mais serão incorporadas depois), a
> apresentação usa deliberadamente um subconjunto enxuto:
> [fluxograma_formulas_wpa.png](../viz/fluxograma_formulas_wpa.png) *(NOVO — substitui
> `fluxograma_metodologia.png`, que tinha só caixas de texto sem números reais e, pior, afirmava
> incorretamente que o pipeline roda testes de significância Mann-Whitney/Kruskal-Wallis, o que ele
> deliberadamente não faz)*, [trocas_por_posicao_por_temporada.png](../viz/trocas_por_posicao_por_temporada.png)
> e as 3 versões de [times_destaque](../viz/times_destaque_2019.png).
>
> **Nota de nomenclatura**: o clustering pré-troca do WPA (`src/wpa_pretrade_clusters.py`) tinha nomes
> próprios ("Rebuilding", "Elite / Contenders"...) em vez dos nomes únicos do resto do projeto — corrigido
> para importar `ARCHETYPE_INFO` de `team_clustering_common.py` como fonte única. Nenhum número mudou (é
> só rótulo interno, não aparecia em nenhum gráfico visível).

### 3.1 Confirmação: foi usada a Contribuição Relativa?

**Sim.** Toda esta frente implementa, ponto a ponto, a metodologia da seção III do artigo
[Kaio-Lucas-de-Sa-2019006850-Contribuicao-Relativa.pdf](../Kaio-Lucas-de-Sa-2019006850-Contribuicao-Relativa.pdf)
("Contribuição Relativa: Detectando Buracos de Elenco na NBA"), aplicada aos dados reais deste repositório
(ESPN `oWPA`/`dWPA`/`tWPA` por jogador-jogo, 2018-19 a 2020-21 — os únicos anos em que a ESPN preencheu esse
campo nos arquivos brutos). Isso está documentado no topo de
[build_wpa_season_profile.py](../src/build_wpa_season_profile.py) e replicado literalmente em
[generate_wpa_paper_figures.py](../src/generate_wpa_paper_figures.py) (que reproduz as Figuras 4, 5, 7, 9 e
10 do artigo original com os dados deste projeto).

### 3.2 As fórmulas, o que significam, e onde aparecem

| Símbolo | Fórmula | Significado |
|---|---|---|
| `WPA_acc` | soma do `tWPA` (ESPN) de um jogador num período | Contribuição acumulada de Win Probability daquele jogador no período |
| `WPA_pos` | soma de `WPA_acc` de todos jogadores de um grupo posicional (Guards/Forwards/Centers) | Contribuição total daquela posição |
| `RCP_pos` | `WPA_pos / WPA_acc(time)` | Fração da contribuição total do time que veio daquela posição |
| `RCP_bar_pos` | `RCP_pos / N` (N = jogos do time no período) | RCP normalizado pelo nº de jogos — permite comparar janelas de tamanhos diferentes |
| `muP` | média do `RCP_bar_pos` das 30 equipes da liga, na mesma data-corte | "Benchmark" de quanto aquela posição contribui, em média, na liga |
| `deltaP` | `RCP_bar_pos(time) − muP` | >0 = **virtude** (posição contribui acima da média da liga); <0 = **lacuna estrutural** |

Posição de cada jogador (Guards/Forwards/Centers) vem do campo `dAvgPos` da ESPN (posição defensiva
contínua, 1=armador...5=pivô), média ponderada por minutos — é o único sinal posicional com cobertura de
elenco completo (o campo `position` da nba.com só cobre os 5 titulares).

**[exemplo_passo_a_passo.png](../viz/exemplo_passo_a_passo.png)** — aplica essa cadeia de 9 passos a **uma
troca real**: James Harden → Brooklyn Nets, 14/01/2021. Cada número na figura foi recalculado ao vivo do
parquet + banco (não copiado dos CSVs já processados), então funciona como conferência independente do
pipeline. Mostra, entre outras coisas, um detalhe contra-intuitivo explicado no próprio gráfico: `RCP_pos`
deu negativo para os armadores do BKN mesmo com `WPA_pos` positivo — porque o `WPA_acc` do time inteiro
estava negativo naquele trecho, e dividir por um total negativo inverte o sinal da razão (propriedade do
formato da métrica, não um erro).

**Réplicas das figuras do artigo original**, aplicadas aos dados 2019-2021 deste projeto:
- **[wpa_bar_distribuicao.png](../viz/wpa_bar_distribuicao.png)** *(NOVO)* — histograma de `WPA_bar`
  (`WPA_acc`/jogos) por jogador-temporada (n=1.818): concentração acentuada perto de zero, replicando a
  Figura 4 do artigo.
- **[wpa_bar_vs_jogos.png](../viz/wpa_bar_vs_jogos.png)** *(NOVO)* — valores extremos de `WPA_bar`
  concentram-se em jogadores com poucos jogos disputados (viés de volume pequeno), replicando a Figura 5.
- **[rcp_bar_distribuicao.png](../viz/rcp_bar_distribuicao.png)** *(NOVO)* — distribuição do `RCP_bar_pos`
  (posição×time×temporada, n=270): sem outliers extremos, ao contrário do `WPA_bar` — porque agregar por
  posição e normalizar por jogos já suaviza os extremos individuais (Figura 7 do artigo).
- **[quadrante_pivo_armador.png](../viz/quadrante_pivo_armador.png)** — `deltaP_Centers` × `deltaP_Guards`,
  por temporada, com os times mais extremos anotados (Figura 9).
- **[evolucao_pre_pos_troca.png](../viz/evolucao_pre_pos_troca.png)** — o mesmo espaço bidimensional, mas
  mostrando a evolução pré→pós de uma troca real (BKN/Harden) em vez de safras anuais (Figura 10).

### 3.3 Classificação de trocas (H1) e objetivo sazonal (H2)

Para cada jogador adquirido em uma troca real de temporada regular (2019-2021), o pipeline
([build_wpa_trade_analysis.py](../src/build_wpa_trade_analysis.py)) calcula o `deltaP` do time adquirente,
**na posição do jogador, usando só jogos ANTES da troca** (nunca dados pós-troca — isso violaria a lógica de
diagnóstico), e classifica:
- `deltaP < 0` → **"Preenche Lacuna"** (a posição já era estruturalmente deficiente antes da troca chegar).
- `deltaP ≥ 0` → **"Redundante"** (a posição já contribuía na média ou acima).

Em paralelo, o time é classificado por **objetivo sazonal** (rank de conferência pré-troca, mesmo instante):
rank 1-6 = Contender, 7-10 = Intermediário, 11-15 = Tanking — extensão própria deste projeto, fora do artigo
original.

**[positional_gaps_by_season.png](../viz/positional_gaps_by_season.png)** — distribuição do `deltaP` entre
as 30 equipes, por posição e temporada (boxplot). A média de `deltaP` é zero por construção (é um desvio da
própria média da liga); o que varia, e é o que importa, é a **dispersão** entre times.

**[trocas_por_posicao_por_temporada.png](../viz/trocas_por_posicao_por_temporada.png)** — contagem de
jogadores adquiridos, por posição e classificação, nas 3 temporadas:

| Temporada | Guards (Lacuna/Redundante) | Forwards (Lacuna/Redundante) | Centers (Lacuna/Redundante) |
|---|---|---|---|
| 2018-19 | 28 / 7 | 18 / 14 | 2 / 6 |
| 2019-20 | 7 / 10 | 14 / 10 | 3 / 6 |
| 2020-21 | 16 / 6 | 10 / 8 | 1 / 7 |

Padrão consistente nas 3 temporadas: aquisições de **Centers** raramente preenchem lacunas reais (a maioria
é redundante), enquanto aquisições de **Guards** e **Forwards** mais frequentemente miram uma posição de
fato deficiente (especialmente em 2018-19 e 2020-21).

**[tabela_trocas_detalhada_2019.png](../viz/tabela_trocas_detalhada_2019.png)**,
**[…_2020.png](../viz/tabela_trocas_detalhada_2020.png)**,
**[…_2021.png](../viz/tabela_trocas_detalhada_2021.png)** — a tabela evento-a-evento por temporada: cada
linha é um jogador adquirido, com data, time, posição, `deltaP` pré-troca (o que gerou a classificação),
classificação, `deltaP` pós-troca (a lacuna realmente fechou depois?), `delta_WPA_time` do time e objetivo
sazonal.

**[times_destaque_2019.png](../viz/times_destaque_2019.png)**,
**[…_2020.png](../viz/times_destaque_2020.png)**,
**[…_2021.png](../viz/times_destaque_2021.png)** — 4 times por temporada (selecionados por critério
estrutural: maior lacuna diagnosticada, ou mais aquisições na janela — nunca por resultado, para não
enviesar a seleção), com um gráfico de barras por jogador adquirido (`deltaP` pré) e um losango marcando o
`deltaP` pós, colorido pelo veredito ("fechou", "melhorou, não fechou", "não melhorou", "não era lacuna").
Ex.: BKN 2020-21 — Harden foi a única aquisição, "Preenche Lacuna", e a lacuna de fato fechou depois.

**[tabela_transicao_perfil.png](../viz/tabela_transicao_perfil.png)** *(NOVO)* — de 90 eventos de troca com
arquétipo pré e pós disponível, 28 (31,1%) mudaram de arquétipo competitivo (cluster) entre o pré e o
pós-troca; a mudança é um pouco mais comum entre trocas "Preenche Lacuna" (36,7%) do que "Redundante"
(20,0%) — consistente com a ideia de que preencher uma lacuna real tem mais chance de mudar o perfil
competitivo do time do que uma troca redundante.

### 3.4 Resultados descritivos (H1 e H2) — sem teste de significância

Importante: com só 3 temporadas reais de dados ESPN completos, **não há poder estatístico para testes de
hipótese** — por isso toda esta seção é deliberadamente descritiva (medianas, %positivo, sem p-valor),
conforme a nota em todos os scripts de visualização.

**[h1_lacuna_vs_redundante_por_temporada.png](../viz/h1_lacuna_vs_redundante_por_temporada.png)** e
**[resumo_descritivo_por_temporada.png](../viz/resumo_descritivo_por_temporada.png)** /
**[tabela_resultados_descritivos.png](../viz/tabela_resultados_descritivos.png)** — `delta_WPA_time`
(taxa de WPA do time, pós menos pré) por classificação:

| Temporada | Preenche Lacuna (mediana, %positivo, n) | Redundante (mediana, %positivo, n) |
|---|---|---|
| 2018-19 | +0,079, 70,4%, n=27 | −0,042, 36,4%, n=11 |
| 2019-20 | −0,043, 50,0%, n=16 | −0,013, 44,4%, n=9 |
| 2020-21 | +0,002, 52,9%, n=17 | −0,067, 20,0%, n=10 |

Em 2 das 3 temporadas (2018-19 e 2020-21), "Preenche Lacuna" teve mediana e %positivo maiores que
"Redundante" — direção consistente com a hipótese de que preencher uma lacuna real tende a ajudar mais o
time do que uma aquisição redundante. 2019-20 é a exceção (ambas as medianas negativas, a temporada
interrompida pela pandemia pode ter distorcido as janelas pós-troca).

**[h2_objetivo_sazonal_por_temporada.png](../viz/h2_objetivo_sazonal_por_temporada.png)** e
**[h2_taxa_sucesso_por_temporada.png](../viz/h2_taxa_sucesso_por_temporada.png)** — o mesmo, por objetivo
sazonal:

| Temporada | Contender | Intermediário | Tanking |
|---|---|---|---|
| 2018-19 | 44,4% positivo (n=9) | 50,0% (n=10) | 73,7% (n=19) |
| 2019-20 | 11,1% (n=9) | 33,3% (n=3) | 76,9% (n=13) |
| 2020-21 | 60,0% (n=10) | 66,7% (n=3) | 21,4% (n=14) |

Sem um padrão estável entre temporadas — times "Tanking" têm a maior taxa de sucesso em 2018-19 e 2019-20,
mas a menor em 2020-21. Isso é esperado dado o N pequeno por célula e reforça por que o projeto não tenta
testar significância aqui: 3 temporadas não sustentam uma conclusão inferencial sobre objetivo sazonal.

### 3.5 Análise final — WPA / Contribuição Relativa

- A metodologia do artigo de Sá (2025) foi implementada de forma fiel (fórmulas idênticas, réplicas das
  figuras originais) e estendida com uma classificação de objetivo sazonal própria.
- Center é a posição menos provável de ter uma aquisição real classificada como "preenche lacuna" nas 3
  temporadas — times raramente reforçam o pivô numa deficiência estrutural real; Guards e Forwards
  concentram as trocas que de fato miram uma lacuna.
- Descritivamente, trocas que preenchem uma lacuna real tendem a ter impacto (`delta_WPA_time`) melhor que
  trocas redundantes em 2 das 3 temporadas — mas, com N pequeno e sem teste de hipótese, isso é um padrão
  observado, não uma conclusão estatística.
- Cerca de 1 em 3 trocas muda o arquétipo competitivo do time (pré→pós), mais frequentemente quando a troca
  preenchia uma lacuna real.

---

## 4. Síntese Integrada

As três frentes se conectam: a **clusterização** fornece o vocabulário de arquétipo competitivo usado tanto
na análise de **desigualdade salarial** (Seção 2.6, mostrando que o efeito Gini×performance é mais forte em
times ofensivos) quanto na análise de **WPA** (Seção 3.3, mostrando quantas trocas mudam o arquétipo do
time). Um fio narrativo possível para o TCC: nem toda troca de trade deadline é igual — o efeito de uma
troca sobre o desempenho parece depender de **(a)** se ela de fato preenche uma lacuna estrutural
diagnosticável antes do fato (WPA/deltaP), e **(b)** de como ela move a desigualdade salarial em quadra
(mais concentração tende a associar-se, moderadamente, a pior performance ajustada) — com ambos os efeitos
sendo mais nítidos nalguns arquétipos competitivos do que noutros.

---

## 5. O que foi criado/alterado nesta sessão

**Scripts novos:**
- [src/generate_k_selection_table.py](../src/generate_k_selection_table.py) → `tabela_escolha_k.png` +
  `data/clustering_k_selection_metrics.csv`
- [src/generate_k4_justification.py](../src/generate_k4_justification.py) →
  `justificativa_k4_elbow_silhouette.png` (justificativa de k=4 baseada só em elbow + Silhouette)
- [src/generate_cluster_archetype_legend.py](../src/generate_cluster_archetype_legend.py) →
  `legenda_arquetipos_cluster.png`
- [src/generate_salary_variable_glossary.py](../src/generate_salary_variable_glossary.py) →
  `salary_inequality_variable_glossary.png`

**Scripts alterados:**
- [src/select_inequality_metric.py](../src/select_inequality_metric.py) — antes salvava uma única imagem
  (`inequality_metric_adequacy.png`, removida); agora salva duas (`inequality_metric_correlation_heatmap.png`,
  `inequality_metric_performance_bar.png`), cada uma com legenda própria explicando o que o gráfico responde.
- [src/team_clustering_common.py](../src/team_clustering_common.py) — `ARCHETYPE_INFO` agora usa um único
  nome em português por arquétipo (Reconstrução, Defensivo, Ofensivo, Elite), nunca um composto tipo
  "Rebuilding / Lottery". Essa é a única fonte de verdade; qualquer gráfico de cluster do projeto herda o
  nome automaticamente.
- [src/cluster_sankey_migration.py](../src/cluster_sankey_migration.py) e
  [src/sankey_additional_plots.py](../src/sankey_additional_plots.py) — corrigido um `next(... if 'Rebuilding'
  in name)` frágil (dependia de substring no nome de exibição) para usar `derive_cluster_archetypes` e
  procurar pela **chave** do arquétipo (`'rebuilding'`/`'elite'`), não pelo texto — necessário porque o nome
  de exibição não tem mais essa palavra em inglês.

**Scripts reexecutados (sem alteração de lógica de negócio) para propagar os nomes novos e/ou atualizar dados:**
- `cluster_sankey_migration.py`, `sankey_additional_plots.py`, `temporal_analysis.py`,
  `pooled_clustering_analysis.py` → `sankey_cluster_migration.png/.html`, `sankey_additional_plots.png`,
  `temporal_evolution.png`, `team_trajectories.png`, `pooled_team_trajectories.png`,
  `pooled_temporal_evolution.png`, `pooled_boxplot_aggregate.png`, `pooled_annual_boxplots.png`,
  `pooled_cluster_scatter_loess.png` (esses 3 últimos existiam no código mas não estavam presentes em
  `viz/`; a reexecução também confirma que os dados vêm do estado atual do banco/CSVs).
- `generate_wpa_paper_figures.py` → gerou `wpa_bar_distribuicao.png`, `wpa_bar_vs_jogos.png`,
  `rcp_bar_distribuicao.png` (réplicas de figuras do artigo original, ausentes de `viz/`).
- `generate_wpa_results_table.py` → gerou `tabela_transicao_perfil.png` (ausente de `viz/`).
