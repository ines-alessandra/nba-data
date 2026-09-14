.PHONY: update
update:
	git pull && \
		uv run python -m src.dl && \
		uv run python -m src.espn && \
		git add data && \
		git commit -m "Latest data: $$(date -u)" && \
		git push

# CI already has a checkout, so we shouldn't need to do a `git pull`
.PHONY: update-ci
update-ci:
	uv run python -m src.dl && \
		uv run python -m src.espn && \
		git add data && \
		git commit -m "Latest data: $$(date -u)" && \
		git push

.PHONY: update-fixtures
update-fixtures:
	uv run tests/capture_responses.py

.PHONY: test
test:
	uv run pytest -v --no-header tests

.PHONY: db-load
db-load:
	python3 src/db_pipeline.py

# --- Pipeline analítico (TCC) -----------------------------------------
# Camada Ouro: recorte de estilo por time-temporada e sua clusterização.
.PHONY: team-features
team-features: db-load
	python3 src/build_team_features.py

.PHONY: clusters
clusters: team-features
	python3 src/fit_clusters.py

.PHONY: k-selection
k-selection: team-features
	python3 src/generate_k_selection_table.py

.PHONY: k4-justification
k4-justification: k-selection
	python3 src/generate_k4_justification.py

# Hipótese 1: desigualdade salarial x desempenho.
.PHONY: trade-activity
trade-activity: db-load
	python3 src/build_trade_activity.py

.PHONY: metric-selection
metric-selection: trade-activity
	python3 src/select_inequality_metric.py

.PHONY: window-robustness
window-robustness: trade-activity
	python3 src/validate_trade_window_robustness.py

.PHONY: salary-inequality-analysis
salary-inequality-analysis: trade-activity
	python3 src/analyze_salary_inequality_trades.py

# Hipóteses 2-4: ajuste funcional, experiência e estrutura da negociação (WPA).
.PHONY: wpa-pretrade-clusters
wpa-pretrade-clusters: db-load clusters
	python3 src/build_wpa_pretrade_clusters.py

.PHONY: wpa-trade-analysis
wpa-trade-analysis: wpa-pretrade-clusters
	python3 src/build_wpa_trade_analysis.py

.PHONY: wpa-trade-analysis-extended
wpa-trade-analysis-extended: wpa-pretrade-clusters
	python3 src/build_wpa_trade_analysis_extended.py

.PHONY: wpa-descriptive
wpa-descriptive: wpa-trade-analysis
	python3 src/describe_wpa_results.py

.PHONY: wpa-hypothesis-tests
wpa-hypothesis-tests: wpa-trade-analysis-extended
	python3 src/test_wpa_hypotheses_extended.py

# Roda o pipeline analítico completo, na ordem de dependência acima.
# Alvos PHONY (não rastreiam arquivo por arquivo): a escala deste projeto
# não justifica a complexidade de um orquestrador com cache de staleness,
# a mesma razão pela qual o Make foi escolhido no lugar do Airflow
# (Seção 2.7.3 da Fundamentação Teórica).
.PHONY: pipeline
pipeline: clusters k4-justification metric-selection window-robustness salary-inequality-analysis wpa-descriptive wpa-hypothesis-tests
	@echo "Pipeline analítico completo."

