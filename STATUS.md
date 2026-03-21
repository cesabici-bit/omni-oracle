# Status -- OmniOracle

## Fase Corrente
F5 IN CORSO -- Discovery reale completato su 551 variabili. Filtri applicati. Prossimo: backtest trading + GitHub pubblico.

## Ultimo Subtask Completato
F5 Fase 0: Full-scale discovery (551 variabili, 4297 ipotesi, 8/8 relazioni note riscoperte).
Filtri post-discovery applicati (identita', lag-0 correlation, stagionalita').
Cross-validation sub-periodi: 2 relazioni ROBUST su 10 candidati trading.

## Prossimo Subtask
1. Investigare serie "Deflation Probability" (STLPPMDEF) per look-ahead bias
2. Backtest sulle 2 relazioni ROBUST (PCE Price -> Price Pressures, Brent -> Price Pressures)
3. Implementare caching risultati pipeline (evitare ricalcolo 50 min)
4. README professionale + GitHub pubblico
5. CI (GitHub Actions) con check-all

## Blockers
- Nessuno. FRED API key funzionante.

## Stato Test
- 63 test passano (pytest, esclusi L5 che richiedono DB non locked)
- Lint (ruff) passa su tutto il codice
- Smoke test: 5/5 PASS (su DB MVP originale)

## Risultati F5 Discovery
- **551 serie ingerite** (253 FRED + 298 World Bank)
- **241 serie dopo quality+stationarity filter**
- **4,297 ipotesi raw** -> **3,484 dopo filtri** (rimossi 813: 36 identita', 777 rho>0.95)
- **1,114 OOS-validated**
- **8/8 relazioni note riscoperte** (Okun, Oil->CPI, FedFunds->Treasury, M2->Inflation, ecc.)
- **2 relazioni ROBUST cross-validated**:
  - PCE Price Index -> Price Pressures (R2: 0.25/0.27 split)
  - Brent Crude -> Price Pressures (R2: 0.20/0.12 split)
- File risultati: results/f5_clean_*.json, results/f5_*.json

## File Modificati (sessione 6)
- src/ingest/fred_expanded.py (nuovo: discovery FRED via search API)
- src/ingest/worldbank.py (espanso: 22 indicatori WB aggiuntivi, modalita' expanded)
- src/config.py (aggiunto spearman_prescreen a PipelineConfig)
- src/pipeline.py (Spearman pre-screen opzionale per scalabilita')
- src/output/trading.py (nuovo: identificazione trading candidates)
- src/output/filters.py (nuovo: filtri identita', stagionalita', cross-validation)
- src/run_f5.py (nuovo: script F5 discovery completo)
- src/run_f5_filter.py (nuovo: script filtri post-discovery)
- data/f5_discovery.duckdb (nuovo: DB con 551 serie)
- results/f5_*.json (nuovo: risultati discovery + trading)

## Log Sessioni
- 2026-03-20: F0 -- Progetto inizializzato.
- 2026-03-21 sess1: F1 completata (ricerca), F2 piano approvato. ST-01 a ST-11 completati. 46 test.
- 2026-03-21 sess2: Ingestione FRED reale (49 serie). Smoke test 4/5. Fix allineamento. ST-13 parziale (L2+L3). 63 test.
- 2026-03-21 sess3: Unicode fix globale. Smoke 5/5 PASS. L4 approvato. ST-13 completo (L5: 10 test). 73 test.
- 2026-03-21 sess4: ST-14 M4 cross-tool verification. verify/ con alt_mi (histogram) + alt_granger (manual OLS). Granger 8/8, MI 7/8. Lint fix. F4 completa.
- 2026-03-21 sess5: Business plan multi-verticale completato. Ricerca mercato (12+ web search). Piano in .claude/plans/vast-crunching-flurry.md.
- 2026-03-21 sess6: F5 Fase 0 completata. Discovery su 551 variabili (FRED expanded + WB expanded). 4297 ipotesi, 8/8 relazioni note, 1114 OOS-validated. Filtri post-discovery: 3484 ipotesi pulite. Cross-validation: 2 relazioni ROBUST. Trading candidates identificati.
