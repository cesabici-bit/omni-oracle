# Status -- OmniOracle

## Fase Corrente
PROGETTO COMPLETATO — Chiuso come portfolio piece. Backtest v2 su 5 ROBUST signals: 0/5 tradabili. Decisione: il tool ha valore come dimostrazione tecnica, non come prodotto standalone.

## Ultimo Subtask Completato
F5 sess14: Backtest v2 su 5 ROBUST signals (Ridge, 60/40 split):
- 0/5 battono random benchmark (nessun Sharpe >2 sigma sopra random)
- Cause: beta microscopico, quasi-tautologie (Imports->TradeBalance), regime flip
- Conclusione: R2 walk-forward alto != segnale tradabile
- Decisione: chiudere progetto, push GitHub finale, README onesto

## Prossimo Subtask
- README professionale con risultati v2 + conclusioni oneste
- Push finale su GitHub
- Nessun ulteriore sviluppo pianificato

## Blockers
- Nessuno. Progetto completato.

## Stato Test
- 118 test passano (pytest, esclusi L5 che richiedono DB)
- 10 errori L5: DuckDB file lock (non bloccanti, richiedono DB dedicato)
- EIA + NOAA fetcher testati con mock
- Lint (ruff) da verificare

## Risultati F5 Discovery v2 (pipeline non-lineare)
- **551 serie ingerite** (253 FRED + 298 World Bank)
- **6,882 ipotesi clean** (pipeline v2: Lagged Spearman + RF walk-forward)
- **30 trading candidates**
- **5 segnali ROBUST** (4 adjusted_robust + 1 raw_robust):
  1. Imports -> Gas Price (lag=8, R2=0.568, adj_robust)
  2. Imports -> Gas Price (lag=3, R2=0.532, adj_robust)
  3. Imports -> Trade Balance (lag=11, R2=0.516, adj_robust)
  4. USD/EUR -> Semiconductor Prod (lag=8, R2=0.225, adj_robust)
  5. Fed Collateral -> Exports (lag=4, R2=0.208, raw_robust only)
- File risultati: results/f5_v2_*.json

## Risultati Backtest
- Backtest v1 INVALIDATI (EC-003, STLPPM tautologico)
- Backtest v2: da eseguire sui 5 ROBUST signals

## GitHub
- Repo: https://github.com/cesabici-bit/omni-oracle (pubblico, MIT)
- data/ e results/ esclusi da .gitignore (open-core model)
- NOTA: repo remoto non aggiornato con fix EC-003

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
- 2026-03-21 sess7: Caching run_f5_filter (0.57s vs 50min). Backtest 2 ROBUST signals (Sharpe +2.19/+1.28). GitHub repo pubblico (cesabici-bit/omni-oracle). README, LICENSE MIT, .gitignore.
- 2026-03-21 sess8: Investigazione STLPPMDEF (EC-003). Fix strutturali: walk-forward validation (sostituisce 50/50 split), OOS lag cap, cointegration-aware Granger. CI GitHub Actions. README pro. Ancora 0 ROBUST, ma #6 Credit Spread->CFNAI vicino (48%). Regime breaks COVID visibili nei R2.
- 2026-03-22 sess9: Regime-aware walk-forward (#6: 48%->55% adj, 3 COVID breaks). 7 nuovi test. Decisione pivot: validation chain lineare (Granger+OLS) incompatibile con MI non-lineare. Prossimo: multi-model OOS (Ridge/RF) + eventuale Transfer Entropy.
- 2026-03-22 sess10: NON-LINEAR PIVOT completato. Granger->Lagged MI, OLS->Ridge+RF. 12 nuovi test (10 lagged MI + 2 multi-model OOS). 82 test totali. Pipeline, models, config, scoring, output tutti aggiornati. Refilter su cache: backward compat OK. Serve --recompute per risultati reali.
- 2026-03-22 sess11: --recompute v2 con pipeline non-lineare (Lagged Spearman + RF walk-forward) su 551 variabili. 6882 ipotesi clean. 5 ROBUST signals (4 adjusted_robust). Top: Imports->Gas Price (R2=0.57, lag=8), USD/EUR->Semiconductor Prod (R2=0.22, lag=8).
- 2026-03-22 sess12: ST-15 domain field aggiunto a DataModel + ST-16 update fetcher per aggiornamento incrementale serie. Nuovi test per domain e update.
- 2026-03-22 sess13: ST-17 EIA fetcher (Energy Information Administration) + ST-18 NOAA fetcher (climate/weather data). 36 nuovi test. 118 test totali (esclusi L5 che richiedono DB).
- 2026-03-22 sess14: Backtest v2 su 5 ROBUST signals (Ridge, 60/40 split). 0/5 tradabili. Decisione: chiudere progetto come portfolio piece. README professionale con risultati onesti. Push GitHub finale.
