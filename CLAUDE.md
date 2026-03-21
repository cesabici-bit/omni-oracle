# CLAUDE.md — OmniOracle

## Progetto
**OmniOracle** — Motore di scoperta automatica di verità statistiche non banali da dati pubblici eterogenei.

## Obiettivo
Costruire un engine che ingerisce decine di migliaia di serie temporali pubbliche cross-domain (economia, clima, salute, trasporti, brevetti, social), scopre automaticamente relazioni causali non banali tramite mutual information + causal discovery, e le presenta con rigore statistico completo (p-value, confidence interval, validazione out-of-sample).

## Stack Tecnico

| Componente | Tecnologia | Versione | Motivo |
|-----------|-----------|---------|--------|
| Linguaggio | Python | 3.12+ | Ecosistema data science maturo |
| Data processing | Pandas / Polars | latest | Gestione serie temporali |
| Statistical screening | Scikit-learn / Scipy | latest | MI, statistical tests |
| Causal discovery | DoWhy / CausalNex | latest | PC algorithm, DAG discovery |
| Information theory | NPEET / sklearn | latest | Mutual information KNN estimator |
| Time series | Statsmodels | latest | Granger causality, VAR models |
| Data ingestion | HTTPX / aiohttp | latest | Async API calls |
| Storage | DuckDB | latest | Analytics su dati colonnari, zero infra |
| Plausibility | Claude API | latest | LLM plausibility scoring |
| Viz | Plotly | latest | Grafici interattivi per output |
| Task orchestration | Luigi / Prefect | latest | Pipeline DAG |

> IMPORTANT: ogni dipendenza DEVE avere entry verificata in `verified-deps.toml`

## Architettura

```
                    FONTI PUBBLICHE
                    (API gov, open data, sensori)
                         │
                    ┌────▼────┐
                    │ INGEST  │  Async fetchers + normalizzazione
                    │ LAYER   │  → X_i(t, geo) serie temporali
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ STORAGE │  DuckDB: tabella universale
                    │         │  feature × tempo × geo
                    └────┬────┘
                         │
                    ┌────▼────────┐
                    │ DISCOVERY   │
                    │ ENGINE      │
                    │             │
                    │ L1: MI screening (scarta 99%)
                    │ L2: Granger + Transfer Entropy
                    │ L3: DAG discovery (PC algo)
                    │ L4: Interaction detection
                    └────┬────────┘
                         │
                    ┌────▼────────┐
                    │ VALIDATION  │
                    │ FILTER      │
                    │             │
                    │ FDR (Benjamini-Hochberg)
                    │ Out-of-sample temporal
                    │ LLM plausibility scoring
                    └────┬────────┘
                         │
                    ┌────▼────────┐
                    │ TRUTH       │
                    │ STORE       │  Verità validate con score,
                    │             │  grafo causale, fonti, CI
                    └────┬────────┘
                         │
                    ┌────▼────┐
                    │ OUTPUT  │  Dashboard, API, report
                    └─────────┘
```

## Moduli Core

```
omni-oracle/
├── src/
│   ├── __init__.py
│   ├── ingest/          # Fetcher per ogni fonte dati
│   │   ├── base.py      # Abstract fetcher
│   │   ├── fred.py      # Federal Reserve (800K+ serie)
│   │   ├── worldbank.py # World Bank (3K indicatori × 200 paesi)
│   │   ├── eurostat.py  # EU statistics
│   │   └── gdelt.py     # Global events
│   ├── storage/         # DuckDB schema + query layer
│   │   ├── schema.py
│   │   └── loader.py
│   ├── discovery/       # Il cuore statistico
│   │   ├── mi_screening.py      # Mutual Information screening
│   │   ├── granger.py           # Granger causality
│   │   ├── transfer_entropy.py  # Non-linear causality
│   │   ├── dag_discovery.py     # PC algorithm / DAG
│   │   └── interaction.py       # Conditional dependencies
│   ├── validation/      # Filtro anti-bullshit
│   │   ├── fdr.py               # Benjamini-Hochberg
│   │   ├── temporal_oos.py      # Out-of-sample validation
│   │   └── plausibility.py      # LLM plausibility scoring
│   ├── scoring/         # Ranking verità
│   │   └── ranker.py
│   └── output/          # Presentazione risultati
│       ├── truth_card.py
│       └── dashboard.py
├── tests/
├── verify/              # M4: verifica con tool alternativo
├── CLAUDE.md
├── verified-deps.toml
├── KNOWN_ISSUES.md
├── STATUS.md
└── Makefile
```

## MVP Scope

### IN (MVP — COMPLETATO)
- [x] Ingestione da 2 fonti (FRED + World Bank)
- [x] 49 serie FRED curate cross-domain
- [x] MI screening su tutte le coppie
- [x] Granger causality sulle coppie sopravvissute
- [x] FDR correction (Benjamini-Hochberg)
- [x] Out-of-sample temporal validation
- [x] Output: hypothesis cards con score, p-value, lag, fonti
- [x] Smoke test 5/5 PASS, golden snapshot L4 approvato

### F5 Roadmap (post-MVP)

#### Fase 0 (Mesi 0-3): Proprietary Trading + Credibilita' [IN PARALLELO]
- [ ] Scaling a 500+ variabili (FRED + World Bank + Eurostat)
- [ ] Identificare top 10 relazioni con lag prevedibile per TRADING
- [ ] Backtest su dati storici (train/test split)
- [ ] Paper trading per 3 mesi
- [ ] Open source core engine su GitHub
- [ ] Blog post / preprint con risultati

#### Fase 1 (Mesi 3-6): Validazione + Community
- [ ] Micro-posizioni reali (se paper trading positivo)
- [ ] API minimale (FastAPI) con free tier
- [ ] Dashboard risultati (Plotly)
- [ ] Outreach accademici + giornalisti dati

#### Fase 2 (Mesi 6-12): Monetizzazione Iniziale
- [ ] Tier Professional ($499/mese)
- [ ] Aggiungere fonti dati (5-10 totali)
- [ ] 50 utenti paganti target

#### Fase 3+ (post-anno 1)
- DAG discovery (PC algorithm)
- Transfer entropy (non-linear)
- LLM plausibility scoring
- Enterprise tier + data licensing
- Scaling a 50K+ variabili

> Business plan completo: `.claude/plans/vast-crunching-flurry.md`

## Oracoli di Dominio

| Livello | Fonte | Uso |
|---------|-------|-----|
| L2 (sanity) | Correlazioni note in letteratura economica (es: oil price → CPI con lag 3-6 mesi) | Verificare che il discovery engine le trovi |
| L2 (sanity) | Granger causality paper seminali (Granger 1969, Toda-Yamamoto 1995) | Verificare implementazione su dataset sintetici |
| L5 (reale) | Paper "alternative data" con risultati replicabili (es: satellite parking → retail earnings) | Verificare che il sistema riscopra verità note |
| L5 (reale) | FRED known relationships (Fed Funds Rate → Unemployment, Okun's Law) | Il sistema deve trovare queste senza che gliele diciamo |

> Questi oracoli sono la base per i test L2/L5. Ogni test DEVE citare la fonte con `# SOURCE:`.

## Subtask Correnti

| # | Subtask | Status |
|---|---------|--------|
| ST-01 | Setup progetto + dipendenze | DONE |
| ST-02 | Data model + DuckDB schema | DONE |
| ST-03 | FRED fetcher | DONE (code), needs FRED_API_KEY for integration test |
| ST-04 | World Bank fetcher | DONE (code), needs integration test |
| ST-05 | Preprocessing (stationarity + quality) | DONE (10 test) |
| ST-06 | MI screening | DONE (7 test, L2+L3) |
| ST-07 | Granger causality | DONE (6 test, L2) |
| ST-08 | FDR correction | DONE (7 test) |
| ST-09 | OOS validation | DONE (6 test, anti-leakage) |
| ST-10 | Scoring + output | DONE |
| ST-11 | Pipeline orchestrator | DONE |
| ST-12 | Smoke test E2E | DONE (5/5 PASS, L4 approvato) |
| ST-13 | Verification L2/L5 | DONE (L2:11, L3:6, L5:10 test) |
| ST-14 | M4 cross-tool verification | DONE (Granger 8/8, MI 7/8) |

## Meccanismi Anti-Allucinazione (M1-M4)

### M1: Dependency Lock
- File: `verified-deps.toml`
- Regola: NESSUNA dipendenza nel codice senza entry verificata via web search

### M2: External Oracle Test Pattern
- Regola: ogni test file DEVE avere almeno 1 test con `# SOURCE:` da oracolo esterno
- Oracoli di questo progetto: vedi tabella sopra
- Critico per questo progetto: le "verità" che il sistema scopre devono includere verità GIÀ NOTE in letteratura. Se non le trova → bug.

### M3: Smoke Before Unit
- Sequenza obbligatoria: smoke test E2E -> unit test -> property-based test
- Lo smoke test produce output leggibile dall'umano e diventa golden snapshot (L4)
- Per OmniOracle: lo smoke test è "ingerisci 100 serie, trova almeno 3 correlazioni note"

### M4: Two-Tool Verification
- Directory `verify/` con script R (o Julia) che calcolano MI e Granger sugli stessi dati
- CI confronta output Python vs R — devono concordare entro tolleranza

## Workflow con Phase Gates

Vedi `genius-lab/CLAUDE.md` per il workflow completo. Phase gates specifici di questo progetto:

### Gate Ricerca (F1) — PASSED
- [x] Almeno 3 web search su competitor (alternative data providers, causal discovery tools)
- [x] Identificati almeno 5 oracoli L2 (relazioni note da riscoprire)
- [x] Scope IN/OUT approvato dall'utente

### Gate Architettura (F2) — PASSED
- [x] Stack verificato (ogni dipendenza in verified-deps.toml)
- [x] Smoke test definito ("cosa vuol dire che funziona?")
- [x] Pre-mortem: 3 rischi identificati con mitigazione

### Gate Implementazione (F3) — PASSED
- [x] `make check-all` verde
- [x] Output ispezionabile dall'utente
- [x] 14 subtask completati (ST-01 → ST-14)

### Gate Verifica (F4) — PASSED
- [x] L1: 46 unit test su storage, stationarity, MI, Granger, FDR, OOS
- [x] L2: 11 test con correlazioni note (`# SOURCE:`)
- [x] L3: 6 property-based test (Hypothesis)
- [x] L4: golden snapshot — smoke test 5/5, lista verita' revisionata dall'umano
- [x] L5: 10 test — sistema riscopre 5 relazioni FRED documentate in letteratura

### Gate Deploy (F5) — IN CORSO
- [x] Business plan multi-verticale completato (marzo 2026)
- [x] Ricerca mercato con 12+ web search, tutte con fonti
- [ ] Discovery reale su 500+ variabili
- [ ] Backtest segnali per proprietary trading
- [ ] README professionale + GitHub pubblico
- [ ] CI (GitHub Actions) con check-all
- [ ] License + CHANGELOG

## Protocollo Correzione Errori

Vedi `genius-lab/CLAUDE.md` per il decision tree completo.
Errori specifici di questo progetto vanno in `KNOWN_ISSUES.md`.

### Rischi specifici OmniOracle
1. **Data dredging**: troppe correlazioni spurie → filtro FDR troppo lasco. Mitigazione: FDR + OOS + plausibility
2. **API rate limits**: fonti che bloccano richieste. Mitigazione: cache aggressiva, retry con backoff
3. **Variabili non stazionarie**: serie con trend producono correlazioni spurie. Mitigazione: test stazionarietà (ADF) + differenziazione

## Checkpoint Utente Obbligatori
- [x] F1: Scope IN/OUT
- [x] F2: Architettura e stack
- [x] F3: Output smoke test (lista verita' trovate)
- [x] F4: Golden snapshot — verita' validate
- [x] Business plan multi-verticale approvato
- [ ] F5: Discovery reale 500+ variabili — output ispezionabile
- [ ] F5: Backtest trading — risultati ispezionabili
- [ ] F5: Go/No-Go per micro-posizioni reali
- [ ] F5: Go/No-Go per GitHub pubblico

## Comandi

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test
pytest tests/ -v

# Check completo
make check-all

# Smoke test
make smoke

# Ingestione dati (MVP)
python -m src.ingest.fred --limit 500
python -m src.ingest.worldbank --limit 300

# Discovery run
python -m src.discovery --input data/ --output results/
```
