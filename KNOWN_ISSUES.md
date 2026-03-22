# Known Issues — OmniOracle

> Questo file persiste tra sessioni. Claude lo legge a inizio sessione per evitare errori ricorrenti.
> Aggiungere OGNI errore significativo con analisi causa radice.

## Formato Entry

```
### EC-NNN: Titolo breve
- **Data**: YYYY-MM-DD
- **Sintomo**: cosa si osserva
- **Causa**: perche' e' successo
- **Fix**: cosa e' stato fatto
- **Prevenzione**: come evitarlo in futuro
- **Status**: OPEN | FIXED | WORKAROUND
```

## Issues

### EC-001: Serie FRED NAPM non esiste
- **Data**: 2026-03-21
- **Sintomo**: `Bad Request. The series does not exist.` durante ingestione NAPM
- **Causa**: La serie NAPM (ISM Manufacturing PMI) e' stata rimossa/rinominata su FRED
- **Fix**: Rimossa dalla lista curata. Le 49 serie rimanenti vengono ingerite correttamente
- **Prevenzione**: Verificare periodicamente che le serie curate esistano ancora su FRED
- **Status**: OPEN (serie da sostituire con alternativa, es. MANEMP o altro PMI proxy)

### EC-002: Unicode chars crash su Windows cp1252
- **Data**: 2026-03-21
- **Sintomo**: `UnicodeEncodeError: 'charmap' codec can't encode character` in smoke.py
- **Causa**: Caratteri Unicode (tick, cross, >=) non supportati da cp1252 (default Windows console)
- **Fix**: Sostituiti con ASCII: [OK], [FAIL], >=
- **Prevenzione**: Non usare emoji/Unicode speciali nel codice che produce output console
- **Status**: FIXED

### EC-003: STLPPM* serie forward-looking — look-ahead bias + circolarita'
- **Data**: 2026-03-21
- **Sintomo**: I 2 segnali ROBUST (PCE->STLPPM, Brent->STLPPM) con Sharpe +2.19/+1.28 erano tautologici
- **Causa**: La famiglia St. Louis Fed Price Pressures (STLPPM, STLPPMDEF) e' un modello FAVAR che: (1) misura la probabilita' di inflazione nei PROSSIMI 12 mesi (forward-looking), (2) usa come input 104 serie tra cui PCE e commodity prices. Trovare "PCE predice Price Pressures" e' come trovare "input predice output del modello" — circolarita', non scoperta causale.
- **Fix**: Aggiunta blacklist in `src/output/filters.py` (prefisso STLPPM). 156 ipotesi rimosse. Re-run cross-validation: 0 segnali ROBUST genuini (l'unico superstite e' un subset identity, vedi EC-004). Backtest precedenti invalidati.
- **Prevenzione**: Prima di includere serie derivate/modello nel pool discovery, verificare (1) se la serie e' forward-looking, (2) quali sono i suoi input. Se gli input del modello sono nel pool, la serie va esclusa. Aggiungere check documentazione FRED per ogni nuova serie.
- **Fonte**: Jackson, Kliesen, Owyang (2015) "A Measure of Price Pressures", Federal Reserve Bank of St. Louis Review, 97(1), pp.25-52
- **Status**: FIXED

### EC-004: Subset identity — Durable Goods ExTransport vs Total Manufacturing
- **Data**: 2026-03-21
- **Sintomo**: Dopo rimozione STLPPM, l'unico segnale "ROBUST" e' Durable Goods Excluding Transportation -> Total Manufacturing (lag 12, R2 0.30/0.24)
- **Causa**: Durable Goods Ex-Transport e' un SOTTOINSIEME contabile di Total Manufacturing. E' come trovare "mele predicono vendite totali di frutta". Il filtro identita' (Jaccard) non lo cattura perche' i nomi sono diversi.
- **Fix**: OPEN — serve un filtro per relazioni subset/component. Non e' un vero segnale di trading.
- **Prevenzione**: Aggiungere filtro "component identity" che rileva relazioni parte-tutto tra serie economiche (es. via metadata FRED su categorie/aggregati)
- **Status**: OPEN
