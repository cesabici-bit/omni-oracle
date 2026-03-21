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
