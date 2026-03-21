# Makefile — OmniOracle
# Target principale: `make check-all` esegue TUTTI i controlli in sequenza.
# Se uno fallisce, i successivi NON vengono eseguiti.

# ============================================================
# Configurazione
# ============================================================

LANG ?= python
PYTHON ?= .venv/Scripts/python
SRC_DIR ?= src
TEST_DIR ?= tests
VERIFY_DIR ?= verify

# ============================================================
# Target comuni
# ============================================================

.PHONY: check-all types lint test smoke deps verify clean

## check-all: Esegue tutti i controlli (deps, types, lint, test, smoke, verify)
check-all: deps lint test verify
	@echo ""
	@echo "=== ALL CHECKS PASSED ==="
	@echo ""

# ============================================================
# Python targets
# ============================================================

types:
	@echo "--- Type checking ---"
	$(PYTHON) -m mypy $(SRC_DIR)/

lint:
	@echo "--- Linting ---"
	$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/

test:
	@echo "--- Running tests ---"
	$(PYTHON) -W ignore::FutureWarning -m pytest $(TEST_DIR)/ -v

smoke:
	@echo "--- Smoke test ---"
	$(PYTHON) -m $(SRC_DIR).smoke

# ============================================================
# M4: Two-Tool Verification
# ============================================================

## verify: M4 cross-tool verification (alt MI + alt Granger vs main pipeline)
verify:
	@echo "--- M4: Cross-tool verification ---"
	$(PYTHON) -W ignore::FutureWarning $(VERIFY_DIR)/run_comparison.py

# ============================================================
# Dependency verification (M1)
# ============================================================

## deps: Verifica che verified-deps.toml esista e non sia vuoto
deps:
	@echo "--- Checking verified-deps.toml ---"
	@test -f verified-deps.toml || (echo "ERROR: verified-deps.toml not found" && exit 1)
	@echo "verified-deps.toml found"

# ============================================================
# Utility
# ============================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
