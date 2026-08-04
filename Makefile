# Convenience targets. `PYTHONPATH=src` lets the experiments import `grok`
# without an editable install; `make install` sets one up if you prefer.
PYTHON ?= python
export PYTHONPATH := src

.PHONY: install test lint reproduce figures paper clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests experiments

# Retrain every model and regenerate every committed figure/table from scratch
# (deterministic; fixed seeds). Slow: the Phase 1 runs dominate (~12 min/seed on
# a laptop GPU). Individual phases can be run directly, e.g.
#   PYTHONPATH=src python experiments/phase2_fourier.py 0
reproduce:
	$(PYTHON) experiments/phase1_grokking.py 0 1 2
	$(PYTHON) experiments/phase2_fourier.py 0
	$(PYTHON) experiments/phase3_circuit.py 0
	$(PYTHON) experiments/phase4_ablation.py 0
	$(PYTHON) experiments/phase5_progress.py 0
	$(PYTHON) experiments/phase6_ablations.py
	$(PYTHON) experiments/phase7_generalization.py
	$(PYTHON) experiments/phase9_scaling.py

figures: reproduce

# Build the technical report to PDF the same way as its sibling project DeepBSDE:
# pandoc + a LaTeX engine (tectonic). Install the toolchain once, e.g.
#   conda create -n paper -c conda-forge pandoc tectonic
# then run this target with that env active (or with pandoc + tectonic on PATH).
paper:
	cd paper && pandoc paper.md -o paper.pdf --pdf-engine=tectonic

clean:
	rm -rf runs __pycache__ .pytest_cache .ruff_cache
