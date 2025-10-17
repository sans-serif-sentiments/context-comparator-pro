PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
PIP ?= $(shell command -v pip3 2>/dev/null || command -v pip 2>/dev/null || echo pip3)
STREAMLIT ?= $(shell command -v streamlit 2>/dev/null || echo streamlit)
RESULTS_DIR ?= results
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV_DIR)/Scripts
else
  VENV_BIN := $(VENV_DIR)/bin
endif

VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
VENV_STREAMLIT := $(VENV_BIN)/streamlit

MODELS ?=
PROFILE ?= summarization
CSV ?= prompts/samples.csv
REPEATS ?= 1
PRESET ?=
EXTRA_ARGS ?=
MSG ?=

.PHONY: help venv activate install run-app bench ensure-results push ui

help:
	@echo "Context Comparator Pro targets:"
	@echo "  make venv               Create a local virtual environment in $(VENV_DIR)/ (override PYTHON=/path/to/python3)"
	@echo "  make activate           Print commands to activate the virtual environment."
	@echo "  make install            Install Python dependencies (uses venv when present)."
	@echo "  make run-app            Launch the Streamlit dashboard."
	@echo "  make ui                 Alias for make run-app."
	@echo "  make bench MODELS=\"m1 m2\" [PROFILE=...] [REPEATS=...] [PRESET=...] [EXTRA_ARGS=...]"
	@echo "  make ensure-results     Create results directory structure."
	@echo "  make push MSG=\"your commit\"    Commit staged changes and push to main."

venv:
	@if ! command -v "$(PYTHON)" >/dev/null 2>&1; then \
		echo "Python interpreter '$(PYTHON)' not found. Set PYTHON=/path/to/python3 when invoking make."; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV_DIR)" ]; then \
		"$(PYTHON)" -m venv "$(VENV_DIR)"; \
		echo "Virtual environment created at $(VENV_DIR)/. Remember to run 'make activate' for activation hints."; \
	else \
		echo "Virtual environment already exists at $(VENV_DIR)/."; \
	fi

activate:
ifeq ($(OS),Windows_NT)
	@echo "Run: .\\$(VENV_BIN)\\activate"
else
	@echo "Run: source $(VENV_BIN)/activate"
endif

install: venv
ifeq ($(wildcard $(VENV_PIP)),)
	"$(PIP)" install -r requirements.txt
else
	"$(VENV_PIP)" install --upgrade pip
	"$(VENV_PIP)" install -r requirements.txt
endif

run-app: ensure-results
ifeq ($(wildcard $(VENV_STREAMLIT)),)
	"$(STREAMLIT)" run app.py
else
	"$(VENV_STREAMLIT)" run app.py
endif

ui: run-app

bench: ensure-results
ifndef MODELS
	$(error Set MODELS="modelA modelB" to run the benchmark)
endif
ifeq ($(wildcard $(VENV_PYTHON)),)
	"$(PYTHON)" bench.py --models $(MODELS) --profile $(PROFILE) --csv $(CSV) --repeats $(REPEATS) $(if $(PRESET),--preset $(PRESET),) $(EXTRA_ARGS)
else
	"$(VENV_PYTHON)" bench.py --models $(MODELS) --profile $(PROFILE) --csv $(CSV) --repeats $(REPEATS) $(if $(PRESET),--preset $(PRESET),) $(EXTRA_ARGS)
endif

ensure-results:
	mkdir -p $(RESULTS_DIR)/logs $(RESULTS_DIR)/outputs $(RESULTS_DIR)/reports

push:
ifndef MSG
	$(error Provide MSG="your commit message" when pushing)
endif
	git status
	git add -A
	git commit -m "$(MSG)"
	git push origin main
