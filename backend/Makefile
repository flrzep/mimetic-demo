SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install run dev test fmt lint clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	uvicorn main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "No tests configured yet"

fmt:
	@echo "Add formatter (black/ruff) if desired"

lint:
	@echo "Add linter (ruff/flake8) if desired"

clean:
	rm -rf $(VENV) __pycache__ *.pyc
