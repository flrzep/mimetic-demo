SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install run dev clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	uvicorn modal_service:app --host 0.0.0.0 --port 9000

dev:
	uvicorn modal_service:app --host 0.0.0.0 --port 9000 --reload

clean:
	rm -rf $(VENV) __pycache__ *.pyc
