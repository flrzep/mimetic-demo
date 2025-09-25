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

# Modal deployment commands
deploy:
	modal deploy modal_service.py

serve:
	modal serve modal_service.py

# Model management commands
download-yolo:
	modal run manage_models.py::download_yolo_model

list-models:
	modal run manage_models.py::list_models

remove-model:
	@echo "Usage: make remove-model MODEL=model_name"
	@if [ -z "$(MODEL)" ]; then echo "Please specify MODEL=model_name"; exit 1; fi
	modal run manage_models.py::remove_model --model-name $(MODEL)

download-custom:
	@echo "Usage: make download-custom REPO=owner/repo FILE=model.onnx NAME=my-model"
	@if [ -z "$(REPO)" ] || [ -z "$(FILE)" ] || [ -z "$(NAME)" ]; then \
		echo "Please specify REPO, FILE, and NAME parameters"; exit 1; fi
	modal run manage_models.py::download_custom_model --repo-id $(REPO) --filename $(FILE) --model-name $(NAME)

clean:
	rm -rf $(VENV) __pycache__ *.pyc
