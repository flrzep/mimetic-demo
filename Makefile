# Top-level Makefile for memetic-demo monorepo

SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help up upd down restart logs ps build clean fe-install be-install modal-install

help:
	@echo "Targets:"
	@echo "  up            - Start all services (foreground)"
	@echo "  upd           - Start all services (detached)"
	@echo "  down          - Stop all services"
	@echo "  restart       - Recreate services"
	@echo "  logs          - Tail logs for all services"
	@echo "  ps            - Show service status"
	@echo "  build         - Build all service images"
	@echo "  clean         - Remove containers, networks, and volumes"
	@echo "  fe-install    - npm install in frontend"
	@echo "  be-install    - pip install -r requirements.txt in backend (venv)"
	@echo "  modal-install - pip install -r requirements.txt in modal-service (venv)"

up:
	$(COMPOSE) up --build

upd:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

restart: down upd

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build --pull

clean:
	$(COMPOSE) down -v --remove-orphans || true
	@echo "Pruning dangling images (optional)"
	-docker image prune -f

fe-install:
	cd frontend && npm install

be-install:
	cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

modal-install:
	cd modal-service && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
