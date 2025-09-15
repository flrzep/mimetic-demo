#!/usr/bin/env bash
set -euo pipefail

echo "[devcontainer] Installing frontend deps..."
cd /workspaces/memetic-demo/frontend
npm install || npm install --legacy-peer-deps

echo "[devcontainer] Installing backend deps..."
cd /workspaces/memetic-demo/backend
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[devcontainer] Installing modal-service deps..."
cd /workspaces/memetic-demo/modal-service
python -m pip install --upgrade pip
pip install -r requirements.txt || true

echo "[devcontainer] Setup complete."
