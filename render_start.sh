#!/usr/bin/env bash
# Use with Render: Build = pip install -r requirements.txt, Start = bash render_start.sh
set -e
export PYTHONUNBUFFERED=1
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
