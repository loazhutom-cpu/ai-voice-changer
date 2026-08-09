#!/usr/bin/env bash
# AI Voice Changer - Development Launcher (Backend + Frontend)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==================================================="
echo "  AI Voice Changer - Starting Dev Environment"
echo "==================================================="

cd "$PROJECT_ROOT"

# Activate Python Virtual Environment if present
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Cleanup child processes on exit
cleanup() {
    echo ""
    echo "[INFO] Shutting down development processes..."
    kill $(jobs -p) 2>/dev/null || true
    echo "[INFO] Shutdown complete."
}
trap cleanup EXIT INT TERM

# 1. Start Python FastAPI Backend in background
echo "[INFO] Starting FastAPI Uvicorn Backend on http://localhost:8000 ..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend readiness
echo "[INFO] Waiting for backend server initialization..."
sleep 2

# 2. Start Electron Frontend
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo "[INFO] Launching Electron Frontend application..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "[INFO] Installing npm packages in frontend..."
        npm install
    fi
    npm run dev || npm start
else
    echo "[NOTICE] Frontend directory 'frontend/' not found. Running backend server only."
    echo "[INFO] Press Ctrl+C to terminate backend server."
    wait $BACKEND_PID
fi
