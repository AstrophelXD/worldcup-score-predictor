#!/usr/bin/env bash
# Local one-click startup: API + Streamlit Dashboard (Linux / macOS / lab host)
#
# Usage:
#   bash scripts/start_local.sh
#   bash scripts/start_local.sh --skip-bootstrap

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SKIP_BOOTSTRAP=0
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-bootstrap)
      SKIP_BOOTSTRAP=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

log() {
  echo "[WorldCup] $*"
}

ensure_bootstrap() {
  if [[ "${SKIP_BOOTSTRAP}" -eq 1 ]]; then
    log "Skip bootstrap (--skip-bootstrap)"
    return
  fi

  local feature_mart="${PROJECT_ROOT}/data/feature_mart/match_features.parquet"
  local checkpoint
  checkpoint="$(find "${PROJECT_ROOT}/artifacts/checkpoints" -maxdepth 1 -name 'baseline_dixon_coles_*.json' ! -name '*world_cup*' 2>/dev/null | sort | tail -n 1 || true)"

  if [[ ! -f "${feature_mart}" ]]; then
    log "Feature mart missing. Running ingest + build_features ..."
    python -m scripts.ingest
    python -m scripts.build_features
  fi

  if [[ -z "${checkpoint}" ]]; then
    log "Checkpoint missing. Running train ..."
    python -m scripts.train
  fi
}

wait_for_api() {
  local url="http://${API_HOST}:${API_PORT}/health"
  local i=0
  while [[ "${i}" -lt "${HEALTH_TIMEOUT}" ]]; do
    if curl -sf "${url}" >/dev/null; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "API not ready after ${HEALTH_TIMEOUT}s: ${url}" >&2
  return 1
}

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    log "Stopping API PID=${API_PID}"
    kill "${API_PID}" 2>/dev/null || true
  fi
}

ensure_bootstrap

export WORLDCUP_API_URL="http://${API_HOST}:${API_PORT}"
log "Starting API at ${WORLDCUP_API_URL}"
python -m uvicorn worldcup.api.main:app --host "${API_HOST}" --port "${API_PORT}" &
API_PID=$!
trap cleanup EXIT INT TERM

wait_for_api
log "API is ready"
log "Starting Streamlit dashboard (Ctrl+C to stop)"
python -m streamlit run "${PROJECT_ROOT}/src/worldcup/dashboard/app.py"
