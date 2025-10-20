#!/usr/bin/env bash

# Exit on error, undefined vars, pipe failures
set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

usage() {
    echo "Usage: $0 [ingest|serve] [additional args...]"
    echo "Commands:"
    echo "  ingest    Run the data ingestion pipeline"
    echo "  serve     Start the FastAPI server (default)"
    echo "  help      Show this help message"
}

# Default command if none provided
COMMAND="${1:-serve}"
shift 2>/dev/null || true

case "${COMMAND}" in
    ingest)
        echo "[entrypoint] Running ingestion pipeline..."
        exec python -m src.ingest "$@"
        ;;
    serve)
        echo "[entrypoint] Starting FastAPI server..."
        exec uvicorn "src.serve:app" --host "0.0.0.0" --port "8000" "$@"
        ;;
    help)
        usage
        exit 0
        ;;
    *)
        echo "[entrypoint] ERROR: Unknown command '${COMMAND}'" >&2
        usage >&2
        exit 1
        ;;
esac
