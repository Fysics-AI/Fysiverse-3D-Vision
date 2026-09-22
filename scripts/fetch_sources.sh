#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

command -v git >/dev/null || {
  echo "git was not found. Install Git before fetching third-party sources." >&2
  exit 1
}

exec python3 "$ROOT/scripts/fetch_sources.py" --project-root "$ROOT" "$@"
