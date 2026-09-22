#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/download_all_models.sh [download_models.py options]

Download all inference model groups and all pinned official source
dependencies. Model files are written under models/ and source checkouts under
third_party/src/.

Example:
  bash scripts/download_all_models.sh
  bash scripts/download_all_models.sh --model-source auto
  bash scripts/download_all_models.sh --model-source modelscope
  bash scripts/download_all_models.sh --token "$HF_TOKEN"
  bash scripts/download_all_models.sh --check-only

`auto` tries configured upstream routes first and switches to a configured
ModelScope repository only after a network timeout/connection failure. Direct
files may have ordered URL routes; MoGe does not require HF_ENDPOINT.
ModelScope-only mode stops before download if any selected model has no
configured equivalent repository. Pinned Git sources independently try
github.com, ghproxy.net, and ghfast.top, then prefer the last healthy route for
later repositories while preserving all fallbacks.
EOF
  exit 0
fi

exec python3 "$ROOT/scripts/download_models.py" \
  --project-root "$ROOT" \
  --model all \
  "$@"
