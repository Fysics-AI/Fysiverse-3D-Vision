#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/download_model.sh MODEL [download_models.py options]

Download one inference model group and all of its transitive model and pinned
official source dependencies. MODEL is one of:
  layout, g2vlm, grounded_sam2, bert_base_uncased, sam3d, moge, dinov2, flux,
  trellis2, trellis_image_large, dinov3_vitl16, rmbg2

Examples:
  bash scripts/download_model.sh layout
  bash scripts/download_model.sh layout --model-source auto
  bash scripts/download_model.sh g2vlm --model-source modelscope
  bash scripts/download_model.sh sam3d --token "$HF_TOKEN"
  bash scripts/download_model.sh sam3d --model-source modelscope \
    --modelscope-token "$MODELSCOPE_API_TOKEN"
  bash scripts/download_model.sh moge --check-only

`auto` tries configured upstream routes first and switches to a configured
ModelScope repository only after a network timeout/connection failure. Direct
files may have ordered URL routes; MoGe automatically tries hf-mirror.com and
then the official Hugging Face URL without requiring HF_ENDPOINT.
ModelScope handles model files only. Pinned Git sources use github.com first,
then ghproxy.net and ghfast.top; a successful route is preferred for later
repositories. Every checkout verifies its commit, required file, official
origin and recursive submodules.
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -gt 0 ]] && exit 0
  exit 2
fi

MODEL="$1"
shift

case "$MODEL" in
  layout|g2vlm|grounded_sam2|bert_base_uncased|sam3d|moge|dinov2|flux|trellis2|trellis_image_large|dinov3_vitl16|rmbg2) ;;
  *)
    echo "Unknown model group: $MODEL" >&2
    usage >&2
    exit 2
    ;;
esac

exec python3 "$ROOT/scripts/download_models.py" \
  --project-root "$ROOT" \
  --model "$MODEL" \
  "$@"
