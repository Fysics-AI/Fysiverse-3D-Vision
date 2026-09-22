#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# These portable defaults match bootstrap_envs.sh, download_all_models.sh, and
# fetch_sources.sh. CLI options take precedence; environment variables let a
# site replace the shell defaults without editing this file. Each *_ENV value
# may be a Conda name or an absolute environment prefix.
DEFAULT_CONDA_BIN="conda"
DEFAULT_LAYOUT_ENV="fysiverse-layout"
DEFAULT_MASK_ENV="fysiverse-mask"
DEFAULT_FLUX_ENV="fysiverse-flux"
DEFAULT_TRELLIS_ENV="fysiverse-trellis2"
DEFAULT_REFINE_ENV="fysiverse-refine"
DEFAULT_MODEL_ROOT="$ROOT/models"
DEFAULT_LAYOUT_MODEL="$DEFAULT_MODEL_ROOT/layout"
DEFAULT_SOURCE_ROOT="$ROOT/third_party/src"

CONDA_BIN="${CONDA_BIN:-$DEFAULT_CONDA_BIN}"
LAYOUT_ENV="${FYSIVERSE_LAYOUT_ENV:-$DEFAULT_LAYOUT_ENV}"
MASK_ENV="${FYSIVERSE_MASK_ENV:-$DEFAULT_MASK_ENV}"
FLUX_ENV="${FYSIVERSE_FLUX_ENV:-$DEFAULT_FLUX_ENV}"
TRELLIS_ENV="${FYSIVERSE_TRELLIS_ENV:-$DEFAULT_TRELLIS_ENV}"
REFINE_ENV="${FYSIVERSE_REFINE_ENV:-$DEFAULT_REFINE_ENV}"
MODEL_ROOT="${FYSIVERSE_MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
SOURCE_ROOT="${FYSIVERSE_SOURCE_ROOT:-$DEFAULT_SOURCE_ROOT}"
LAYOUT_MODEL_FROM_DEFAULT=1
LAYOUT_MODEL="$DEFAULT_LAYOUT_MODEL"
if [[ -n "${FYSIVERSE_LAYOUT_MODEL:-}" ]]; then
  LAYOUT_MODEL="$FYSIVERSE_LAYOUT_MODEL"
  LAYOUT_MODEL_FROM_DEFAULT=0
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec python3 -m fysiverse.infer --help
fi

arguments=("$@")
HAS_CONDA_BIN=0
HAS_LAYOUT_ENV=0
HAS_MASK_ENV=0
HAS_FLUX_ENV=0
HAS_TRELLIS_ENV=0
HAS_REFINE_ENV=0
HAS_MODEL_ROOT=0
HAS_LAYOUT_MODEL=0
HAS_SOURCE_ROOT=0
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --conda-bin)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--conda-bin requires a value" >&2
        exit 2
      fi
      CONDA_BIN="${arguments[index + 1]}"
      HAS_CONDA_BIN=1
      index=$((index + 1))
      ;;
    --conda-bin=*)
      CONDA_BIN="${arguments[index]#*=}"
      HAS_CONDA_BIN=1
      ;;
    --layout-env)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--layout-env requires a value" >&2
        exit 2
      fi
      LAYOUT_ENV="${arguments[index + 1]}"
      HAS_LAYOUT_ENV=1
      index=$((index + 1))
      ;;
    --layout-env=*)
      LAYOUT_ENV="${arguments[index]#*=}"
      HAS_LAYOUT_ENV=1
      ;;
    --mask-env)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--mask-env requires a value" >&2
        exit 2
      fi
      MASK_ENV="${arguments[index + 1]}"
      HAS_MASK_ENV=1
      index=$((index + 1))
      ;;
    --mask-env=*)
      MASK_ENV="${arguments[index]#*=}"
      HAS_MASK_ENV=1
      ;;
    --flux-env)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--flux-env requires a value" >&2
        exit 2
      fi
      FLUX_ENV="${arguments[index + 1]}"
      HAS_FLUX_ENV=1
      index=$((index + 1))
      ;;
    --flux-env=*)
      FLUX_ENV="${arguments[index]#*=}"
      HAS_FLUX_ENV=1
      ;;
    --trellis-env)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--trellis-env requires a value" >&2
        exit 2
      fi
      TRELLIS_ENV="${arguments[index + 1]}"
      HAS_TRELLIS_ENV=1
      index=$((index + 1))
      ;;
    --trellis-env=*)
      TRELLIS_ENV="${arguments[index]#*=}"
      HAS_TRELLIS_ENV=1
      ;;
    --refine-env)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--refine-env requires a value" >&2
        exit 2
      fi
      REFINE_ENV="${arguments[index + 1]}"
      HAS_REFINE_ENV=1
      index=$((index + 1))
      ;;
    --refine-env=*)
      REFINE_ENV="${arguments[index]#*=}"
      HAS_REFINE_ENV=1
      ;;
    --model-root)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--model-root requires a value" >&2
        exit 2
      fi
      MODEL_ROOT="${arguments[index + 1]}"
      HAS_MODEL_ROOT=1
      index=$((index + 1))
      ;;
    --model-root=*)
      MODEL_ROOT="${arguments[index]#*=}"
      HAS_MODEL_ROOT=1
      ;;
    --layout-model)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--layout-model requires a value" >&2
        exit 2
      fi
      LAYOUT_MODEL="${arguments[index + 1]}"
      HAS_LAYOUT_MODEL=1
      index=$((index + 1))
      ;;
    --layout-model=*)
      LAYOUT_MODEL="${arguments[index]#*=}"
      HAS_LAYOUT_MODEL=1
      ;;
    --source-root)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--source-root requires a value" >&2
        exit 2
      fi
      SOURCE_ROOT="${arguments[index + 1]}"
      HAS_SOURCE_ROOT=1
      index=$((index + 1))
      ;;
    --source-root=*)
      SOURCE_ROOT="${arguments[index]#*=}"
      HAS_SOURCE_ROOT=1
      ;;
  esac
done

# A custom model root changes the Layout fallback as well. An explicit
# FYSIVERSE_LAYOUT_MODEL or --layout-model remains the individual override.
if ((HAS_LAYOUT_MODEL == 0 && LAYOUT_MODEL_FROM_DEFAULT == 1)); then
  LAYOUT_MODEL="$MODEL_ROOT/layout"
fi

INFERENCE_ARGUMENTS=("$@")
if ((HAS_CONDA_BIN == 0)); then INFERENCE_ARGUMENTS+=(--conda-bin "$CONDA_BIN"); fi
if ((HAS_LAYOUT_ENV == 0)); then INFERENCE_ARGUMENTS+=(--layout-env "$LAYOUT_ENV"); fi
if ((HAS_MASK_ENV == 0)); then INFERENCE_ARGUMENTS+=(--mask-env "$MASK_ENV"); fi
if ((HAS_FLUX_ENV == 0)); then INFERENCE_ARGUMENTS+=(--flux-env "$FLUX_ENV"); fi
if ((HAS_TRELLIS_ENV == 0)); then INFERENCE_ARGUMENTS+=(--trellis-env "$TRELLIS_ENV"); fi
if ((HAS_REFINE_ENV == 0)); then INFERENCE_ARGUMENTS+=(--refine-env "$REFINE_ENV"); fi
if ((HAS_MODEL_ROOT == 0)); then INFERENCE_ARGUMENTS+=(--model-root "$MODEL_ROOT"); fi
if ((HAS_LAYOUT_MODEL == 0)); then INFERENCE_ARGUMENTS+=(--layout-model "$LAYOUT_MODEL"); fi
if ((HAS_SOURCE_ROOT == 0)); then INFERENCE_ARGUMENTS+=(--source-root "$SOURCE_ROOT"); fi

command -v "$CONDA_BIN" >/dev/null || {
  echo "conda was not found. Set CONDA_BIN, pass --conda-bin, or install Miniconda/Mamba first." >&2
  exit 1
}

CONDA_SELECTOR=(-n "$LAYOUT_ENV")
case "$LAYOUT_ENV" in
  /*|./*|../*|*/*)
    CONDA_SELECTOR=(-p "$(realpath -m "$LAYOUT_ENV")")
    ;;
esac

exec "$CONDA_BIN" run --no-capture-output "${CONDA_SELECTOR[@]}" \
  python -m fysiverse.infer "${INFERENCE_ARGUMENTS[@]}"
