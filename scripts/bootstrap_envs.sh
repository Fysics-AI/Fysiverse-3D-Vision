#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_CHANNEL="${FYSIVERSE_CONDA_CHANNEL:-conda-forge}"
CONDA_REPODATA_FN="${FYSIVERSE_CONDA_REPODATA_FN:-current_repodata.json}"
PYPI_INDEX="${FYSIVERSE_PYPI_INDEX:-https://pypi.org/simple}"
PIP_TIMEOUT="${FYSIVERSE_PIP_TIMEOUT:-120}"
PIP_RETRIES="${FYSIVERSE_PIP_RETRIES:-5}"
PIP_CACHE_DIR="${FYSIVERSE_PIP_CACHE_DIR:-$ROOT/.cache/pip}"
CUDA_CONSTRAINTS="${FYSIVERSE_CUDA_CONSTRAINTS:-$ROOT/configs/constraints-cu-runtime.txt}"
TRELLIS_CONSTRAINTS="${FYSIVERSE_TRELLIS_CONSTRAINTS:-$ROOT/configs/constraints-trellis2-cu124.txt}"
LAYOUT_ENV="${FYSIVERSE_LAYOUT_ENV:-fysiverse-layout}"
MASK_ENV="${FYSIVERSE_MASK_ENV:-fysiverse-mask}"
FLUX_ENV="${FYSIVERSE_FLUX_ENV:-fysiverse-flux}"
TRELLIS_ENV="${FYSIVERSE_TRELLIS_ENV:-fysiverse-trellis2}"
REFINE_ENV="${FYSIVERSE_REFINE_ENV:-fysiverse-refine}"
PYTHON_VERSION="${FYSIVERSE_PYTHON_VERSION:-3.10}"
FLUX_PYTHON_VERSION="${FYSIVERSE_FLUX_PYTHON_VERSION:-3.12}"
LAYOUT_TORCH_INDEX="${LAYOUT_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
FLUX_TORCH_INDEX="${FLUX_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
TRELLIS_TORCH_INDEX="${TRELLIS_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
REFINE_TORCH_INDEX="${REFINE_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"
MIN_FREE_GB_FOR_ENV="${FYSIVERSE_MIN_FREE_GB_FOR_ENV:-30}"
MIN_CUDA="${FYSIVERSE_MIN_CUDA:-12.4}"
MAX_JOBS="${MAX_JOBS:-8}"
FLASH_ATTN_CUDA_ARCHS="${FYSIVERSE_FLASH_ATTN_CUDA_ARCHS:-}"
MODEL_SOURCE="${FYSIVERSE_MODEL_SOURCE:-auto}"
NETWORK_TIMEOUT="${FYSIVERSE_NETWORK_TIMEOUT:-30}"
SOURCE_TRANSFER_TIMEOUT="${FYSIVERSE_SOURCE_TRANSFER_TIMEOUT:-1800}"
export MAX_JOBS

DOWNLOAD_MODELS=0
SOURCE_CHECK_ONLY=0
ALLOW_NO_GPU=0
SKIP_BLENDER=0
ONLY="all"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_envs.sh [options]

Create and fully install five new inference environments. The installer never
reuses or modifies an existing Conda environment. By default, it fetches all
pinned official source repositories and installs every environment, but it
does not download model weights.

Options:
  --only ROLE          Install one role: layout, mask, flux, trellis, or refine
  --download-models    Download model artifacts needed by the selected role(s)
  --model-source MODE  Model provider: auto, upstream, or modelscope
  --network-timeout S  Model route limit, HF metadata, Git probe/stall timeout
  --source-transfer-timeout S  Wall-clock limit for each Git transfer attempt
  --fetch-sources      Fetch pinned sources (default; kept for compatibility)
  --skip-source-fetch  Use existing sources after validating files and commits
  --allow-no-gpu       Dependency-only setup; skip CUDA extension compilation
  --skip-blender       Skip Blender validation for dependency-only preparation
  -h, --help           Show this help

Environment variables:
  FYSIVERSE_LAYOUT_ENV, FYSIVERSE_MASK_ENV, FYSIVERSE_FLUX_ENV,
  FYSIVERSE_TRELLIS_ENV, FYSIVERSE_REFINE_ENV, FYSIVERSE_MIN_CUDA,
  FYSIVERSE_CONDA_CHANNEL, FYSIVERSE_CONDA_REPODATA_FN,
  FYSIVERSE_PYPI_INDEX, FYSIVERSE_PIP_TIMEOUT, FYSIVERSE_PIP_RETRIES,
  FYSIVERSE_PIP_CACHE_DIR, FYSIVERSE_CUDA_CONSTRAINTS,
  FYSIVERSE_TRELLIS_CONSTRAINTS,
  FYSIVERSE_FLUX_PYTHON_VERSION, LAYOUT_TORCH_INDEX, FLUX_TORCH_INDEX,
  TRELLIS_TORCH_INDEX, REFINE_TORCH_INDEX, KAOLIN_FIND_LINKS,
  FYSIVERSE_MODEL_SOURCE, FYSIVERSE_NETWORK_TIMEOUT,
  FYSIVERSE_SOURCE_TRANSFER_TIMEOUT,
  FYSIVERSE_FLASH_ATTN_CUDA_ARCHS, MAX_JOBS, POST_REFINE_BLENDER
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || { echo "--only requires a role" >&2; exit 2; }
      ONLY="$2"
      shift 2
      ;;
    --download-models) DOWNLOAD_MODELS=1; shift ;;
    --model-source)
      [[ $# -ge 2 ]] || { echo "--model-source requires a mode" >&2; exit 2; }
      MODEL_SOURCE="$2"
      shift 2
      ;;
    --network-timeout)
      [[ $# -ge 2 ]] || { echo "--network-timeout requires seconds" >&2; exit 2; }
      NETWORK_TIMEOUT="$2"
      shift 2
      ;;
    --source-transfer-timeout)
      [[ $# -ge 2 ]] || { echo "--source-transfer-timeout requires seconds" >&2; exit 2; }
      SOURCE_TRANSFER_TIMEOUT="$2"
      shift 2
      ;;
    --fetch-sources) SOURCE_CHECK_ONLY=0; shift ;;
    --skip-source-fetch) SOURCE_CHECK_ONLY=1; shift ;;
    --allow-no-gpu) ALLOW_NO_GPU=1; shift ;;
    --skip-blender) SKIP_BLENDER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ONLY" in
  all|layout|mask|flux|trellis|refine) ;;
  *) echo "Invalid --only role: $ONLY" >&2; usage >&2; exit 2 ;;
esac
case "$MODEL_SOURCE" in
  auto|upstream|modelscope) ;;
  *) echo "Invalid --model-source: $MODEL_SOURCE" >&2; exit 2 ;;
esac

command -v "$CONDA_BIN" >/dev/null || {
  echo "conda was not found. Set CONDA_BIN or install Miniconda/Mamba first." >&2
  exit 1
}

selected() {
  [[ "$ONLY" == "all" || "$ONLY" == "$1" ]]
}

env_exists() {
  "$CONDA_BIN" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$1"
}

assert_new_environment_targets() {
  local -a targets=()
  local -A owners=()
  local item role env_name
  selected layout && targets+=("layout:$LAYOUT_ENV")
  selected mask && targets+=("mask:$MASK_ENV")
  selected flux && targets+=("flux:$FLUX_ENV")
  selected trellis && targets+=("trellis:$TRELLIS_ENV")
  selected refine && targets+=("refine:$REFINE_ENV")

  for item in "${targets[@]}"; do
    role="${item%%:*}"
    env_name="${item#*:}"
    if [[ -n "${owners[$env_name]:-}" ]]; then
      echo "[env] target name is shared by multiple roles: $env_name (${owners[$env_name]}, $role)" >&2
      echo "Set a distinct FYSIVERSE_<ROLE>_ENV value for every selected role." >&2
      exit 1
    fi
    owners[$env_name]="$role"
    if env_exists "$env_name"; then
      echo "[env] refusing to modify existing Conda environment: $env_name ($role)" >&2
      echo "Choose an unused name with FYSIVERSE_${role^^}_ENV and run again." >&2
      exit 1
    fi
  done
  echo "[env] all selected target names are unused; existing environments will not be modified"
}

# Reject every conflicting target before system checks, source downloads, or
# environment creation so an all-role invocation cannot partially start.
assert_new_environment_targets

SYSTEM_CHECK_ARGS=(
  --min-free-gb "$MIN_FREE_GB_FOR_ENV"
  --min-cuda "$MIN_CUDA"
  --path "$ROOT"
  --conda-bin "$CONDA_BIN"
  --blender "${POST_REFINE_BLENDER:-blender}"
)
if [[ "$ALLOW_NO_GPU" == 1 ]]; then
  SYSTEM_CHECK_ARGS+=(--allow-no-gpu)
else
  SYSTEM_CHECK_ARGS+=(--require-nvcc)
fi
if [[ "$SKIP_BLENDER" == 1 || ( "$ONLY" != "all" && "$ONLY" != "refine" ) ]]; then
  SYSTEM_CHECK_ARGS+=(--skip-blender)
fi
python3 "$ROOT/scripts/check_system.py" "${SYSTEM_CHECK_ARGS[@]}"

create_env() {
  local env_name="$1"
  local python_version="$2"
  if env_exists "$env_name"; then
    echo "[env] refusing to modify environment created by another process: $env_name" >&2
    exit 1
  fi
  echo "[env] create new $env_name (Python $python_version)"
  "$CONDA_BIN" create -n "$env_name" --override-channels \
    --channel "$CONDA_CHANNEL" --repodata-fn "$CONDA_REPODATA_FN" \
    "python=$python_version" pip -y
}

pip_in() {
  local env_name="$1"
  shift
  local constraint_file=""
  local -a pip_environment=(
    PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL="$PYPI_INDEX"
    PIP_EXTRA_INDEX_URL= PIP_TRUSTED_HOST=
    PIP_CACHE_DIR="$PIP_CACHE_DIR"
    PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" PIP_RETRIES="$PIP_RETRIES"
  )
  case "$env_name" in
    "$LAYOUT_ENV"|"$MASK_ENV") constraint_file="$CUDA_CONSTRAINTS" ;;
    "$TRELLIS_ENV") constraint_file="$TRELLIS_CONSTRAINTS" ;;
  esac
  if [[ -n "$constraint_file" ]]; then
    [[ -f "$constraint_file" ]] || {
      echo "[pip] constraint file not found: $constraint_file" >&2
      exit 1
    }
    pip_environment+=(PIP_CONSTRAINT="$constraint_file")
  fi
  mkdir -p "$PIP_CACHE_DIR"
  "$CONDA_BIN" run --no-capture-output -n "$env_name" env \
    "${pip_environment[@]}" \
    python -m pip "$@"
}

install_torch_stack() {
  local env_name="$1"
  local torch_spec="$2"
  local torchvision_spec="$3"
  local torch_index="${4%/}"
  # Resolve small generic dependencies through the configured PyPI route.
  # The CUDA index then supplies only torch, torchvision and NVIDIA runtimes.
  pip_in "$env_name" install \
    "filelock>=3" "typing-extensions>=4.8" "networkx>=3,<4" \
    "jinja2>=3,<4" "fsspec>=2024" "sympy==1.13.1" \
    "numpy==1.26.4" "Pillow>=10,<11"
  pip_in "$env_name" install "$torch_spec" "$torchvision_spec" \
    --index-url "$torch_index"
}

require_source() {
  local label="$1"
  local path="$2"
  [[ -e "$path" ]] || {
    echo "[$label] required source path is missing: $path" >&2
    echo "Run without --skip-source-fetch or run scripts/fetch_sources.sh first." >&2
    exit 1
  }
}

install_cuda_extensions() {
  local role="$1"
  local env_name="$2"
  local -a extension_args=(--role "$role" --env "$env_name" --target-is-new)
  if [[ "$ALLOW_NO_GPU" == 1 ]]; then extension_args+=(--allow-no-gpu); fi
  if [[ "$SKIP_BLENDER" == 1 ]]; then extension_args+=(--skip-blender); fi
  FYSIVERSE_PYPI_INDEX="$PYPI_INDEX" \
  FYSIVERSE_PIP_TIMEOUT="$PIP_TIMEOUT" \
  FYSIVERSE_PIP_RETRIES="$PIP_RETRIES" \
  FYSIVERSE_PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  FYSIVERSE_CUDA_CONSTRAINTS="$CUDA_CONSTRAINTS" \
  FYSIVERSE_TRELLIS_CONSTRAINTS="$TRELLIS_CONSTRAINTS" \
  FYSIVERSE_FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" \
  KAOLIN_FIND_LINKS="$KAOLIN_FIND_LINKS" \
    bash "$ROOT/scripts/install_cuda_extensions.sh" "${extension_args[@]}"
}

fetch_sources_for_selection() {
  local source_args=(
    --network-timeout "$NETWORK_TIMEOUT"
    --transfer-timeout "$SOURCE_TRANSFER_TIMEOUT"
  )
  if [[ "$SOURCE_CHECK_ONLY" == 1 ]]; then
    source_args+=(--check-only)
  fi
  case "$ONLY" in
    all) python3 "$ROOT/scripts/fetch_sources.py" --model all "${source_args[@]}" ;;
    layout) python3 "$ROOT/scripts/fetch_sources.py" --model layout "${source_args[@]}" ;;
    mask) python3 "$ROOT/scripts/fetch_sources.py" --model grounded_sam2 "${source_args[@]}" ;;
    flux) python3 "$ROOT/scripts/fetch_sources.py" --model flux "${source_args[@]}" ;;
    trellis) python3 "$ROOT/scripts/fetch_sources.py" --model trellis2 "${source_args[@]}" ;;
    refine) python3 "$ROOT/scripts/fetch_sources.py" --source nvdiffrast "${source_args[@]}" ;;
  esac
}

fetch_sources_for_selection

if selected layout; then create_env "$LAYOUT_ENV" "$PYTHON_VERSION"; fi
if selected mask; then create_env "$MASK_ENV" "$PYTHON_VERSION"; fi
if selected flux; then create_env "$FLUX_ENV" "$FLUX_PYTHON_VERSION"; fi
if selected trellis; then create_env "$TRELLIS_ENV" "$PYTHON_VERSION"; fi
if selected refine; then create_env "$REFINE_ENV" "$PYTHON_VERSION"; fi

if selected layout; then
  echo "[layout] install PyTorch and shared inference dependencies into $LAYOUT_ENV"
  install_torch_stack "$LAYOUT_ENV" "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "$LAYOUT_TORCH_INDEX"
  pip_in "$LAYOUT_ENV" install -r "$ROOT/requirements/layout.txt"

  require_source layout "$ROOT/third_party/src/utils3d-moge/pyproject.toml"
  require_source layout "$ROOT/third_party/src/MoGe/pyproject.toml"
  require_source layout "$ROOT/third_party/src/sam-3d-objects/pyproject.toml"
  pip_in "$LAYOUT_ENV" install --no-deps -e "$ROOT/third_party/src/utils3d-moge"
  # The layout runtime prepends the pinned MoGe and SAM3D roots itself. Their
  # package metadata includes optional Web UI, cloud, notebook, Blender-Python,
  # and development dependencies that are not used by Layout inference.

  install_cuda_extensions layout "$LAYOUT_ENV"
  pip_in "$LAYOUT_ENV" install -e "$ROOT" --no-deps
fi

if selected mask; then
  echo "[mask] install Grounded-SAM2 and GroundingDINO into $MASK_ENV"
  install_torch_stack "$MASK_ENV" "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "$LAYOUT_TORCH_INDEX"
  pip_in "$MASK_ENV" install -r "$ROOT/requirements/mask.txt"
  install_cuda_extensions mask "$MASK_ENV"
fi

if selected flux; then
  echo "[flux] install FLUX.2 runtime into $FLUX_ENV"
  install_torch_stack "$FLUX_ENV" "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "$FLUX_TORCH_INDEX"
  require_source flux "$ROOT/third_party/src/diffusers/src/diffusers/__init__.py"
  pip_in "$FLUX_ENV" install --no-deps -e "$ROOT/third_party/src/diffusers"
  pip_in "$FLUX_ENV" install -r "$ROOT/requirements/flux.txt"
fi

if selected trellis; then
  echo "[trellis] install TRELLIS.2 runtime and pinned CUDA extensions into $TRELLIS_ENV"
  install_torch_stack "$TRELLIS_ENV" "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "$TRELLIS_TORCH_INDEX"
  pip_in "$TRELLIS_ENV" install -r "$ROOT/requirements/trellis.txt"
  require_source trellis "$ROOT/third_party/src/TRELLIS.2/trellis2/pipelines/__init__.py"
  require_source trellis "$ROOT/third_party/src/utils3d-trellis/pyproject.toml"
  pip_in "$TRELLIS_ENV" install --no-deps -e "$ROOT/third_party/src/utils3d-trellis"
  install_cuda_extensions trellis "$TRELLIS_ENV"
fi

if selected refine; then
  echo "[refine] install post-refinement runtime into $REFINE_ENV"
  install_torch_stack "$REFINE_ENV" "torch==2.5.0+cu121" "torchvision==0.20.0+cu121" "$REFINE_TORCH_INDEX"
  pip_in "$REFINE_ENV" install -r "$ROOT/requirements/refine.txt"
  install_cuda_extensions refine "$REFINE_ENV"
fi

echo "[env] validate dependency metadata for every selected environment"
if selected layout; then pip_in "$LAYOUT_ENV" check; fi
if selected mask; then pip_in "$MASK_ENV" check; fi
if selected flux; then pip_in "$FLUX_ENV" check; fi
if selected trellis; then pip_in "$TRELLIS_ENV" check; fi
if selected refine; then pip_in "$REFINE_ENV" check; fi

if [[ "$DOWNLOAD_MODELS" == 1 ]]; then
  model_selection="$ONLY"
  case "$ONLY" in
    all) model_selection="all" ;;
    mask) model_selection="grounded_sam2" ;;
    trellis) model_selection="trellis2" ;;
    refine)
      model_selection=""
      echo "[model] refinement has no separate model artifact"
      ;;
  esac
  if [[ -n "$model_selection" ]]; then
    downloader_env="$LAYOUT_ENV"
    if ! env_exists "$downloader_env"; then
      case "$ONLY" in
        mask) downloader_env="$MASK_ENV" ;;
        flux) downloader_env="$FLUX_ENV" ;;
        trellis) downloader_env="$TRELLIS_ENV" ;;
      esac
    fi
    echo "[model] download configured selection: $model_selection"
    "$CONDA_BIN" run -n "$downloader_env" python "$ROOT/scripts/download_models.py" \
      --model "$model_selection" \
      --model-source "$MODEL_SOURCE" \
      --network-timeout "$NETWORK_TIMEOUT" \
      --skip-source-fetch
  fi
fi

cat <<EOF

Environment installation finished.

Selected role: $ONLY
Model download source: $MODEL_SOURCE
Environment names:
  layout:  $LAYOUT_ENV
  mask:    $MASK_ENV
  flux:    $FLUX_ENV
  trellis: $TRELLIS_ENV
  refine:  $REFINE_ENV

EOF

if [[ "$ALLOW_NO_GPU" == 1 ]]; then
  cat <<'EOF'
This was a dependency-only setup. CUDA extensions were not compiled, so the
selected environment is not ready for GPU inference. Re-run on a supported
CUDA host without --allow-no-gpu.
EOF
else
  cat <<'EOF'
After model downloads, validate the complete installation with:
  python scripts/preflight.py
EOF
fi
