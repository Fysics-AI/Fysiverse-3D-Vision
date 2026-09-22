#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
PYPI_INDEX="${FYSIVERSE_PYPI_INDEX:-https://pypi.org/simple}"
PIP_TIMEOUT="${FYSIVERSE_PIP_TIMEOUT:-120}"
PIP_RETRIES="${FYSIVERSE_PIP_RETRIES:-5}"
PIP_CACHE_DIR="${FYSIVERSE_PIP_CACHE_DIR:-$ROOT/.cache/pip}"
CUDA_CONSTRAINTS="${FYSIVERSE_CUDA_CONSTRAINTS:-$ROOT/configs/constraints-cu-runtime.txt}"
TRELLIS_CONSTRAINTS="${FYSIVERSE_TRELLIS_CONSTRAINTS:-$ROOT/configs/constraints-trellis2-cu124.txt}"
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"
FLASH_ATTN_CUDA_ARCHS="${FYSIVERSE_FLASH_ATTN_CUDA_ARCHS:-}"
GROUNDINGDINO_ARCH_PATCH="${FYSIVERSE_GROUNDINGDINO_ARCH_PATCH:-$ROOT/patches/groundingdino-torch-cuda-arch-list.patch}"
MIN_FREE_GB="${FYSIVERSE_MIN_FREE_GB_FOR_ENV:-30}"
MIN_CUDA="${FYSIVERSE_MIN_CUDA:-12.4}"
MAX_JOBS="${MAX_JOBS:-8}"
export MAX_JOBS

ROLE=""
ENV_NAME=""
ALLOW_NO_GPU=0
SKIP_BLENDER=0
CONFIRM_NEW_ENV=0

usage() {
  cat <<'EOF'
Usage: scripts/install_cuda_extensions.sh --role ROLE --env NAME --target-is-new [options]

Install the source-built backend/CUDA packages for one newly created
Fysiverse environment. This script never creates, deletes, repairs, or updates
an environment implicitly; --target-is-new is required as an explicit guard.

Roles:
  layout   PyTorch3D, gsplat, flash-attn, Kaolin, and xformers
  mask     Grounded-SAM2 and GroundingDINO source packages/CUDA operators
  trellis  flash-attn, CuMesh, FlexGEMM, o-voxel, nvdiffrast, nvdiffrec
  refine   nvdiffrast

Options:
  --allow-no-gpu  Install only the CPU-capable Mask source packages; skip CUDA
                  compilation for Layout, TRELLIS.2, and Refine.
  --skip-blender  Skip Blender validation for dependency-only preparation.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) [[ $# -ge 2 ]] || { echo "--role requires a value" >&2; exit 2; }; ROLE="$2"; shift 2 ;;
    --env) [[ $# -ge 2 ]] || { echo "--env requires a value" >&2; exit 2; }; ENV_NAME="$2"; shift 2 ;;
    --target-is-new) CONFIRM_NEW_ENV=1; shift ;;
    --allow-no-gpu) ALLOW_NO_GPU=1; shift ;;
    --skip-blender) SKIP_BLENDER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ROLE" in layout|mask|trellis|refine) ;; *) echo "Invalid --role: $ROLE" >&2; usage >&2; exit 2 ;; esac
[[ -n "$ENV_NAME" ]] || { echo "--env is required" >&2; exit 2; }
[[ "$CONFIRM_NEW_ENV" == 1 ]] || {
  echo "Refusing to modify a Conda environment without --target-is-new." >&2
  echo "Create an unused environment name first; never point this helper at an existing working environment." >&2
  exit 1
}
command -v "$CONDA_BIN" >/dev/null || { echo "conda was not found; set CONDA_BIN" >&2; exit 1; }
"$CONDA_BIN" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME" || {
  echo "Conda environment does not exist: $ENV_NAME" >&2
  exit 1
}

system_check_args=(
  --path "$ROOT" --min-free-gb "$MIN_FREE_GB" --min-cuda "$MIN_CUDA"
  --conda-bin "$CONDA_BIN"
)
if [[ "$ALLOW_NO_GPU" == 1 ]]; then
  system_check_args+=(--allow-no-gpu)
else
  system_check_args+=(--require-nvcc)
fi
if [[ "$ROLE" == "refine" && "$SKIP_BLENDER" != 1 ]]; then
  system_check_args+=(--blender "${POST_REFINE_BLENDER:-blender}")
else
  system_check_args+=(--skip-blender)
fi
python3 "$ROOT/scripts/check_system.py" "${system_check_args[@]}"

pip_in() {
  local constraint_file=""
  local -a pip_environment=(
    PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL="$PYPI_INDEX"
    PIP_EXTRA_INDEX_URL= PIP_TRUSTED_HOST=
    PIP_CACHE_DIR="$PIP_CACHE_DIR"
    PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" PIP_RETRIES="$PIP_RETRIES"
  )
  case "$ROLE" in
    layout|mask) constraint_file="$CUDA_CONSTRAINTS" ;;
    trellis) constraint_file="$TRELLIS_CONSTRAINTS" ;;
  esac
  if [[ -n "$constraint_file" ]]; then
    [[ -f "$constraint_file" ]] || { echo "constraint file not found: $constraint_file" >&2; exit 1; }
    pip_environment+=(PIP_CONSTRAINT="$constraint_file")
  fi
  mkdir -p "$PIP_CACHE_DIR"
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" env \
    "${pip_environment[@]}" python -m pip "$@"
}

require_source() {
  [[ -e "$2" ]] || {
    echo "[$ROLE] required pinned source path is missing: $2" >&2
    echo "Run scripts/fetch_sources.sh or follow the manual source commands in README.md." >&2
    exit 1
  }
}

cuda_archs_in() {
  "$CONDA_BIN" run -n "$ENV_NAME" python -c \
    'import torch; print(";".join(sorted({f"{major}{minor}" for major, minor in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))})))'
}

torch_cuda_archs_in() {
  "$CONDA_BIN" run -n "$ENV_NAME" python -c \
    'import torch; print(";".join(sorted({f"{major}.{minor}" for major, minor in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))})))'
}

verify_groundingdino_cuda() {
  local source_root="$ROOT/third_party/src/Grounded-SAM-2"
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" env \
    PYTHONPATH="$source_root" python -c \
    'import torch; from grounding_dino.groundingdino import _C; d="cuda"; value=torch.ones((1,1,1,1),device=d); shapes=torch.tensor([[1,1]],dtype=torch.long,device=d); starts=torch.tensor([0],dtype=torch.long,device=d); locations=torch.full((1,1,1,1,1,2),0.5,device=d); weights=torch.ones((1,1,1,1,1),device=d); output=_C.ms_deform_attn_forward(value,shapes,starts,locations,weights,1); assert output.shape == (1,1,1) and torch.allclose(output,torch.ones_like(output)); print(f"[mask] GroundingDINO CUDA operator ready on sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")'
}

case "$ROLE" in
  layout)
    require_source pytorch3d "$ROOT/third_party/src/pytorch3d/setup.py"
    require_source gsplat "$ROOT/third_party/src/gsplat/setup.py"
    if [[ "$ALLOW_NO_GPU" == 1 ]]; then
      echo "[layout] CUDA extensions skipped by --allow-no-gpu" >&2
    else
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/pytorch3d"
      pip_in install "kaolin==0.17.0" -f "$KAOLIN_FIND_LINKS"
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/gsplat"
      pip_in install "flash-attn==2.8.3" --no-build-isolation --no-deps
      pip_in install "xformers==0.0.28.post3" --no-deps
    fi
    ;;
  mask)
    require_source grounded_sam2 "$ROOT/third_party/src/Grounded-SAM-2/setup.py"
    require_source grounding_dino "$ROOT/third_party/src/Grounded-SAM-2/grounding_dino/setup.py"
    if [[ "$ALLOW_NO_GPU" == 1 ]]; then
      export SAM2_BUILD_CUDA=0
    else
      export SAM2_BUILD_ALLOW_ERRORS=0
    fi
    pip_in install --no-build-isolation -e "$ROOT/third_party/src/Grounded-SAM-2"
    if [[ "$ALLOW_NO_GPU" == 1 ]]; then
      pip_in install --no-build-isolation -e "$ROOT/third_party/src/Grounded-SAM-2/grounding_dino"
    else
      mask_cuda_archs="$(torch_cuda_archs_in)"
      [[ -n "$mask_cuda_archs" ]] || { echo "[mask] cannot detect a CUDA architecture" >&2; exit 1; }
      [[ -f "$GROUNDINGDINO_ARCH_PATCH" ]] || {
        echo "[mask] GroundingDINO CUDA architecture patch not found: $GROUNDINGDINO_ARCH_PATCH" >&2
        exit 1
      }
      echo "[mask] GroundingDINO CUDA architectures: $mask_cuda_archs"
      grounding_setup="$ROOT/third_party/src/Grounded-SAM-2/grounding_dino/setup.py"
      grounding_setup_backup="$(mktemp "${TMPDIR:-/tmp}/groundingdino-setup.XXXXXX")"
      cp "$grounding_setup" "$grounding_setup_backup"
      restore_grounding_setup() {
        cp "$grounding_setup_backup" "$grounding_setup"
        rm -f -- "$grounding_setup_backup"
      }
      trap restore_grounding_setup EXIT
      git -C "$ROOT/third_party/src/Grounded-SAM-2" apply --check "$GROUNDINGDINO_ARCH_PATCH"
      git -C "$ROOT/third_party/src/Grounded-SAM-2" apply "$GROUNDINGDINO_ARCH_PATCH"
      for artifact in "$ROOT"/third_party/src/Grounded-SAM-2/grounding_dino/groundingdino/_C*.so; do
        [[ -e "$artifact" ]] && rm -f -- "$artifact"
      done
      TORCH_CUDA_ARCH_LIST="$mask_cuda_archs" \
        pip_in install --force-reinstall --no-build-isolation --no-deps \
          -e "$ROOT/third_party/src/Grounded-SAM-2/grounding_dino"
      restore_grounding_setup
      trap - EXIT
      verify_groundingdino_cuda
    fi
    ;;
  trellis)
    require_source trellis "$ROOT/third_party/src/TRELLIS.2/o-voxel/setup.py"
    require_source cumesh "$ROOT/third_party/src/CuMesh/setup.py"
    require_source flexgemm "$ROOT/third_party/src/FlexGEMM/setup.py"
    require_source nvdiffrast "$ROOT/third_party/src/nvdiffrast/setup.py"
    require_source nvdiffrec "$ROOT/third_party/src/nvdiffrec/setup.py"
    if [[ "$ALLOW_NO_GPU" == 1 ]]; then
      echo "[trellis] CUDA extensions skipped by --allow-no-gpu" >&2
    else
      trellis_flash_archs="$FLASH_ATTN_CUDA_ARCHS"
      if [[ -z "$trellis_flash_archs" ]]; then trellis_flash_archs="$(cuda_archs_in)"; fi
      [[ -n "$trellis_flash_archs" ]] || { echo "[trellis] cannot detect a CUDA architecture" >&2; exit 1; }
      echo "[trellis] flash-attn CUDA architectures: $trellis_flash_archs"
      FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="$trellis_flash_archs" \
        pip_in install "flash-attn==2.7.3" --no-build-isolation --no-deps
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/CuMesh"
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/FlexGEMM"
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/TRELLIS.2/o-voxel"
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/nvdiffrast"
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/nvdiffrec"
    fi
    ;;
  refine)
    require_source nvdiffrast "$ROOT/third_party/src/nvdiffrast/setup.py"
    if [[ "$ALLOW_NO_GPU" == 1 ]]; then
      echo "[refine] nvdiffrast compilation skipped by --allow-no-gpu" >&2
    else
      pip_in install --no-build-isolation --no-deps -e "$ROOT/third_party/src/nvdiffrast"
    fi
    ;;
esac

echo "[$ROLE] source-built backend/CUDA installation finished in $ENV_NAME"
