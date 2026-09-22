#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_PYTHON="${POST_REFINE_ADAPTER_PYTHON:-python3}"
BLENDER="${POST_REFINE_BLENDER:-blender}"
CONDA_BIN="${POST_REFINE_CONDA_BIN:-conda}"
CONDA_ENV="${POST_REFINE_CONDA_ENV:-${FYSIVERSE_REFINE_ENV:-fysiverse-refine}}"

if [[ -n "${POST_REFINE_PROJECT_ROOT:-}" ]]; then
  DEFAULT_RUNNER="$POST_REFINE_PROJECT_ROOT/run_case.sh"
  DEFAULT_FYSIVERSE_ROOT="$POST_REFINE_PROJECT_ROOT/vendor/fysiverse_3d"
else
  DEFAULT_RUNNER="$SCRIPT_ROOT/run_case_local.sh"
  DEFAULT_FYSIVERSE_ROOT="$SCRIPT_ROOT/vendor/fysiverse_3d"
fi
REFINER_RUNNER="${POST_REFINE_RUNNER:-$DEFAULT_RUNNER}"
FYSIVERSE_ROOT="${POST_REFINE_FYSIVERSE_ROOT:-$DEFAULT_FYSIVERSE_ROOT}"

PIPELINE_MANIFEST=""
SCENE_GLB=""
IMAGE=""
MASK_DIR=""
CAMERA_JSON=""
STAGE_JSON=""
CASE_NAME=""
OUTPUT_ROOT=""
FINAL_GLB=""
OVERWRITE=0
ALLOW_PREVIEW_CAMERA=0
COPY_MODE="${POST_REFINE_COPY_MODE:-hardlink}"

WORKERS="${POST_REFINE_WORKERS:-1}"
OPT_MAX_SIDE="${POST_REFINE_OPT_MAX_SIDE:-256}"
POSITION_STEPS="${POST_REFINE_POSITION_STEPS:-30}"
YAW_STEPS="${POST_REFINE_YAW_STEPS:-20}"
SCALE_STEPS="${POST_REFINE_SCALE_STEPS:-15}"
RENDER_SAMPLES="${POST_REFINE_RENDER_SAMPLES:-32}"
EXPORT_NORMALS="${POST_REFINE_EXPORT_NORMALS:-0}"

usage() {
  cat <<'EOF'
Usage: run.sh --case-name NAME --camera-json PATH --output-root DIR --final-glb PATH [inputs]

Inputs:
  --pipeline-manifest PATH          FysicsMagic OOD manifest (preferred)
  --scene-glb PATH --image PATH --mask-dir DIR
                                    Direct input mode
  --stage-json PATH                 Optional categories/bboxes sidecar

Controls:
  --overwrite                      Reset this task's case/run directories
  --allow-preview-camera           Operational test only; not an accuracy result
  --copy-mode hardlink|copy

Environment:
  POST_REFINE_EXPORT_NORMALS=0     Omit explicit normals to avoid flat-shaded vertex expansion
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline-manifest) PIPELINE_MANIFEST="$2"; shift 2 ;;
    --scene-glb) SCENE_GLB="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --mask-dir) MASK_DIR="$2"; shift 2 ;;
    --camera-json) CAMERA_JSON="$2"; shift 2 ;;
    --stage-json) STAGE_JSON="$2"; shift 2 ;;
    --case-name) CASE_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --final-glb) FINAL_GLB="$2"; shift 2 ;;
    --copy-mode) COPY_MODE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --allow-preview-camera) ALLOW_PREVIEW_CAMERA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CASE_NAME" || -z "$CAMERA_JSON" || -z "$OUTPUT_ROOT" || -z "$FINAL_GLB" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "$PIPELINE_MANIFEST" && ( -z "$SCENE_GLB" || -z "$IMAGE" || -z "$MASK_DIR" ) ]]; then
  echo "Provide --pipeline-manifest or all of --scene-glb/--image/--mask-dir" >&2
  exit 2
fi
if [[ "$COPY_MODE" != "hardlink" && "$COPY_MODE" != "copy" ]]; then
  echo "Unsupported copy mode: $COPY_MODE" >&2
  exit 2
fi
case "$EXPORT_NORMALS" in
  0|false|FALSE|no|NO|off|OFF) EXPORT_NORMALS=0 ;;
  1|true|TRUE|yes|YES|on|ON) EXPORT_NORMALS=1 ;;
  *) echo "POST_REFINE_EXPORT_NORMALS must be a boolean, got: $EXPORT_NORMALS" >&2; exit 2 ;;
esac
for path in "$SCRIPT_ROOT/post_refine_adapter.py" "$SCRIPT_ROOT/export_refined_glb.py" "$REFINER_RUNNER" "$FYSIVERSE_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required post-refine dependency is missing: $path" >&2
    exit 1
  fi
done
command -v "$BLENDER" >/dev/null 2>&1 || { echo "Blender executable not found: $BLENDER" >&2; exit 1; }
command -v "$CONDA_BIN" >/dev/null 2>&1 || { echo "Conda executable not found: $CONDA_BIN" >&2; exit 1; }

OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
FINAL_GLB="$(realpath -m "$FINAL_GLB")"
CASE_INPUT_ROOT="$OUTPUT_ROOT/case_inputs"
CASE_DIR="$CASE_INPUT_ROOT/$CASE_NAME"
RUNS_ROOT="$OUTPUT_ROOT/runs"
RUN_DIR="$RUNS_ROOT/$CASE_NAME"
INTEGRATION_MANIFEST="$CASE_DIR/integration_manifest.json"
EXPORT_REPORT="$OUTPUT_ROOT/refined_glb_export.json"
SUMMARY="$OUTPUT_ROOT/post_refine_summary.json"
LOG="$OUTPUT_ROOT/post_refine.log"
mkdir -p "$OUTPUT_ROOT"

prepare_args=(
  "$SCRIPT_ROOT/post_refine_adapter.py" prepare
  --case-name "$CASE_NAME"
  --case-dir "$CASE_DIR"
  --run-dir "$RUN_DIR"
  --camera-json "$CAMERA_JSON"
  --copy-mode "$COPY_MODE"
)
if [[ -n "$PIPELINE_MANIFEST" ]]; then prepare_args+=(--pipeline-manifest "$PIPELINE_MANIFEST"); fi
if [[ -n "$SCENE_GLB" ]]; then prepare_args+=(--scene-glb "$SCENE_GLB"); fi
if [[ -n "$IMAGE" ]]; then prepare_args+=(--image "$IMAGE"); fi
if [[ -n "$MASK_DIR" ]]; then prepare_args+=(--mask-dir "$MASK_DIR"); fi
if [[ -n "$STAGE_JSON" ]]; then prepare_args+=(--stage-json "$STAGE_JSON"); fi
if [[ "$OVERWRITE" == "1" ]]; then prepare_args+=(--overwrite); fi
if [[ "$ALLOW_PREVIEW_CAMERA" == "1" ]]; then prepare_args+=(--allow-preview-camera); fi

run_pipeline() {
  echo "[post-refine] prepare case=$CASE_NAME"
  "$ADAPTER_PYTHON" "${prepare_args[@]}"

  echo "[post-refine] run Fysivese refinement via $REFINER_RUNNER -> $RUN_DIR"
  DATASET_ROOT="$CASE_INPUT_ROOT" \
  FYSIVERSE_ROOT="$FYSIVERSE_ROOT" \
  BLENDER="$BLENDER" \
  CONDA_BIN="$CONDA_BIN" \
  CONDA_ENV="$CONDA_ENV" \
  OUTPUT_ROOT="$RUNS_ROOT" \
  WORKERS="$WORKERS" \
  OPT_MAX_SIDE="$OPT_MAX_SIDE" \
  POSITION_STEPS="$POSITION_STEPS" \
  YAW_STEPS="$YAW_STEPS" \
  SCALE_STEPS="$SCALE_STEPS" \
  RENDER_SAMPLES="$RENDER_SAMPLES" \
    bash "$REFINER_RUNNER" "$CASE_NAME"

  echo "[post-refine] export refined GLB -> $FINAL_GLB"
  export_args=(
    --input-blend "$RUN_DIR/separated.blend"
    --output-glb "$FINAL_GLB"
    --report "$EXPORT_REPORT"
  )
  if [[ "$EXPORT_NORMALS" == "1" ]]; then export_args+=(--export-normals); fi
  "$BLENDER" -b --python-exit-code 1 --python "$SCRIPT_ROOT/export_refined_glb.py" -- "${export_args[@]}"

  "$ADAPTER_PYTHON" "$SCRIPT_ROOT/post_refine_adapter.py" finalize \
    --integration-manifest "$INTEGRATION_MANIFEST" \
    --run-dir "$RUN_DIR" \
    --refined-glb "$FINAL_GLB" \
    --export-report "$EXPORT_REPORT" \
    --output "$SUMMARY"
  echo "[post-refine] done: $FINAL_GLB"
  echo "[post-refine] summary: $SUMMARY"
}

run_pipeline 2>&1 | tee "$LOG"
