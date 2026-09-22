#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${1:-3d_future_test_0000000}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset_cases}"
FYSIVERSE_ROOT="${FYSIVERSE_ROOT:-${PROJECT_ROOT}/vendor/fysiverse_3d}"
BLENDER="${BLENDER:-blender}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-${FYSIVERSE_REFINE_ENV:-fysiverse-refine}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"

WORKERS="${WORKERS:-1}"
OPT_MAX_SIDE="${OPT_MAX_SIDE:-256}"
POSITION_STEPS="${POSITION_STEPS:-30}"
YAW_STEPS="${YAW_STEPS:-20}"
SCALE_STEPS="${SCALE_STEPS:-15}"
RENDER_SAMPLES="${RENDER_SAMPLES:-32}"

CASE_DIR="${DATASET_ROOT}/${CASE_NAME}"
WORK_DIR="${OUTPUT_ROOT}/${CASE_NAME}"

if [[ ! -d "${CASE_DIR}" ]]; then
  echo "Missing case directory: ${CASE_DIR}" >&2
  exit 1
fi

INPUT_IMAGE="${CASE_DIR}/input_rgb.png"
if [[ ! -f "${INPUT_IMAGE}" ]]; then
  INPUT_IMAGE="${CASE_DIR}/input_rgb.jpg"
fi
if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Missing input image under ${CASE_DIR}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
cp "${INPUT_IMAGE}" "${WORK_DIR}/observed.png"

echo "[1/10] prepare masks, manifest, fixed camera"
if [[ -n "${MASK_DIR_NAME:-}" ]]; then
  python3 "${PROJECT_ROOT}/scripts/prepare_trial_inputs.py" \
    --case-dir "${CASE_DIR}" \
    --output-dir "${WORK_DIR}" \
    --mask-dir-name "${MASK_DIR_NAME}"
else
  python3 "${PROJECT_ROOT}/scripts/prepare_trial_inputs.py" \
    --case-dir "${CASE_DIR}" \
    --output-dir "${WORK_DIR}"
fi

echo "[2/10] import dataset GLB"
"${BLENDER}" -b --python "${PROJECT_ROOT}/scripts/blender_import_dataset_glb.py" -- \
  --case-dir "${CASE_DIR}" \
  --output-blend "${WORK_DIR}/imported.blend" \
  --report "${WORK_DIR}/import_report.json"

echo "[3/10] export nvdiffrast mesh"
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/scripts/blender_export_nvdiffrast_mesh.py" -- \
  --input "${WORK_DIR}/imported.blend" \
  --output "${WORK_DIR}/object_mesh.npz"

echo "[4/10] optimize object RTS"
CONDA_SELECTOR=(-n "${CONDA_ENV}")
case "${CONDA_ENV}" in
  /*|./*|../*|*/*)
    CONDA_SELECTOR=(-p "$(realpath -m "${CONDA_ENV}")")
    ;;
esac
"${CONDA_BIN}" run "${CONDA_SELECTOR[@]}" python "${FYSIVERSE_ROOT}/scripts/optimize_object_poses_nvdiffrast.py" \
  --mesh "${WORK_DIR}/object_mesh.npz" \
  --target-mask "${WORK_DIR}/label_mask.png" \
  --final-manifest "${WORK_DIR}/alignment_manifest.json" \
  --output-json "${WORK_DIR}/object_pose_optimization.json" \
  --preview-dir "${WORK_DIR}/object_pose_previews" \
  --input-image "${INPUT_IMAGE}" \
  --camera-optimization "${WORK_DIR}/fixed_camera_optimization.json" \
  --workers "${WORKERS}" \
  --max-side "${OPT_MAX_SIDE}" \
  --position-steps "${POSITION_STEPS}" \
  --yaw-steps "${YAW_STEPS}" \
  --scale-steps "${SCALE_STEPS}" \
  --translation-initial-grid 3 \
  --yaw-initial-samples 3 \
  --scale-initial-samples 3 \
  --phase-accept-iou-drop 0.02 \
  --max-translation 0.25 \
  --max-yaw-deg 25 \
  --max-scale-delta 0.15 \
  --log-every 10

echo "[5/10] apply object RTS"
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/scripts/blender_apply_object_pose_optimization.py" -- \
  --input "${WORK_DIR}/imported.blend" \
  --object-pose-optimization "${WORK_DIR}/object_pose_optimization.json" \
  --output "${WORK_DIR}/optimized.blend"

echo "[6/10] apply ground snap and convex separation"
python3 "${PROJECT_ROOT}/scripts/make_separation_plan.py" \
  --output "${WORK_DIR}/separation_plan.json"
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/scripts/blender_apply_moge_stage.py" -- \
  --input "${WORK_DIR}/optimized.blend" \
  --plan "${WORK_DIR}/separation_plan.json" \
  --output "${WORK_DIR}/separated.blend" \
  --report "${WORK_DIR}/separation_report.json"
"${BLENDER}" -b --python "${PROJECT_ROOT}/scripts/blender_scene_summary.py" -- \
  --input "${WORK_DIR}/separated.blend" \
  --output "${WORK_DIR}/separated_summary.json"

echo "[7/10] render before and after metric views"
mkdir -p "${WORK_DIR}/metrics_before" "${WORK_DIR}/metrics_after"
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/experiment/quantitative/blender_render_metrics.py" -- \
  --input "${WORK_DIR}/imported.blend" \
  --camera-optimization "${WORK_DIR}/fixed_camera_optimization.json" \
  --rgb-output "${WORK_DIR}/metrics_before/rgb.png" \
  --mask-dir "${WORK_DIR}/metrics_before/masks" \
  --report "${WORK_DIR}/metrics_before/render_report.json" \
  --samples "${RENDER_SAMPLES}" \
  --compute-backend CUDA
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/experiment/quantitative/blender_render_metrics.py" -- \
  --input "${WORK_DIR}/separated.blend" \
  --camera-optimization "${WORK_DIR}/fixed_camera_optimization.json" \
  --rgb-output "${WORK_DIR}/metrics_after/rgb.png" \
  --mask-dir "${WORK_DIR}/metrics_after/masks" \
  --report "${WORK_DIR}/metrics_after/render_report.json" \
  --samples "${RENDER_SAMPLES}" \
  --compute-backend CUDA

echo "[8/10] compute penetration metrics"
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/experiment/quantitative/blender_penetration_metric.py" -- \
  --input "${WORK_DIR}/imported.blend" \
  --output "${WORK_DIR}/metrics_before/penetration_metrics.json" \
  --target-edge-ratio 0.05
"${BLENDER}" -b --python "${FYSIVERSE_ROOT}/experiment/quantitative/blender_penetration_metric.py" -- \
  --input "${WORK_DIR}/separated.blend" \
  --output "${WORK_DIR}/metrics_after/penetration_metrics.json" \
  --target-edge-ratio 0.05

echo "[9/10] compute render mask mIoU"
python3 "${PROJECT_ROOT}/scripts/compute_mask_iou.py" \
  --label-mask "${WORK_DIR}/label_mask.png" \
  --before-mask-dir "${WORK_DIR}/metrics_before/masks" \
  --after-mask-dir "${WORK_DIR}/metrics_after/masks" \
  --before-penetration "${WORK_DIR}/metrics_before/penetration_metrics.json" \
  --after-penetration "${WORK_DIR}/metrics_after/penetration_metrics.json" \
  --output "${WORK_DIR}/metrics_summary.json"

echo "[10/10] done"
python3 - "${WORK_DIR}/metrics_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
for split in ("before", "after"):
    print(
        f"{split}: mIoU={data[split]['mIoU']:.4f}, "
        f"R_pen={data[split]['penetration']['r_pen']:.4f}"
    )
print(f"summary: {path}")
PY
