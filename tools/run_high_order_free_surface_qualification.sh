#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
SOLVER=${SVMULTIPHYSICS_EXECUTABLE:-"${REPO}/build/svMultiPhysics-build/bin/svmultiphysics"}
OUT_DIR=${HIGH_ORDER_QUALIFICATION_OUT_DIR:-"${REPO}/Documentation/qualification_logs"}
STAMP=${HIGH_ORDER_QUALIFICATION_STAMP:-$(date +%Y%m%d)}
SMOKE="${REPO}/tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py"

mkdir -p "${OUT_DIR}"

PRESERVE_ARGS=()
if [[ "${HIGH_ORDER_QUALIFICATION_PRESERVE_RUN_DIR:-0}" != "0" ]]; then
  PRESERVE_ARGS+=(--preserve-run-dir)
fi

run_gate() {
  local label=$1
  local json_name=$2
  shift 2
  local json_path="${OUT_DIR}/${json_name}"
  local stdout_path="${json_path%.json}.out"

  printf '[high-order qualification] running %s\n' "${label}"
  "${PYTHON_BIN}" "${SMOKE}" \
    --solver "${SOLVER}" \
    "$@" \
    --qualification-log "${json_path}" \
    "${PRESERVE_ARGS[@]}" \
    > "${stdout_path}" 2>&1
  printf '[high-order qualification] passed %s -> %s\n' "${label}" "${json_path}"
}

run_gate \
  "2D free-surface production gate" \
  "high_order_free_surface_production_gate_${STAMP}.json" \
  --high-order-production-qualification

run_gate \
  "2D capillary projected-curvature gate" \
  "high_order_capillary_projection_sloshing2d_smoke_${STAMP}.json" \
  --high-order-capillary-projection-smoke

run_gate \
  "2D volume-corrected free-surface motion gate" \
  "high_order_volume_corrected_sloshing2d_${STAMP}.json" \
  --high-order-volume-corrected-motion-smoke

run_gate \
  "2D visible free-surface motion demonstration gate" \
  "high_order_visible_motion_tilt2d_${STAMP}.json" \
  --high-order-visible-motion-demo

run_gate \
  "MPI2 free-surface motion gate" \
  "high_order_sloshing2d_mpi2_motion_smoke_${STAMP}.json" \
  --high-order-mpi-motion-smoke

run_gate \
  "MPI2 2D free-surface production gate" \
  "high_order_free_surface_mpi2_production_gate_${STAMP}.json" \
  --high-order-mpi-production-qualification

run_gate \
  "curved Tetra10 3D simplex solver gate" \
  "high_order_curved_tet3d_simplex_smoke_${STAMP}.json" \
  --high-order-curved-3d-simplex-smoke

run_gate \
  "D18/D38 3D benchmark qualification gate" \
  "high_order_d18_d38_3d_auto_subcell_fsils_qualification_${STAMP}.json" \
  --high-order-3d-benchmark-qualification

if [[ "${HIGH_ORDER_QUALIFICATION_INCLUDE_PROFILE:-0}" != "0" ]]; then
  run_gate \
    "D18 MPI2 first-profile 3D benchmark qualification gate" \
    "high_order_d18_3d_profile_mpi2_blockschur_${STAMP}.json" \
    --high-order-3d-benchmark-profile-qualification \
    --case d18 \
    --mpi-ranks 2

  run_gate \
    "D38 MPI2 first-profile 3D benchmark qualification gate" \
    "high_order_d38_3d_profile_mpi2_blockschur_${STAMP}.json" \
    --high-order-3d-benchmark-profile-qualification \
    --case d38 \
    --mpi-ranks 2
fi

printf '[high-order qualification] all gates passed\n'
