#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage: tools/run_phase14_moving_mesh_validation.sh [--skip-mpi] [--skip-physics] [--dry-run]

Runs the Phase 14 moving-mesh validation evidence checks and writes logs under:
  Documentation/qualification_logs/phase14_moving_mesh/latest/

Environment overrides:
  FE_BUILD_DIR       Default: build-fe-check
  MESH_BUILD_DIR     Default: build-mesh-tests
  PHYSICS_BUILD_DIR  Default: build-physics-gcc13-check
  JOBS               Default: 4
  PHASE14_OUT_DIR    Default: Documentation/qualification_logs/phase14_moving_mesh
EOF
}

shell_quote() {
  printf "%q" "$1"
}

md_escape() {
  local text="$1"
  text="${text//|/\\|}"
  printf "%s" "$text"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if repo_root="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null)"; then
  REPO="$repo_root"
else
  REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

FE_BUILD_DIR="${FE_BUILD_DIR:-build-fe-check}"
MESH_BUILD_DIR="${MESH_BUILD_DIR:-build-mesh-tests}"
PHYSICS_BUILD_DIR="${PHYSICS_BUILD_DIR:-build-physics-gcc13-check}"
JOBS="${JOBS:-4}"
OUT_BASE="${PHASE14_OUT_DIR:-$REPO/Documentation/qualification_logs/phase14_moving_mesh}"
OUT_DIR="$OUT_BASE/latest"
LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.md"

SKIP_MPI=0
SKIP_PHYSICS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-mpi)
      SKIP_MPI=1
      shift
      ;;
    --skip-physics)
      SKIP_PHYSICS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

rm -rf "$OUT_DIR"
mkdir -p "$LOG_DIR"

cat > "$SUMMARY" <<EOF
# Phase 14 Moving-Mesh Validation Evidence

- Generated: $(date --iso-8601=seconds)
- Repository: $REPO
- FE build: $FE_BUILD_DIR
- Mesh build: $MESH_BUILD_DIR
- Physics build: $PHYSICS_BUILD_DIR
- Jobs: $JOBS
- MPI checks: $(if [[ "$SKIP_MPI" == "1" ]]; then echo "skipped"; else echo "enabled"; fi)
- Physics checks: $(if [[ "$SKIP_PHYSICS" == "1" ]]; then echo "skipped"; else echo "enabled"; fi)
- Dry run: $(if [[ "$DRY_RUN" == "1" ]]; then echo "yes"; else echo "no"; fi)

## Results

EOF

FAILURES=0

record_skip() {
  local name="$1"
  local reason="$2"
  {
    echo "- SKIP \`$name\`: $reason"
    echo
  } >> "$SUMMARY"
  echo "SKIP $name: $reason"
}

run_check() {
  local name="$1"
  local description="$2"
  local command="$3"
  local log_file="$LOG_DIR/$name.log"
  local rc=0

  echo "RUN  $name"
  echo "$command" > "$log_file"
  echo >> "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    {
      echo "- DRY-RUN \`$name\`: $description"
      echo "  - Command: \`$(md_escape "$command")\`"
      echo "  - Log: \`logs/$name.log\`"
      echo
    } >> "$SUMMARY"
    return 0
  fi

  bash -lc "cd $(shell_quote "$REPO") && $command" >> "$log_file" 2>&1
  rc=$?

  if [[ "$rc" -eq 0 ]]; then
    {
      echo "- PASS \`$name\`: $description"
      echo "  - Command: \`$(md_escape "$command")\`"
      echo "  - Log: \`logs/$name.log\`"
      echo
    } >> "$SUMMARY"
  else
    FAILURES=$((FAILURES + 1))
    {
      echo "- FAIL \`$name\`: $description"
      echo "  - Exit code: $rc"
      echo "  - Command: \`$(md_escape "$command")\`"
      echo "  - Log: \`logs/$name.log\`"
      echo
    } >> "$SUMMARY"
  fi

  return 0
}

FE_SYSTEMS_BIN="$FE_BUILD_DIR/test_fe_systems"
FE_GEOMETRY_BIN="$FE_BUILD_DIR/test_fe_geometry"
FE_TIMESTEPPING_BIN="$FE_BUILD_DIR/test_fe_timestepping"
FE_FORMS_BIN="$FE_BUILD_DIR/test_fe_forms"
PHYSICS_BIN="$PHYSICS_BUILD_DIR/test_physics"

run_check \
  "build_fe_phase14" \
  "Build FE targets that exercise Phase 14 moving-mesh infrastructure evidence." \
  "cmake --build $(shell_quote "$FE_BUILD_DIR") --target test_fe_geometry test_fe_timestepping test_fe_forms test_fe_systems -j $(shell_quote "$JOBS")"

if [[ "$SKIP_PHYSICS" == "1" ]]; then
  record_skip "build_physics_moving_domain" "physics checks disabled by --skip-physics"
else
  run_check \
    "build_physics_moving_domain" \
    "Build focused moving-domain physics validation target." \
    "cmake --build $(shell_quote "$PHYSICS_BUILD_DIR") --target test_physics -j $(shell_quote "$JOBS")"
fi

run_check \
  "fe_phase14_focused" \
  "Focused current-geometry assembly and matrix-free revision tracking tests added for Phase 14." \
  "$(shell_quote "$FE_SYSTEMS_BIN") --gtest_filter='FESystem.PrescribedMovingMeshVectorMassRespectsCurrentGeometry:OperatorBackends.MatrixFreeMassTracksCurrentGeometryRevisionWithoutRefetch' --gtest_color=no"

run_check \
  "fe_systems_broad" \
  "Broader affected FE systems qualification covering moving geometry, restart, adaptivity, search, operators, and contact kernels." \
  "$(shell_quote "$FE_SYSTEMS_BIN") --gtest_filter='FESystem.*:FEAdaptivityTransfer.*:FEMovingMeshRestart.*:OperatorBackends.*:SearchAccess.MeshSearchAccess_*:ContactPenaltyKernel.*:SurfaceContactKernel.*' --gtest_color=no"

run_check \
  "fe_geometry_all" \
  "Full FE geometry test suite, including frame geometry and moving-domain geometry utilities." \
  "$(shell_quote "$FE_GEOMETRY_BIN") --gtest_color=no"

run_check \
  "fe_timestepping_all" \
  "Full FE time-stepping test suite, including trial/accepted/rollback state coverage available in this build." \
  "$(shell_quote "$FE_TIMESTEPPING_BIN") --gtest_color=no"

run_check \
  "fe_forms_moving_domain" \
  "Forms vocabulary tests for moving-domain required data, frame-aware terms, and lowering hooks." \
  "$(shell_quote "$FE_FORMS_BIN") --gtest_filter='FormVocabularyTest.*' --gtest_color=no"

if [[ "$SKIP_PHYSICS" == "1" ]]; then
  record_skip "physics_moving_domain" "physics checks disabled by --skip-physics"
else
  run_check \
    "physics_moving_domain" \
    "Focused moving-domain physics term checks for ALE advection, Navier-Stokes, and FSI-style interface motion." \
    "$(shell_quote "$PHYSICS_BIN") --gtest_filter='MovingDomainPhysics.*' --gtest_color=no"
fi

if [[ "$SKIP_MPI" == "1" ]]; then
  record_skip "fe_mpi_ctest" "MPI checks disabled by --skip-mpi"
  record_skip "mesh_mpi_ctest" "MPI checks disabled by --skip-mpi"
else
  run_check \
    "fe_mpi_ctest" \
    "Representative FE MPI tests, including moving-mesh backend MPI coverage." \
    "ctest --test-dir $(shell_quote "$FE_BUILD_DIR") --output-on-failure --timeout 300 -j1 -L MPI"

  run_check \
    "mesh_mpi_ctest" \
    "Representative Mesh MPI tests covering current coordinates, motion, migration, restart, repartition, and distributed semantics." \
    "ctest --test-dir $(shell_quote "$MESH_BUILD_DIR") --output-on-failure --timeout 300 -j1 -R 'MPI|_4ranks|GhostCoordinateExchange|DistributedSemantics|RebalanceParMetis|Migration|PVTU|StartupParMetis|PartitionQualityMetis|MotionMPI|MovingAdaptivity|MovingMeshRestart'"
fi

cat >> "$SUMMARY" <<EOF
## Conclusion

$(if [[ "$FAILURES" -eq 0 ]]; then echo "All enabled checks passed."; else echo "$FAILURES enabled check(s) failed. Inspect the referenced logs before marking Phase 14 evidence complete."; fi)
EOF

echo
echo "summary=$SUMMARY"
echo "logs=$LOG_DIR"

if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi
