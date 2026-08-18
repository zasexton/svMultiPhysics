#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FE_BUILD_DIR="${FE_BUILD_DIR:-${ROOT_DIR}/build-fe-check}"
PHYSICS_BUILD_DIR="${PHYSICS_BUILD_DIR:-${ROOT_DIR}/build-physics-check}"
JOBS="${JOBS:-2}"
PERF_SMOKE="${PERF_SMOKE:-1}"
PERF_GATE="${PERF_GATE:-0}"
PERF_MAX_JIT_OVER_AD_RATIO="${PERF_MAX_JIT_OVER_AD_RATIO:-1.50}"

require_build_dir() {
  local build_dir="$1"
  if [[ ! -d "${build_dir}" ]]; then
    echo "Missing build directory: ${build_dir}" >&2
    exit 1
  fi
}

run_ctest() {
  local build_dir="$1"
  local regex="$2"
  ctest --test-dir "${build_dir}" -R "${regex}" --output-on-failure
}

run_timed_gtest() {
  local label="$1"
  local output_file="$2"
  shift 2

  /usr/bin/time -f '%e %U %S' -o "${output_file}" "$@"
  read -r wall user sys < "${output_file}"
  echo "${label}: wall=${wall} user=${user} sys=${sys}"
}

require_build_dir "${FE_BUILD_DIR}"

cmake --build "${FE_BUILD_DIR}" \
  --target test_fe_forms test_fe_systems test_fe_assembly test_fe_movingmesh \
           test_fe_timestepping test_fe_mpi test_fe_assembly_mpi test_fe_movingmesh_mpi \
  -j "${JOBS}"

"${FE_BUILD_DIR}/test_fe_forms" \
  --gtest_filter='FormVocabularyTest.SymbolicCurrentGeometryTangentMatchesAD:FormVocabularyTest.SymbolicMeshVelocityGeometryTangentMatchesAD:FormVocabularyTest.SymbolicBoundaryGeometryTangentMatchesAD:FormVocabularyTest.SymbolicInteriorGeometryTangentMatchesAD:FormVocabularyTest.SymbolicCurrentGeometryTangentJITMatchesInterpreter:FormVocabularyTest.SymbolicBoundaryGeometryTangentJITMatchesInterpreter:FormVocabularyTest.SymbolicInteriorGeometryTangentJITMatchesInterpreter:FormVocabularyTest.SymbolicInterfaceGeometryTangentJITCompilesTangentDispatch'

"${FE_BUILD_DIR}/test_fe_systems" \
  --gtest_filter='FormsInstaller.MovingMeshSymbolicWithADCheckInstallsJITPrimaryAndPassesRuntimeReference:FormsInstaller.MovingMeshSymbolicJITAssemblyMatchesADReferenceAssembly:FESystemMortar.MovingMeshInterfaceSymbolicInterpreterMatchesADReference:FESystemMortar.MovingMeshInterfaceSymbolicJITMatchesADReference:FESystemMortar.MovingMeshInterfaceSymbolicWithADCheckUsesJITPrimary'

if [[ "${PERF_SMOKE}" != "0" ]]; then
  if command -v /usr/bin/time >/dev/null 2>&1; then
    ad_time_file="$(mktemp)"
    jit_time_file="$(mktemp)"
    trap 'rm -f "${ad_time_file:-}" "${jit_time_file:-}"' EXIT

    run_timed_gtest 'moving-mesh AD-vs-symbolic parity smoke' "${ad_time_file}" \
      "${FE_BUILD_DIR}/test_fe_forms" \
      --gtest_filter='FormVocabularyTest.SymbolicCurrentGeometryTangentMatchesAD:FormVocabularyTest.SymbolicMeshVelocityGeometryTangentMatchesAD:FormVocabularyTest.SymbolicBoundaryGeometryTangentMatchesAD:FormVocabularyTest.SymbolicInteriorGeometryTangentMatchesAD'

    run_timed_gtest 'moving-mesh symbolic-JIT parity smoke' "${jit_time_file}" \
      "${FE_BUILD_DIR}/test_fe_forms" \
      --gtest_filter='FormVocabularyTest.SymbolicCurrentGeometryTangentJITMatchesInterpreter:FormVocabularyTest.SymbolicBoundaryGeometryTangentJITMatchesInterpreter:FormVocabularyTest.SymbolicInteriorGeometryTangentJITMatchesInterpreter:FormVocabularyTest.SymbolicInterfaceGeometryTangentJITCompilesTangentDispatch'

    if [[ "${PERF_GATE}" != "0" ]]; then
      ad_wall="$(awk '{print $1}' "${ad_time_file}")"
      jit_wall="$(awk '{print $1}' "${jit_time_file}")"
      awk -v ad="${ad_wall}" -v jit="${jit_wall}" -v ratio="${PERF_MAX_JIT_OVER_AD_RATIO}" '
        BEGIN {
          limit = ad * ratio;
          if (jit > limit) {
            printf("symbolic-JIT smoke exceeded perf gate: jit=%g ad=%g ratio_limit=%g\n", jit, ad, ratio) > "/dev/stderr";
            exit 1;
          }
        }'
    fi
  fi
fi

"${FE_BUILD_DIR}/test_fe_systems" \
  --gtest_filter='*GeometryTangentPath*:*ALE*:*Symbolic*:FormsInstaller.MovingMesh*'

run_ctest "${FE_BUILD_DIR}" \
  'FE_Forms_Tests|FE_Systems_Tests|FE_MovingMesh_Tests|FE_Assembly_Tests|FE_TimeStepping_Tests'

run_ctest "${FE_BUILD_DIR}" \
  'test_fe_movingmesh_mpi_mpi_2|test_fe_assembly_mpi_mpi_2|test_fe_mpi_mpi_2|test_fe_assembly_mpi_mpi_4|test_fe_mpi_mpi_4'

if [[ -d "${PHYSICS_BUILD_DIR}" ]]; then
  cmake --build "${PHYSICS_BUILD_DIR}" --target test_physics -j "${JOBS}"

  "${PHYSICS_BUILD_DIR}/test_physics" \
    --gtest_filter='*MovingDomain*:*MovingMesh*:*NavierStokes*'

  run_ctest "${PHYSICS_BUILD_DIR}" 'Physics'
fi
