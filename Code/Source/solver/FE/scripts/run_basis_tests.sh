#!/bin/bash
#
# Canonical helper for rebuilding and running the FE Basis unit-test suite.
#

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../../../../.." && pwd )"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-unit/svMultiPhysics-build}"

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "Configuring build directory: ${BUILD_DIR}"
    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
fi

if [ "${1:-}" = "--clean" ]; then
    echo "Cleaning build directory target graph"
    cmake --build "${BUILD_DIR}" --target clean
fi

echo "Building test_fe_basis"
cmake --build "${BUILD_DIR}" --target test_fe_basis -j"$(nproc)"

echo "Listing basis tests"
"${BUILD_DIR}/bin/test_fe_basis" --gtest_list_tests > /dev/null

echo "Running basis tests through ctest"
(
    cd "${BUILD_DIR}/Source/solver/FE"
    ctest --output-on-failure -R FE_Basis_Tests
)
