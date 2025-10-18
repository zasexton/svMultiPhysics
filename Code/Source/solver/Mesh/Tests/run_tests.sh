#!/bin/bash
# Script to build and run DistributedMesh unit tests

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "   DistributedMesh Test Suite Runner"
echo "========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MESH_DIR="$SCRIPT_DIR/.."
BUILD_DIR="$MESH_DIR/build_tests"

# Parse command line arguments
NUM_PROCS=2
RUN_ADVANCED=1
CLEAN_BUILD=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-procs)
            NUM_PROCS="$2"
            shift 2
            ;;
        --basic-only)
            RUN_ADVANCED=0
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -n, --num-procs N    Number of MPI processes (default: 2)"
            echo "  --basic-only         Run only basic tests"
            echo "  --clean              Clean build before compiling"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
echo -e "${GREEN}Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake -DMESH_BUILD_TESTS=ON \
      -DMESH_ENABLE_MPI=ON \
      -DMESH_ENABLE_VTK=ON \
      -DUSE_SYSTEM_VTK=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      "$MESH_DIR" || {
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
}

# Build the tests
echo -e "${GREEN}Building tests...${NC}"
make -j$(nproc) test_DistributedMesh test_DistributedMesh_Advanced 2>&1 | tee build.log || {
    echo -e "${RED}Build failed! Check build.log for details.${NC}"
    exit 1
}

# Check if executables exist
if [ ! -f "Tests/Unit/Core/test_DistributedMesh" ]; then
    echo -e "${RED}Basic test executable not found!${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "         Running Unit Tests"
echo "========================================="
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local test_exe=$2

    echo -e "${YELLOW}Running $test_name with $NUM_PROCS processes...${NC}"

    if [ -f "$test_exe" ]; then
        mpirun -n $NUM_PROCS "$test_exe" 2>&1 | tee "${test_name}.log"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}✓ $test_name PASSED${NC}"
            return 0
        else
            echo -e "${RED}✗ $test_name FAILED${NC}"
            return 1
        fi
    else
        echo -e "${RED}Test executable not found: $test_exe${NC}"
        return 1
    fi
}

# Track test results
FAILED_TESTS=()

# Run basic tests
if ! run_test "Basic DistributedMesh Tests" "Tests/Unit/Core/test_DistributedMesh"; then
    FAILED_TESTS+=("Basic Tests")
fi

# Run advanced tests if requested
if [ $RUN_ADVANCED -eq 1 ]; then
    if ! run_test "Advanced DistributedMesh Tests" "Tests/Unit/Core/test_DistributedMesh_Advanced"; then
        FAILED_TESTS+=("Advanced Tests")
    fi
fi

echo ""
echo "========================================="
echo "            Test Summary"
echo "========================================="

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All tests PASSED! ✓${NC}"
    echo "MPI Processes used: $NUM_PROCS"
    exit 0
else
    echo -e "${RED}The following tests FAILED:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "Check the log files for details."
    exit 1
fi
