#!/bin/bash
#
# Script to run FE Math module tests with code coverage analysis
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FE Math Module Coverage Analysis${NC}"
echo -e "${GREEN}========================================${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FE_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${FE_DIR}/build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Clean previous coverage data
echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
find . -name "*.gcda" -delete 2>/dev/null || true
find . -name "*.gcno" -delete 2>/dev/null || true
rm -rf coverage coverage.info 2>/dev/null || true

# Configure with coverage enabled
echo -e "${YELLOW}Configuring with coverage enabled...${NC}"
cmake -DFE_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..

# Build tests
echo -e "${YELLOW}Building tests...${NC}"
make -j$(nproc) test_fe_math

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
./test_fe_math --gtest_brief=1

# Generate coverage report
echo -e "${YELLOW}Generating coverage report...${NC}"

# Capture coverage data
lcov --capture --directory . --output-file coverage.info --no-external

# Remove unwanted files from coverage
lcov --remove coverage.info '/usr/*' --output-file coverage.info
lcov --remove coverage.info '*/test*' --output-file coverage.info
lcov --remove coverage.info '*/Tests/*' --output-file coverage.info
lcov --remove coverage.info '*/ThirdParty/*' --output-file coverage.info
lcov --remove coverage.info '*/build/*' --output-file coverage.info

# Generate HTML report
genhtml coverage.info --output-directory coverage --title "FE Math Module Coverage" --legend --show-details

# Display summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Coverage Summary:${NC}"
echo -e "${GREEN}========================================${NC}"
lcov --summary coverage.info

# Get coverage percentage
COVERAGE_PCT=$(lcov --summary coverage.info 2>&1 | grep "lines" | grep -oE '[0-9]+\.[0-9]+%' | head -1)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Overall Line Coverage: ${COVERAGE_PCT}${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if coverage meets target
TARGET=90
CURRENT=$(echo $COVERAGE_PCT | sed 's/%//')
if (( $(echo "$CURRENT >= $TARGET" | bc -l) )); then
    echo -e "${GREEN}✓ Coverage target ($TARGET%) achieved!${NC}"
else
    echo -e "${YELLOW}⚠ Coverage ($CURRENT%) below target ($TARGET%)${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Report generated in: ${BUILD_DIR}/coverage/index.html${NC}"
echo -e "${GREEN}========================================${NC}"

# Open report in browser if available
if command -v xdg-open &> /dev/null; then
    echo -e "${YELLOW}Opening report in browser...${NC}"
    xdg-open coverage/index.html
elif command -v open &> /dev/null; then
    echo -e "${YELLOW}Opening report in browser...${NC}"
    open coverage/index.html
fi