# Unit Tests for Mesh Folder

This subfolder contains unit tests for the Mesh folder components.

## Purpose

Unit tests verify the correctness of individual components in isolation. Each subfolder of the Mesh directory should have corresponding unit tests here.

## Organization

Tests are organized by subfolder to match the Mesh directory structure:

```
Tests/Unit/
├── Core/           # Tests for Core components (MeshBase, MeshTypes, etc.)
├── Topology/       # Tests for Topology components (CellShape, CellTopology, etc.)
├── Geometry/       # Tests for Geometry components (MeshGeometry)
├── Boundary/       # Tests for Boundary components (BoundaryDetector, BoundaryKey, etc.)
├── Fields/         # Tests for Fields components
├── Labels/         # Tests for Labels components
├── IO/             # Tests for IO components
├── Search/         # Tests for Search components
├── Validation/     # Tests for Validation components
├── Algorithms/     # Tests for Algorithms components
├── Adaptivity/     # Tests for Adaptivity components
├── Constraints/    # Tests for Constraints components
└── Observer/       # Tests for Observer components
```

## Testing Framework

Tests should use a standard C++ testing framework such as:
- **Google Test** (recommended)
- **Catch2**
- **doctest**

## Writing Tests

### File Naming Convention

Test files should follow the pattern: `test_<component_name>.cpp`

Examples:
- `test_MeshBase.cpp` - Tests for MeshBase class
- `test_BoundaryDetector.cpp` - Tests for BoundaryDetector class
- `test_MeshGeometry.cpp` - Tests for MeshGeometry class

### Test Structure

Each test file should:
1. Include the component being tested
2. Include the testing framework headers
3. Create test fixtures for shared setup/teardown
4. Write individual test cases for each method/functionality

### Example Test Template

```cpp
#include "gtest/gtest.h"
#include "Boundary/BoundaryDetector.h"
#include "Core/MeshBase.h"

namespace svmp {
namespace test {

// Test fixture for BoundaryDetector tests
class BoundaryDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test mesh
        // Initialize test data
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper methods
    MeshBase create_tet_mesh() {
        // Create a simple tetrahedral mesh
        // ...
    }
};

// Test: Detect boundary for a simple tet mesh
TEST_F(BoundaryDetectorTest, DetectBoundarySimpleTet) {
    MeshBase mesh = create_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_TRUE(info.has_boundary());
    EXPECT_EQ(info.boundary_faces.size(), 4);  // Single tet has 4 boundary faces
}

// Test: Closed mesh has no boundary
TEST_F(BoundaryDetectorTest, ClosedMeshNoBoundary) {
    MeshBase mesh = create_closed_mesh();
    BoundaryDetector detector(mesh);

    EXPECT_TRUE(detector.is_closed_mesh());
}

} // namespace test
} // namespace svmp
```

## Building and Running Tests

### CMake Integration

Tests should be built using CMake with the following structure:

```cmake
# In Tests/Unit/CMakeLists.txt
find_package(GTest REQUIRED)

# Add test executable
add_executable(mesh_unit_tests
    Core/test_MeshBase.cpp
    Boundary/test_BoundaryDetector.cpp
    Geometry/test_MeshGeometry.cpp
    # ... more test files
)

target_link_libraries(mesh_unit_tests
    mesh_lib
    GTest::gtest
    GTest::gtest_main
)

# Register tests with CTest
add_test(NAME MeshUnitTests COMMAND mesh_unit_tests)
```

### Running Tests

```bash
# Build tests
cd build
cmake ..
make mesh_unit_tests

# Run all unit tests
./mesh_unit_tests

# Run specific test suite
./mesh_unit_tests --gtest_filter=BoundaryDetectorTest.*

# Run with verbose output
./mesh_unit_tests --gtest_verbose
```

## Test Coverage

Unit tests should cover:

### Core Functionality
- Constructor/destructor behavior
- Method input validation
- Edge cases (empty inputs, boundary conditions)
- Error handling

### Topology Tests
- Cell shape queries
- Face/edge extraction
- Connectivity relationships
- Canonical orderings

### Geometry Tests
- Center computations (cell, face, edge)
- Normal computations (correct orientation)
- Measure computations (length, area, volume)
- Bounding box calculations
- Vector operations

### Boundary Tests
- Boundary detection accuracy
- Incidence counting
- Connected component extraction
- Non-manifold detection
- Right-hand rule orientation

### Fields Tests
- Field creation and destruction
- Data access patterns
- Configuration switching
- Field metadata

### Validation Tests
- Mesh quality metrics
- Topology validation
- Geometric validity checks

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Descriptive Names**: Use clear, descriptive test names that explain what is being tested
3. **Arrange-Act-Assert**: Structure tests with setup, execution, and verification phases
4. **Test One Thing**: Each test should verify a single behavior or property
5. **Use Fixtures**: Share common setup code using test fixtures
6. **Mock External Dependencies**: Mock file I/O, MPI, or other external dependencies
7. **Test Error Conditions**: Don't just test the happy path, test error cases too
8. **Document Complex Tests**: Add comments explaining non-obvious test logic

## Continuous Integration

Unit tests should:
- Run automatically on every commit
- Block merges if tests fail
- Report test coverage metrics
- Be fast (entire suite should run in seconds to minutes)

## Test Data

For tests requiring mesh data:
- Create simple meshes programmatically in tests
- Keep test meshes small (< 100 elements)
- Store reference test meshes in `Tests/Unit/data/` if needed
- Document the structure of any reference meshes used
