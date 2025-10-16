# Mesh Tests

This directory contains tests for the Mesh folder components.

## Test Organization

```
Tests/
├── Unit/           # Unit tests for individual components
└── Integration/    # Integration tests (future)
```

## Unit Tests

Located in `Unit/`, these tests verify the correctness of individual components in isolation. Unit tests should:

- Test a single component or class
- Run quickly (milliseconds per test)
- Not depend on external resources (files, network, etc.)
- Be deterministic and reproducible
- Not depend on other tests

See `Unit/README.md` for detailed information about writing and running unit tests.

## Integration Tests (Future)

Integration tests will verify that multiple components work together correctly. These tests:

- Test interactions between multiple components
- May use realistic mesh data
- May test file I/O operations
- May take longer to run
- Verify end-to-end workflows

## Running Tests

### Build Tests

```bash
# From Mesh folder
mkdir build && cd build
cmake ..
make

# Build specific test targets
make mesh_unit_tests
```

### Run Tests

```bash
# Run all unit tests
./Tests/Unit/mesh_unit_tests

# Run specific test suite
./Tests/Unit/mesh_unit_tests --gtest_filter=BoundaryDetectorTest.*

# Run with verbose output
./Tests/Unit/mesh_unit_tests --gtest_verbose

# Run tests using CTest
ctest --output-on-failure -L mesh
```

## Test Coverage

To generate test coverage reports:

```bash
# Configure with coverage enabled
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make
make mesh_unit_tests

# Run tests
ctest

# Generate coverage report (requires lcov)
make coverage
```

## Continuous Integration

Tests should be run automatically:
- On every commit
- Before merging pull requests
- Nightly for comprehensive test suites

CI should:
- Block merges if tests fail
- Report test coverage metrics
- Track test performance over time

## Adding New Tests

When adding new functionality to the Mesh folder:

1. **Write tests first** (TDD approach recommended)
2. Place tests in corresponding subfolder under `Unit/`
3. Follow naming convention: `test_<ComponentName>.cpp`
4. Add test file to `Unit/CMakeLists.txt`
5. Ensure tests pass before committing

## Test Requirements

All code in the Mesh folder should have:

- **Unit test coverage** for all public APIs
- **Edge case testing** (empty inputs, boundary conditions, etc.)
- **Error handling tests** (invalid inputs, exceptional cases)
- **Performance tests** for critical algorithms (optional, may be in separate benchmarks)

## Testing Framework

Tests use **Google Test** (gtest) as the primary testing framework.

### Installing Google Test

**Ubuntu/Debian:**
```bash
sudo apt-get install libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

**Fedora:**
```bash
sudo dnf install gtest-devel
```

**macOS:**
```bash
brew install googletest
```

**From source:**
```bash
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake ..
make
sudo make install
```

## Test Guidelines

### Good Test Characteristics

- **Fast**: Tests should run in milliseconds
- **Isolated**: No dependencies on other tests
- **Repeatable**: Same result every time
- **Self-checking**: Use assertions, not manual verification
- **Timely**: Written at the same time as code

### Test Naming

Use descriptive names that explain what is being tested:

```cpp
TEST_F(BoundaryDetectorTest, SingleTetHasFourBoundaryFaces) { ... }
TEST_F(MeshGeometryTest, TriangleAreaRightAngle) { ... }
TEST_F(CellTopologyTest, TetHasFourTriangularFaces) { ... }
```

### Test Structure (Arrange-Act-Assert)

```cpp
TEST_F(ComponentTest, DescriptiveName) {
    // Arrange: Set up test data
    MeshBase mesh = create_test_mesh();
    Component component(mesh);

    // Act: Perform the operation
    auto result = component.do_something();

    // Assert: Verify the result
    EXPECT_EQ(result, expected_value);
}
```

## Best Practices

1. **One assertion per test** (when practical)
2. **Test behavior, not implementation**
3. **Use fixtures for shared setup**
4. **Mock external dependencies**
5. **Test error conditions**
6. **Keep tests simple and readable**
7. **Avoid test interdependencies**
8. **Document complex test logic**

## Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- [Unit Testing Best Practices](https://en.wikipedia.org/wiki/Unit_testing)
