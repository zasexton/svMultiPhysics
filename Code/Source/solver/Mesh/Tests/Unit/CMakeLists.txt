# Unit Tests for Mesh Folder
cmake_minimum_required(VERSION 3.15)

# Find testing framework (Google Test recommended)
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found. Unit tests will not be built.")
    message(STATUS "To enable unit tests, install Google Test:")
    message(STATUS "  - Ubuntu/Debian: sudo apt-get install libgtest-dev")
    message(STATUS "  - Fedora: sudo dnf install gtest-devel")
    message(STATUS "  - macOS: brew install googletest")
    return()
endif()

message(STATUS "Google Test found. Building unit tests...")

# Collect all test source files
set(TEST_SOURCES
    # Core tests
    # Core/test_MeshBase.cpp
    # Core/test_MeshTypes.cpp

    # Topology tests
    # Topology/test_CellShape.cpp
    # Topology/test_CellTopology.cpp

    # Geometry tests
    # Geometry/test_MeshGeometry.cpp

    # Boundary tests
    # Boundary/test_BoundaryDetector.cpp
    # Boundary/test_BoundaryKey.cpp
    # Boundary/test_BoundaryComponent.cpp

    # Fields tests
    # Fields/test_FieldManager.cpp

    # Labels tests
    # Labels/test_MeshLabels.cpp

    # IO tests
    # IO/test_MeshIO.cpp

    # Search tests
    # Search/test_MeshSearch.cpp

    # Validation tests
    # Validation/test_MeshValidation.cpp

    # Algorithms tests
    # Algorithms/test_MeshAlgorithms.cpp

    # Adaptivity tests
    # Adaptivity/test_MeshAdaptivity.cpp

    # Constraints tests
    # Constraints/test_MeshConstraints.cpp

    # Observer tests
    # Observer/test_MeshObserver.cpp
)

# Only build if there are test sources
if(TEST_SOURCES)
    # Create test executable
    add_executable(mesh_unit_tests ${TEST_SOURCES})

    # Link against mesh library and Google Test
    target_link_libraries(mesh_unit_tests
        PRIVATE
            mesh_lib  # Assumes mesh library target name
            GTest::gtest
            GTest::gtest_main
    )

    # Set C++ standard
    target_compile_features(mesh_unit_tests PRIVATE cxx_std_17)

    # Include directories
    target_include_directories(mesh_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..  # Mesh folder root
    )

    # Enable testing
    enable_testing()

    # Register tests with CTest
    add_test(NAME MeshUnitTests COMMAND mesh_unit_tests)

    # Set test properties
    set_tests_properties(MeshUnitTests PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "unit;mesh"
    )

    message(STATUS "Unit tests configured: mesh_unit_tests")
else()
    message(STATUS "No test source files found. Add test files and uncomment them in CMakeLists.txt")
endif()

# Optional: Add custom target to run tests
add_custom_target(run_mesh_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L mesh
    DEPENDS mesh_unit_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Mesh unit tests..."
)
