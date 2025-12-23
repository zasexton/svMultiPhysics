# =============================================================================
# FE Dofs Module Unit Tests
# =============================================================================

# Test sources
set(DOFS_TEST_SOURCES
    test_main.cpp
    test_DofIndexSet.cpp
    test_SubspaceView.cpp
    test_DofMap.cpp
    test_DofHandler.cpp
    test_EntityDofMap.cpp
    test_FieldDofMap.cpp
    test_DofConstraints.cpp
    test_ConstrainedAssembly.cpp
    test_DofGraph.cpp
    test_BlockDofMap.cpp
    test_DofNumbering.cpp
    test_DofTools.cpp
    test_GhostDofManager.cpp
)

# Create test executable
add_executable(test_fe_dofs ${DOFS_TEST_SOURCES})

# Link dependencies
target_link_libraries(test_fe_dofs
    PRIVATE
        fe_lib
        GTest::gtest
        GTest::gtest_main
)

# Include directories
target_include_directories(test_fe_dofs
    PRIVATE
        ${CMAKE_SOURCE_DIR}/Code/Source/solver
)

# Set C++ standard
set_target_properties(test_fe_dofs PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(test_fe_dofs
    PROPERTIES
        LABELS "FE;Dofs;Unit"
)

# Custom target for running just Dofs tests
add_custom_target(run_dofs_tests
    COMMAND test_fe_dofs --gtest_color=yes
    DEPENDS test_fe_dofs
    COMMENT "Running FE Dofs module unit tests"
)
