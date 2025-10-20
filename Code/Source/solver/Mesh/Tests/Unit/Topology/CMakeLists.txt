# Unit tests for CellTopology (no MPI, optional GTest)

cmake_minimum_required(VERSION 3.12)

# Unified GTest enable (find or fetch)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/EnableGTest.cmake)

if(NOT GTest_FOUND)
    message(STATUS "GTest unavailable. Skipping Topology unit tests.")
    return()
endif()

set(SRC
    test_CellTopology.cpp
)

add_executable(test_CellTopology ${SRC})
target_link_libraries(test_CellTopology
    PRIVATE
        svmesh
        GTest::gtest
        GTest::gtest_main
)
# Avoid embedding RPATHs that include Conda paths; rely on system lookup
set_target_properties(test_CellTopology PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_RPATH ""
)
target_compile_features(test_CellTopology PRIVATE cxx_std_17)
target_include_directories(test_CellTopology PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

add_test(NAME test_CellTopology COMMAND test_CellTopology)
set_tests_properties(test_CellTopology PROPERTIES LABELS "unit;mesh;topology")
