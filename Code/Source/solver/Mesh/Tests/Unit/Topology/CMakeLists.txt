# Unit tests for CellTopology (no MPI, optional GTest)

cmake_minimum_required(VERSION 3.12)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found. Skipping Topology unit tests.")
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
target_compile_features(test_CellTopology PRIVATE cxx_std_17)
target_include_directories(test_CellTopology PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

add_test(NAME test_CellTopology COMMAND test_CellTopology)
set_tests_properties(test_CellTopology PROPERTIES LABELS "unit;mesh;topology")
