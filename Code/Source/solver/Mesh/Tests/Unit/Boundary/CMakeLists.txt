cmake_minimum_required(VERSION 3.12)

# Unified GTest enable (find or fetch)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/EnableGTest.cmake)

if(NOT GTest_FOUND)
    message(STATUS "GTest unavailable. Skipping Boundary unit tests.")
    return()
endif()

add_executable(test_BoundaryDetector test_BoundaryDetector.cpp)
target_link_libraries(test_BoundaryDetector PRIVATE svmesh GTest::gtest GTest::gtest_main)
target_compile_features(test_BoundaryDetector PRIVATE cxx_std_17)
target_include_directories(test_BoundaryDetector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set_target_properties(test_BoundaryDetector PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_RPATH ""
)

add_test(NAME test_BoundaryDetector COMMAND test_BoundaryDetector)
set_tests_properties(test_BoundaryDetector PROPERTIES LABELS "unit;mesh;boundary")

