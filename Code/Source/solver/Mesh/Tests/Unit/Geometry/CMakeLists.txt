cmake_minimum_required(VERSION 3.12)

# Unified GTest enable (find or fetch)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/EnableGTest.cmake)

if(NOT GTest_FOUND)
    message(STATUS "GTest unavailable. Skipping Geometry unit tests.")
    return()
endif()

add_executable(test_MeshGeometry test_MeshGeometry.cpp)
target_link_libraries(test_MeshGeometry PRIVATE svmesh GTest::gtest GTest::gtest_main)
target_compile_features(test_MeshGeometry PRIVATE cxx_std_17)
target_include_directories(test_MeshGeometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set_target_properties(test_MeshGeometry PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_RPATH ""
)

add_test(NAME test_MeshGeometry COMMAND test_MeshGeometry)
set_tests_properties(test_MeshGeometry PROPERTIES LABELS "unit;mesh;geometry")

# PolyGeometry tests
add_executable(test_PolyGeometry test_PolyGeometry.cpp)
target_link_libraries(test_PolyGeometry PRIVATE svmesh GTest::gtest GTest::gtest_main)
target_compile_features(test_PolyGeometry PRIVATE cxx_std_17)
target_include_directories(test_PolyGeometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set_target_properties(test_PolyGeometry PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_RPATH ""
)

add_test(NAME test_PolyGeometry COMMAND test_PolyGeometry)
set_tests_properties(test_PolyGeometry PROPERTIES LABELS "unit;mesh;geometry")
