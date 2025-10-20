cmake_minimum_required(VERSION 3.12)

if(NOT MESH_ENABLE_VTK OR NOT VTK_FOUND)
    message(STATUS "Skipping VTK parity tests (VTK disabled or not found)")
    return()
endif()

# Unified GTest enable (find or fetch)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/EnableGTest.cmake)
if(NOT GTest_FOUND)
    message(STATUS "GTest unavailable. Skipping VTK parity tests.")
    return()
endif()

add_executable(test_VTKParity test_VTKParity.cpp)
target_link_libraries(test_VTKParity PRIVATE svmesh GTest::gtest GTest::gtest_main)
target_compile_features(test_VTKParity PRIVATE cxx_std_17)
target_include_directories(test_VTKParity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set_target_properties(test_VTKParity PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_RPATH ""
)

add_test(NAME test_VTKParity COMMAND test_VTKParity)
set_tests_properties(test_VTKParity PROPERTIES LABELS "unit;mesh;io;vtk")
