# Copyright (c) Stanford University, The Regents of the University of California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.

cmake_minimum_required(VERSION 3.14)

# ========================
# Search Module Unit Tests
# ========================

# Find required packages
find_package(Threads REQUIRED)

# Enable/locate GoogleTest
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/EnableGTest.cmake)
if(NOT GTest_FOUND)
    message(STATUS "GTest unavailable. Skipping Search unit tests.")
    return()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/Code/Source/solver
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/Mesh
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/Mesh/Core
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/Mesh/Search
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/Mesh/Topology
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/Mesh/Geometry
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/Mesh/Observer
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ========================
# SearchPrimitives Test
# ========================
add_executable(test_SearchPrimitives
    test_SearchPrimitives.cpp
    ../../../Search/SearchPrimitives.cpp
)

target_link_libraries(test_SearchPrimitives
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_SearchPrimitives PRIVATE cxx_std_17)

add_test(NAME SearchPrimitives_Tests
         COMMAND test_SearchPrimitives)

# ========================
# SearchBuilders Test
# ========================
add_executable(test_SearchBuilders
    test_SearchBuilders.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_SearchBuilders
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_SearchBuilders PRIVATE cxx_std_17)

add_test(NAME SearchBuilders_Tests
         COMMAND test_SearchBuilders)

# ========================
# UniformGridAccel Test
# ========================
add_executable(test_UniformGridAccel
    test_UniformGridAccel.cpp
    ../../../Search/UniformGridAccel.cpp
    ../../../Search/SearchAccel.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Search/MeshSearch.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_UniformGridAccel
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_UniformGridAccel PRIVATE cxx_std_17)

add_test(NAME UniformGridAccel_Tests
         COMMAND test_UniformGridAccel)

# ========================
# KDTreeAccel Test
# ========================
add_executable(test_KDTreeAccel
    test_KDTreeAccel.cpp
    ../../../Search/KDTreeAccel.cpp
    ../../../Search/SearchAccel.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/DistributedMesh.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_KDTreeAccel
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_KDTreeAccel PRIVATE cxx_std_17)

add_test(NAME KDTreeAccel_Tests
         COMMAND test_KDTreeAccel)

# ========================
# OctreeAccel Test
# ========================
add_executable(test_OctreeAccel
    test_OctreeAccel.cpp
    ../../../Search/OctreeAccel.cpp
    ../../../Search/SearchAccel.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/DistributedMesh.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_OctreeAccel
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_OctreeAccel PRIVATE cxx_std_17)

add_test(NAME OctreeAccel_Tests
         COMMAND test_OctreeAccel)

# ========================
# BVHAccel Test
# ========================
add_executable(test_BVHAccel
    test_BVHAccel.cpp
    ../../../Search/BVHAccel.cpp
    ../../../Search/SearchAccel.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/DistributedMesh.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_BVHAccel
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_BVHAccel PRIVATE cxx_std_17)

add_test(NAME BVHAccel_Tests
         COMMAND test_BVHAccel)

# ========================
# RTreeAccel Test
# ========================
add_executable(test_RTreeAccel
    test_RTreeAccel.cpp
    ../../../Search/RTreeAccel.cpp
    ../../../Search/SearchAccel.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/DistributedMesh.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_RTreeAccel
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_RTreeAccel PRIVATE cxx_std_17)

add_test(NAME RTreeAccel_Tests
         COMMAND test_RTreeAccel)

# ========================
# MeshSearch Integration Test
# ========================
add_executable(test_MeshSearch
    test_MeshSearch.cpp
    ../../../Search/MeshSearch.cpp
    ../../../Search/UniformGridAccel.cpp
    ../../../Search/KDTreeAccel.cpp
    ../../../Search/OctreeAccel.cpp
    ../../../Search/BVHAccel.cpp
    ../../../Search/RTreeAccel.cpp
    ../../../Search/SearchAccel.cpp
    ../../../Search/SearchBuilders.cpp
    ../../../Search/SearchPrimitives.cpp
    ../../../Core/MeshBase.cpp
    ../../../Core/MeshFactory.cpp
    ../../../Topology/MeshTopology.cpp
    ../../../Topology/CellShape.cpp
    ../../../Topology/CellOrder.cpp
    ../../../Geometry/MeshGeometry.cpp
    ../../../Observer/MeshObserver.cpp
    ../../../Observer/MeshEvent.cpp
)

target_link_libraries(test_MeshSearch
    PRIVATE GTest::gtest GTest::gtest_main
    PRIVATE Threads::Threads
    PRIVATE ${CMAKE_DL_LIBS}
)

target_compile_features(test_MeshSearch PRIVATE cxx_std_17)

add_test(NAME MeshSearch_Integration_Tests
         COMMAND test_MeshSearch)

# ========================
# Test Properties
# ========================

# Set test properties (optional timeout, labels, etc.)
set_tests_properties(
    SearchPrimitives_Tests
    SearchBuilders_Tests
    UniformGridAccel_Tests
    KDTreeAccel_Tests
    OctreeAccel_Tests
    BVHAccel_Tests
    RTreeAccel_Tests
    MeshSearch_Integration_Tests
    PROPERTIES
        TIMEOUT 60
        LABELS "Search;Unit"
)

# ========================
# Coverage Settings (if enabled)
# ========================
if(ENABLE_COVERAGE)
    target_compile_options(test_SearchPrimitives PRIVATE --coverage)
    target_link_options(test_SearchPrimitives PRIVATE --coverage)

    target_compile_options(test_SearchBuilders PRIVATE --coverage)
    target_link_options(test_SearchBuilders PRIVATE --coverage)

    target_compile_options(test_UniformGridAccel PRIVATE --coverage)
    target_link_options(test_UniformGridAccel PRIVATE --coverage)

    target_compile_options(test_KDTreeAccel PRIVATE --coverage)
    target_link_options(test_KDTreeAccel PRIVATE --coverage)

    target_compile_options(test_OctreeAccel PRIVATE --coverage)
    target_link_options(test_OctreeAccel PRIVATE --coverage)

    target_compile_options(test_BVHAccel PRIVATE --coverage)
    target_link_options(test_BVHAccel PRIVATE --coverage)

    target_compile_options(test_RTreeAccel PRIVATE --coverage)
    target_link_options(test_RTreeAccel PRIVATE --coverage)

    target_compile_options(test_MeshSearch PRIVATE --coverage)
    target_link_options(test_MeshSearch PRIVATE --coverage)
endif()

# ========================
# Installation (if needed)
# ========================
if(INSTALL_TESTS)
    install(TARGETS
        test_SearchPrimitives
        test_SearchBuilders
        test_UniformGridAccel
        test_KDTreeAccel
        test_OctreeAccel
        test_BVHAccel
        test_RTreeAccel
        test_MeshSearch
        DESTINATION bin/tests/search
    )
endif()
