# Copyright (c) Stanford University, The Regents of the University of California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.
#
# CMakeLists.txt for svMultiPhysics Physics Infrastructure
# This file supports both standalone builds and integration with the main project.

cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------
# Project setup for standalone vs integrated builds
#------------------------------------------------------------------------------

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(svMultiPhysicsPhysics
        VERSION 1.0.0
        DESCRIPTION "Physics formulations and constitutive models for svMultiPhysics"
        LANGUAGES CXX
    )
    set(PHYSICS_STANDALONE_BUILD ON)
    message(STATUS "Physics: Building in STANDALONE mode")
else()
    set(PHYSICS_STANDALONE_BUILD OFF)
    message(STATUS "Physics: Building as part of svMultiPhysics")
endif()

#------------------------------------------------------------------------------
# Compiler settings
#------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# Build options
#------------------------------------------------------------------------------

option(PHYSICS_BUILD_SHARED "Build Physics as shared library" OFF)
option(PHYSICS_BUILD_TESTS "Build Physics unit tests" ${PHYSICS_STANDALONE_BUILD})
option(PHYSICS_WITH_MESH "Enable Mesh integration (link svmesh when available)" ON)
option(PHYSICS_USE_SYSTEM_GTEST "Use system-installed Google Test" ON)

#------------------------------------------------------------------------------
# Dependencies (FE is required; Mesh is optional)
#------------------------------------------------------------------------------

# If Mesh is enabled, ensure svmesh is available before enabling FE_WITH_MESH.
if(PHYSICS_WITH_MESH AND (NOT TARGET svmesh))
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../Mesh/CMakeLists.txt)
        # Make Mesh lightweight when built as a dependency of Physics.
        if(NOT DEFINED MESH_ENABLE_VTK)
            set(MESH_ENABLE_VTK OFF CACHE BOOL "Disable VTK for Mesh when building via Physics" FORCE)
        endif()
        if(NOT DEFINED MESH_BUILD_TESTS)
            set(MESH_BUILD_TESTS OFF CACHE BOOL "Disable Mesh tests when building via Physics" FORCE)
        endif()

        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../Mesh
            ${CMAKE_BINARY_DIR}/thirdparty/svmesh
        )
    else()
        find_package(svmesh QUIET)
    endif()
endif()

# Ensure FE is available; Physics requires FE/Assembly + FE/Systems + FE/Forms.
if(NOT TARGET svfe)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../FE/CMakeLists.txt)
        # Force-enable the FE modules required by Physics.
        set(FE_ENABLE_ASSEMBLY ON CACHE BOOL "Enable FE Assembly/Systems/Forms for Physics" FORCE)

        # If Physics is built with Mesh, enable FE's Mesh types/API to avoid
        # conflicting svmp::CellFamily/Configuration definitions in headers.
        if(PHYSICS_WITH_MESH AND TARGET svmesh AND (NOT DEFINED FE_WITH_MESH))
            set(FE_WITH_MESH ON CACHE BOOL "Enable FE Mesh integration for Physics dependency" FORCE)
        endif()

        # Keep standalone builds deterministic by avoiding optional heavyweight deps by default.
        if(NOT DEFINED FE_ENABLE_TRILINOS)
            set(FE_ENABLE_TRILINOS OFF CACHE BOOL "Disable Trilinos for Physics dependency" FORCE)
        endif()
        if(NOT DEFINED FE_ENABLE_PETSC)
            set(FE_ENABLE_PETSC OFF CACHE BOOL "Disable PETSc for Physics dependency" FORCE)
        endif()
        if(NOT DEFINED FE_ENABLE_SYMENGINE)
            set(FE_ENABLE_SYMENGINE OFF CACHE BOOL "Disable SymEngine for Physics dependency" FORCE)
        endif()
        if(NOT DEFINED FE_ENABLE_METIS)
            set(FE_ENABLE_METIS OFF CACHE BOOL "Disable internal METIS for Physics dependency" FORCE)
        endif()
        if(NOT DEFINED FE_ENABLE_PARMETIS)
            set(FE_ENABLE_PARMETIS OFF CACHE BOOL "Disable internal ParMETIS for Physics dependency" FORCE)
        endif()

        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../FE
            ${CMAKE_BINARY_DIR}/thirdparty/svfe
        )
    else()
        find_package(svfe REQUIRED)
    endif()
endif()

#------------------------------------------------------------------------------
# Library target
#------------------------------------------------------------------------------

set(PHYSICS_PUBLIC_HEADERS
    Physics.h
    Core/Domain.h
    Core/ParameterSchema.h
    Core/PhysicsModule.h
    Formulations/Poisson/PoissonModule.h
    Materials/Common/TensorOps.h
    Materials/Fluid/CarreauYasudaViscosity.h
    Materials/Fluid/NewtonianViscosity.h
    Materials/Solid/LinearElasticStress.h
    Materials/Solid/NeoHookeanPK1.h
)

set(PHYSICS_SOURCES
    Physics.cpp
    Core/ParameterSchema.cpp
    Formulations/Poisson/PoissonModule.cpp
    Materials/Fluid/CarreauYasudaViscosity.cpp
    Materials/Fluid/NewtonianViscosity.cpp
    Materials/Solid/LinearElasticStress.cpp
    Materials/Solid/NeoHookeanPK1.cpp
)

if(PHYSICS_BUILD_SHARED)
    add_library(svphysics SHARED ${PHYSICS_PUBLIC_HEADERS} ${PHYSICS_SOURCES})
    set_target_properties(svphysics PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
else()
    add_library(svphysics STATIC ${PHYSICS_PUBLIC_HEADERS} ${PHYSICS_SOURCES})
endif()

add_library(svphysics::svphysics ALIAS svphysics)

set_target_properties(svphysics PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(svphysics
    PUBLIC
        # Include root is the solver/ directory so consumers can write:
        #   #include "Physics/..."
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include/svphysics>
)

target_link_libraries(svphysics PUBLIC svfe::svfe)

if(PHYSICS_WITH_MESH AND TARGET svmesh)
    target_link_libraries(svphysics PUBLIC svmesh)
    target_compile_definitions(svphysics PUBLIC SVMP_PHYSICS_WITH_MESH=1)
else()
    target_compile_definitions(svphysics PUBLIC SVMP_PHYSICS_WITH_MESH=0)
endif()

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

if(PHYSICS_BUILD_TESTS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableGTest.cmake)

    set(PHYSICS_TEST_SOURCES
        Tests/Unit/test_main.cpp
        Tests/Unit/test_PoissonModule.cpp
        Tests/Unit/test_FluidViscosityModels.cpp
        Tests/Unit/test_SolidMaterialModels.cpp
        Tests/Unit/test_FormIntegrationJacobian.cpp
        Tests/Unit/test_CoreUtilities.cpp
        Tests/Unit/test_VtpMeshSupport.cpp
    )

    add_executable(test_physics ${PHYSICS_TEST_SOURCES})
    target_link_libraries(test_physics PRIVATE svphysics ${GTEST_LIBRARIES})
    add_test(NAME Physics_Tests COMMAND test_physics)
endif()

#------------------------------------------------------------------------------
# Installation (standalone builds)
#------------------------------------------------------------------------------

if(PHYSICS_STANDALONE_BUILD)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS svphysics
        EXPORT svphysicsTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/svphysics/Physics
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hpp"
        PATTERN "Tests" EXCLUDE
        PATTERN "cmake" EXCLUDE
        PATTERN "build" EXCLUDE
    )

    # CMake package export requires that link dependencies are either imported
    # targets (from find_package) or also exported. When Physics builds svfe
    # in-tree as a convenience dependency, it is not install-exportable here.
    get_target_property(_svphysics_svfe_imported svfe IMPORTED)
    if(_svphysics_svfe_imported)
        install(EXPORT svphysicsTargets
            FILE svphysicsTargets.cmake
            NAMESPACE svphysics::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svphysics
        )

        configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/svphysicsConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/svphysicsConfig.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svphysics
        )

        write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/svphysicsConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion
        )

        install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/svphysicsConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/svphysicsConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svphysics
        )
    else()
        message(STATUS "Physics: svfe is built in-tree; skipping svphysics CMake package export (install svfe separately and rebuild Physics to export).")
    endif()
endif()
