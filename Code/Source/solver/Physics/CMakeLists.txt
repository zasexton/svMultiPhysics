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
# Clang/libstdc++ compatibility (Ubuntu Clang 14 vs libstdc++ 13)
#
# FE applies the same workaround for its own targets. Physics also compiles
# Mesh-backed FE headers directly, so it needs the workaround on Physics targets
# as well when using older Clang with newer libstdc++ headers.
#------------------------------------------------------------------------------
set(_SVMP_PHYSICS_IS_CLANG FALSE)
set(_SVMP_PHYSICS_CLANG_MAJOR 0)
set(_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_SVMP_PHYSICS_IS_CLANG TRUE)
    if(CMAKE_CXX_COMPILER_VERSION MATCHES "^([0-9]+)\\.")
        set(_SVMP_PHYSICS_CLANG_MAJOR "${CMAKE_MATCH_1}")
    endif()
endif()

if(NOT _SVMP_PHYSICS_IS_CLANG AND NOT CMAKE_CROSSCOMPILING)
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" "--version"
        RESULT_VARIABLE _svmp_physics_cxx_version_rc
        OUTPUT_VARIABLE _svmp_physics_cxx_version_out
        ERROR_VARIABLE _svmp_physics_cxx_version_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(_svmp_physics_cxx_version_rc EQUAL 0)
        set(_svmp_physics_cxx_version_text
            "${_svmp_physics_cxx_version_out}\n${_svmp_physics_cxx_version_err}")
        if(_svmp_physics_cxx_version_text MATCHES "clang version")
            set(_SVMP_PHYSICS_IS_CLANG TRUE)
            if(_svmp_physics_cxx_version_text MATCHES "clang version ([0-9]+)\\.")
                set(_SVMP_PHYSICS_CLANG_MAJOR "${CMAKE_MATCH_1}")
            endif()
            message(STATUS "Physics: NOTE: CMake reports compiler ID '${CMAKE_CXX_COMPILER_ID}', but '${CMAKE_CXX_COMPILER}' identifies as Clang; consider deleting the build directory to refresh compiler detection")
        endif()
    endif()
endif()

if(_SVMP_PHYSICS_IS_CLANG AND _SVMP_PHYSICS_CLANG_MAJOR LESS 16 AND
   EXISTS "/usr/include/c++/11" AND
   EXISTS "/usr/include/x86_64-linux-gnu/c++/11")
    message(STATUS "Physics: Clang <16 detected; using libstdc++-11 headers to avoid <chrono> build failure with newer libstdc++")
    list(APPEND _SVMP_PHYSICS_LIBSTDCXX11_OPTIONS
        $<$<COMPILE_LANGUAGE:CXX>:-nostdinc++>
        $<$<COMPILE_LANGUAGE:CXX>:-isystem/usr/include/c++/11>
        $<$<COMPILE_LANGUAGE:CXX>:-isystem/usr/include/x86_64-linux-gnu/c++/11>
        $<$<COMPILE_LANGUAGE:CXX>:-isystem/usr/include/c++/11/backward>
    )
endif()

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
    Core/EquationModuleInput.h
    Core/EquationModuleRegistry.h
    Core/JITRuntimePolicy.h
    Core/ParameterSchema.h
    Core/PhysicsJITPolicy.h
    Core/PhysicsModule.h
    Core/TemporalValues.h
)

set(PHYSICS_SOURCES
    Physics.cpp
    Core/EquationModuleRegistry.cpp
    Core/ParameterSchema.cpp
    Core/PhysicsModule.cpp
    Core/TemporalValues.cpp
)

#------------------------------------------------------------------------------
# Auto-discover formulation/material sources so adding new physics does not
# require editing this top-level CMakeLists.txt.
#------------------------------------------------------------------------------

file(GLOB_RECURSE PHYSICS_FORMULATION_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Formulations/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Formulations/*.hpp"
)
file(GLOB_RECURSE PHYSICS_FORMULATION_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Formulations/*.cpp"
)
list(FILTER PHYSICS_FORMULATION_HEADERS EXCLUDE REGEX ".*/(Tests|Examples)/.*")
list(FILTER PHYSICS_FORMULATION_SOURCES EXCLUDE REGEX ".*/(Tests|Examples)/.*")

file(GLOB_RECURSE PHYSICS_MATERIAL_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Materials/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Materials/*.hpp"
)
file(GLOB_RECURSE PHYSICS_MATERIAL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Materials/*.cpp"
)
list(FILTER PHYSICS_MATERIAL_HEADERS EXCLUDE REGEX ".*/(Tests|Examples)/.*")
list(FILTER PHYSICS_MATERIAL_SOURCES EXCLUDE REGEX ".*/(Tests|Examples)/.*")

file(GLOB_RECURSE PHYSICS_COUPLING_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Coupling/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Coupling/*.hpp"
)
file(GLOB_RECURSE PHYSICS_COUPLING_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Coupling/*.cpp"
)
list(FILTER PHYSICS_COUPLING_HEADERS EXCLUDE REGEX ".*/(Tests|Examples)/.*")
list(FILTER PHYSICS_COUPLING_SOURCES EXCLUDE REGEX ".*/(Tests|Examples)/.*")

list(APPEND PHYSICS_PUBLIC_HEADERS
    ${PHYSICS_FORMULATION_HEADERS}
    ${PHYSICS_MATERIAL_HEADERS}
    ${PHYSICS_COUPLING_HEADERS}
)
list(APPEND PHYSICS_SOURCES
    ${PHYSICS_FORMULATION_SOURCES}
    ${PHYSICS_MATERIAL_SOURCES}
    ${PHYSICS_COUPLING_SOURCES}
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

if(_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS)
    target_compile_options(svphysics PRIVATE ${_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS})
endif()

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

    set(PHYSICS_COUPLING_HEADER_TEST_SOURCES)
    foreach(_physics_coupling_header IN LISTS PHYSICS_COUPLING_HEADERS)
        file(RELATIVE_PATH
            _physics_coupling_header_include
            "${CMAKE_CURRENT_SOURCE_DIR}/.."
            "${_physics_coupling_header}"
        )
        string(MAKE_C_IDENTIFIER
            "${_physics_coupling_header_include}"
            _physics_coupling_header_id
        )
        set(_physics_coupling_header_source
            "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${_physics_coupling_header_id}.cpp"
        )
        file(WRITE
            "${_physics_coupling_header_source}"
            "#include \"${_physics_coupling_header_include}\"\n"
            "int ${_physics_coupling_header_id}_compile_check() { return 0; }\n"
        )
        list(APPEND
            PHYSICS_COUPLING_HEADER_TEST_SOURCES
            "${_physics_coupling_header_source}"
        )
    endforeach()

    add_library(test_physics_coupling_headers OBJECT
        ${PHYSICS_COUPLING_HEADER_TEST_SOURCES}
    )
    target_link_libraries(test_physics_coupling_headers PRIVATE svphysics)
    if(_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS)
        target_compile_options(test_physics_coupling_headers
            PRIVATE
                ${_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS}
        )
    endif()

    set(PHYSICS_TEST_SOURCES
        Tests/Unit/test_main.cpp
        Tests/Unit/test_PoissonModule.cpp
        Tests/Unit/test_PoissonSquareSteady.cpp
        Tests/Unit/test_FluidViscosityModels.cpp
        Tests/Unit/test_SolidMaterialModels.cpp
        Tests/Unit/test_UstructModule.cpp
        Tests/Unit/test_FormIntegrationJacobian.cpp
        Tests/Unit/test_NavierStokesPressureGauge.cpp
        Tests/Unit/test_NavierStokesLegacyBCs.cpp
        Tests/Unit/test_NavierStokesOutletFactory.cpp
        Tests/Unit/test_MovingDomainPhysics.cpp
        Tests/Unit/test_CoreUtilities.cpp
        Tests/Unit/test_VtpMeshSupport.cpp
        Tests/Unit/Coupling/test_FSICouplingModule.cpp
        Tests/Unit/Coupling/test_ThermalInterfaceCouplingModule.cpp
    )

    add_executable(test_physics ${PHYSICS_TEST_SOURCES})
    add_dependencies(test_physics test_physics_coupling_headers)
    target_link_libraries(test_physics PRIVATE svphysics ${GTEST_LIBRARIES})
    if(_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS)
        target_compile_options(test_physics PRIVATE ${_SVMP_PHYSICS_LIBSTDCXX11_OPTIONS})
    endif()
    add_test(NAME Physics_Tests COMMAND test_physics)

    # Optional MPI-launched tests (requires mpiexec and building with MPI-enabled deps).
    find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun)
    if(MPIEXEC_EXECUTABLE)
        add_test(
            NAME Physics_Tests_MPI_2
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_physics>
                    --gtest_filter=PoissonSquareSteadyMPI.*:PoissonModuleMPI.*:MovingDomainPhysicsMPI.*
        )
        set_tests_properties(Physics_Tests_MPI_2 PROPERTIES
            TIMEOUT 180
            PROCESSORS 2
            LABELS "MPI;physics"
        )
    else()
        message(STATUS "Physics: MPI executor not found; MPI tests will not be registered")
    endif()
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
