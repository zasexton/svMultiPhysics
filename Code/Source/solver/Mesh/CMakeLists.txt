# Copyright (c) 2014-2015 The Regents of the University of California.
# All Rights Reserved.
#
# CMakeLists.txt for svMultiPhysics Mesh Infrastructure
# This file can be used for standalone builds or as part of the main project

cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------
# Environment sanitization: ignore active Conda/Miniconda environments
# to avoid accidental linkage against $CONDA_PREFIX libraries (e.g., libX11).
# This keeps the Mesh build deterministic on systems with Conda enabled.
#------------------------------------------------------------------------------

# Helper to remove any entries under a given prefix from a CMake list variable
function(_svmp_filter_out_prefix list_var prefix)
    if(NOT DEFINED ${list_var})
        return()
    endif()
    set(_svmp_new_list "")
    foreach(_p IN LISTS ${list_var})
        if(NOT _p STREQUAL "")
            string(FIND "${_p}" "${prefix}" _svmp_pos)
            if(_svmp_pos EQUAL -1)
                list(APPEND _svmp_new_list "${_p}")
            endif()
        endif()
    endforeach()
    set(${list_var} "${_svmp_new_list}" PARENT_SCOPE)
endfunction()

# If running inside Conda, prefer system toolchains and libraries by
# (1) ignoring the Conda prefix in CMake's find_* searches and
# (2) filtering common search path variables to drop $CONDA_PREFIX entries.
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(_SVMP_CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    message(STATUS "svmesh: Detected Conda environment at: ${_SVMP_CONDA_PREFIX}")

    # Avoid find_package/find_library picking up Conda packages
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_SVMP_CONDA_PREFIX}")
    if(WIN32)
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_SVMP_CONDA_PREFIX}/Library")
    endif()

    # Filter path-like variables that may have been pre-seeded by the shell
    _svmp_filter_out_prefix(CMAKE_PREFIX_PATH        "${_SVMP_CONDA_PREFIX}")
    _svmp_filter_out_prefix(CMAKE_LIBRARY_PATH       "${_SVMP_CONDA_PREFIX}")
    _svmp_filter_out_prefix(CMAKE_INCLUDE_PATH       "${_SVMP_CONDA_PREFIX}")
    _svmp_filter_out_prefix(CMAKE_PROGRAM_PATH       "${_SVMP_CONDA_PREFIX}")

    # Prefer $ORIGIN-based RPATH to avoid embedding absolute Conda paths
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

# Set CMake policies for compatibility with fetched VTK
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)  # find_package() uses <PackageName>_ROOT variables
endif()
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)  # MSVC runtime library flags are selected by an abstraction
endif()

# Only set project if this is a standalone build (top-level CMakeLists.txt)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(svMultiPhysicsMesh CXX)
    set(MESH_STANDALONE_BUILD ON)
else()
    set(MESH_STANDALONE_BUILD OFF)
endif()

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Options for optional features
option(MESH_ENABLE_MPI "Enable MPI support for distributed meshes" ON)
option(MESH_ENABLE_VTK "Enable VTK I/O support" ON)
option(USE_SYSTEM_VTK "Use system VTK instead of fetching via FetchContent" OFF)
# VTK fetch/build options (applied when MESH_ENABLE_VTK=ON and USE_SYSTEM_VTK=OFF)
set(VTK_GIT_TAG "v9.2.6" CACHE STRING "VTK Git tag to fetch (when not using system VTK)")
option(MESH_VTK_ENABLE_RENDERING "Enable VTK Rendering group when fetching VTK" OFF)
option(MESH_VTK_ENABLE_QT "Enable VTK Qt group when fetching VTK" OFF)
option(MESH_VTK_ENABLE_WEB "Enable VTK Web group when fetching VTK" OFF)
option(MESH_VTK_ENABLE_IOXML "Enable VTK IOXML module when fetching VTK" ON)
option(MESH_VTK_ENABLE_IOLEGACY "Enable VTK IOLegacy module when fetching VTK" ON)
option(MESH_BUILD_TESTS "Build mesh tests" OFF)
option(MESH_BUILD_SHARED "Build mesh as shared library" OFF)
option(MESH_ENABLE_PYTHON "Build Python bindings (pybind11)" OFF)
option(MESH_ENABLE_COVERAGE "Enable code coverage analysis (requires gcov/lcov)" OFF)
option(MESH_ENABLE_VALGRIND "Enable memory analysis with valgrind" OFF)
option(MESH_ENABLE_GTEST "Enable building GTest-based unit tests" ON)
option(USE_SYSTEM_GTEST "Use system-installed GTest (no fetch)" ON)

# For standalone builds, set up include directories
if(MESH_STANDALONE_BUILD)
    # Include the solver directory so relative includes work
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

#------------------------------------------------------------------------------
# Find dependencies
#------------------------------------------------------------------------------

# MPI (optional, only needed for DistributedMesh)
if(MESH_ENABLE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        message(STATUS "MPI found - enabling distributed mesh support")
        include_directories(${MPI_CXX_INCLUDE_DIRS})
    else()
        message(WARNING "MPI not found - distributed mesh features will be disabled")
        set(MESH_ENABLE_MPI OFF)
    endif()
else()
    message(STATUS "MPI support disabled")
endif()

# VTK (optional, only needed for VTK I/O)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnableVTK.cmake)

#------------------------------------------------------------------------------
# Coverage support
#------------------------------------------------------------------------------

if(MESH_ENABLE_COVERAGE)
    if(NOT MESH_BUILD_TESTS)
        message(WARNING "Coverage enabled but tests are not being built. Setting MESH_BUILD_TESTS=ON")
        set(MESH_BUILD_TESTS ON)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "Enabling code coverage instrumentation")
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)

        # Find required tools
        find_program(GCOV_EXECUTABLE gcov)
        find_program(LCOV_EXECUTABLE lcov)
        find_program(GENHTML_EXECUTABLE genhtml)

        if(NOT LCOV_EXECUTABLE)
            message(WARNING "lcov not found - coverage report generation will not be available")
        endif()
        if(NOT GENHTML_EXECUTABLE)
            message(WARNING "genhtml not found - HTML coverage report will not be available")
        endif()
    else()
        message(WARNING "Code coverage requires GNU or Clang compiler")
        set(MESH_ENABLE_COVERAGE OFF)
    endif()
endif()

#------------------------------------------------------------------------------
# Valgrind support
#------------------------------------------------------------------------------

if(MESH_ENABLE_VALGRIND)
    find_program(VALGRIND_EXECUTABLE valgrind)
    if(NOT VALGRIND_EXECUTABLE)
        message(WARNING "valgrind not found - memory analysis will not be available")
        set(MESH_ENABLE_VALGRIND OFF)
    else()
        message(STATUS "Valgrind found at: ${VALGRIND_EXECUTABLE}")
        # Set common valgrind options
        set(VALGRIND_OPTIONS
            --leak-check=full
            --show-leak-kinds=all
            --track-origins=yes
            --verbose
            --log-file=valgrind_%p.log
            --suppressions=${CMAKE_CURRENT_SOURCE_DIR}/cmake/valgrind.supp
            CACHE STRING "Options for valgrind memory analysis"
        )
    endif()
endif()

#------------------------------------------------------------------------------
# Source files
#------------------------------------------------------------------------------

# Core components
set(MESH_CORE_SOURCES
    Core/MeshTypes.h
    Core/MeshBase.h
    Core/MeshBase.cpp
    Core/InterfaceMesh.h
)

# Add DistributedMesh only if MPI is available
if(MESH_ENABLE_MPI)
    list(APPEND MESH_CORE_SOURCES
        Core/DistributedMesh.h
        Core/DistributedMesh.cpp
    )
endif()

# Topology
set(MESH_TOPOLOGY_SOURCES
    Topology/CellShape.h
    Topology/CellShape.cpp
    Topology/CellTopology.h
    Topology/CellTopology.cpp
    Topology/MeshTopology.h
    Topology/MeshTopology.cpp
)

# Geometry
set(MESH_GEOMETRY_SOURCES
    Geometry/MeshGeometry.h
    Geometry/MeshGeometry.cpp
    Geometry/MeshQuality.h
    Geometry/MeshQuality.cpp
    Geometry/MeshOrientation.h
    Geometry/GeometryConfig.h
    Geometry/PolyGeometry.h
    Geometry/PolyGeometry.cpp
)

# Boundary
set(MESH_BOUNDARY_SOURCES
    Boundary/BoundaryKey.h
    Boundary/BoundaryComponent.h
    Boundary/BoundaryDetector.h
    Boundary/BoundaryDetector.cpp
)

# Fields
set(MESH_FIELDS_SOURCES
    Fields/MeshFields.h
    Fields/MeshFields.cpp
    Fields/MeshFieldDescriptor.h
)

# Labels
set(MESH_LABELS_SOURCES
    Labels/MeshLabels.h
    Labels/MeshLabels.cpp
)

# Search
set(MESH_SEARCH_SOURCES
    Search/MeshSearch.h
    Search/MeshSearch.cpp
    Search/SearchAccel.h
    Search/SearchAccel.cpp
    Search/SearchPrimitives.h
    Search/SearchPrimitives.cpp
    Search/SearchBuilders.h
    Search/SearchBuilders.cpp
    Search/UniformGridAccel.h
    Search/UniformGridAccel.cpp
    Search/KDTreeAccel.h
    Search/KDTreeAccel.cpp
)

list(APPEND MESH_SEARCH_SOURCES
    Search/OctreeAccel.h
    Search/OctreeAccel.cpp
    Search/RTreeAccel.h
    Search/RTreeAccel.cpp
    Search/BVHAccel.h
    Search/BVHAccel.cpp
)

# Validation
set(MESH_VALIDATION_SOURCES
    Validation/MeshValidation.h
    Validation/MeshValidation.cpp
    Validation/SpatialHashing.h
)

# Observer
set(MESH_OBSERVER_SOURCES
    Observer/MeshObserver.h
    Observer/ScopedSubscription.h
    Observer/ObserverRegistry.h
    Observer/ObserverRegistry.cpp
    Observer/EventCounterObserver.h
    Observer/EventCounterObserver.cpp
    Observer/EventLogObserver.h
    Observer/EventLogObserver.cpp
)

# I/O (basic, always included)
set(MESH_IO_SOURCES
    IO/MeshIO.h
    # Note: MeshIO.cpp is excluded due to interface mismatch - work in progress
    # IO/MeshIO.cpp
)

# VTK I/O (only if VTK is available)
if(MESH_ENABLE_VTK)
    list(APPEND MESH_IO_SOURCES
        IO/VTKReader.h
        IO/VTKReader.cpp
        IO/VTKWriter.h
        IO/VTKWriter.cpp
    )
endif()

# Main mesh interface
set(MESH_MAIN_SOURCES
    Mesh.h
)

# Combine all sources
set(MESH_ALL_SOURCES
    ${MESH_CORE_SOURCES}
    ${MESH_TOPOLOGY_SOURCES}
    ${MESH_GEOMETRY_SOURCES}
    ${MESH_BOUNDARY_SOURCES}
    ${MESH_FIELDS_SOURCES}
    ${MESH_LABELS_SOURCES}
    ${MESH_SEARCH_SOURCES}
    ${MESH_VALIDATION_SOURCES}
    ${MESH_OBSERVER_SOURCES}
    ${MESH_IO_SOURCES}
    ${MESH_MAIN_SOURCES}
)

#------------------------------------------------------------------------------
# Build library
#------------------------------------------------------------------------------

if(MESH_BUILD_SHARED)
    add_library(svmesh SHARED ${MESH_ALL_SOURCES})
    message(STATUS "Building svmesh as shared library")
else()
    add_library(svmesh STATIC ${MESH_ALL_SOURCES})
    message(STATUS "Building svmesh as static library")
endif()

# Set library properties
set_target_properties(svmesh PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER "Mesh.h"
    LINKER_LANGUAGE CXX
)

# Compile definitions
if(MESH_ENABLE_MPI)
    target_compile_definitions(svmesh PUBLIC MESH_HAS_MPI)
endif()

if(MESH_ENABLE_VTK)
    target_compile_definitions(svmesh PUBLIC MESH_HAS_VTK)
endif()

#------------------------------------------------------------------------------
# Link libraries
#------------------------------------------------------------------------------

if(MESH_ENABLE_MPI AND MPI_FOUND)
    target_link_libraries(svmesh PUBLIC ${MPI_CXX_LIBRARIES})
endif()

if(MESH_ENABLE_VTK AND VTK_FOUND)
    target_link_libraries(svmesh PUBLIC ${VTK_LIBRARIES})

    # VTK module autoinit for proper initialization (VTK 9.x)
    if(COMMAND vtk_module_autoinit)
        vtk_module_autoinit(
            TARGETS svmesh
            MODULES ${VTK_LIBRARIES}
        )
    endif()
endif()

#------------------------------------------------------------------------------
# Include directories for consumers of this library
#------------------------------------------------------------------------------

target_include_directories(svmesh PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)

# Propagate test build flag as a preprocessor define so headers can adapt
if(MESH_BUILD_TESTS)
    target_compile_definitions(svmesh PUBLIC MESH_BUILD_TESTS)
endif()

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------

if(MESH_STANDALONE_BUILD)
    include(GNUInstallDirs)

    # Install library
    install(TARGETS svmesh
        EXPORT svmeshTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/svmesh
    )

    # Install headers
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/svmesh
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "test_compile" EXCLUDE
        PATTERN "build" EXCLUDE
    )

    # Install export targets
    install(EXPORT svmeshTargets
        FILE svmeshTargets.cmake
        NAMESPACE svmesh::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svmesh
    )

    # Create and install package config file
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/svmeshConfigVersion.cmake"
        VERSION 1.0.0
        COMPATIBILITY AnyNewerVersion
    )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/svmeshConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/svmeshConfig.cmake"
        @ONLY
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/svmeshConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/svmeshConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svmesh
    )
endif()

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

if(MESH_BUILD_TESTS)
    message(STATUS "Building mesh tests")
    # Enable CTest at top-level so tests are discoverable from the build root
    enable_testing()
    add_subdirectory(Tests)
    # Legacy test_compile directory (if it exists)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_compile)
        add_subdirectory(test_compile)
    endif()

    #--------------------------------------------------------------------------
    # Coverage targets (only if coverage is enabled)
    #--------------------------------------------------------------------------

    if(MESH_ENABLE_COVERAGE AND LCOV_EXECUTABLE)
        # Create coverage baseline
        add_custom_target(coverage-baseline
            COMMAND ${LCOV_EXECUTABLE} --capture --initial
                    --directory ${CMAKE_BINARY_DIR}
                    --output-file ${CMAKE_BINARY_DIR}/coverage-base.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Capturing coverage baseline..."
        )

        # Run tests and capture coverage
        add_custom_target(coverage-run
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running tests for coverage analysis..."
        )

        # Capture coverage data after test run
        add_custom_target(coverage-capture
            COMMAND ${LCOV_EXECUTABLE} --capture
                    --directory ${CMAKE_BINARY_DIR}
                    --output-file ${CMAKE_BINARY_DIR}/coverage-test.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Capturing coverage data..."
        )

        # Combine baseline and test coverage
        add_custom_target(coverage-combine
            COMMAND ${LCOV_EXECUTABLE}
                    --add-tracefile ${CMAKE_BINARY_DIR}/coverage-base.info
                    --add-tracefile ${CMAKE_BINARY_DIR}/coverage-test.info
                    --output-file ${CMAKE_BINARY_DIR}/coverage-total.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Combining coverage data..."
        )

        # Remove external and system files from coverage
        add_custom_target(coverage-filter
            COMMAND ${LCOV_EXECUTABLE}
                    --remove ${CMAKE_BINARY_DIR}/coverage-total.info
                    '/usr/*' '*/build/*' '*/Tests/*' '*/test_compile/*' '*/ThirdParty/*' '*/_deps/*'
                    --output-file ${CMAKE_BINARY_DIR}/coverage-filtered.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Filtering coverage data..."
        )

        # Generate HTML report
        if(GENHTML_EXECUTABLE)
            add_custom_target(coverage-html
                COMMAND ${GENHTML_EXECUTABLE}
                        ${CMAKE_BINARY_DIR}/coverage-filtered.info
                        --output-directory ${CMAKE_BINARY_DIR}/coverage-report
                        --title "svMultiPhysics Mesh Library Coverage Report"
                        --legend --show-details --demangle-cpp
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating HTML coverage report..."
            )
        endif()

        # Main coverage target that runs all steps
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} --build . --target coverage-baseline
            COMMAND ${CMAKE_COMMAND} --build . --target coverage-run
            COMMAND ${CMAKE_COMMAND} --build . --target coverage-capture
            COMMAND ${CMAKE_COMMAND} --build . --target coverage-combine
            COMMAND ${CMAKE_COMMAND} --build . --target coverage-filter
            COMMAND ${GENHTML_EXECUTABLE}
                    ${CMAKE_BINARY_DIR}/coverage-filtered.info
                    --output-directory ${CMAKE_BINARY_DIR}/coverage-report
                    --title "svMultiPhysics Mesh Library Coverage Report"
                    --legend --show-details --demangle-cpp
            COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated in: ${CMAKE_BINARY_DIR}/coverage-report/index.html"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating complete coverage report..."
        )

        # Clean coverage data
        add_custom_target(coverage-clean
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage-report
            COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.info
            COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.gcda
            COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.gcno
            COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
            COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcno" -delete
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Cleaning coverage data..."
        )
    endif()

    #--------------------------------------------------------------------------
    # Valgrind memory analysis targets
    #--------------------------------------------------------------------------

    if(MESH_ENABLE_VALGRIND AND VALGRIND_EXECUTABLE)
        # Run all tests under valgrind
        add_custom_target(valgrind
            COMMAND ${CMAKE_CTEST_COMMAND}
                    --overwrite MemoryCheckCommand=${VALGRIND_EXECUTABLE}
                    --overwrite MemoryCheckCommandOptions="${VALGRIND_OPTIONS}"
                    -T MemCheck --output-on-failure
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running tests under valgrind memory checker..."
        )

        # Run specific test under valgrind (usage: make valgrind-test_DistributedMesh)
        # Create individual valgrind targets for each test executable
        if(TARGET test_DistributedMesh)
            add_custom_target(valgrind-test_DistributedMesh
                COMMAND ${VALGRIND_EXECUTABLE} ${VALGRIND_OPTIONS}
                        ${CMAKE_BINARY_DIR}/Tests/Unit/Core/test_DistributedMesh
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running test_DistributedMesh under valgrind..."
            )
        endif()

        if(TARGET test_DistributedMesh_Advanced)
            add_custom_target(valgrind-test_DistributedMesh_Advanced
                COMMAND ${VALGRIND_EXECUTABLE} ${VALGRIND_OPTIONS}
                        ${CMAKE_BINARY_DIR}/Tests/Unit/Core/test_DistributedMesh_Advanced
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running test_DistributedMesh_Advanced under valgrind..."
            )
        endif()

        # MPI tests under valgrind (requires special handling)
        if(MPI_FOUND)
            add_custom_target(valgrind-mpi
                COMMAND mpirun -n 2 ${VALGRIND_EXECUTABLE} ${VALGRIND_OPTIONS}
                        ${CMAKE_BINARY_DIR}/Tests/Unit/Core/test_DistributedMesh
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running MPI tests under valgrind..."
            )
        endif()

        # Clean valgrind logs
        add_custom_target(valgrind-clean
            COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/valgrind*.log
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Cleaning valgrind logs..."
        )
    endif()
endif()

#------------------------------------------------------------------------------
# Python bindings (optional, requires pybind11)
#------------------------------------------------------------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnablePython.cmake)

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------

message(STATUS "")
message(STATUS "svMultiPhysics Mesh Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  MPI support: ${MESH_ENABLE_MPI}")
message(STATUS "  VTK support: ${MESH_ENABLE_VTK}")
if(MESH_ENABLE_VTK)
    message(STATUS "    Use system VTK: ${USE_SYSTEM_VTK}")
    if(NOT USE_SYSTEM_VTK)
        message(STATUS "    VTK version: ${VTK_GIT_TAG}")
        message(STATUS "    VTK options (fetch):")
        message(STATUS "      Rendering: ${MESH_VTK_ENABLE_RENDERING}")
        message(STATUS "      Qt:        ${MESH_VTK_ENABLE_QT}")
        message(STATUS "      Web:       ${MESH_VTK_ENABLE_WEB}")
        message(STATUS "      IOXML:     ${MESH_VTK_ENABLE_IOXML}")
        message(STATUS "      IOLegacy:  ${MESH_VTK_ENABLE_IOLEGACY}")
    endif()
endif()
message(STATUS "  Build tests: ${MESH_BUILD_TESTS}")
if(MESH_BUILD_TESTS)
    message(STATUS "    GTest enabled: ${MESH_ENABLE_GTEST}")
    message(STATUS "    Use system GTest: ${USE_SYSTEM_GTEST}")
    message(STATUS "    Coverage analysis: ${MESH_ENABLE_COVERAGE}")
    if(MESH_ENABLE_COVERAGE AND LCOV_EXECUTABLE)
        message(STATUS "      lcov found: ${LCOV_EXECUTABLE}")
        message(STATUS "      Use 'make coverage' to generate coverage report")
    endif()
    message(STATUS "    Valgrind analysis: ${MESH_ENABLE_VALGRIND}")
    if(MESH_ENABLE_VALGRIND AND VALGRIND_EXECUTABLE)
        message(STATUS "      valgrind found: ${VALGRIND_EXECUTABLE}")
        message(STATUS "      Use 'make valgrind' to run memory analysis")
    endif()
endif()
message(STATUS "  Build shared: ${MESH_BUILD_SHARED}")
message(STATUS "  Standalone build: ${MESH_STANDALONE_BUILD}")
message(STATUS "")
