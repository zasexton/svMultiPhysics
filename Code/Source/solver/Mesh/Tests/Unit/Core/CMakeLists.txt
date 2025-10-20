# Copyright (c) Stanford University, The Regents of the University of California, and others.
# All Rights Reserved.
#
# CMakeLists.txt for DistributedMesh unit tests

cmake_minimum_required(VERSION 3.12)

# Test executables
set(TEST_SOURCES
    test_DistributedMesh.cpp
    test_DistributedMesh_Advanced.cpp
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/Core
    ${CMAKE_SOURCE_DIR}/Fields
    ${CMAKE_SOURCE_DIR}/Topology
    ${CMAKE_SOURCE_DIR}/Geometry
    ${CMAKE_SOURCE_DIR}/../..  # For solver directory
)

# Create test executables
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    # Link with mesh library and dependencies
    target_link_libraries(${test_name}
        svmesh
        ${MPI_CXX_LIBRARIES}
    )

    # Set MPI compile flags
    target_compile_options(${test_name} PRIVATE ${MPI_CXX_COMPILE_FLAGS})

    # Avoid embedding RPATHs that might include Conda paths; MPI/system libs
    # should be resolvable via the runtime linker on developer machines.
    set_target_properties(${test_name} PROPERTIES
        SKIP_BUILD_RPATH TRUE
        BUILD_RPATH ""
    )

    # Add as a test
    add_test(NAME ${test_name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS})

    # For running with more ranks
    add_test(NAME ${test_name}_4ranks
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS})
endforeach()

# Enable testing
enable_testing()

# Custom target to run all tests
add_custom_target(run_dmesh_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_DistributedMesh test_DistributedMesh_Advanced
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DistributedMesh unit tests..."
)
