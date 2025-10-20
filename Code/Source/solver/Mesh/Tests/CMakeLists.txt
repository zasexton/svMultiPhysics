# Copyright (c) Stanford University, The Regents of the University of California, and others.
# All Rights Reserved.
#
# CMakeLists.txt for Mesh unit tests

cmake_minimum_required(VERSION 3.12)

# Enable testing early so subdirectories can register tests
enable_testing()

# Add Core tests (DistributedMesh) only if MPI is enabled and found
if(MESH_ENABLE_MPI AND MPI_FOUND)
    add_subdirectory(Unit/Core)
else()
    message(STATUS "Skipping Unit/Core tests (MPI not enabled or not found)")
endif()

# Add Topology unit tests (no MPI)
add_subdirectory(Unit/Topology)

# Add Geometry tests (no MPI)
add_subdirectory(Unit/Geometry)

# Add Boundary tests (no MPI)
add_subdirectory(Unit/Boundary)

# Add IO unit tests (only when VTK is enabled and found)
add_subdirectory(Unit/IO)


# Future test subdirectories can be added here
# add_subdirectory(Unit/Topology)
# add_subdirectory(Unit/Geometry)
# add_subdirectory(Unit/Fields)
# add_subdirectory(Unit/IO)

# add_subdirectory(Unit/Validation)
# add_subdirectory(Integration)
# add_subdirectory(Performance)

# Custom target to run all mesh tests
add_custom_target(run_all_mesh_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all Mesh unit tests..."
)
