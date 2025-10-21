################################################################################
# Unified Mesh Tests CMake
# - Auto-discovers tests under Unit/* and builds/runs them.
# - Handles three kinds of tests:
#     1) MPI tests (Unit/Core): run via mpiexec with multiple ranks
#     2) GTest-based tests (include <gtest/gtest.h>): link GTest main
#     3) Standalone tests (define their own main): no GTest linkage
################################################################################

cmake_minimum_required(VERSION 3.12)

# Make sure testing is enabled from this scope
enable_testing()

# GTest configuration is controlled via top-level options:
#   -DMESH_ENABLE_GTEST=ON|OFF
#   -DUSE_SYSTEM_GTEST=ON|OFF (when OFF and ENABLE=ON, googletest is fetched)

# Provide GTest targets if available (no-op if not found)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/EnableGTest.cmake)

# Basic variables
set(MESH_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(MESH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Collect all test source files recursively
file(GLOB_RECURSE ALL_TEST_SOURCES
     RELATIVE ${MESH_TEST_ROOT}
     ${MESH_TEST_ROOT}/Unit/*/test_*.cpp)

if(NOT ALL_TEST_SOURCES)
  message(STATUS "No test sources found under ${MESH_TEST_ROOT}/Unit")
  return()
endif()

message(STATUS "Discovered Mesh unit tests:")
foreach(_s IN LISTS ALL_TEST_SOURCES)
  message(STATUS "  - ${_s}")
endforeach()

set(_built_tests)

foreach(src_rel IN LISTS ALL_TEST_SOURCES)
  # Absolute path and name parts
  set(src_abs ${MESH_TEST_ROOT}/${src_rel})
  get_filename_component(test_name ${src_abs} NAME_WE)
  get_filename_component(_src_dir ${src_abs} DIRECTORY)
  get_filename_component(_category ${_src_dir} NAME)

  # Special handling for MPI tests in Unit/Core
  if(_category STREQUAL "Core")
    if(MESH_ENABLE_MPI AND MPI_FOUND)
      add_executable(${test_name} ${src_abs})
      target_link_libraries(${test_name} PRIVATE svmesh ${MPI_CXX_LIBRARIES})
      if(MPI_CXX_COMPILE_FLAGS)
        target_compile_options(${test_name} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
      endif()
      target_compile_features(${test_name} PRIVATE cxx_std_17)
      target_include_directories(${test_name} PRIVATE ${MESH_ROOT_DIR})
      set_target_properties(${test_name} PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_RPATH "")

      # Register MPI tests (2 and 4 ranks)
      if(MPIEXEC_EXECUTABLE)
        add_test(NAME ${test_name}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(${test_name} PROPERTIES LABELS "unit;mesh;${_category}")

        add_test(NAME ${test_name}_4ranks
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(${test_name}_4ranks PROPERTIES LABELS "unit;mesh;${_category}")
      else()
        message(WARNING "MPIEXEC_EXECUTABLE not defined; cannot register ${test_name} as a test")
      endif()

      list(APPEND _built_tests ${test_name})
      message(STATUS "Enabling ${src_rel} [mpi]")
    else()
      message(STATUS "Skipping ${src_rel} (MPI not enabled or not found)")
    endif()
    continue()
  endif()

  # For non-Core tests, detect if the test defines its own main and/or uses GTest
  file(READ ${src_abs} _src_contents)
  set(_has_main FALSE)
  # Use simple substring search to avoid regex escaping pitfalls
  string(FIND "${_src_contents}" "int main(" _main_pos)
  if(NOT _main_pos EQUAL -1)
    set(_has_main TRUE)
  endif()

  set(_uses_gtest FALSE)
  if(_src_contents MATCHES "gtest/gtest.h")
    set(_uses_gtest TRUE)
  endif()


  # Optional: skip IO parity test unless VTK is enabled and found
  if(_category STREQUAL "IO" AND (NOT MESH_ENABLE_VTK OR NOT VTK_FOUND))
    message(STATUS "Skipping ${src_rel} (VTK disabled or not found)")
    continue()
  endif()

  # Build the test target based on framework detection
  if(_uses_gtest AND NOT GTest_FOUND)
    message(STATUS "GTest not found; skipping ${src_rel}")
    continue()
  endif()

  add_executable(${test_name} ${src_abs})
  target_link_libraries(${test_name} PRIVATE svmesh)
  if(_uses_gtest)
    # Only link gtest_main when the test does not provide its own main()
    if(_has_main)
      target_link_libraries(${test_name} PRIVATE GTest::gtest)
    else()
      target_link_libraries(${test_name} PRIVATE GTest::gtest GTest::gtest_main)
    endif()
  endif()
  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_include_directories(${test_name} PRIVATE ${MESH_ROOT_DIR})
  set_target_properties(${test_name} PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_RPATH "")

  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS "unit;mesh;${_category}")
  list(APPEND _built_tests ${test_name})
  # Helpful status line so users see which tests are enabled
  if(_uses_gtest)
    if(_has_main)
      message(STATUS "Enabling ${src_rel} [gtest + own main]")
    else()
      message(STATUS "Enabling ${src_rel} [gtest]")
    endif()
  else()
    if(_has_main)
      message(STATUS "Enabling ${src_rel} [standalone]")
    else()
      message(STATUS "Enabling ${src_rel} [standalone-no-main]")
    endif()
  endif()
endforeach()

if(_built_tests)
  # Aggregate target to build all tests
  add_custom_target(build_mesh_tests DEPENDS ${_built_tests})

  # Custom target to run all mesh tests with verbose output
  add_custom_target(run_all_mesh_tests
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Running all Mesh unit tests...")
else()
  message(STATUS "No Mesh tests enabled for build.")
endif()
