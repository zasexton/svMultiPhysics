# EnableGTest.cmake
# Unified GTest discovery/fetch for Mesh tests

cmake_minimum_required(VERSION 3.12)

option(MESH_FETCH_GTEST "Fetch GoogleTest if not found on system" ON)

# Try system GTest first
find_package(GTest QUIET)

if(NOT GTest_FOUND AND MESH_FETCH_GTEST)
  message(STATUS "GTest not found on system; fetching via FetchContent...")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
  )
  # Prevent install rules from googletest
  set(INSTALL_GTEST OFF CACHE BOOL "Disable gtest install" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "Disable gmock" FORCE)
  set(BUILD_GTEST ON CACHE BOOL "Enable gtest" FORCE)
  FetchContent_MakeAvailable(googletest)

  # After add_subdirectory, imported targets GTest::gtest and GTest::gtest_main exist
  set(GTest_FOUND TRUE)
endif()

if(GTest_FOUND)
  message(STATUS "GTest enabled: using ${GTEST_LIBRARIES}")
else()
  message(STATUS "GTest not available. Tests depending on GTest will be skipped.")
endif()

