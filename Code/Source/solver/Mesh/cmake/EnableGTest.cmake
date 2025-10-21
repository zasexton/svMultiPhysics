cmake_minimum_required(VERSION 3.12)

# Unified GTest discovery/fetch for Mesh tests

# Respect top-level options if defined; otherwise provide sane defaults
if(NOT DEFINED MESH_ENABLE_GTEST)
  option(MESH_ENABLE_GTEST "Enable building GTest-based unit tests" ON)
endif()

if(NOT DEFINED USE_SYSTEM_GTEST)
  option(USE_SYSTEM_GTEST "Use system-installed GTest (no fetch)" ON)
endif()

# Backward-compat: map deprecated MESH_FETCH_GTEST if provided
if(DEFINED MESH_FETCH_GTEST)
  message(WARNING "MESH_FETCH_GTEST is deprecated. Use MESH_ENABLE_GTEST and USE_SYSTEM_GTEST instead.")
  if(NOT DEFINED MESH_ENABLE_GTEST)
    set(MESH_ENABLE_GTEST ON CACHE BOOL "Enable building GTest-based unit tests" FORCE)
  endif()
  if(MESH_FETCH_GTEST)
    # Old behavior: fetch when requested
    if(NOT DEFINED USE_SYSTEM_GTEST)
      set(USE_SYSTEM_GTEST OFF CACHE BOOL "Use system-installed GTest (no fetch)" FORCE)
    endif()
  else()
    # Old behavior: don't fetch; prefer system
    if(NOT DEFINED USE_SYSTEM_GTEST)
      set(USE_SYSTEM_GTEST ON CACHE BOOL "Use system-installed GTest (no fetch)" FORCE)
    endif()
  endif()
endif()

# If gtest disabled, signal not found and exit
if(NOT MESH_ENABLE_GTEST)
  set(GTest_FOUND FALSE PARENT_SCOPE)
  set(GTest_FOUND FALSE)
  message(STATUS "GTest disabled via MESH_ENABLE_GTEST=OFF. GTest-based tests will be skipped.")
  return()
endif()

# Try system first when requested
if(USE_SYSTEM_GTEST)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    message(STATUS "GTest (system) found.")
  else()
    message(STATUS "GTest not found on system. Set -DUSE_SYSTEM_GTEST=OFF to fetch.")
  endif()
else()
  # Prefer system, but fallback to fetch
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "Fetching GoogleTest via FetchContent (USE_SYSTEM_GTEST=OFF)...")
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
    set(GTest_FOUND TRUE)
  else()
    message(STATUS "GTest (system) found.")
  endif()
endif()

if(GTest_FOUND)
  # After add_subdirectory or system find, imported targets should exist
  message(STATUS "GTest enabled: targets available (GTest::gtest, GTest::gtest_main)")
else()
  message(STATUS "GTest not available. Tests depending on GTest will be skipped.")
endif()
