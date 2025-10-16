# EnablePython.cmake
# Python bindings configuration and pybind11 fetching logic for the svMultiPhysics Mesh library
#
# This module handles:
#   - System pybind11 detection
#   - FetchContent-based pybind11 fetching with deferred execution
#   - Python bindings setup via pybind11
#
# Expected input variables:
#   MESH_ENABLE_PYTHON          - Enable/disable Python bindings
#   PYBIND11_GIT_TAG            - pybind11 version to fetch (optional)
#
# Output variables:
#   pybind11_FOUND              - TRUE if pybind11 is available
#
# Side effects:
#   - Adds python subdirectory if pybind11 is available

if(MESH_ENABLE_PYTHON)
    # Try to find a system pybind11 first
    find_package(pybind11 QUIET CONFIG)

    # If not found, attempt to fetch pybind11 via FetchContent (requires CMake >= 3.11)
    if(NOT pybind11_FOUND)
        include(FetchContent OPTIONAL)
        if(COMMAND FetchContent_Declare)
            # Allow override of pybind11 tag via -DPYBIND11_GIT_TAG=...
            if(NOT DEFINED PYBIND11_GIT_TAG)
                set(PYBIND11_GIT_TAG "v2.11.1" CACHE STRING "pybind11 Git tag to fetch")
            endif()

            message(STATUS "pybind11 target version: ${PYBIND11_GIT_TAG}")

            # Suppress policy warnings from pybind11's use of deprecated FindPython modules
            # CMP0148: FindPythonInterp and FindPythonLibs are deprecated (use FindPython instead)
            if(POLICY CMP0148)
                cmake_policy(SET CMP0148 OLD)
            endif()

            FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG        ${PYBIND11_GIT_TAG}
                GIT_SHALLOW    TRUE
            )

            # Check current pybind11 cache status
            FetchContent_GetProperties(pybind11)

            # Determine if we need to fetch pybind11 (similar to VTK deferred fetch logic)
            set(NEED_PYBIND11_FETCH FALSE)
            set(PYBIND11_FETCH_APPROVED FALSE)

            if(NOT pybind11_POPULATED)
                # pybind11 source not fetched yet - check if this is first configure
                if(NOT DEFINED MESH_PYBIND11_FETCH_APPROVED)
                    # First configure - don't auto-fetch, let user review settings
                    message(STATUS "pybind11 ${PYBIND11_GIT_TAG} will be fetched on next configure")
                    message(STATUS "  -> Run 'cmake ..' again to fetch pybind11")
                    message(STATUS "  -> Or use 'ccmake ..' to change PYBIND11_GIT_TAG before fetching")
                    set(MESH_PYBIND11_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch pybind11 (set automatically)" FORCE)
                    set(NEED_PYBIND11_FETCH FALSE)
                else()
                    # Second configure - user has had chance to change settings
                    message(STATUS "pybind11 not found - fetching via FetchContent")
                    set(NEED_PYBIND11_FETCH TRUE)
                    set(PYBIND11_FETCH_APPROVED TRUE)
                endif()
            else()
                # pybind11 source exists - check if already configured
                if(TARGET pybind11::pybind11)
                    message(STATUS "pybind11 ${PYBIND11_GIT_TAG} already cached - using existing build")
                    set(NEED_PYBIND11_FETCH FALSE)
                    set(PYBIND11_FETCH_APPROVED TRUE)
                else()
                    message(STATUS "pybind11 ${PYBIND11_GIT_TAG} source cached but not configured - configuring...")
                    set(NEED_PYBIND11_FETCH FALSE)
                    set(PYBIND11_FETCH_APPROVED TRUE)
                endif()
            endif()

            # Fetch pybind11 if needed (only if approved)
            if(NEED_PYBIND11_FETCH AND PYBIND11_FETCH_APPROVED)
                # Use FetchContent_MakeAvailable if available (CMake 3.14+)
                # Falls back to manual populate for older CMake versions
                if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.14")
                    FetchContent_MakeAvailable(pybind11)
                else()
                    # Fallback for CMake 3.11-3.13
                    FetchContent_Populate(pybind11)
                    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
                endif()
                message(STATUS "pybind11 fetched successfully")
            elseif(NOT NEED_PYBIND11_FETCH AND PYBIND11_FETCH_APPROVED AND pybind11_POPULATED)
                # Already fetched, just need to add subdirectory
                if(NOT TARGET pybind11::pybind11)
                    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
                endif()
            endif()

            # Mark pybind11 as available if configured
            if(PYBIND11_FETCH_APPROVED AND (TARGET pybind11::pybind11))
                set(pybind11_FOUND TRUE)
            endif()
        else()
            message(WARNING "pybind11 not found and FetchContent not available (requires CMake >= 3.11) - Python bindings will not be built")
        endif()
    endif()

    # If pybind11 is available (found or fetched), enable bindings
    if(pybind11_FOUND OR TARGET pybind11::pybind11)
        message(STATUS "Enabling Python bindings (pybind11)")
        add_subdirectory(python)
    else()
        message(WARNING "Python bindings disabled: pybind11 is not available")
    endif()
endif()
