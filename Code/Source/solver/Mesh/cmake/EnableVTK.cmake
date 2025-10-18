# EnableVTK.cmake
# VTK configuration and fetching logic for the svMultiPhysics Mesh library
#
# This module handles:
#   - System VTK detection
#   - FetchContent-based VTK fetching with deferred execution
#   - VTK build configuration
#   - VTK library setup
#
# Expected input variables:
#   MESH_ENABLE_VTK              - Enable/disable VTK support
#   USE_SYSTEM_VTK               - Use system VTK vs FetchContent
#   VTK_GIT_TAG                  - VTK version to fetch
#   MESH_VTK_ENABLE_RENDERING    - Enable rendering modules
#   MESH_VTK_ENABLE_QT           - Enable Qt modules
#   MESH_VTK_ENABLE_WEB          - Enable web modules
#   MESH_VTK_ENABLE_IOXML        - Enable IOXML module
#   MESH_VTK_ENABLE_IOLEGACY     - Enable IOLegacy module
#
# Output variables:
#   MESH_ENABLE_VTK              - May be set to OFF if VTK not available
#   VTK_FOUND                    - TRUE if VTK is available
#   VTK_LIBRARIES                - List of VTK libraries to link

if(MESH_ENABLE_VTK)
    if(USE_SYSTEM_VTK)
        # Use system-installed VTK
        # Prefer distro-installed VTK over Conda when both are present.
        # Conda VTK packages can be inconsistent (missing optional modules
        # like IOFFMPEG) and cause hard errors during find_package().

        # Temporarily ignore $CONDA_PREFIX in CMake's prefix search so we do not
        # default to a Conda VTK. Restore the variable after the lookup.
        set(_SVMP_OLD_IGNORE_PREFIX "${CMAKE_IGNORE_PREFIX_PATH}")
        if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
            list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
            if(WIN32)
                # On Windows, CMake packages often live under .../Library
                list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
            endif()
            message(STATUS "EnableVTK: Ignoring Conda prefix for VTK lookup: $ENV{CONDA_PREFIX}")
        endif()

        # Also guard against VTK_DIR pointing into the Conda prefix
        if(DEFINED VTK_DIR AND DEFINED ENV{CONDA_PREFIX})
            string(FIND "${VTK_DIR}" "$ENV{CONDA_PREFIX}" _svmp_vtkdir_in_conda)
            if(NOT _svmp_vtkdir_in_conda EQUAL -1)
                message(STATUS "EnableVTK: VTK_DIR points into Conda prefix; ignoring it: ${VTK_DIR}")
                unset(VTK_DIR CACHE)
                unset(VTK_DIR)
            endif()
        endif()

        # Try to find VTK, catching any errors in VTK's config files
        set(VTK_FIND_ERROR FALSE)
        find_package(VTK QUIET)

        # Restore previous ignore list to avoid affecting other packages (e.g., GTest)
        set(CMAKE_IGNORE_PREFIX_PATH "${_SVMP_OLD_IGNORE_PREFIX}")

        if(VTK_FOUND)
            # Additional check: verify VTK libraries are actually accessible
            message(STATUS "System VTK found (version ${VTK_VERSION}) - enabling VTK I/O support")
            if(VTK_USE_FILE AND EXISTS ${VTK_USE_FILE})
                include(${VTK_USE_FILE})
            endif()

            # For VTK 9.x, ensure VTK_LIBRARIES is set if not already
            if(NOT VTK_LIBRARIES OR VTK_LIBRARIES STREQUAL "")
                set(VTK_LIBRARIES
                    VTK::CommonCore
                    VTK::CommonDataModel
                    VTK::IOCore
                    VTK::IOLegacy
                    VTK::IOXML
                    VTK::IOParallelXML
                )
                message(STATUS "Set VTK libraries: ${VTK_LIBRARIES}")
            else()
                message(STATUS "Using system VTK libraries: ${VTK_LIBRARIES}")
            endif()
        else()
            message(WARNING "System VTK not found or has configuration errors - VTK I/O features will be disabled")
            message(WARNING "To fetch VTK automatically, set USE_SYSTEM_VTK=OFF")
            set(MESH_ENABLE_VTK OFF)
        endif()
    else()
        # Fetch VTK via FetchContent (requires CMake >= 3.11)
        include(FetchContent OPTIONAL)
        if(COMMAND FetchContent_Declare)
            message(STATUS "VTK target version: ${VTK_GIT_TAG}")

            # Set policy defaults for VTK subdirectory (fixes KWSys compatibility and warnings)
            # These suppress policy warnings from VTK 9.2.6 when using newer CMake versions
            set(CMAKE_POLICY_DEFAULT_CMP0025 NEW)  # Compiler id for Apple Clang
            set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)  # project() command manages VERSION variables
            set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)  # find_package() uses <PackageName>_ROOT variables
            set(CMAKE_POLICY_DEFAULT_CMP0146 NEW)  # Use GLVND OpenGL libraries
            set(CMAKE_POLICY_DEFAULT_CMP0174 NEW)  # cmake_parse_arguments() empty string handling
            set(CMAKE_POLICY_DEFAULT_CMP0177 NEW)  # install() DESTINATION paths are normalized

            # Tell CMake to accept VTK's old cmake_minimum_required
            # CMake 3.27+ removed support for cmake_minimum_required < 3.5
            # We need to tell it to work anyway for VTK's KWSys utility
            if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
                set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING "Minimum policy version for subprojects" FORCE)
            endif()

            # Configure VTK build options (minimal build for I/O only)
            set(VTK_BUILD_TESTING OFF CACHE BOOL "Disable VTK testing" FORCE)
            set(VTK_BUILD_EXAMPLES OFF CACHE BOOL "Disable VTK examples" FORCE)
            set(VTK_BUILD_DOCUMENTATION OFF CACHE BOOL "Disable VTK documentation" FORCE)

            # Disable modules not needed for mesh I/O (fixes compilation errors/warnings)
            set(VTK_MODULE_ENABLE_VTK_GeovisCore NO CACHE STRING "Disable geographic visualization" FORCE)
            set(VTK_MODULE_ENABLE_VTK_IOGeoJSON NO CACHE STRING "Disable GeoJSON I/O" FORCE)
            set(VTK_MODULE_ENABLE_VTK_libproj NO CACHE STRING "Disable libproj" FORCE)
            set(VTK_MODULE_ENABLE_VTK_IOCGNS NO CACHE STRING "Disable CGNS I/O" FORCE)
            set(VTK_MODULE_ENABLE_VTK_cgns NO CACHE STRING "Disable CGNS library" FORCE)
            set(VTK_MODULE_ENABLE_VTK_IOImage NO CACHE STRING "Disable image I/O" FORCE)
            set(VTK_MODULE_ENABLE_VTK_IOMovie NO CACHE STRING "Disable movie I/O" FORCE)

            # Group toggles
            if(MESH_VTK_ENABLE_RENDERING)
                set(VTK_GROUP_ENABLE_Rendering WANT CACHE STRING "Enable rendering" FORCE)
            else()
                set(VTK_GROUP_ENABLE_Rendering DONT_WANT CACHE STRING "Disable rendering" FORCE)
            endif()
            if(MESH_VTK_ENABLE_WEB)
                set(VTK_GROUP_ENABLE_Web WANT CACHE STRING "Enable web" FORCE)
            else()
                set(VTK_GROUP_ENABLE_Web DONT_WANT CACHE STRING "Disable web" FORCE)
            endif()
            if(MESH_VTK_ENABLE_QT)
                set(VTK_GROUP_ENABLE_Qt WANT CACHE STRING "Enable Qt" FORCE)
            else()
                set(VTK_GROUP_ENABLE_Qt DONT_WANT CACHE STRING "Disable Qt" FORCE)
            endif()
            # Modules
            if(MESH_VTK_ENABLE_IOXML)
                set(VTK_MODULE_ENABLE_VTK_IOXML YES CACHE STRING "Enable VTK XML I/O" FORCE)
            else()
                set(VTK_MODULE_ENABLE_VTK_IOXML NO CACHE STRING "Disable VTK XML I/O" FORCE)
            endif()
            if(MESH_VTK_ENABLE_IOLEGACY)
                set(VTK_MODULE_ENABLE_VTK_IOLegacy YES CACHE STRING "Enable VTK Legacy I/O" FORCE)
            else()
                set(VTK_MODULE_ENABLE_VTK_IOLegacy NO CACHE STRING "Disable VTK Legacy I/O" FORCE)
            endif()

            # Declare VTK fetch source
            FetchContent_Declare(
                vtk
                GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk.git
                GIT_TAG        ${VTK_GIT_TAG}
                GIT_SHALLOW    TRUE
            )

            # Check current VTK cache status
            FetchContent_GetProperties(vtk)

            # Determine if we need to fetch/build VTK
            set(NEED_VTK_FETCH FALSE)
            set(NEED_VTK_BUILD FALSE)
            set(VTK_FETCH_APPROVED FALSE)

            if(NOT vtk_POPULATED)
                # VTK source not fetched yet - check if this is first configure
                if(NOT DEFINED MESH_VTK_FETCH_APPROVED)
                    # First configure - don't auto-fetch, let user review settings
                    message(STATUS "VTK ${VTK_GIT_TAG} will be fetched on next configure")
                    message(STATUS "  -> Run 'cmake ..' again to fetch and build VTK")
                    message(STATUS "  -> Or use 'ccmake ..' to change VTK_GIT_TAG before fetching")
                    set(MESH_VTK_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch VTK (set automatically)" FORCE)
                    set(NEED_VTK_FETCH FALSE)
                    set(NEED_VTK_BUILD FALSE)
                else()
                    # Second configure - user has had chance to change settings
                    message(STATUS "VTK not found in build/_deps - will fetch from GitLab")
                    set(NEED_VTK_FETCH TRUE)
                    set(NEED_VTK_BUILD TRUE)
                    set(VTK_FETCH_APPROVED TRUE)
                endif()
            else()
                # VTK source exists - check if version matches
                if(DEFINED MESH_VTK_LAST_FETCHED_TAG AND NOT "${MESH_VTK_LAST_FETCHED_TAG}" STREQUAL "${VTK_GIT_TAG}")
                    # Version changed - check if user has approved the change
                    if(NOT DEFINED MESH_VTK_FETCH_APPROVED OR NOT MESH_VTK_FETCH_APPROVED)
                        message(STATUS "VTK version changed: ${MESH_VTK_LAST_FETCHED_TAG} -> ${VTK_GIT_TAG}")
                        message(STATUS "  -> Run 'cmake ..' again to fetch new version")
                        message(STATUS "  -> Or use 'ccmake ..' to change VTK_GIT_TAG")
                        set(MESH_VTK_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch VTK (set automatically)" FORCE)
                        set(NEED_VTK_FETCH FALSE)
                        set(NEED_VTK_BUILD FALSE)
                    else()
                        message(STATUS "VTK version changed: ${MESH_VTK_LAST_FETCHED_TAG} -> ${VTK_GIT_TAG}")
                        message(STATUS "Removing old VTK and fetching new version...")
                        file(REMOVE_RECURSE "${vtk_SOURCE_DIR}" "${vtk_BINARY_DIR}")
                        set(vtk_POPULATED FALSE)
                        set(NEED_VTK_FETCH TRUE)
                        set(NEED_VTK_BUILD TRUE)
                        set(VTK_FETCH_APPROVED TRUE)
                        # Reset approval for next version change
                        set(MESH_VTK_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch VTK (set automatically)" FORCE)
                    endif()
                else()
                    # Same version - check if already built
                    if(TARGET VTK::CommonCore)
                        message(STATUS "VTK ${VTK_GIT_TAG} already cached and built - using existing build")
                        set(NEED_VTK_FETCH FALSE)
                        set(NEED_VTK_BUILD FALSE)
                        set(VTK_FETCH_APPROVED TRUE)
                    else()
                        message(STATUS "VTK ${VTK_GIT_TAG} source cached but not built - configuring...")
                        set(NEED_VTK_FETCH FALSE)
                        set(NEED_VTK_BUILD TRUE)
                        set(VTK_FETCH_APPROVED TRUE)
                    endif()
                endif()
            endif()

            # Fetch and/or build VTK if needed (only if approved)
            if((NEED_VTK_FETCH OR NEED_VTK_BUILD) AND VTK_FETCH_APPROVED)
                if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.14")
                    # Modern API - handles both fetch and configure
                    if(NEED_VTK_FETCH)
                        message(STATUS "Fetching VTK ${VTK_GIT_TAG} (this will take a few minutes)...")
                    endif()
                    FetchContent_MakeAvailable(vtk)
                    if(NEED_VTK_FETCH)
                        message(STATUS "VTK source fetched successfully")
                    endif()
                else()
                    # Fallback for CMake 3.11-3.13
                    if(NEED_VTK_FETCH)
                        message(STATUS "Fetching VTK ${VTK_GIT_TAG} (this will take a few minutes)...")
                        FetchContent_Populate(vtk)
                        message(STATUS "VTK source fetched successfully")
                    endif()
                    if(NEED_VTK_BUILD)
                        message(STATUS "Configuring VTK build...")
                        add_subdirectory(${vtk_SOURCE_DIR} ${vtk_BINARY_DIR})
                    endif()
                endif()
            endif()

            # Check if VTK is actually available
            if(VTK_FETCH_APPROVED)
                # VTK is now available - set up VTK libraries
                set(VTK_FOUND TRUE)

                # Set VTK_LIBRARIES to the modules we need for I/O
                set(VTK_LIBRARIES
                    VTK::CommonCore
                    VTK::CommonDataModel
                    VTK::IOCore
                )

                # Add optional I/O modules if enabled
                if(MESH_VTK_ENABLE_IOLEGACY)
                    list(APPEND VTK_LIBRARIES VTK::IOLegacy)
                endif()
                if(MESH_VTK_ENABLE_IOXML)
                    list(APPEND VTK_LIBRARIES VTK::IOXML VTK::IOParallelXML)
                endif()

                # Save the current tag for future comparisons
                set(MESH_VTK_LAST_FETCHED_TAG "${VTK_GIT_TAG}" CACHE STRING "Last fetched VTK Git tag" FORCE)
                message(STATUS "VTK I/O support enabled")
                message(STATUS "VTK libraries: ${VTK_LIBRARIES}")
            else()
                # VTK not fetched yet - disable VTK support temporarily
                message(STATUS "VTK support temporarily disabled (pending fetch)")
                set(MESH_ENABLE_VTK OFF)
            endif()
        else()
            message(WARNING "FetchContent not available (requires CMake >= 3.11)")
            message(WARNING "Cannot fetch VTK automatically. Either:")
            message(WARNING "  1. Install VTK system-wide and set USE_SYSTEM_VTK=ON")
            message(WARNING "  2. Upgrade to CMake 3.11 or newer")
            message(WARNING "  3. Disable VTK with MESH_ENABLE_VTK=OFF")
            set(MESH_ENABLE_VTK OFF)
        endif()
    endif()
else()
    message(STATUS "VTK support disabled")
endif()
