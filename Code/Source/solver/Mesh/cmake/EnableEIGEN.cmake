# EnableEIGEN.cmake
# Eigen configuration and fetching logic for the svMultiPhysics Mesh library
#
# This module handles:
#   - System Eigen detection
#   - FetchContent-based Eigen fetching with deferred execution
#   - Eigen include directory setup
#
# Expected input variables:
#   MESH_ENABLE_EIGEN            - Enable/disable Eigen support
#   USE_SYSTEM_EIGEN             - Use system Eigen vs FetchContent
#   EIGEN_GIT_TAG                - Eigen version to fetch (default: 3.4.0)
#
# Output variables:
#   MESH_ENABLE_EIGEN            - May be set to OFF if Eigen not available
#   EIGEN_FOUND                  - TRUE if Eigen is available
#   EIGEN3_INCLUDE_DIR           - Path to Eigen include directory
#

if(MESH_ENABLE_EIGEN)
    if(USE_SYSTEM_EIGEN)
        # Use system-installed Eigen
        # Prefer distro-installed Eigen over Conda when both are present.
        # Conda Eigen packages can be inconsistent and cause hard errors during find_package().

        # Temporarily ignore $CONDA_PREFIX in CMake's prefix search so we do not
        # default to a Conda Eigen. Restore the variable after the lookup.
        set(_SVMP_OLD_IGNORE_PREFIX "${CMAKE_IGNORE_PREFIX_PATH}")
        if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
            list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
            if(WIN32)
                # On Windows, CMake packages often live under .../Library
                list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
            endif()
            message(STATUS "EnableEIGEN: Ignoring Conda prefix for Eigen lookup: $ENV{CONDA_PREFIX}")
        endif()

        # Also guard against Eigen3_DIR pointing into the Conda prefix
        if(DEFINED Eigen3_DIR AND DEFINED ENV{CONDA_PREFIX})
            string(FIND "${Eigen3_DIR}" "$ENV{CONDA_PREFIX}" _svmp_eigendir_in_conda)
            if(NOT _svmp_eigendir_in_conda EQUAL -1)
                message(STATUS "EnableEIGEN: Eigen3_DIR points into Conda prefix; ignoring it: ${Eigen3_DIR}")
                unset(Eigen3_DIR CACHE)
                unset(Eigen3_DIR)
            endif()
        endif()

        # Try to find Eigen3, catching any errors in Eigen's config files
        find_package(Eigen3 QUIET NO_MODULE)

        # Restore previous ignore list to avoid affecting other packages
        set(CMAKE_IGNORE_PREFIX_PATH "${_SVMP_OLD_IGNORE_PREFIX}")

        if(Eigen3_FOUND OR EIGEN3_FOUND)
            # Additional check: verify Eigen include directory is actually accessible
            if(EIGEN3_INCLUDE_DIR)
                set(EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
            elseif(TARGET Eigen3::Eigen)
                get_target_property(EIGEN_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
            endif()

            if(EIGEN_INCLUDE_DIR)
                message(STATUS "System Eigen found (version ${Eigen3_VERSION}) - enabling Eigen linear algebra support")
                message(STATUS "Eigen include directory: ${EIGEN_INCLUDE_DIR}")
                set(EIGEN_FOUND TRUE)
                set(EIGEN3_INCLUDE_DIR "${EIGEN_INCLUDE_DIR}")

                # Create imported target if it doesn't exist
                if(NOT TARGET Eigen3::Eigen)
                    add_library(Eigen3::Eigen INTERFACE IMPORTED)
                    set_target_properties(Eigen3::Eigen PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
                    )
                endif()
            else()
                message(WARNING "System Eigen found but include directory not accessible")
                set(MESH_ENABLE_EIGEN OFF)
            endif()
        else()
            message(WARNING "System Eigen not found or has configuration errors - Eigen linear algebra features will be disabled")
            message(WARNING "To fetch Eigen automatically, set USE_SYSTEM_EIGEN=OFF")
            set(MESH_ENABLE_EIGEN OFF)
        endif()
    else()
        # Fetch Eigen via FetchContent (requires CMake >= 3.11)
        include(FetchContent OPTIONAL)
        if(COMMAND FetchContent_Declare)
            # Set default Eigen version if not specified
            if(NOT DEFINED EIGEN_GIT_TAG)
                set(EIGEN_GIT_TAG "3.4.0" CACHE STRING "Eigen Git tag to fetch" FORCE)
            endif()

            message(STATUS "Eigen target version: ${EIGEN_GIT_TAG}")

            # Set policy defaults for Eigen subdirectory
            # These suppress policy warnings from Eigen when using newer CMake versions
            set(CMAKE_POLICY_DEFAULT_CMP0025 NEW)  # Compiler id for Apple Clang
            set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)  # project() command manages VERSION variables
            set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)  # find_package() uses <PackageName>_ROOT variables

            # Configure Eigen build options (header-only library, minimal configuration)
            set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen documentation" FORCE)
            set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing" FORCE)
            set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkg-config" FORCE)
            set(BUILD_TESTING OFF CACHE BOOL "Disable testing in Eigen" FORCE)

            # Declare Eigen fetch source
            FetchContent_Declare(
                eigen
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG        ${EIGEN_GIT_TAG}
                GIT_SHALLOW    TRUE
            )

            # Check current Eigen cache status
            FetchContent_GetProperties(eigen)

            # Determine if we need to fetch/populate Eigen
            set(NEED_EIGEN_FETCH FALSE)
            set(NEED_EIGEN_POPULATE FALSE)
            set(EIGEN_FETCH_APPROVED FALSE)

            if(NOT eigen_POPULATED)
                # Eigen source not fetched yet - check if this is first configure
                if(NOT DEFINED MESH_EIGEN_FETCH_APPROVED)
                    # First configure - don't auto-fetch, let user review settings
                    message(STATUS "Eigen ${EIGEN_GIT_TAG} will be fetched on next configure")
                    message(STATUS "  -> Run 'cmake ..' again to fetch Eigen")
                    message(STATUS "  -> Or use 'ccmake ..' to change EIGEN_GIT_TAG before fetching")
                    set(MESH_EIGEN_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch Eigen (set automatically)" FORCE)
                    set(NEED_EIGEN_FETCH FALSE)
                    set(NEED_EIGEN_POPULATE FALSE)
                else()
                    # Second configure - user has had chance to change settings
                    message(STATUS "Eigen not found in build/_deps - will fetch from GitLab")
                    set(NEED_EIGEN_FETCH TRUE)
                    set(NEED_EIGEN_POPULATE TRUE)
                    set(EIGEN_FETCH_APPROVED TRUE)
                endif()
            else()
                # Eigen source exists - check if version matches
                if(DEFINED MESH_EIGEN_LAST_FETCHED_TAG AND NOT "${MESH_EIGEN_LAST_FETCHED_TAG}" STREQUAL "${EIGEN_GIT_TAG}")
                    # Version changed - check if user has approved the change
                    if(NOT DEFINED MESH_EIGEN_FETCH_APPROVED OR NOT MESH_EIGEN_FETCH_APPROVED)
                        message(STATUS "Eigen version changed: ${MESH_EIGEN_LAST_FETCHED_TAG} -> ${EIGEN_GIT_TAG}")
                        message(STATUS "  -> Run 'cmake ..' again to fetch new version")
                        message(STATUS "  -> Or use 'ccmake ..' to change EIGEN_GIT_TAG")
                        set(MESH_EIGEN_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch Eigen (set automatically)" FORCE)
                        set(NEED_EIGEN_FETCH FALSE)
                        set(NEED_EIGEN_POPULATE FALSE)
                    else()
                        message(STATUS "Eigen version changed: ${MESH_EIGEN_LAST_FETCHED_TAG} -> ${EIGEN_GIT_TAG}")
                        message(STATUS "Removing old Eigen and fetching new version...")
                        file(REMOVE_RECURSE "${eigen_SOURCE_DIR}" "${eigen_BINARY_DIR}")
                        set(eigen_POPULATED FALSE)
                        set(NEED_EIGEN_FETCH TRUE)
                        set(NEED_EIGEN_POPULATE TRUE)
                        set(EIGEN_FETCH_APPROVED TRUE)
                        # Reset approval for next version change
                        set(MESH_EIGEN_FETCH_APPROVED FALSE CACHE BOOL "Approval to fetch Eigen (set automatically)" FORCE)
                    endif()
                else()
                    # Same version - check if already populated
                    if(TARGET Eigen3::Eigen)
                        message(STATUS "Eigen ${EIGEN_GIT_TAG} already cached and configured - using existing fetch")
                        set(NEED_EIGEN_FETCH FALSE)
                        set(NEED_EIGEN_POPULATE FALSE)
                        set(EIGEN_FETCH_APPROVED TRUE)
                    else()
                        message(STATUS "Eigen ${EIGEN_GIT_TAG} source cached but not configured - setting up...")
                        set(NEED_EIGEN_FETCH FALSE)
                        set(NEED_EIGEN_POPULATE TRUE)
                        set(EIGEN_FETCH_APPROVED TRUE)
                    endif()
                endif()
            endif()

            # Fetch and/or populate Eigen if needed (only if approved)
            if((NEED_EIGEN_FETCH OR NEED_EIGEN_POPULATE) AND EIGEN_FETCH_APPROVED)
                if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.14")
                    # Modern API - handles both fetch and configure
                    if(NEED_EIGEN_FETCH)
                        message(STATUS "Fetching Eigen ${EIGEN_GIT_TAG} (this will take a moment)...")
                    endif()
                    FetchContent_MakeAvailable(eigen)
                    if(NEED_EIGEN_FETCH)
                        message(STATUS "Eigen source fetched successfully")
                    endif()
                else()
                    # Fallback for CMake 3.11-3.13
                    if(NEED_EIGEN_FETCH)
                        message(STATUS "Fetching Eigen ${EIGEN_GIT_TAG} (this will take a moment)...")
                        FetchContent_Populate(eigen)
                        message(STATUS "Eigen source fetched successfully")
                    endif()
                    if(NEED_EIGEN_POPULATE)
                        # Eigen is header-only, so we just need to set include directories
                        # and create imported target manually
                        set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")

                        if(NOT TARGET Eigen3::Eigen)
                            add_library(Eigen3::Eigen INTERFACE IMPORTED)
                            set_target_properties(Eigen3::Eigen PROPERTIES
                                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
                            )
                        endif()
                    endif()
                endif()
            endif()

            # Check if Eigen is actually available
            if(EIGEN_FETCH_APPROVED)
                # Eigen is now available - set up Eigen include directory
                set(EIGEN_FOUND TRUE)

                # Get Eigen include directory from target or source
                if(TARGET Eigen3::Eigen)
                    get_target_property(_eigen_include_dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
                    if(_eigen_include_dir)
                        set(EIGEN3_INCLUDE_DIR "${_eigen_include_dir}")
                    elseif(eigen_SOURCE_DIR)
                        set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
                    endif()
                elseif(eigen_SOURCE_DIR)
                    set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
                    # Create target if it doesn't exist
                    if(NOT TARGET Eigen3::Eigen)
                        add_library(Eigen3::Eigen INTERFACE IMPORTED)
                        set_target_properties(Eigen3::Eigen PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
                        )
                    endif()
                endif()

                # Save the current tag for future comparisons
                set(MESH_EIGEN_LAST_FETCHED_TAG "${EIGEN_GIT_TAG}" CACHE STRING "Last fetched Eigen Git tag" FORCE)
                message(STATUS "Eigen linear algebra support enabled")
                message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
            else()
                # Eigen not fetched yet - disable Eigen support temporarily
                message(STATUS "Eigen support temporarily disabled (pending fetch)")
                set(MESH_ENABLE_EIGEN OFF)
            endif()
        else()
            message(WARNING "FetchContent not available (requires CMake >= 3.11)")
            message(WARNING "Cannot fetch Eigen automatically. Either:")
            message(WARNING "  1. Install Eigen system-wide and set USE_SYSTEM_EIGEN=ON")
            message(WARNING "  2. Upgrade to CMake 3.11 or newer")
            message(WARNING "  3. Disable Eigen with MESH_ENABLE_EIGEN=OFF")
            set(MESH_ENABLE_EIGEN OFF)
        endif()
    endif()
else()
    message(STATUS "Eigen support disabled")
endif()
