# EnableEigen.cmake - Configure Eigen3 for FE Math module
#
# This module finds and configures Eigen3 for use in the FE library's Math module
# Eigen provides high-performance dense linear algebra and expression templates

if(FE_ENABLE_EIGEN)
    message(STATUS "FE: Configuring Eigen3 support for Math module")

    # Try to find system Eigen first
    find_package(Eigen3 3.3 QUIET)

    if(Eigen3_FOUND)
        message(STATUS "FE: Found system Eigen3 at ${EIGEN3_INCLUDE_DIR}")
    else()
        # If not found, use FetchContent to download Eigen
        message(STATUS "FE: System Eigen3 not found, fetching from GitLab")

        include(FetchContent)

        FetchContent_Declare(
            Eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )

        # Configure Eigen build options
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
        set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "")
        set(BUILD_TESTING OFF CACHE BOOL "")

        FetchContent_MakeAvailable(Eigen)

        # Set Eigen3 variables for consistency
        set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
        set(Eigen3_FOUND TRUE)
    endif()

    # Add Eigen include directories to the FE library target
    if(TARGET svfe)
        target_include_directories(svfe PUBLIC ${EIGEN3_INCLUDE_DIR})
        target_compile_definitions(svfe PUBLIC FE_HAS_EIGEN)
    endif()

    # Export Eigen3 dependency information
    set(FE_EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} CACHE INTERNAL "FE Eigen include directory")

    message(STATUS "FE: Eigen3 configuration complete")
endif()