# EnableGTest.cmake - Configure Google Test for FE library testing
#
# This module finds or fetches Google Test for unit testing the FE library

if(FE_BUILD_TESTS)
    message(STATUS "FE: Configuring Google Test for unit testing")

    if(FE_USE_SYSTEM_GTEST)
        # Try to find system-installed Google Test
        # Prefer CMake's FindGTest module over environment-provided CONFIG packages
        # (e.g., Conda), which can inject incompatible libraries/RUNPATHs.
        if(DEFINED GTest_DIR AND (GTest_DIR MATCHES "miniconda" OR GTest_DIR MATCHES "conda"))
            message(STATUS "FE: Ignoring Conda-provided GTest package at ${GTest_DIR}")
            unset(GTest_DIR CACHE)
        endif()

        set(_fe_saved_ignore_prefix_path "${CMAKE_IGNORE_PREFIX_PATH}")
        set(_fe_saved_ignore_path "${CMAKE_IGNORE_PATH}")
        if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
            list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
            list(APPEND CMAKE_IGNORE_PATH "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/include")
        endif()

        find_package(GTest QUIET MODULE)

        set(CMAKE_IGNORE_PREFIX_PATH "${_fe_saved_ignore_prefix_path}")
        set(CMAKE_IGNORE_PATH "${_fe_saved_ignore_path}")

        if(GTest_FOUND)
            message(STATUS "FE: Found system Google Test")
            # Normalize to a consistent variable used throughout the FE build.
            if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
                set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
            elseif(TARGET GTest::gtest)
                set(GTEST_LIBRARIES GTest::gtest)
            endif()
        else()
            message(WARNING "FE: System Google Test not found, will fetch from GitHub")
            set(FE_USE_SYSTEM_GTEST OFF)
        endif()
    endif()

    if(NOT FE_USE_SYSTEM_GTEST)
        # Fetch Google Test from GitHub
        include(FetchContent)

        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )

        # Prevent Google Test from overriding our compiler/linker options
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)

        # Set variables for consistency
        set(GTest_FOUND TRUE)
        set(GTEST_LIBRARIES gtest gtest_main)
        set(GTEST_INCLUDE_DIRS ${googletest_SOURCE_DIR}/googletest/include)
    endif()

    # Enable CTest
    enable_testing()
    include(CTest)

    # Define a function to simplify adding FE tests
    function(add_fe_test test_name)
        set(options)
        set(oneValueArgs)
        set(multiValueArgs SOURCES LINKS)
        cmake_parse_arguments(FE_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        add_executable(${test_name} ${FE_TEST_SOURCES})
        target_link_libraries(${test_name}
            PRIVATE
            svfe
            ${GTEST_LIBRARIES}
            ${FE_TEST_LINKS}
        )

        # Add test to CTest
        add_test(NAME ${test_name} COMMAND ${test_name})

        # Set test properties for better output
        set_tests_properties(${test_name} PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "GTEST_COLOR=yes"
        )

        # Group tests in IDE
        set_target_properties(${test_name} PROPERTIES
            FOLDER "Tests/FE"
        )
    endfunction()

    message(STATUS "FE: Google Test configuration complete")
endif()
