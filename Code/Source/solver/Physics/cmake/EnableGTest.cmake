# EnableGTest.cmake - Configure Google Test for Physics library testing
#
# This module finds or fetches Google Test for unit testing the Physics library.

if(PHYSICS_BUILD_TESTS)
    message(STATUS "Physics: Configuring Google Test for unit testing")

    if(PHYSICS_USE_SYSTEM_GTEST)
        find_package(GTest QUIET)
        if(GTest_FOUND)
            message(STATUS "Physics: Found system Google Test")
            if(TARGET GTest::gtest)
                set(GTEST_LIBRARIES GTest::gtest)
            endif()
        else()
            message(WARNING "Physics: System Google Test not found, will fetch from GitHub")
            set(PHYSICS_USE_SYSTEM_GTEST OFF)
        endif()
    endif()

    if(NOT PHYSICS_USE_SYSTEM_GTEST)
        include(FetchContent)

        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )

        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)

        set(GTest_FOUND TRUE)
        set(GTEST_LIBRARIES gtest)
        set(GTEST_INCLUDE_DIRS ${googletest_SOURCE_DIR}/googletest/include)
    endif()

    enable_testing()
    include(CTest)
endif()
