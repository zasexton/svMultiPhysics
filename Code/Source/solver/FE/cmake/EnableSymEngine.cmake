#
# EnableSymEngine.cmake - Configure SymEngine for symbolic FE spaces
#
# This module provides an optional dependency on SymEngine, used for
# constructing rational H(div) spaces on pyramids and minimal H(curl)
# spaces on wedge/pyramid elements. When SymEngine is available, the
# FE library is built with FE_HAS_SYMENGINE defined and linked against
# the SymEngine C++ library.
#

if(FE_ENABLE_SYMENGINE)
    message(STATUS "FE: Configuring SymEngine support for symbolic FE spaces")

    # Try to find an existing SymEngine installation first
    find_package(SymEngine QUIET)

    if(SymEngine_FOUND)
        message(STATUS "FE: Found system SymEngine")
        set(FE_SYMENGINE_TARGET SymEngine::symengine)
    else()
        message(STATUS "FE: SymEngine not found, fetching via FetchContent")

        include(FetchContent)

        FetchContent_Declare(
            SymEngine
            GIT_REPOSITORY https://github.com/symengine/symengine.git
            GIT_TAG        v0.11.1
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )

        # Disable SymEngine tests and extras for this embed
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

        # SymEngine v0.11.1 uses cmake_minimum_required < 3.5 which is deprecated.
        # Set this policy to allow the older CMake version requirement.
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version" FORCE)

        FetchContent_MakeAvailable(SymEngine)

        # Prefer the non-namespaced symengine target created by SymEngine's build.
        if(TARGET symengine)
            set(FE_SYMENGINE_TARGET symengine)
            # SymEngine is a third-party dependency; silence its warnings so that
            # FE warning flags (-Wall, -Wextra, -Wsign-conversion, etc.) do not
            # flood the build with external diagnostics.
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(symengine PRIVATE -w)
            endif()
        elseif(TARGET SymEngine::symengine)
            set(FE_SYMENGINE_TARGET SymEngine::symengine)
        else()
            message(FATAL_ERROR "FE: SymEngine FetchContent completed but no library target was found")
        endif()
    endif()

    if(TARGET svfe AND FE_SYMENGINE_TARGET)
        target_link_libraries(svfe PUBLIC ${FE_SYMENGINE_TARGET})
        target_compile_definitions(svfe PUBLIC FE_HAS_SYMENGINE)
        message(STATUS "FE: SymEngine linked to svfe with FE_HAS_SYMENGINE defined")
    endif()

endif()
