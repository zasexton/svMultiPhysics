# EnableLLVMJIT.cmake - Configure LLVM OrcJIT support for FE/Forms
#
# This module discovers an LLVM installation and records the include
# directories, compile definitions, and libraries needed to build/link the
# FE/Forms OrcJIT backend when FE_ENABLE_LLVM_JIT=ON.
#
# This module is intentionally side-effect free: it does not modify any targets
# directly because it is typically included before the `svfe` target is created.

if(FE_ENABLE_LLVM_JIT)
    message(STATUS "FE: Configuring LLVM OrcJIT support for Forms")

    # Require a modern LLVM with the ORC (LLJIT) APIs we plan to use.
    # NOTE: The implementation is written to tolerate minor API differences
    # between LLVM 14+ (e.g., ObjectCache and JIT event listener hooks).
    set(_SVMP_FE_LLVM_JIT_MIN_VERSION "14.0")

    # Prefer the LLVM CMake package config (works on Linux/macOS/Windows).
    find_package(LLVM ${_SVMP_FE_LLVM_JIT_MIN_VERSION} CONFIG REQUIRED)

    if(NOT LLVM_PACKAGE_VERSION AND LLVM_VERSION_MAJOR)
        set(LLVM_PACKAGE_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
    endif()

    message(STATUS "FE: Found LLVM ${LLVM_PACKAGE_VERSION} (LLVM_DIR='${LLVM_DIR}')")

    set(FE_LLVM_DIR "${LLVM_DIR}" CACHE INTERNAL "FE LLVM config directory")
    set(FE_LLVM_VERSION "${LLVM_PACKAGE_VERSION}" CACHE INTERNAL "FE LLVM version string")
    set(FE_LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}" CACHE INTERNAL "FE LLVM include directories")

    if(DEFINED LLVM_ENABLE_RTTI)
        set(FE_LLVM_ENABLE_RTTI "${LLVM_ENABLE_RTTI}" CACHE INTERNAL "FE LLVM RTTI setting")
    endif()
    if(DEFINED LLVM_ENABLE_EH)
        set(FE_LLVM_ENABLE_EH "${LLVM_ENABLE_EH}" CACHE INTERNAL "FE LLVM exceptions setting")
    endif()

    # Choose LLVM components needed by the OrcJIT backend. Keep this list
    # minimal: llvm_map_components_to_libnames expands dependencies.
    set(_svmp_fe_llvm_jit_components
        orcjit
        native
        support
        core
        passes
    )

    llvm_map_components_to_libnames(_svmp_fe_llvm_jit_libs ${_svmp_fe_llvm_jit_components})

    set(FE_LLVM_JIT_LIBRARIES "${_svmp_fe_llvm_jit_libs}" CACHE INTERNAL "FE LLVM libraries for OrcJIT")
    set(FE_LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS}" CACHE INTERNAL "FE LLVM system libraries")

    # Compile definitions needed by LLVM headers + optional defs from LLVMConfig.
    set(_svmp_fe_llvm_jit_compile_defs
        __STDC_CONSTANT_MACROS
        __STDC_FORMAT_MACROS
        __STDC_LIMIT_MACROS
    )

    if(LLVM_DEFINITIONS)
        # Normalize to a token list; LLVM_DEFINITIONS may be a list or a raw string.
        string(REPLACE ";" " " _svmp_fe_llvm_def_string "${LLVM_DEFINITIONS}")
        separate_arguments(_svmp_fe_llvm_def_tokens NATIVE_COMMAND "${_svmp_fe_llvm_def_string}")

        foreach(tok IN LISTS _svmp_fe_llvm_def_tokens)
            if(tok MATCHES "^-D(.+)")
                list(APPEND _svmp_fe_llvm_jit_compile_defs "${CMAKE_MATCH_1}")
            elseif(tok MATCHES "^/D(.+)")
                list(APPEND _svmp_fe_llvm_jit_compile_defs "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _svmp_fe_llvm_jit_compile_defs)
    set(FE_LLVM_JIT_COMPILE_DEFINITIONS "${_svmp_fe_llvm_jit_compile_defs}" CACHE INTERNAL "FE compile definitions for LLVM JIT")

    message(STATUS "FE: LLVM JIT configuration complete")
endif()
