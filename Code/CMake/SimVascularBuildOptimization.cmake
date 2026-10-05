# Opt-in build-optimization options
#-----------------------------------------------------------------------------
# The defaults leave the production build unchanged (CMAKE_BUILD_TYPE=Release:
# -O3 for the compiler's generic target, no LTO, no PGO).  Measured effects and
# the recommended configuration are in
# Source/solver/FE/Docs/BuildOptimization.md.
#
#   SV_ENABLE_LTO=ON              link-time optimization of every target
#   SV_PGO=GENERATE|USE           profile-guided optimization (GCC >= 11)
#   SV_PGO_PROFILE_DIR=<dir>      profile data written by GENERATE, read by USE
#
# Both keep the results bitwise identical to the default build.
#
# These options must be set before the first target is created; they apply to
# every C and C++ target of the build, third-party libraries included.
#-----------------------------------------------------------------------------

option(SV_ENABLE_LTO "Build all targets with link-time optimization" OFF)

set(SV_PGO "OFF" CACHE STRING
  "Profile-guided optimization: OFF, GENERATE (instrumented build) or USE")
set_property(CACHE SV_PGO PROPERTY STRINGS OFF GENERATE USE)

set(SV_PGO_PROFILE_DIR "" CACHE PATH
  "PGO profile data directory (empty: <build>/pgo-profile)")

set(_sv_opt_summary "")

#-----------------------------------------------------------------------------
# Link-time optimization
if(SV_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _sv_ipo_supported OUTPUT _sv_ipo_output LANGUAGES C CXX)
  if(NOT _sv_ipo_supported)
    message(FATAL_ERROR "SV_ENABLE_LTO=ON, but the compiler does not support IPO/LTO: ${_sv_ipo_output}")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  list(APPEND _sv_opt_summary "LTO")
endif()

#-----------------------------------------------------------------------------
# Profile-guided optimization
string(TOUPPER "${SV_PGO}" _sv_pgo_mode)
if(NOT _sv_pgo_mode STREQUAL "OFF" AND NOT _sv_pgo_mode STREQUAL "")
  if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "SV_PGO=${SV_PGO} is implemented for GCC 11 or newer")
  endif()
  set(_sv_pgo_dir "${SV_PGO_PROFILE_DIR}")
  if("${_sv_pgo_dir}" STREQUAL "")
    set(_sv_pgo_dir "${CMAKE_BINARY_DIR}/pgo-profile")
  endif()
  # Profile files are named after the object paths relative to the build
  # directory, so a USE build in another directory finds the profile of a
  # GENERATE build with the same layout.
  if(_sv_pgo_mode STREQUAL "GENERATE")
    set(_sv_pgo_compile "-fprofile-generate=${_sv_pgo_dir}" "-fprofile-prefix-path=${CMAKE_BINARY_DIR}")
    set(_sv_pgo_link "-fprofile-generate=${_sv_pgo_dir}")
  elseif(_sv_pgo_mode STREQUAL "USE")
    if(NOT IS_DIRECTORY "${_sv_pgo_dir}")
      message(FATAL_ERROR "SV_PGO=USE: profile directory '${_sv_pgo_dir}' does not exist")
    endif()
    # -fprofile-partial-training keeps code that the training run did not
    # execute optimized normally (instead of optimized for size).
    set(_sv_pgo_compile "-fprofile-use=${_sv_pgo_dir}" "-fprofile-prefix-path=${CMAKE_BINARY_DIR}"
      "-fprofile-partial-training" "-Wno-missing-profile" "-Wno-error=coverage-mismatch")
    set(_sv_pgo_link "-fprofile-use=${_sv_pgo_dir}" "-fprofile-partial-training")
  else()
    message(FATAL_ERROR "SV_PGO must be OFF, GENERATE or USE; got '${SV_PGO}'")
  endif()
  foreach(_sv_flag IN LISTS _sv_pgo_compile)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:${_sv_flag}>)
  endforeach()
  add_link_options(${_sv_pgo_link})
  list(APPEND _sv_opt_summary "PGO ${_sv_pgo_mode} (${_sv_pgo_dir})")
endif()

if(_sv_opt_summary)
  list(JOIN _sv_opt_summary ", " _sv_opt_summary)
  message(STATUS "svMultiPhysics build optimization: ${_sv_opt_summary}")
endif()
