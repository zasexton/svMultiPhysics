# EnableMUMPS.cmake - optional MUMPS (distributed sparse direct solver)
#
# FE_ENABLE_MUMPS=ON (default OFF) links an external, parallel, double
# precision MUMPS build and defines FE_HAS_MUMPS=1 on svfe.  MUMPS is never
# vendored; point the build at an installation with
#
#   -DSV_MUMPS_DIR=<prefix>            (prefix/include/dmumps_c.h, prefix/lib/libdmumps.a,
#                                       libmumps_common.a, libpord.a)
#
# MUMPS itself needs ScaLAPACK, BLAS/LAPACK, the METIS it was configured with,
# the MPI Fortran bindings and the Fortran runtime.  They are searched with the
# hints below; SV_MUMPS_EXTRA_LIBRARIES (a list of libraries or linker flags)
# replaces that search when set.
#
#   -DSV_MUMPS_SCALAPACK_DIR=<prefix>  (lib/libscalapack.*)
#   -DSV_MUMPS_BLAS_DIR=<prefix>       (lib/libopenblas.* or liblapack + libblas)
#   -DSV_MUMPS_METIS_DIR=<prefix>      (lib/libmetis.*; empty: MUMPS built without METIS)
#
# Results:  FE_MUMPS_INCLUDE_DIR, FE_MUMPS_LIBRARIES (cache, internal).

if(NOT FE_ENABLE_MUMPS)
    return()
endif()

if(NOT FE_ENABLE_MPI)
    message(FATAL_ERROR "FE_ENABLE_MUMPS=ON requires FE_ENABLE_MPI=ON")
endif()

set(SV_MUMPS_DIR "" CACHE PATH "MUMPS installation prefix (include/dmumps_c.h, lib/libdmumps.a)")
set(SV_MUMPS_SCALAPACK_DIR "" CACHE PATH "ScaLAPACK prefix used by MUMPS")
set(SV_MUMPS_BLAS_DIR "" CACHE PATH "BLAS/LAPACK prefix used by MUMPS (OpenBLAS or reference)")
set(SV_MUMPS_METIS_DIR "" CACHE PATH "METIS prefix used by MUMPS (empty: none)")
set(SV_MUMPS_EXTRA_LIBRARIES "" CACHE STRING
    "Libraries MUMPS depends on; replaces the ScaLAPACK/BLAS/METIS/MPI-Fortran/Fortran-runtime search")

find_path(FE_MUMPS_INCLUDE_DIR dmumps_c.h
    HINTS ${SV_MUMPS_DIR} ENV MUMPS_DIR
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
foreach(_fe_mumps_lib dmumps mumps_common pord)
    find_library(FE_MUMPS_${_fe_mumps_lib}_LIBRARY ${_fe_mumps_lib}
        HINTS ${SV_MUMPS_DIR} ENV MUMPS_DIR
        PATH_SUFFIXES lib lib64 PORD/lib
        NO_DEFAULT_PATH)
    if(NOT FE_MUMPS_${_fe_mumps_lib}_LIBRARY)
        message(FATAL_ERROR "FE_ENABLE_MUMPS: lib${_fe_mumps_lib} not found under SV_MUMPS_DIR='${SV_MUMPS_DIR}'")
    endif()
endforeach()
if(NOT FE_MUMPS_INCLUDE_DIR)
    message(FATAL_ERROR "FE_ENABLE_MUMPS: dmumps_c.h not found under SV_MUMPS_DIR='${SV_MUMPS_DIR}'")
endif()

set(_fe_mumps_libs
    ${FE_MUMPS_dmumps_LIBRARY}
    ${FE_MUMPS_mumps_common_LIBRARY}
    ${FE_MUMPS_pord_LIBRARY})

if(NOT SV_MUMPS_EXTRA_LIBRARIES STREQUAL "")
    list(APPEND _fe_mumps_libs ${SV_MUMPS_EXTRA_LIBRARIES})
else()
    find_library(FE_MUMPS_SCALAPACK_LIBRARY scalapack
        HINTS ${SV_MUMPS_SCALAPACK_DIR} ENV SCALAPACK_DIR PATH_SUFFIXES lib lib64)
    find_library(FE_MUMPS_OPENBLAS_LIBRARY openblas
        HINTS ${SV_MUMPS_BLAS_DIR} PATH_SUFFIXES lib lib64)
    if(NOT FE_MUMPS_SCALAPACK_LIBRARY)
        message(FATAL_ERROR "FE_ENABLE_MUMPS: ScaLAPACK not found (set SV_MUMPS_SCALAPACK_DIR or SV_MUMPS_EXTRA_LIBRARIES)")
    endif()
    list(APPEND _fe_mumps_libs ${FE_MUMPS_SCALAPACK_LIBRARY})
    if(FE_MUMPS_OPENBLAS_LIBRARY)
        list(APPEND _fe_mumps_libs ${FE_MUMPS_OPENBLAS_LIBRARY})
    else()
        find_library(FE_MUMPS_LAPACK_LIBRARY lapack HINTS ${SV_MUMPS_BLAS_DIR} PATH_SUFFIXES lib lib64)
        find_library(FE_MUMPS_BLAS_LIBRARY blas HINTS ${SV_MUMPS_BLAS_DIR} PATH_SUFFIXES lib lib64)
        if(NOT FE_MUMPS_LAPACK_LIBRARY OR NOT FE_MUMPS_BLAS_LIBRARY)
            message(FATAL_ERROR "FE_ENABLE_MUMPS: BLAS/LAPACK not found (set SV_MUMPS_BLAS_DIR or SV_MUMPS_EXTRA_LIBRARIES)")
        endif()
        list(APPEND _fe_mumps_libs ${FE_MUMPS_LAPACK_LIBRARY} ${FE_MUMPS_BLAS_LIBRARY})
    endif()
    if(NOT SV_MUMPS_METIS_DIR STREQUAL "")
        find_library(FE_MUMPS_METIS_LIBRARY metis HINTS ${SV_MUMPS_METIS_DIR} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
        if(NOT FE_MUMPS_METIS_LIBRARY)
            message(FATAL_ERROR "FE_ENABLE_MUMPS: libmetis not found under SV_MUMPS_METIS_DIR='${SV_MUMPS_METIS_DIR}'")
        endif()
        list(APPEND _fe_mumps_libs ${FE_MUMPS_METIS_LIBRARY})
    endif()
    # MPI Fortran bindings next to the MPI C library, and the Fortran runtime of
    # the compiler family (MUMPS is Fortran).
    set(_fe_mumps_mpi_dirs "")
    foreach(_fe_mpi_lib IN LISTS MPI_C_LIBRARIES FE_MPI_CXX_LIBRARIES)
        if(EXISTS "${_fe_mpi_lib}")
            get_filename_component(_fe_mpi_dir "${_fe_mpi_lib}" DIRECTORY)
            list(APPEND _fe_mumps_mpi_dirs "${_fe_mpi_dir}")
        endif()
    endforeach()
    foreach(_fe_mpif mpi_usempif08 mpi_usempi_ignore_tkr mpi_mpifh)
        find_library(FE_MUMPS_${_fe_mpif}_LIBRARY ${_fe_mpif} HINTS ${_fe_mumps_mpi_dirs})
        if(FE_MUMPS_${_fe_mpif}_LIBRARY)
            list(APPEND _fe_mumps_libs ${FE_MUMPS_${_fe_mpif}_LIBRARY})
        endif()
    endforeach()
    if(NOT FE_MUMPS_mpi_mpifh_LIBRARY)
        message(FATAL_ERROR "FE_ENABLE_MUMPS: MPI Fortran library mpi_mpifh not found (set SV_MUMPS_EXTRA_LIBRARIES)")
    endif()
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgfortran.so
        OUTPUT_VARIABLE _fe_mumps_gfortran OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT EXISTS "${_fe_mumps_gfortran}")
        message(FATAL_ERROR "FE_ENABLE_MUMPS: libgfortran not found next to ${CMAKE_CXX_COMPILER} (set SV_MUMPS_EXTRA_LIBRARIES)")
    endif()
    list(APPEND _fe_mumps_libs ${_fe_mumps_gfortran})
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libquadmath.so
        OUTPUT_VARIABLE _fe_mumps_quadmath OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${_fe_mumps_quadmath}")
        list(APPEND _fe_mumps_libs ${_fe_mumps_quadmath})
    endif()
endif()

set(FE_MUMPS_LIBRARIES ${_fe_mumps_libs} CACHE INTERNAL "FE MUMPS link libraries")
message(STATUS "FE: MUMPS enabled: include=${FE_MUMPS_INCLUDE_DIR}")
message(STATUS "FE: MUMPS libraries: ${FE_MUMPS_LIBRARIES}")
