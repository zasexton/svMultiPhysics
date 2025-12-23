# EnableMPI.cmake - Configure MPI support for FE library
#
# This module configures MPI for distributed DOF maps and parallel assembly

if(FE_ENABLE_MPI)
    message(STATUS "FE: Configuring MPI support for distributed operations")

    # CMake's FindMPI module merges stderr into stdout for wrapper interrogation.
    # In sandboxed environments, OpenMPI can emit warnings (e.g. opal_ifinit socket failures)
    # that then break FindMPI's parsing of include directories. Detect that case and fall back
    # to a manual wrapper-based configuration path.
    find_program(FE_MPI_CXX_WRAPPER
        NAMES mpicxx mpic++ mpiCC
    )

    set(_fe_mpi_use_manual OFF)
    if(FE_MPI_CXX_WRAPPER)
        execute_process(
            COMMAND ${FE_MPI_CXX_WRAPPER} -showme:compile
            OUTPUT_VARIABLE _fe_mpi_showme_out OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE  _fe_mpi_showme_err OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _fe_mpi_showme_rc
        )
        if(_fe_mpi_showme_err MATCHES "opal_ifinit: socket\\(\\) failed" OR
           _fe_mpi_showme_out MATCHES "opal_ifinit: socket\\(\\) failed") # defensive
            set(_fe_mpi_use_manual ON)
            message(WARNING "FE: OpenMPI wrapper emitted warnings; using manual MPI configuration (bypassing FindMPI).")
        endif()
    endif()

    if(NOT _fe_mpi_use_manual)
        find_package(MPI REQUIRED COMPONENTS CXX)
    endif()

    if(NOT _fe_mpi_use_manual AND MPI_CXX_FOUND)
        message(STATUS "FE: MPI found")
        message(STATUS "    MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
        message(STATUS "    MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "    MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")

        if(TARGET svfe)
            # Prefer the imported target to ensure correct include/link settings.
            target_link_libraries(svfe PUBLIC MPI::MPI_CXX)
            target_compile_definitions(svfe PUBLIC FE_HAS_MPI)
            target_compile_definitions(svfe PUBLIC MESH_ENABLE_MPI)  # For compatibility with Mesh library
        endif()

        # Set up MPI compiler wrappers if requested
        if(FE_USE_MPI_WRAPPERS)
            set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
            message(STATUS "FE: Using MPI compiler wrapper: ${CMAKE_CXX_COMPILER}")
        endif()

        # Export MPI flags for downstream usage
        set(FE_MPI_CXX_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS} CACHE INTERNAL "FE MPI include directories")
        set(FE_MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES} CACHE INTERNAL "FE MPI libraries")

    elseif(_fe_mpi_use_manual)
        if(NOT FE_MPI_CXX_WRAPPER)
            message(FATAL_ERROR "FE: MPI requested but no MPI C++ compiler wrapper (mpicxx) was found")
        endif()

        message(STATUS "FE: Configuring MPI via wrapper interrogation: ${FE_MPI_CXX_WRAPPER}")

        execute_process(
            COMMAND ${FE_MPI_CXX_WRAPPER} -showme:incdirs
            OUTPUT_VARIABLE _fe_mpi_incdirs_out OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE  _fe_mpi_incdirs_err OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _fe_mpi_incdirs_rc
        )
        if(NOT _fe_mpi_incdirs_rc EQUAL 0 OR _fe_mpi_incdirs_out STREQUAL "")
            # Fallback to compile flags parsing if incdirs is unavailable.
            execute_process(
                COMMAND ${FE_MPI_CXX_WRAPPER} -showme:compile
                OUTPUT_VARIABLE _fe_mpi_incdirs_out OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_VARIABLE  _fe_mpi_incdirs_err OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _fe_mpi_incdirs_rc
            )
        endif()

        separate_arguments(_fe_mpi_inc_tokens NATIVE_COMMAND "${_fe_mpi_incdirs_out}")
        set(_fe_mpi_include_dirs "")
        foreach(tok IN LISTS _fe_mpi_inc_tokens)
            if(tok MATCHES "^-I(.+)")
                set(_dir "${CMAKE_MATCH_1}")
            else()
                set(_dir "${tok}")
            endif()
            if(IS_DIRECTORY "${_dir}")
                list(APPEND _fe_mpi_include_dirs "${_dir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _fe_mpi_include_dirs)

        execute_process(
            COMMAND ${FE_MPI_CXX_WRAPPER} -showme:libdirs
            OUTPUT_VARIABLE _fe_mpi_libdirs_out OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE  _fe_mpi_libdirs_err OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _fe_mpi_libdirs_rc
        )
        separate_arguments(_fe_mpi_libdir_tokens NATIVE_COMMAND "${_fe_mpi_libdirs_out}")
        set(_fe_mpi_libdirs "")
        foreach(dir IN LISTS _fe_mpi_libdir_tokens)
            if(IS_DIRECTORY "${dir}")
                list(APPEND _fe_mpi_libdirs "${dir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _fe_mpi_libdirs)

        execute_process(
            COMMAND ${FE_MPI_CXX_WRAPPER} -showme:libs
            OUTPUT_VARIABLE _fe_mpi_libs_out OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE  _fe_mpi_libs_err OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _fe_mpi_libs_rc
        )
        separate_arguments(_fe_mpi_lib_tokens NATIVE_COMMAND "${_fe_mpi_libs_out}")

        set(_fe_mpi_lib_paths "")
        foreach(lib IN LISTS _fe_mpi_lib_tokens)
            if(lib STREQUAL "")
                continue()
            endif()
            unset(_fe_mpi_lib CACHE)
            find_library(_fe_mpi_lib NAMES ${lib} PATHS ${_fe_mpi_libdirs})
            if(NOT _fe_mpi_lib)
                message(FATAL_ERROR "FE: Could not locate MPI library '${lib}' (search paths: ${_fe_mpi_libdirs})")
            endif()
            list(APPEND _fe_mpi_lib_paths "${_fe_mpi_lib}")
        endforeach()

        if(TARGET svfe)
            target_include_directories(svfe PUBLIC ${_fe_mpi_include_dirs})
            target_link_libraries(svfe PUBLIC ${_fe_mpi_lib_paths})
            target_compile_definitions(svfe PUBLIC FE_HAS_MPI)
            target_compile_definitions(svfe PUBLIC MESH_ENABLE_MPI)
        endif()

        set(FE_MPI_CXX_INCLUDE_DIRS ${_fe_mpi_include_dirs} CACHE INTERNAL "FE MPI include directories")
        set(FE_MPI_CXX_LIBRARIES ${_fe_mpi_lib_paths} CACHE INTERNAL "FE MPI libraries")
    else()
        message(FATAL_ERROR "FE: MPI requested but not found")
    endif()

    # For testing with MPI
    find_program(MPIEXEC_EXECUTABLE
        NAMES mpiexec mpirun
    )

    if(MPIEXEC_EXECUTABLE)
        message(STATUS "FE: Found MPI executor: ${MPIEXEC_EXECUTABLE}")
        set(FE_MPIEXEC ${MPIEXEC_EXECUTABLE} CACHE INTERNAL "FE MPI executor")

        # Define a function for adding MPI tests
        function(add_fe_mpi_test test_name num_procs)
            if(TARGET ${test_name})
                add_test(
                    NAME ${test_name}_mpi_${num_procs}
                    COMMAND ${MPIEXEC_EXECUTABLE} -np ${num_procs} $<TARGET_FILE:${test_name}>
                )
                set_tests_properties(${test_name}_mpi_${num_procs} PROPERTIES
                    TIMEOUT 120
                    PROCESSORS ${num_procs}
                    LABELS "MPI"
                )
            endif()
        endfunction()
    else()
        message(WARNING "FE: MPI executor not found, MPI tests will not be available")
    endif()

    message(STATUS "FE: MPI configuration complete")
endif()
