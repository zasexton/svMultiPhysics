# SvmpTestEntries.cmake - CTest registration for GoogleTest executables
#
# svmp_add_gtest() registers one GoogleTest executable as one or more CTest
# entries so that `ctest -j <N>` can run a long executable on several cores:
#
#   svmp_add_gtest(<name>
#       TARGET <executable target>
#       [MPI_RANKS <n>]          launch with `mpiexec -n <n>`; PROCESSORS <n>
#       [COST <seconds>]         serial run time of the whole entry (integer);
#                                orders the first `ctest -j` run and sets the
#                                shard count
#       [SHARDS <n>]             explicit shard count (takes precedence over COST)
#       [FILTER <gtest filter>]  passed as --gtest_filter=<filter>
#       [ARGS <arg>...]          further executable arguments
#       [TIMEOUT <seconds>]
#       [LABELS <label>...]
#       [ENVIRONMENT <VAR=value>...])
#
# With SVMP_TEST_SHARDING=ON (default) an entry whose shard count is above one
# is registered as <name>_shard_<i>_of_<n>, i = 0..n-1, running the executable
# with GTEST_TOTAL_SHARDS=<n> and GTEST_SHARD_INDEX=<i>.  GoogleTest assigns
# the cases selected by the filter to shards round-robin, so every case runs in
# exactly one shard and cases added later are picked up without CMake changes.
# MPI entries are sharded the same way: every rank sees the same shard.
# To rerun one shard by hand, export the two variables shown above.
# SVMP_TEST_SHARDING=OFF registers one entry per call under <name>, as before.
#
# With SVMP_TEST_ISOLATION=ON (default) every entry runs in its own working
# directory, <binary dir>/test_work/<entry>, with TMPDIR pointing to a "tmp"
# directory inside it, so files that tests write to the working directory or
# to std::filesystem::temp_directory_path() cannot collide between entries that
# run at the same time.  The Open MPI session directory stays under
# SVMP_TEST_MPI_TMPDIR, because Unix socket paths below a deep TMPDIR can exceed
# the operating system limit.
#
# Properties such as ENVIRONMENT must be passed through this function: a later
# set_tests_properties(<name> ...) does not reach sharded entries and would
# replace the isolation environment of a single entry.

include_guard(GLOBAL)

option(SVMP_TEST_SHARDING
    "Split long GoogleTest executables into several CTest entries (GoogleTest sharding) for ctest -j; OFF registers one entry per executable"
    ON)
option(SVMP_TEST_ISOLATION
    "Run each CTest entry in its own working directory with its own TMPDIR"
    ON)
set(SVMP_TEST_SHARD_SECONDS "60" CACHE STRING
    "Target serial run time of one shard in seconds; an entry with COST c gets ceil(c / this) shards")
set(SVMP_TEST_MAX_SHARDS "32" CACHE STRING
    "Upper bound on the number of shards of one entry")
set(SVMP_TEST_MPIEXEC_PREFLAGS "--bind-to;none" CACHE STRING
    "mpiexec flags for MPI test entries (before the executable); --bind-to none keeps concurrent MPI entries from binding their ranks to the same cores")
set(SVMP_TEST_MPI_TMPDIR "/tmp" CACHE PATH
    "Open MPI session directory base (OMPI_MCA_orte_tmpdir_base) for isolated entries; empty leaves it unset")

function(_svmp_test_entry entry)
    cmake_parse_arguments(PARSE_ARGV 1 _e "" "PROCESSORS;COST;TIMEOUT" "COMMAND;LABELS;ENVIRONMENT")

    set(_env ${_e_ENVIRONMENT})
    if(SVMP_TEST_ISOLATION)
        set(_work "${CMAKE_CURRENT_BINARY_DIR}/test_work/${entry}")
        file(MAKE_DIRECTORY "${_work}/tmp")
        list(APPEND _env "TMPDIR=${_work}/tmp")
        if(NOT SVMP_TEST_MPI_TMPDIR STREQUAL "")
            list(APPEND _env "OMPI_MCA_orte_tmpdir_base=${SVMP_TEST_MPI_TMPDIR}")
        endif()
        add_test(NAME ${entry} COMMAND ${_e_COMMAND} WORKING_DIRECTORY "${_work}")
    else()
        add_test(NAME ${entry} COMMAND ${_e_COMMAND})
    endif()

    set_property(TEST ${entry} PROPERTY PROCESSORS ${_e_PROCESSORS})
    if(_env)
        set_property(TEST ${entry} PROPERTY ENVIRONMENT ${_env})
    endif()
    if(_e_LABELS)
        set_property(TEST ${entry} PROPERTY LABELS ${_e_LABELS})
    endif()
    if(_e_TIMEOUT)
        set_property(TEST ${entry} PROPERTY TIMEOUT ${_e_TIMEOUT})
    endif()
    if(_e_COST)
        set_property(TEST ${entry} PROPERTY COST ${_e_COST})
    endif()
endfunction()

function(svmp_add_gtest name)
    cmake_parse_arguments(PARSE_ARGV 1 _sg ""
        "TARGET;MPI_RANKS;COST;SHARDS;FILTER;TIMEOUT"
        "ARGS;LABELS;ENVIRONMENT")
    if(_sg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "svmp_add_gtest(${name}): unknown arguments: ${_sg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT _sg_TARGET OR NOT TARGET ${_sg_TARGET})
        message(FATAL_ERROR "svmp_add_gtest(${name}): TARGET '${_sg_TARGET}' is not an existing target")
    endif()

    set(_command)
    set(_processors 1)
    set(_labels ${_sg_LABELS})
    if(_sg_MPI_RANKS)
        if(NOT MPIEXEC_EXECUTABLE)
            message(FATAL_ERROR "svmp_add_gtest(${name}): MPI_RANKS given but MPIEXEC_EXECUTABLE is not set")
        endif()
        set(_np_flag "${MPIEXEC_NUMPROC_FLAG}")
        if(_np_flag STREQUAL "")
            set(_np_flag "-n")
        endif()
        list(APPEND _command ${MPIEXEC_EXECUTABLE} ${_np_flag} ${_sg_MPI_RANKS}
                             ${SVMP_TEST_MPIEXEC_PREFLAGS} ${MPIEXEC_PREFLAGS})
        set(_processors ${_sg_MPI_RANKS})
        if(NOT "MPI" IN_LIST _labels)
            list(APPEND _labels MPI)
        endif()
    endif()
    list(APPEND _command $<TARGET_FILE:${_sg_TARGET}>)
    if(_sg_MPI_RANKS)
        list(APPEND _command ${MPIEXEC_POSTFLAGS})
    endif()
    if(DEFINED _sg_FILTER AND NOT _sg_FILTER STREQUAL "")
        list(APPEND _command "--gtest_filter=${_sg_FILTER}")
    endif()
    list(APPEND _command ${_sg_ARGS})

    set(_shards 1)
    if(SVMP_TEST_SHARDING)
        if(_sg_SHARDS)
            set(_shards ${_sg_SHARDS})
        elseif(_sg_COST AND SVMP_TEST_SHARD_SECONDS GREATER 0)
            math(EXPR _shards "(${_sg_COST} + ${SVMP_TEST_SHARD_SECONDS} - 1) / ${SVMP_TEST_SHARD_SECONDS}")
        endif()
        if(SVMP_TEST_MAX_SHARDS GREATER 0 AND _shards GREATER SVMP_TEST_MAX_SHARDS)
            set(_shards ${SVMP_TEST_MAX_SHARDS})
        endif()
        if(_shards LESS 1)
            set(_shards 1)
        endif()
    endif()

    # Optional keywords are only passed with a value (see policy CMP0174).
    set(_timeout)
    if(_sg_TIMEOUT)
        set(_timeout TIMEOUT ${_sg_TIMEOUT})
    endif()

    if(_shards EQUAL 1)
        set(_cost)
        if(_sg_COST)
            set(_cost COST ${_sg_COST})
        endif()
        _svmp_test_entry(${name}
            COMMAND ${_command}
            PROCESSORS ${_processors}
            ${_cost}
            ${_timeout}
            LABELS ${_labels}
            ENVIRONMENT ${_sg_ENVIRONMENT})
        return()
    endif()

    set(_shard_cost)
    if(_sg_COST)
        math(EXPR _value "(${_sg_COST} + ${_shards} - 1) / ${_shards}")
        set(_shard_cost COST ${_value})
    endif()
    string(LENGTH "${_shards}" _width)
    math(EXPR _last "${_shards} - 1")
    foreach(_index RANGE 0 ${_last})
        set(_padded "000${_index}")
        string(LENGTH "${_padded}" _plen)
        math(EXPR _start "${_plen} - ${_width}")
        string(SUBSTRING "${_padded}" ${_start} ${_width} _padded)
        _svmp_test_entry(${name}_shard_${_padded}_of_${_shards}
            COMMAND ${_command}
            PROCESSORS ${_processors}
            ${_shard_cost}
            ${_timeout}
            LABELS ${_labels} shard
            ENVIRONMENT ${_sg_ENVIRONMENT}
                        GTEST_TOTAL_SHARDS=${_shards}
                        GTEST_SHARD_INDEX=${_index})
    endforeach()
endfunction()
