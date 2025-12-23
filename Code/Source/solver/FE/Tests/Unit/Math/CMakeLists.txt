# CMakeLists.txt for FE Math Module Unit Tests
cmake_minimum_required(VERSION 3.14)

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for optimization and warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

    # Enable vectorization reports in debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-missed")
    endif()
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/Code/Source/solver
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/FE
    ${CMAKE_SOURCE_DIR}/Code/Source/solver/FE/Math
    ${GTEST_INCLUDE_DIRS}
)

# Define test sources
set(MATH_TEST_SOURCES
    test_TypeTraits.cpp
    test_Vector.cpp
    test_Matrix.cpp
    test_VoigtNotation.cpp
    test_Rotations.cpp
    test_Eigensolvers.cpp
    test_Interpolation.cpp
    test_QuadratureSupport.cpp
    test_SIMD.cpp
    test_ExpressionOps.cpp
    test_VectorExpr.cpp
    test_MatrixExpr.cpp
    test_Tensor.cpp
    test_MathUtils.cpp
    test_MathConstants.cpp
    test_LU.cpp
)

# Create test executable for each test file
foreach(test_source ${MATH_TEST_SOURCES})
    # Extract test name from filename
    string(REPLACE ".cpp" "" test_name ${test_source})

    # Create test executable
    add_executable(${test_name} ${test_source})

    # Link with Google Test and pthread
    target_link_libraries(${test_name}
        GTest::GTest
        GTest::Main
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )

    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})

    # Set test properties
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 30
        LABELS "Math;Unit"
    )
endforeach()

# Create a single executable that runs all tests
add_executable(test_Math_all
    ${MATH_TEST_SOURCES}
    test_main.cpp  # Main file for combined test executable
)

target_link_libraries(test_Math_all
    GTest::GTest
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

add_test(NAME Math_all COMMAND test_Math_all)
set_tests_properties(Math_all PROPERTIES
    TIMEOUT 300
    LABELS "Math;Unit;All"
)

# Performance benchmarks (optional, requires Google Benchmark)
find_package(benchmark QUIET)
if(benchmark_FOUND)
    set(MATH_BENCHMARK_SOURCES
        benchmark_Vector.cpp
        benchmark_Matrix.cpp
        benchmark_SIMD.cpp
        benchmark_Eigensolvers.cpp
        benchmark_LU.cpp
    )

    foreach(bench_source ${MATH_BENCHMARK_SOURCES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${bench_source})
            string(REPLACE ".cpp" "" bench_name ${bench_source})
            add_executable(${bench_name} ${bench_source})
            target_link_libraries(${bench_name}
                benchmark::benchmark
                benchmark::benchmark_main
                Threads::Threads
            )
        endif()
    endforeach()
endif()

# Coverage target (requires gcov/lcov)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    option(ENABLE_COVERAGE "Enable code coverage" OFF)
    if(ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E remove_directory coverage
            COMMAND ${CMAKE_COMMAND} -E make_directory coverage
            COMMAND lcov --capture --directory . --output-file coverage/coverage.info
            COMMAND lcov --remove coverage/coverage.info '/usr/*' '*/test/*' --output-file coverage/coverage.info
            COMMAND genhtml coverage/coverage.info --output-directory coverage
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
        )
    endif()
endif()

# Valgrind memory check target
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(memcheck
        COMMAND ${CMAKE_CTEST_COMMAND}
            --test-action memcheck
            --memcheck-command ${VALGRIND_EXECUTABLE}
            --memcheck-command-options "--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose"
        COMMENT "Running tests with Valgrind memory checker"
    )
endif()

# Custom target to run all math tests
add_custom_target(test_math
    COMMAND ${CMAKE_CTEST_COMMAND} -L Math -V
    DEPENDS ${MATH_TEST_SOURCES}
    COMMENT "Running all Math module unit tests"
)

# Installation rules (optional)
install(TARGETS test_Math_all
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/tests/FE/Math
)

# Print configuration summary
message(STATUS "====================================")
message(STATUS "FE Math Tests Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  GTest found: ${GTEST_FOUND}")
message(STATUS "  Benchmark found: ${benchmark_FOUND}")
message(STATUS "  Valgrind found: ${VALGRIND_EXECUTABLE}")
message(STATUS "  Coverage enabled: ${ENABLE_COVERAGE}")
message(STATUS "====================================")