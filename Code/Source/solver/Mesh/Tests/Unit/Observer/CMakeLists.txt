# Observer Unit Tests

# Include the cmake module to enable Google Test
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/EnableGTest.cmake)

if(NOT GTest_FOUND)
  message(STATUS "GTest unavailable. Skipping Observer unit tests.")
  return()
endif()

# Test executables for Observer components
add_executable(test_MeshObserver test_MeshObserver.cpp)
add_executable(test_ObserverRegistry test_ObserverRegistry.cpp ../../../Observer/ObserverRegistry.cpp)
add_executable(test_ScopedSubscription test_ScopedSubscription.cpp)
add_executable(test_EventLogObserver test_EventLogObserver.cpp ../../../Observer/EventLogObserver.cpp)
add_executable(test_EventCounterObserver test_EventCounterObserver.cpp ../../../Observer/EventCounterObserver.cpp)

# Set include directories
target_include_directories(test_MeshObserver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_include_directories(test_ObserverRegistry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_include_directories(test_ScopedSubscription PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_include_directories(test_EventLogObserver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_include_directories(test_EventCounterObserver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

# Link with Google Test
target_link_libraries(test_MeshObserver PRIVATE GTest::gtest GTest::gtest_main pthread)
target_link_libraries(test_ObserverRegistry PRIVATE GTest::gtest GTest::gtest_main pthread)
target_link_libraries(test_ScopedSubscription PRIVATE GTest::gtest GTest::gtest_main pthread)
target_link_libraries(test_EventLogObserver PRIVATE GTest::gtest GTest::gtest_main pthread)
target_link_libraries(test_EventCounterObserver PRIVATE GTest::gtest GTest::gtest_main pthread)

# Add tests to CTest
add_test(NAME MeshObserver COMMAND test_MeshObserver)
add_test(NAME ObserverRegistry COMMAND test_ObserverRegistry)
add_test(NAME ScopedSubscription COMMAND test_ScopedSubscription)
add_test(NAME EventLogObserver COMMAND test_EventLogObserver)
add_test(NAME EventCounterObserver COMMAND test_EventCounterObserver)

# Set test properties
set_tests_properties(MeshObserver PROPERTIES
    LABELS "Unit;Observer"
    TIMEOUT 30
)

set_tests_properties(ObserverRegistry PROPERTIES
    LABELS "Unit;Observer"
    TIMEOUT 30
)

set_tests_properties(ScopedSubscription PROPERTIES
    LABELS "Unit;Observer"
    TIMEOUT 30
)

set_tests_properties(EventLogObserver PROPERTIES
    LABELS "Unit;Observer"
    TIMEOUT 30
)

set_tests_properties(EventCounterObserver PROPERTIES
    LABELS "Unit;Observer"
    TIMEOUT 30
)

# Combined test suite executable (optional)
add_executable(test_ObserverSuite
    test_MeshObserver.cpp
    test_ObserverRegistry.cpp
    test_ScopedSubscription.cpp
    test_EventLogObserver.cpp
    test_EventCounterObserver.cpp
    ../../../Observer/ObserverRegistry.cpp
    ../../../Observer/EventLogObserver.cpp
    ../../../Observer/EventCounterObserver.cpp
)

target_include_directories(test_ObserverSuite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_link_libraries(test_ObserverSuite PRIVATE GTest::gtest GTest::gtest_main pthread)
add_test(NAME ObserverSuite COMMAND test_ObserverSuite)
set_tests_properties(ObserverSuite PROPERTIES
    LABELS "Unit;Observer;Suite"
    TIMEOUT 60
)

# Enable code coverage if requested
if(ENABLE_COVERAGE)
    target_compile_options(test_MeshObserver PRIVATE --coverage)
    target_link_options(test_MeshObserver PRIVATE --coverage)

    target_compile_options(test_ObserverRegistry PRIVATE --coverage)
    target_link_options(test_ObserverRegistry PRIVATE --coverage)

    target_compile_options(test_ScopedSubscription PRIVATE --coverage)
    target_link_options(test_ScopedSubscription PRIVATE --coverage)

    target_compile_options(test_EventLogObserver PRIVATE --coverage)
    target_link_options(test_EventLogObserver PRIVATE --coverage)

    target_compile_options(test_EventCounterObserver PRIVATE --coverage)
    target_link_options(test_EventCounterObserver PRIVATE --coverage)

    target_compile_options(test_ObserverSuite PRIVATE --coverage)
    target_link_options(test_ObserverSuite PRIVATE --coverage)
endif()

# Add compile definitions for test builds
target_compile_definitions(test_MeshObserver PRIVATE MESH_TEST_BUILD)
target_compile_definitions(test_ObserverRegistry PRIVATE MESH_TEST_BUILD)
target_compile_definitions(test_ScopedSubscription PRIVATE MESH_TEST_BUILD)
target_compile_definitions(test_EventLogObserver PRIVATE MESH_TEST_BUILD)
target_compile_definitions(test_EventCounterObserver PRIVATE MESH_TEST_BUILD)
target_compile_definitions(test_ObserverSuite PRIVATE MESH_TEST_BUILD)

# Set C++ standard
set_target_properties(test_MeshObserver test_ObserverRegistry test_ScopedSubscription
                     test_EventLogObserver test_EventCounterObserver test_ObserverSuite
                     PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Add custom target to run all Observer tests
add_custom_target(run_observer_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Observer" -V
    DEPENDS test_MeshObserver test_ObserverRegistry test_ScopedSubscription
            test_EventLogObserver test_EventCounterObserver
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all Observer unit tests..."
)

# Add memory check target if valgrind is available
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(memcheck_observer_tests
        COMMAND ${VALGRIND_EXECUTABLE} --leak-check=full --show-leak-kinds=all
                --track-origins=yes --error-exitcode=1
                $<TARGET_FILE:test_ObserverSuite>
        DEPENDS test_ObserverSuite
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running Observer tests with memory leak detection..."
    )
endif()

# Print configuration summary
message(STATUS "Observer Unit Tests configured:")
message(STATUS "  - MeshObserver tests")
message(STATUS "  - ObserverRegistry tests")
message(STATUS "  - ScopedSubscription tests")
message(STATUS "  - EventLogObserver tests")
message(STATUS "  - EventCounterObserver tests")
message(STATUS "  - Combined ObserverSuite test")
