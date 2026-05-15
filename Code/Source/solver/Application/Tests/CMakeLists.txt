cmake_minimum_required(VERSION 3.12)

find_package(GTest QUIET)
if(NOT GTest_FOUND AND NOT TARGET GTest::gtest)
  message(FATAL_ERROR "svapplication tests require GoogleTest")
endif()

enable_testing()

add_executable(test_application
  Unit/test_ActiveDomainOutput.cpp
  Unit/test_EquationTranslator.cpp
  Unit/test_LevelSetEquationTranslator.cpp
  Unit/test_LegacyParameterStubs.cpp
  Unit/test_MeshCollection.cpp
  Unit/test_MeshTranslator.cpp
  Unit/test_OpenVesselExamples.cpp
  Unit/test_OpenVesselStabilizationMetadata.cpp
  ../Core/ActiveDomainOutput.cpp
  ../Core/MeshCollection.cpp
  ../Translators/EquationTranslator.cpp
  ../Translators/LevelSetEquationTranslator.cpp
  ../Translators/MeshTranslator.cpp
  ../../Parameters.cpp
  ../../consts.cpp
)

set_target_properties(test_application PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

target_link_libraries(test_application
  PRIVATE
    svphysics
    ${TINYXML_LIBRARY_NAME}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_application PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(test_application PRIVATE -Wl,--gc-sections)
endif()

if(TARGET GTest::gtest_main)
  target_link_libraries(test_application PRIVATE GTest::gtest_main)
elseif(TARGET GTest::gtest)
  target_link_libraries(test_application PRIVATE GTest::gtest)
else()
  target_link_libraries(test_application PRIVATE ${GTEST_LIBRARIES})
endif()

add_test(NAME Application_Tests COMMAND test_application)
