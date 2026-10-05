cmake_minimum_required(VERSION 3.12)

find_package(GTest QUIET)
if(NOT GTest_FOUND AND NOT TARGET GTest::gtest)
  message(FATAL_ERROR "svapplication tests require GoogleTest")
endif()

enable_testing()

add_executable(test_application
  Unit/test_ApplicationDriverLevelSetWorkflows.cpp
  Unit/test_ActiveDomainOutput.cpp
  Unit/test_EquationTranslator.cpp
  Unit/test_FreeSurfaceEnergyLedger.cpp
  Unit/test_LevelSetCurvatureSamples.cpp
  Unit/test_LevelSetCutConfiguration.cpp
  Unit/test_LevelSetEquationTranslator.cpp
  Unit/test_LevelSetMaintenanceConfiguration.cpp
  Unit/test_LevelSetMaintenanceHistory.cpp
  Unit/test_LevelSetPdeVelocityExtension.cpp
  Unit/test_LegacyParameterStubs.cpp
  Unit/test_MeshCollection.cpp
  Unit/test_MeshTranslator.cpp
  Unit/test_NearestPointIndex.cpp
  Unit/test_OpenVesselExamples.cpp
  Unit/test_OpenVesselStabilizationMetadata.cpp
  ../Core/ActiveDomainOutput.cpp
  ../Core/FreeSurfaceEnergyLedger.cpp
  ../Core/FreeSurfaceSemiImplicitReference.cpp
  ../Core/LevelSetCutConfiguration.cpp
  ../Core/LevelSetMaintenanceConfiguration.cpp
  ../Core/LevelSetCurvatureSamples.cpp
  ../Core/LevelSetMaintenanceHistory.cpp
  ../Core/LevelSetMaintenanceTransactionConsensus.cpp
  ../Core/LevelSetPdeVelocityExtension.cpp
  ../Core/LevelSetVelocityExtensionMap.cpp
  ../Core/MeshCollection.cpp
  ../Core/SimulationBuilder.cpp
  ../Translators/EquationTranslator.cpp
  ../Translators/LevelSetConfigurationParsing.cpp
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

if(FE_ENABLE_MPI AND MPIEXEC_EXECUTABLE)
  add_executable(test_application_mpi
    Unit/test_ApplicationDriverLevelSetWorkflowsMPI.cpp
    Unit/test_LevelSetMaintenanceTransactionConsensusMPI.cpp
    Unit/test_LevelSetPdeVelocityExtensionMPI.cpp
    Unit/test_LegacyParameterStubs.cpp
    ../Core/ActiveDomainOutput.cpp
    ../Core/FreeSurfaceEnergyLedger.cpp
    ../Core/FreeSurfaceSemiImplicitReference.cpp
    ../Core/LevelSetCutConfiguration.cpp
    ../Core/LevelSetMaintenanceConfiguration.cpp
    ../Core/LevelSetCurvatureSamples.cpp
    ../Core/LevelSetMaintenanceHistory.cpp
    ../Core/LevelSetMaintenanceTransactionConsensus.cpp
    ../Core/LevelSetPdeVelocityExtension.cpp
    ../Core/LevelSetVelocityExtensionMap.cpp
    ../Core/MeshCollection.cpp
    ../Core/SimulationBuilder.cpp
    ../Translators/EquationTranslator.cpp
    ../Translators/LevelSetConfigurationParsing.cpp
    ../Translators/LevelSetEquationTranslator.cpp
    ../Translators/MeshTranslator.cpp
    ../../Parameters.cpp
    ../../consts.cpp
  )

  set_target_properties(test_application_mpi PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_link_libraries(test_application_mpi
    PRIVATE
      svphysics
      ${TINYXML_LIBRARY_NAME}
  )
  if(TARGET MPI::MPI_CXX)
    target_link_libraries(test_application_mpi PRIVATE MPI::MPI_CXX)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_application_mpi PRIVATE
      -ffunction-sections
      -fdata-sections
    )
    target_link_options(test_application_mpi PRIVATE -Wl,--gc-sections)
  endif()

  # The MPI test owns MPI_Init/MPI_Finalize, so intentionally omit gtest_main.
  if(TARGET GTest::gtest)
    target_link_libraries(test_application_mpi PRIVATE GTest::gtest)
  else()
    set(_application_mpi_gtest_libs ${GTEST_LIBRARIES})
    list(REMOVE_ITEM _application_mpi_gtest_libs gtest_main GTest::gtest_main)
    target_link_libraries(test_application_mpi PRIVATE
      ${_application_mpi_gtest_libs}
    )
  endif()

  add_test(
    NAME Application_MPI_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_application_mpi>
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(Application_MPI_2 PROPERTIES
    PROCESSORS 2
    ENVIRONMENT
      "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  )

  add_test(
    NAME Application_LevelSetMaintenanceConsensus_MPI_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_application_mpi>
            --gtest_filter=LevelSetMaintenanceTransactionConsensusMPI.TwoRankCommitRejectAndContentRevisionAgreement
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(
    Application_LevelSetMaintenanceConsensus_MPI_2
    PROPERTIES
      PROCESSORS 2
      TIMEOUT 1200
      ENVIRONMENT
        "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  )

  add_test(
    NAME Application_LevelSetMaintenanceConsensus_MPI_4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_application_mpi>
            --gtest_filter=LevelSetMaintenanceTransactionConsensusMPI.FourRankCommitRejectAndContentRevisionAgreement
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(
    Application_LevelSetMaintenanceConsensus_MPI_4
    PROPERTIES
      PROCESSORS 4
      TIMEOUT 1200
      ENVIRONMENT
        "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  )

  # End-to-end rank-count check of unfitted free-surface runs: generated
  # static-drop, capillary-wave and sessile-drop decks, a few steps of
  # svmultiphysics on 1, 2 and 4 ranks, fields and Newton counts compared
  # with the 1-rank run.  Exit code 77 (no numpy for the generators) is a skip.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  set(_free_surface_benchmarks
    "${CMAKE_SOURCE_DIR}/../tests/cases/fluid/free_surface_benchmarks")
  if(Python3_Interpreter_FOUND AND TARGET ${SV_MULTIPHYSICS_EXE} AND
     EXISTS "${_free_surface_benchmarks}")
    string(REPLACE ";" " " _free_surface_mpiexec_preflags "${MPIEXEC_PREFLAGS}")
    add_test(
      NAME Application_FreeSurfaceRankConsistency_MPI_4
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/Support/free_surface_rank_consistency.py
              --solver $<TARGET_FILE:${SV_MULTIPHYSICS_EXE}>
              --mpiexec ${MPIEXEC_EXECUTABLE}
              "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
              "--mpiexec-preflags=${_free_surface_mpiexec_preflags}"
              --benchmarks ${_free_surface_benchmarks}
              --work-dir ${CMAKE_CURRENT_BINARY_DIR}/free_surface_rank_consistency
              --ranks 1 2 4
              --steps 2
    )
    set_tests_properties(
      Application_FreeSurfaceRankConsistency_MPI_4
      PROPERTIES
        PROCESSORS 4
        TIMEOUT 3600
        SKIP_RETURN_CODE 77
        ENVIRONMENT
          "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    )
  endif()
endif()
