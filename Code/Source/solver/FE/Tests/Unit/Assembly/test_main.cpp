/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_main.cpp
 * @brief Main entry point for Assembly module unit tests
 */

#include <gtest/gtest.h>

#if FE_HAS_MPI
#include <mpi.h>
#include <cstdlib>
#include <string_view>
#endif

int main(int argc, char** argv) {
#if FE_HAS_MPI
    // This unit-test binary is intended to run in serial. When FE is built with MPI support,
    // avoid initializing MPI by default so the tests remain runnable in restricted environments.
    bool mpi_initialized_here = false;
    const char* init_mpi_env = std::getenv("SVMP_FE_TEST_INIT_MPI");
    if (init_mpi_env && std::string_view(init_mpi_env) == "1") {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
            MPI_Init(&argc, &argv);
            mpi_initialized_here = true;
        }
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

#if FE_HAS_MPI
    if (mpi_initialized_here) {
        MPI_Finalize();
    }
#endif

    return result;
}
