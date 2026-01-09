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
#endif

int main(int argc, char** argv) {
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(&argc, &argv);
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

#if FE_HAS_MPI
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Finalize();
    }
#endif

    return result;
}
