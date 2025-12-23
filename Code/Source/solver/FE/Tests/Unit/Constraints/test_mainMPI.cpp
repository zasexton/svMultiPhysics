/**
 * @file test_mainMPI.cpp
 * @brief MPI-aware test main for FE/Constraints distributed unit tests
 */

#include <gtest/gtest.h>

#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}

