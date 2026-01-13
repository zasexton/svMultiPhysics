/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#  include <cstdlib>
#endif

int main(int argc, char** argv)
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    auto has_env = [](const char* key) -> bool { return std::getenv(key) != nullptr; };

    const bool launched_under_mpi =
        // Open MPI
        has_env("OMPI_COMM_WORLD_SIZE") || has_env("OMPI_COMM_WORLD_RANK") ||
        // MPICH / Intel MPI / MVAPICH
        has_env("PMI_SIZE") || has_env("PMI_RANK") || has_env("PMIX_RANK") ||
        has_env("MV2_COMM_WORLD_SIZE") || has_env("MV2_COMM_WORLD_RANK") ||
        // Slurm (srun)
        has_env("SLURM_PROCID") || has_env("SLURM_NTASKS") || has_env("SLURM_JOB_ID");

    bool we_initialized_mpi = false;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized && launched_under_mpi) {
        MPI_Init(&argc, &argv);
        we_initialized_mpi = true;
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    // Avoid duplicate GTest output when running under `mpiexec -np N ...` by
    // silencing the default result printer on non-root ranks.
    int mpi_initialized_after = 0;
    MPI_Initialized(&mpi_initialized_after);
    if (mpi_initialized_after) {
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1 && rank != 0) {
            auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
            delete listeners.Release(listeners.default_result_printer());
        }
    }
#endif

    const int result = RUN_ALL_TESTS();

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
    if (we_initialized_mpi) {
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Finalize();
        }
    }
#endif

    return result;
}
