#include <gtest/gtest.h>

#include <cstdlib>

#if FE_HAS_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#if FE_HAS_MPI
    bool owns_mpi_initialization = false;
    if (std::getenv("SVMP_FE_RUN_MPI_TESTS") != nullptr) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) {
            int provided = MPI_THREAD_SINGLE;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
            owns_mpi_initialization = true;
        }
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

#if FE_HAS_MPI
    if (owns_mpi_initialization) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized == 0) {
            MPI_Finalize();
        }
    }
#endif

    return result;
}
