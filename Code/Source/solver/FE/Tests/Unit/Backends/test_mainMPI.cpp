/**
 * @file test_mainMPI.cpp
 * @brief MPI-aware test main for FE/Backends distributed unit tests
 */

#include <gtest/gtest.h>

#if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
#include <Tpetra_Core.hpp>
#endif

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
#include <petscsys.h>
#endif

#include <mpi.h>

int main(int argc, char** argv)
{
#if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
    // Prefer the Trilinos scope guard when available so Tpetra is initialized correctly.
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
#else
    MPI_Init(&argc, &argv);
#endif

    ::testing::InitGoogleTest(&argc, argv);

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
    PetscBool petsc_was_initialized = PETSC_FALSE;
    PetscInitialized(&petsc_was_initialized);
    PetscBool petsc_initialized_here = PETSC_FALSE;
    if (!petsc_was_initialized) {
        PetscInitialize(&argc, &argv, nullptr, nullptr);
        petsc_initialized_here = PETSC_TRUE;
    }
#endif

    const int rc = RUN_ALL_TESTS();

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
    if (petsc_initialized_here) {
        PetscFinalize();
    }
#endif

#if !defined(FE_HAS_TRILINOS) || !FE_HAS_TRILINOS
    MPI_Finalize();
#endif

    return rc;
}
