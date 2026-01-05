/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file test_main.cpp
 * @brief Main entry point for Backends module unit tests
 */

#include <gtest/gtest.h>

#if defined(FE_HAS_FSILS)
#include <mpi.h>
#endif

#if defined(FE_HAS_PETSC)
#include <petscsys.h>
#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

#if defined(FE_HAS_FSILS)
    int mpi_was_initialized = 0;
    MPI_Initialized(&mpi_was_initialized);
    bool mpi_initialized_here = false;
    if (!mpi_was_initialized) {
        MPI_Init(&argc, &argv);
        mpi_initialized_here = true;
    }
#endif

#if defined(FE_HAS_PETSC)
    PetscBool petsc_was_initialized = PETSC_FALSE;
    PetscInitialized(&petsc_was_initialized);
    PetscBool petsc_initialized_here = PETSC_FALSE;
    if (!petsc_was_initialized) {
        PetscInitialize(&argc, &argv, nullptr, nullptr);
        petsc_initialized_here = PETSC_TRUE;
    }
#endif

    const int result = RUN_ALL_TESTS();

#if defined(FE_HAS_PETSC)
    if (petsc_initialized_here) {
        PetscFinalize();
    }
#endif

#if defined(FE_HAS_FSILS)
    if (mpi_initialized_here) {
        MPI_Finalize();
    }
#endif

    return result;
}
