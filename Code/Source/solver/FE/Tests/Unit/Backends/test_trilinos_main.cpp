/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#if defined(FE_HAS_TRILINOS)
#include <Tpetra_Core.hpp>
#endif

int main(int argc, char** argv)
{
#if defined(FE_HAS_TRILINOS)
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

