/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Utils/BackendOptions.h"

namespace svmp::FE::backends {

TEST(BackendOptions, ToString)
{
    EXPECT_EQ(solverMethodToString(SolverMethod::Direct), "direct");
    EXPECT_EQ(solverMethodToString(SolverMethod::CG), "cg");
    EXPECT_EQ(solverMethodToString(SolverMethod::BiCGSTAB), "bicgstab");
    EXPECT_EQ(solverMethodToString(SolverMethod::GMRES), "gmres");
    EXPECT_EQ(solverMethodToString(SolverMethod::FGMRES), "fgmres");
    EXPECT_EQ(solverMethodToString(SolverMethod::BlockSchur), "block-schur");

    EXPECT_EQ(preconditionerToString(PreconditionerType::None), "none");
    EXPECT_EQ(preconditionerToString(PreconditionerType::Diagonal), "diagonal");
    EXPECT_EQ(preconditionerToString(PreconditionerType::ILU), "ilu");
    EXPECT_EQ(preconditionerToString(PreconditionerType::AMG), "amg");
    EXPECT_EQ(preconditionerToString(PreconditionerType::RowColumnScaling), "row-column-scaling");
    EXPECT_EQ(preconditionerToString(PreconditionerType::FieldSplit), "field-split");

    EXPECT_EQ(fieldSplitKindToString(FieldSplitKind::Additive), "additive");
    EXPECT_EQ(fieldSplitKindToString(FieldSplitKind::Multiplicative), "multiplicative");
    EXPECT_EQ(fieldSplitKindToString(FieldSplitKind::Schur), "schur");
}

} // namespace svmp::FE::backends
