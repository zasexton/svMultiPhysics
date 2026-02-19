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
    EXPECT_EQ(solverMethodToString(SolverMethod::PGMRES), "pgmres");
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

// --- BlockLayout tests ---

TEST(BlockLayout, DefaultConstruction)
{
    BlockLayout layout{};
    EXPECT_TRUE(layout.blocks.empty());
    EXPECT_EQ(layout.totalComponents(), 0);
    EXPECT_FALSE(layout.hasSaddlePoint());
    EXPECT_EQ(layout.findBlock("velocity"), nullptr);
}

TEST(BlockLayout, NavierStokes2D)
{
    // NS 2D: velocity(2) + pressure(1) = dof=3
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 2});
    layout.blocks.push_back({"pressure", 2, 1});
    layout.momentum_block = 0;
    layout.constraint_block = 1;

    EXPECT_EQ(layout.totalComponents(), 3);
    EXPECT_TRUE(layout.hasSaddlePoint());

    const auto* vel = layout.findBlock("velocity");
    ASSERT_NE(vel, nullptr);
    EXPECT_EQ(vel->start_component, 0);
    EXPECT_EQ(vel->n_components, 2);

    const auto* pres = layout.findBlock("pressure");
    ASSERT_NE(pres, nullptr);
    EXPECT_EQ(pres->start_component, 2);
    EXPECT_EQ(pres->n_components, 1);

    EXPECT_EQ(layout.findBlock("temperature"), nullptr);
}

TEST(BlockLayout, NavierStokes3D)
{
    // NS 3D: velocity(3) + pressure(1) = dof=4
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3});
    layout.blocks.push_back({"pressure", 3, 1});
    layout.momentum_block = 0;
    layout.constraint_block = 1;

    EXPECT_EQ(layout.totalComponents(), 4);
    EXPECT_TRUE(layout.hasSaddlePoint());

    EXPECT_EQ(layout.blocks[static_cast<std::size_t>(*layout.momentum_block)].n_components, 3);
    EXPECT_EQ(layout.blocks[static_cast<std::size_t>(*layout.constraint_block)].n_components, 1);
}

TEST(BlockLayout, FSI)
{
    // FSI: displacement(3) + velocity(3) + pressure(1) = dof=7
    // Saddle-point between velocity (block 1) and pressure (block 2).
    BlockLayout layout{};
    layout.blocks.push_back({"displacement", 0, 3});
    layout.blocks.push_back({"velocity", 3, 3});
    layout.blocks.push_back({"pressure", 6, 1});
    layout.momentum_block = 1;
    layout.constraint_block = 2;

    EXPECT_EQ(layout.totalComponents(), 7);
    EXPECT_TRUE(layout.hasSaddlePoint());

    const auto& mb = layout.blocks[static_cast<std::size_t>(*layout.momentum_block)];
    EXPECT_EQ(mb.name, "velocity");
    EXPECT_EQ(mb.start_component, 3);
    EXPECT_EQ(mb.n_components, 3);

    const auto& cb = layout.blocks[static_cast<std::size_t>(*layout.constraint_block)];
    EXPECT_EQ(cb.name, "pressure");
    EXPECT_EQ(cb.start_component, 6);
    EXPECT_EQ(cb.n_components, 1);
}

TEST(BlockLayout, MultiSpeciesTransport)
{
    // N scalar blocks, no saddle-point (e.g., multi-species transport).
    BlockLayout layout{};
    layout.blocks.push_back({"c1", 0, 1});
    layout.blocks.push_back({"c2", 1, 1});
    layout.blocks.push_back({"c3", 2, 1});

    EXPECT_EQ(layout.totalComponents(), 3);
    EXPECT_FALSE(layout.hasSaddlePoint());

    ASSERT_NE(layout.findBlock("c2"), nullptr);
    EXPECT_EQ(layout.findBlock("c2")->start_component, 1);
}

TEST(BlockLayout, ThermoMechanical)
{
    // Thermo-mechanical: displacement(3) + temperature(1), no saddle-point.
    BlockLayout layout{};
    layout.blocks.push_back({"displacement", 0, 3});
    layout.blocks.push_back({"temperature", 3, 1});

    EXPECT_EQ(layout.totalComponents(), 4);
    EXPECT_FALSE(layout.hasSaddlePoint());
}

TEST(BlockLayout, InvalidSaddlePointIndices)
{
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3});
    layout.blocks.push_back({"pressure", 3, 1});

    // Out-of-range indices.
    layout.momentum_block = 5;
    layout.constraint_block = 1;
    EXPECT_FALSE(layout.hasSaddlePoint());

    // Negative index.
    layout.momentum_block = -1;
    layout.constraint_block = 1;
    EXPECT_FALSE(layout.hasSaddlePoint());

    // Only one set.
    layout.momentum_block = 0;
    layout.constraint_block = std::nullopt;
    EXPECT_FALSE(layout.hasSaddlePoint());
}

TEST(BlockLayout, SolverOptionsBlockLayout)
{
    SolverOptions opts{};
    EXPECT_FALSE(opts.block_layout.has_value());

    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3});
    layout.blocks.push_back({"pressure", 3, 1});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    opts.block_layout = layout;

    ASSERT_TRUE(opts.block_layout.has_value());
    EXPECT_EQ(opts.block_layout->totalComponents(), 4);
    EXPECT_TRUE(opts.block_layout->hasSaddlePoint());
}

} // namespace svmp::FE::backends
