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

// --- BlockRole tests ---

TEST(BlockRole, ToStringCoversAllValues)
{
    EXPECT_EQ(blockRoleToString(BlockRole::Generic), "Generic");
    EXPECT_EQ(blockRoleToString(BlockRole::PrimaryField), "PrimaryField");
    EXPECT_EQ(blockRoleToString(BlockRole::ConstraintField), "ConstraintField");
    EXPECT_EQ(blockRoleToString(BlockRole::AuxiliaryField), "AuxiliaryField");
}

TEST(BlockRole, DefaultRoleIsGeneric)
{
    BlockDescriptor desc;
    desc.name = "test";
    EXPECT_EQ(desc.role, BlockRole::Generic);
}

TEST(BlockLayout, FindBlockByRole)
{
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3, BlockRole::PrimaryField});
    layout.blocks.push_back({"pressure", 3, 1, BlockRole::ConstraintField});
    layout.blocks.push_back({"temperature", 4, 1, BlockRole::AuxiliaryField});

    const auto* primary = layout.findBlockByRole(BlockRole::PrimaryField);
    ASSERT_NE(primary, nullptr);
    EXPECT_EQ(primary->name, "velocity");

    const auto* constraint = layout.findBlockByRole(BlockRole::ConstraintField);
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->name, "pressure");

    const auto* aux = layout.findBlockByRole(BlockRole::AuxiliaryField);
    ASSERT_NE(aux, nullptr);
    EXPECT_EQ(aux->name, "temperature");

    // No second PrimaryField
    EXPECT_EQ(layout.findBlockByRole(BlockRole::Generic), nullptr);
}

TEST(BlockLayout, PrimaryFieldBlockWithRoleAnnotation)
{
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3, BlockRole::PrimaryField});
    layout.blocks.push_back({"pressure", 3, 1, BlockRole::ConstraintField});

    const auto* primary = layout.primaryFieldBlock();
    ASSERT_NE(primary, nullptr);
    EXPECT_EQ(primary->name, "velocity");

    const auto* constraint = layout.constraintFieldBlock();
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->name, "pressure");
}

TEST(BlockLayout, PrimaryFieldBlockFallsBackToMomentumBlock)
{
    // No BlockRole annotation, uses legacy momentum_block/constraint_block indices
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3});
    layout.blocks.push_back({"pressure", 3, 1});
    layout.momentum_block = 0;
    layout.constraint_block = 1;

    const auto* primary = layout.primaryFieldBlock();
    ASSERT_NE(primary, nullptr);
    EXPECT_EQ(primary->name, "velocity");

    const auto* constraint = layout.constraintFieldBlock();
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->name, "pressure");
}

TEST(BlockLayout, PrimaryFieldBlockReturnsNullWhenNoAnnotation)
{
    BlockLayout layout{};
    layout.blocks.push_back({"temperature", 0, 1});
    // No roles, no momentum/constraint indices
    EXPECT_EQ(layout.primaryFieldBlock(), nullptr);
    EXPECT_EQ(layout.constraintFieldBlock(), nullptr);
}

TEST(BlockLayout, RoleAnnotationBackwardCompatible)
{
    // Existing aggregate initialization without role (defaults to Generic)
    BlockLayout layout{};
    layout.blocks.push_back({"velocity", 0, 3});
    EXPECT_EQ(layout.blocks[0].role, BlockRole::Generic);
}

// --- TimeIntegrationDescriptor tests ---

TEST(TimeIntegrationDescriptor, DefaultConstruction)
{
    TimeIntegrationDescriptor desc{};
    EXPECT_TRUE(desc.fields.empty());
    EXPECT_TRUE(desc.global_scheme.empty());
    EXPECT_EQ(desc.maxDerivativeOrder(), 0);
    EXPECT_EQ(desc.maxHistoryDepth(), 0);
}

TEST(TimeIntegrationDescriptor, FirstOrderSystem)
{
    // Heat equation: 1 field, first-order time derivative
    TimeIntegrationDescriptor desc{};
    desc.global_scheme = "BDF2";
    desc.fields.push_back({FieldId(1), "temperature", 1, 2, ""});

    EXPECT_EQ(desc.maxDerivativeOrder(), 1);
    EXPECT_EQ(desc.maxHistoryDepth(), 2);
    EXPECT_EQ(desc.fields[0].name, "temperature");
}

TEST(TimeIntegrationDescriptor, SecondOrderSystem)
{
    // Elastodynamics: displacement with 2nd-order time derivative
    TimeIntegrationDescriptor desc{};
    desc.global_scheme = "Newmark";
    desc.fields.push_back({FieldId(1), "displacement", 2, 1, ""});

    EXPECT_EQ(desc.maxDerivativeOrder(), 2);
    EXPECT_EQ(desc.maxHistoryDepth(), 1);
}

TEST(TimeIntegrationDescriptor, MixedOrderSystem)
{
    // FSI-like: displacement (2nd-order) + pressure (1st-order) + temperature (1st-order)
    TimeIntegrationDescriptor desc{};
    desc.global_scheme = "GenAlpha";
    desc.fields.push_back({FieldId(1), "displacement", 2, 2, ""});
    desc.fields.push_back({FieldId(2), "pressure", 1, 1, ""});
    desc.fields.push_back({FieldId(3), "temperature", 1, 3, "BDF2"});

    EXPECT_EQ(desc.maxDerivativeOrder(), 2);
    EXPECT_EQ(desc.maxHistoryDepth(), 3);

    // Verify scheme override
    EXPECT_TRUE(desc.fields[0].scheme_override.empty());
    EXPECT_EQ(desc.fields[2].scheme_override, "BDF2");
}

TEST(TimeIntegrationDescriptor, SteadyState)
{
    // Steady-state: 0th-order derivative
    TimeIntegrationDescriptor desc{};
    desc.fields.push_back({FieldId(1), "velocity", 0, 1, ""});
    desc.fields.push_back({FieldId(2), "pressure", 0, 1, ""});

    EXPECT_EQ(desc.maxDerivativeOrder(), 0);
    EXPECT_EQ(desc.maxHistoryDepth(), 1);
}

// --- SolverOptions new fields tests ---

TEST(SolverOptions, BlockRoleNamesDefaultEmpty)
{
    SolverOptions opts{};
    EXPECT_TRUE(opts.block_role_names.empty());
    EXPECT_FALSE(opts.time_integration.has_value());
}

TEST(SolverOptions, BlockRoleNamesPopulated)
{
    SolverOptions opts{};
    opts.block_role_names.emplace_back(BlockRole::PrimaryField, "velocity");
    opts.block_role_names.emplace_back(BlockRole::ConstraintField, "pressure");

    ASSERT_EQ(opts.block_role_names.size(), 2u);
    EXPECT_EQ(opts.block_role_names[0].first, BlockRole::PrimaryField);
    EXPECT_EQ(opts.block_role_names[0].second, "velocity");
    EXPECT_EQ(opts.block_role_names[1].first, BlockRole::ConstraintField);
    EXPECT_EQ(opts.block_role_names[1].second, "pressure");
}

TEST(SolverOptions, TimeIntegrationOptional)
{
    SolverOptions opts{};
    EXPECT_FALSE(opts.time_integration.has_value());

    TimeIntegrationDescriptor desc{};
    desc.global_scheme = "GenAlpha";
    desc.fields.push_back({FieldId(1), "velocity", 1, 1, ""});
    opts.time_integration = desc;

    ASSERT_TRUE(opts.time_integration.has_value());
    EXPECT_EQ(opts.time_integration->global_scheme, "GenAlpha");
    EXPECT_EQ(opts.time_integration->maxDerivativeOrder(), 1);
}

} // namespace svmp::FE::backends
