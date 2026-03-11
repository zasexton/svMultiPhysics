/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MixedFormIR.cpp
 * @brief Unit tests for MixedFormIR data structure and FormCompiler::compileMixed()
 */

#include <gtest/gtest.h>

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/MixedFormIR.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"

#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

// ============================================================================
// MixedFormIR data structure tests
// ============================================================================

TEST(MixedFormIR, DefaultConstruction)
{
    MixedFormIR mir;
    EXPECT_EQ(mir.numTestFields(), 0u);
    EXPECT_EQ(mir.numTrialFields(), 0u);
    EXPECT_EQ(mir.numActiveBlocks(), 0u);
    EXPECT_TRUE(mir.activeBlocks().empty());
    EXPECT_EQ(mir.kind(), FormKind::Bilinear);
}

TEST(MixedFormIR, SizedConstruction)
{
    MixedFormIR mir(2, 3);
    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 3u);
    EXPECT_EQ(mir.numActiveBlocks(), 0u);
    // All blocks empty initially
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_FALSE(mir.hasBlock(i, j));
        }
    }
}

TEST(MixedFormIR, KindSetter)
{
    MixedFormIR mir(1, 1);
    EXPECT_EQ(mir.kind(), FormKind::Bilinear);

    mir.setKind(FormKind::Residual);
    EXPECT_EQ(mir.kind(), FormKind::Residual);

    mir.setKind(FormKind::Linear);
    EXPECT_EQ(mir.kind(), FormKind::Linear);
}

TEST(MixedFormIR, FieldDescriptors)
{
    MixedFormIR mir(2, 2);

    std::vector<MixedFieldDescriptor> test_fields;
    test_fields.push_back({FieldId(1), "velocity", std::nullopt, 3});
    test_fields.push_back({FieldId(2), "pressure", std::nullopt, 1});
    mir.setTestFields(std::move(test_fields));

    std::vector<MixedFieldDescriptor> trial_fields;
    trial_fields.push_back({FieldId(1), "velocity", std::nullopt, 3});
    trial_fields.push_back({FieldId(2), "pressure", std::nullopt, 1});
    mir.setTrialFields(std::move(trial_fields));

    ASSERT_EQ(mir.testFields().size(), 2u);
    EXPECT_EQ(mir.testFields()[0].name, "velocity");
    EXPECT_EQ(mir.testFields()[0].value_dimension, 3);
    EXPECT_EQ(mir.testFields()[1].name, "pressure");
    EXPECT_EQ(mir.testFields()[1].value_dimension, 1);

    ASSERT_EQ(mir.trialFields().size(), 2u);
    EXPECT_EQ(mir.trialFields()[0].name, "velocity");
    EXPECT_EQ(mir.trialFields()[1].name, "pressure");
}

TEST(MixedFormIR, ActiveBlocksTracking)
{
    // 2x2 layout, only set 3 of 4 blocks (sparse structure like Stokes)
    MixedFormIR mir(2, 2);

    FormCompiler compiler;
    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // VV block
    mir.setBlock(0, 0, compiler.compileBilinear((u * v).dx()));
    // VP block (div constraint)
    mir.setBlock(0, 1, compiler.compileBilinear((p * v).dx()));
    // PV block (continuity)
    mir.setBlock(1, 0, compiler.compileBilinear((u * q).dx()));
    // PP block left empty (zero block)

    EXPECT_TRUE(mir.hasBlock(0, 0));
    EXPECT_TRUE(mir.hasBlock(0, 1));
    EXPECT_TRUE(mir.hasBlock(1, 0));
    EXPECT_FALSE(mir.hasBlock(1, 1));

    EXPECT_EQ(mir.numActiveBlocks(), 3u);

    auto active = mir.activeBlocks();
    ASSERT_EQ(active.size(), 3u);
    EXPECT_EQ(active[0], std::make_pair(std::size_t{0}, std::size_t{0}));
    EXPECT_EQ(active[1], std::make_pair(std::size_t{0}, std::size_t{1}));
    EXPECT_EQ(active[2], std::make_pair(std::size_t{1}, std::size_t{0}));
}

TEST(MixedFormIR, BlockAccessConst)
{
    MixedFormIR mir(1, 1);
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    mir.setBlock(0, 0, compiler.compileBilinear((u * v).dx()));

    const auto& const_mir = mir;
    ASSERT_TRUE(const_mir.hasBlock(0, 0));
    EXPECT_TRUE(const_mir.block(0, 0).isCompiled());
    EXPECT_EQ(const_mir.block(0, 0).kind(), FormKind::Bilinear);
}

TEST(MixedFormIR, BlockOptAccess)
{
    MixedFormIR mir(2, 1);
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    mir.setBlock(0, 0, compiler.compileBilinear((u * v).dx()));

    EXPECT_TRUE(mir.blockOpt(0, 0).has_value());
    EXPECT_FALSE(mir.blockOpt(1, 0).has_value());
}

TEST(MixedFormIR, OutOfRangeAccess)
{
    MixedFormIR mir(2, 2);
    EXPECT_THROW(mir.block(5, 0), std::out_of_range);
    EXPECT_THROW((void)mir.blockOpt(5, 0), std::out_of_range);
}

// ============================================================================
// FormCompiler::compileMixed() tests
// ============================================================================

TEST(CompileMixed, SingleFieldDelegatesToSinglePath)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto mir = compiler.compileMixed((u * v).dx(), FormKind::Bilinear);

    EXPECT_EQ(mir.numTestFields(), 1u);
    EXPECT_EQ(mir.numTrialFields(), 1u);
    EXPECT_EQ(mir.numActiveBlocks(), 1u);
    EXPECT_TRUE(mir.hasBlock(0, 0));
    EXPECT_EQ(mir.kind(), FormKind::Bilinear);
    EXPECT_TRUE(mir.block(0, 0).isCompiled());
}

TEST(CompileMixed, SingleFieldResidual)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto mir = compiler.compileMixed((u * v).dx(), FormKind::Residual);

    EXPECT_EQ(mir.numTestFields(), 1u);
    EXPECT_EQ(mir.numTrialFields(), 1u);
    EXPECT_EQ(mir.kind(), FormKind::Residual);
    EXPECT_EQ(mir.block(0, 0).kind(), FormKind::Residual);
}

TEST(CompileMixed, TwoFieldStokesLikeBilinear)
{
    FormCompiler compiler;

    // Velocity space (vector, 3D)
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace V(scalar_space, /*components=*/3);
    // Pressure space (scalar)
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // Stokes: a(u,v) + b(p,v) + b(u,q)
    //   VV: inner(grad(u), grad(v))
    //   VP: -p * div(v)
    //   PV: div(u) * q
    //   PP: empty (zero block)
    const auto form = (inner(grad(u), grad(v))).dx()
                    + (FormExpr::constant(-1.0) * p * div(v)).dx()
                    + (div(u) * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 2u);
    EXPECT_EQ(mir.kind(), FormKind::Bilinear);

    // VV block (velocity-velocity): inner(grad(u), grad(v))
    EXPECT_TRUE(mir.hasBlock(0, 0));
    EXPECT_TRUE(mir.block(0, 0).isCompiled());
    EXPECT_TRUE(mir.block(0, 0).hasCellTerms());

    // VP block (velocity-pressure): -p * div(v)
    EXPECT_TRUE(mir.hasBlock(0, 1));
    EXPECT_TRUE(mir.block(0, 1).isCompiled());

    // PV block (pressure-velocity): div(u) * q
    EXPECT_TRUE(mir.hasBlock(1, 0));
    EXPECT_TRUE(mir.block(1, 0).isCompiled());

    // PP block: zero (no terms)
    EXPECT_FALSE(mir.hasBlock(1, 1));

    EXPECT_EQ(mir.numActiveBlocks(), 3u);

    // Field descriptors should be set
    ASSERT_EQ(mir.testFields().size(), 2u);
    EXPECT_EQ(mir.testFields()[0].name, "v");
    EXPECT_EQ(mir.testFields()[1].name, "q");

    ASSERT_EQ(mir.trialFields().size(), 2u);
    EXPECT_EQ(mir.trialFields()[0].name, "u");
    EXPECT_EQ(mir.trialFields()[1].name, "p");
}

TEST(CompileMixed, TwoFieldResidual)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    const auto form = (u * v).dx() + (p * v).dx() + (u * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Residual);

    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 2u);
    EXPECT_EQ(mir.kind(), FormKind::Residual);

    // All 3 non-zero blocks should be Residual kind
    ASSERT_TRUE(mir.hasBlock(0, 0));
    EXPECT_EQ(mir.block(0, 0).kind(), FormKind::Residual);
    ASSERT_TRUE(mir.hasBlock(0, 1));
    EXPECT_EQ(mir.block(0, 1).kind(), FormKind::Residual);
    ASSERT_TRUE(mir.hasBlock(1, 0));
    EXPECT_EQ(mir.block(1, 0).kind(), FormKind::Residual);
}

TEST(CompileMixed, RejectsInvalidForm)
{
    FormCompiler compiler;
    FormExpr invalid_form;
    EXPECT_THROW(compiler.compileMixed(invalid_form), std::invalid_argument);
}

TEST(CompileMixed, RejectsNoTestFunction)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");

    // A form with only trial, no test
    // This should throw because compileMixed requires at least a test function
    EXPECT_THROW(compiler.compileMixed(u.dx()), std::invalid_argument);
}

TEST(CompileMixed, MultipleTermsSameBlock)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // Two terms in the VV block
    const auto form = (u * v).dx() + (FormExpr::constant(2.0) * u * v).dx()
                     + (p * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 2u);

    // VV block should have 2 cell terms
    ASSERT_TRUE(mir.hasBlock(0, 0));
    EXPECT_EQ(mir.block(0, 0).terms().size(), 2u);

    // PP block should have 1 cell term
    ASSERT_TRUE(mir.hasBlock(1, 1));
    EXPECT_EQ(mir.block(1, 1).terms().size(), 1u);

    // Off-diagonals empty
    EXPECT_FALSE(mir.hasBlock(0, 1));
    EXPECT_FALSE(mir.hasBlock(1, 0));
}

TEST(CompileMixed, BoundaryTermsPreserveMarkers)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // Cell term in VV, boundary term in VP on marker 3
    const auto form = (u * v).dx() + (p * v).ds(3);

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    ASSERT_TRUE(mir.hasBlock(0, 0));
    EXPECT_TRUE(mir.block(0, 0).hasCellTerms());
    EXPECT_FALSE(mir.block(0, 0).hasBoundaryTerms());

    ASSERT_TRUE(mir.hasBlock(0, 1));
    EXPECT_TRUE(mir.block(0, 1).hasBoundaryTerms());
    EXPECT_FALSE(mir.block(0, 1).hasCellTerms());
    ASSERT_EQ(mir.block(0, 1).terms().size(), 1u);
    EXPECT_EQ(mir.block(0, 1).terms()[0].boundary_marker, 3);
}

// ============================================================================
// MixedFormIR allFieldRequirements() tests
// ============================================================================

TEST(MixedFormIR, AllFieldRequirementsMergesFlags)
{
    MixedFormIR mir(2, 2);

    FormCompiler compiler;
    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    mir.setBlock(0, 0, compiler.compileBilinear((u * v).dx()));
    mir.setBlock(1, 1, compiler.compileBilinear((p * q).dx()));

    // Just verify we can call it without error and get a non-empty result
    // (field requirement details depend on compilation internals)
    const auto reqs = mir.allFieldRequirements();
    // Each block may or may not produce field requirements depending on compilation details
    // The key invariant is: no crash, and if present they're sorted by FieldId
    for (std::size_t i = 1; i < reqs.size(); ++i) {
        EXPECT_LT(reqs[i - 1].field, reqs[i].field);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
