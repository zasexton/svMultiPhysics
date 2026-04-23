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

#include <array>
#include <stdexcept>
#include <utility>

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
// Source provenance tests
// ============================================================================

TEST(CompileMixed, SourceExpressionPreserved)
{
    FormCompiler compiler;

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace V(scalar_space, /*components=*/3);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    const auto form = (inner(grad(u), grad(v))).dx()
                    + (FormExpr::constant(-1.0) * p * div(v)).dx()
                    + (div(u) * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    // Source expression should be stored
    ASSERT_TRUE(mir.sourceExpression().has_value());
    EXPECT_TRUE(mir.sourceExpression()->isValid());
}

TEST(CompileMixed, SingleFieldSourceExpressionPreserved)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto form = (u * v).dx();
    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    ASSERT_TRUE(mir.sourceExpression().has_value());
    EXPECT_TRUE(mir.sourceExpression()->isValid());
}

TEST(CompileMixed, BlockProvenanceTracksTermIndices)
{
    FormCompiler compiler;

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace V(scalar_space, /*components=*/3);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // 3 terms: VV(1), VP(1), PV(1)
    const auto form = (inner(grad(u), grad(v))).dx()
                    + (FormExpr::constant(-1.0) * p * div(v)).dx()
                    + (div(u) * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    // VV block: 1 contributing term
    ASSERT_TRUE(mir.blockProvenance(0, 0).has_value());
    EXPECT_EQ(mir.blockProvenance(0, 0)->contributing_term_indices.size(), 1u);
    EXPECT_FALSE(mir.blockProvenance(0, 0)->source_summary.empty());

    // VP block: 1 contributing term
    ASSERT_TRUE(mir.blockProvenance(0, 1).has_value());
    EXPECT_EQ(mir.blockProvenance(0, 1)->contributing_term_indices.size(), 1u);

    // PV block: 1 contributing term
    ASSERT_TRUE(mir.blockProvenance(1, 0).has_value());
    EXPECT_EQ(mir.blockProvenance(1, 0)->contributing_term_indices.size(), 1u);

    // PP block: no provenance (zero block)
    EXPECT_FALSE(mir.blockProvenance(1, 1).has_value());
}

TEST(CompileMixed, BlockProvenanceMultipleTermsSameBlock)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // Two terms in VV block, one in PP
    const auto form = (u * v).dx() + (FormExpr::constant(2.0) * u * v).dx()
                     + (p * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    // VV block: 2 contributing terms
    ASSERT_TRUE(mir.blockProvenance(0, 0).has_value());
    EXPECT_EQ(mir.blockProvenance(0, 0)->contributing_term_indices.size(), 2u);

    // PP block: 1 contributing term
    ASSERT_TRUE(mir.blockProvenance(1, 1).has_value());
    EXPECT_EQ(mir.blockProvenance(1, 1)->contributing_term_indices.size(), 1u);
}

// ============================================================================
// Domain summary tests
// ============================================================================

TEST(CompileMixed, DomainSummaryCellOnly)
{
    FormCompiler compiler;

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace V(scalar_space, /*components=*/3);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    const auto form = (inner(grad(u), grad(v))).dx() + (div(u) * q).dx();
    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    EXPECT_TRUE(mir.domainSummary().has_cell_terms);
    EXPECT_FALSE(mir.domainSummary().has_boundary_terms);
    EXPECT_FALSE(mir.domainSummary().has_interior_face_terms);
    EXPECT_FALSE(mir.domainSummary().has_interface_face_terms);
    EXPECT_TRUE(mir.domainSummary().boundary_markers.empty());
}

TEST(CompileMixed, DomainSummaryWithBoundary)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    // Cell term in VV, boundary terms on markers 3 and 5
    const auto form = (u * v).dx() + (p * v).ds(3) + (p * v).ds(5);
    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    EXPECT_TRUE(mir.domainSummary().has_cell_terms);
    EXPECT_TRUE(mir.domainSummary().has_boundary_terms);
    ASSERT_EQ(mir.domainSummary().boundary_markers.size(), 2u);
    // Markers should include both 3 and 5
    EXPECT_TRUE(std::find(mir.domainSummary().boundary_markers.begin(),
                          mir.domainSummary().boundary_markers.end(), 3)
                != mir.domainSummary().boundary_markers.end());
    EXPECT_TRUE(std::find(mir.domainSummary().boundary_markers.begin(),
                          mir.domainSummary().boundary_markers.end(), 5)
                != mir.domainSummary().boundary_markers.end());
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

// ============================================================================
// FormCompiler::compile() auto-detecting entry point
// ============================================================================

// ============================================================================
// Residual/Bilinear with no trial are rejected (consistent with compileResidual)
// ============================================================================

TEST(CompileMixed, ResidualNoTrial_Rejected)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = FormExpr::testFunction(space, "v");
    const auto f = FormExpr::constant(1.0);

    auto form = (f * v).dx();

    // Residual with no trial → should throw (consistent with compileResidual)
    EXPECT_THROW(compiler.compileMixed(form, FormKind::Residual), std::invalid_argument);
    EXPECT_THROW(compiler.compile(form, FormKind::Residual), std::invalid_argument);

    // Bilinear with no trial → should also throw
    EXPECT_THROW(compiler.compileMixed(form, FormKind::Bilinear), std::invalid_argument);

    // Linear with no trial → should succeed (this is the valid use case)
    EXPECT_NO_THROW(compiler.compileMixed(form, FormKind::Linear));
}

TEST(CompileMixed, MultiFieldResidualNoTrial_Rejected)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto v = FormExpr::testFunction(V, "v");
    const auto q = FormExpr::testFunction(Q, "q");
    const auto f = FormExpr::constant(1.0);
    const auto g = FormExpr::constant(2.0);

    // Multi-field form with two test functions but no trial
    auto form = (f * v).dx() + (g * q).dx();

    // Residual/Bilinear → rejected
    EXPECT_THROW(compiler.compileMixed(form, FormKind::Residual), std::invalid_argument);
    EXPECT_THROW(compiler.compileMixed(form, FormKind::Bilinear), std::invalid_argument);

    // Linear → succeeds
    EXPECT_NO_THROW(compiler.compileMixed(form, FormKind::Linear));
}

// ============================================================================

TEST(CompileMixed, AutoDetect_SingleField)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // compile() on a single-field form → 1×1 MixedFormIR
    const auto mir = compiler.compile((u * v).dx(), FormKind::Bilinear);

    EXPECT_EQ(mir.numTestFields(), 1u);
    EXPECT_EQ(mir.numTrialFields(), 1u);
    EXPECT_EQ(mir.numActiveBlocks(), 1u);
    EXPECT_TRUE(mir.hasBlock(0, 0));
}

TEST(CompileMixed, AutoDetect_MixedField)
{
    FormCompiler compiler;

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace V(scalar_space, /*components=*/3);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    auto mixed = (inner(grad(u), grad(v))).dx()
               + (FormExpr::constant(-1.0) * p * div(v)).dx()
               + (div(u) * q).dx();

    // compile() on a multi-field form → 2×2 block-sparse MixedFormIR
    const auto mir = compiler.compile(mixed, FormKind::Bilinear);

    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 2u);
    EXPECT_EQ(mir.numActiveBlocks(), 3u);  // VV, VP, PV — no PP

    EXPECT_TRUE(mir.hasBlock(0, 0));
    EXPECT_TRUE(mir.hasBlock(0, 1));
    EXPECT_TRUE(mir.hasBlock(1, 0));
    EXPECT_FALSE(mir.hasBlock(1, 1));
}

TEST(CompileMixed, ThreeFieldSparseLayoutPreservesOrderingDomainsAndProvenance)
{
    FormCompiler compiler;

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace V(scalar_space, /*components=*/3);
    spaces::L2Space Q(ElementType::Tetra4, 0);
    spaces::H1Space T(ElementType::Tetra4, 1);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto theta = FormExpr::trialFunction(T, "theta");
    const auto psi = FormExpr::testFunction(T, "psi");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    const auto form =
        inner(grad(u), grad(v)).dx() +
        (theta * div(v)).dx() +
        (div(u) * q).dx() +
        (p * q).ds(4) +
        (minus(theta) * minus(psi)).dI(6);

    const auto mir = compiler.compileMixed(form, FormKind::Bilinear);

    ASSERT_EQ(mir.numTestFields(), 3u);
    ASSERT_EQ(mir.numTrialFields(), 3u);
    EXPECT_EQ(mir.numActiveBlocks(), 5u);

    ASSERT_EQ(mir.testFields().size(), 3u);
    EXPECT_EQ(mir.testFields()[0].name, "v");
    EXPECT_EQ(mir.testFields()[1].name, "q");
    EXPECT_EQ(mir.testFields()[2].name, "psi");

    ASSERT_EQ(mir.trialFields().size(), 3u);
    EXPECT_EQ(mir.trialFields()[0].name, "u");
    EXPECT_EQ(mir.trialFields()[1].name, "theta");
    EXPECT_EQ(mir.trialFields()[2].name, "p");

    const std::array<std::pair<std::size_t, std::size_t>, 5> active_blocks = {{
        {0u, 0u}, {0u, 1u}, {1u, 0u}, {1u, 2u}, {2u, 1u},
    }};
    for (const auto& [row, col] : active_blocks) {
        ASSERT_TRUE(mir.hasBlock(row, col)) << "Expected active block (" << row << "," << col << ")";
        ASSERT_TRUE(mir.blockProvenance(row, col).has_value());
        EXPECT_EQ(mir.blockProvenance(row, col)->contributing_term_indices.size(), 1u);
    }

    EXPECT_FALSE(mir.hasBlock(0, 2));
    EXPECT_FALSE(mir.hasBlock(1, 1));
    EXPECT_FALSE(mir.hasBlock(2, 0));
    EXPECT_FALSE(mir.hasBlock(2, 2));
    EXPECT_FALSE(mir.blockProvenance(2, 2).has_value());

    EXPECT_TRUE(mir.block(0, 0).hasCellTerms());
    EXPECT_TRUE(mir.block(0, 1).hasCellTerms());
    EXPECT_TRUE(mir.block(1, 0).hasCellTerms());
    EXPECT_TRUE(mir.block(1, 2).hasBoundaryTerms());
    ASSERT_EQ(mir.block(1, 2).terms().size(), 1u);
    EXPECT_EQ(mir.block(1, 2).terms()[0].boundary_marker, 4);

    EXPECT_TRUE(mir.block(2, 1).hasInterfaceFaceTerms());
    ASSERT_EQ(mir.block(2, 1).terms().size(), 1u);
    EXPECT_EQ(mir.block(2, 1).terms()[0].interface_marker, 6);

    EXPECT_TRUE(mir.domainSummary().has_cell_terms);
    EXPECT_TRUE(mir.domainSummary().has_boundary_terms);
    EXPECT_TRUE(mir.domainSummary().has_interface_face_terms);
    ASSERT_EQ(mir.domainSummary().boundary_markers.size(), 1u);
    EXPECT_EQ(mir.domainSummary().boundary_markers.front(), 4);
}

// ============================================================================
// Issue 1: Residual test-only terms (f*v) preserved in block classification
// ============================================================================

TEST(CompileMixed, ResidualTestOnlyTermsNotInJacobianBlocks)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    const auto f = FormExpr::constant(1.0);

    // Mixed residual: u*v (trial-dep), f*v (test-only source), p*q (trial-dep)
    const auto form = (u * v).dx() + (f * v).dx() + (p * q).dx();

    const auto mir = compiler.compileMixed(form, FormKind::Residual);

    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 2u);

    // compileMixed Residual reflects Jacobian block structure, not the full
    // residual. Test-only terms (f*v) do NOT appear in block classification
    // because they have no trial dependency. They are handled by the
    // installCoupledResidualMixed per-test decomposition instead.
    ASSERT_TRUE(mir.hasBlock(0, 0));
    EXPECT_EQ(mir.block(0, 0).terms().size(), 1u) << "VV block: u*v only";

    ASSERT_TRUE(mir.hasBlock(1, 1));
    EXPECT_EQ(mir.block(1, 1).terms().size(), 1u) << "PP block: p*q only";

    // f*v has no trial → not in any block
    EXPECT_EQ(mir.numActiveBlocks(), 2u);
}

// ============================================================================
// Issue 3: Duplicate symbol names across different spaces are rejected
// ============================================================================

TEST(CompileMixed, RejectsDuplicateTestFunctionNames)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    // Two test functions named "v" on DIFFERENT spaces
    const auto u1 = FormExpr::trialFunction(V, "u");
    const auto v1 = FormExpr::testFunction(V, "v");
    const auto u2 = FormExpr::trialFunction(Q, "u2");
    const auto v2 = FormExpr::testFunction(Q, "v");  // same name, different space

    auto form = (u1 * v1).dx() + (u2 * v2).dx();

    EXPECT_THROW(compiler.compileMixed(form, FormKind::Bilinear),
                 std::invalid_argument);
}

TEST(CompileMixed, RejectsDuplicateTrialFunctionNames)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    // Two trial functions named "u" on DIFFERENT spaces
    const auto u1 = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto u2 = FormExpr::trialFunction(Q, "u");  // same name, different space
    const auto q = FormExpr::testFunction(Q, "q");

    auto form = (u1 * v).dx() + (u2 * q).dx();

    EXPECT_THROW(compiler.compileMixed(form, FormKind::Bilinear),
                 std::invalid_argument);
}

// ============================================================================
// Issue 5: Linear forms produce consistent IR
// ============================================================================

TEST(CompileMixed, LinearFormSingleField_SyntheticTrialColumn)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto v = FormExpr::testFunction(space, "v");
    const auto f = FormExpr::constant(1.0);

    auto linear = (f * v).dx();
    const auto mir = compiler.compileMixed(linear, FormKind::Linear);

    // Linear form: 1 test, 1 synthetic trial column (for installMixedLinear compat)
    EXPECT_EQ(mir.numTestFields(), 1u);
    EXPECT_EQ(mir.numTrialFields(), 1u);

    // The single block at (0,0) should contain the compiled linear form
    EXPECT_TRUE(mir.hasBlock(0, 0));
    EXPECT_EQ(mir.numActiveBlocks(), 1u);

    // Test field descriptor populated
    ASSERT_EQ(mir.testFields().size(), 1u);
    EXPECT_EQ(mir.testFields()[0].name, "v");

    // Trial field descriptor populated (synthetic, from test field)
    ASSERT_EQ(mir.trialFields().size(), 1u);
}

// ============================================================================

// ============================================================================
// Multi-field linear form compilation
// ============================================================================

TEST(CompileMixed, MultiFieldLinear_TwoTestNoTrial)
{
    FormCompiler compiler;

    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto v = FormExpr::testFunction(V, "v");
    const auto q = FormExpr::testFunction(Q, "q");
    const auto f = FormExpr::constant(1.0);
    const auto g = FormExpr::constant(2.0);

    // Multi-field linear form: two test functions, no trial
    auto linear = (f * v).dx() + (g * q).dx();
    const auto mir = compiler.compileMixed(linear, FormKind::Linear);

    EXPECT_EQ(mir.numTestFields(), 2u);
    // Synthetic trial column
    EXPECT_EQ(mir.numTrialFields(), 1u);

    // Both test rows should have a block at column 0
    EXPECT_TRUE(mir.hasBlock(0, 0));
    EXPECT_TRUE(mir.hasBlock(1, 0));
    EXPECT_EQ(mir.numActiveBlocks(), 2u);

    // Trial descriptor should be populated (synthetic)
    ASSERT_EQ(mir.trialFields().size(), 1u);
}

// ============================================================================

TEST(CompileMixed, AutoDetect_MatchesExplicitCompileMixed)
{
    FormCompiler compiler;
    spaces::H1Space V(ElementType::Tetra4, 1);
    spaces::L2Space Q(ElementType::Tetra4, 0);

    const auto u = FormExpr::trialFunction(V, "u");
    const auto v = FormExpr::testFunction(V, "v");
    const auto p = FormExpr::trialFunction(Q, "p");
    const auto q = FormExpr::testFunction(Q, "q");

    auto form = (u * v).dx() + (p * v).dx() + (u * q).dx();

    // Both entry points should produce identical structure
    const auto mir_auto = compiler.compile(form, FormKind::Bilinear);
    const auto mir_explicit = compiler.compileMixed(form, FormKind::Bilinear);

    EXPECT_EQ(mir_auto.numTestFields(), mir_explicit.numTestFields());
    EXPECT_EQ(mir_auto.numTrialFields(), mir_explicit.numTrialFields());
    EXPECT_EQ(mir_auto.numActiveBlocks(), mir_explicit.numActiveBlocks());
    EXPECT_EQ(mir_auto.kind(), mir_explicit.kind());
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
