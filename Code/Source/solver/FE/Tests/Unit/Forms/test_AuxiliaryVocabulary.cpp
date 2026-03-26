/**
 * @file test_AuxiliaryVocabulary.cpp
 * @brief Unit tests for the generalized auxiliary symbolic vocabulary (Phase 12)
 *
 * Covers FormExpr factories, Vocabulary.h helpers, PointEvaluator evaluation,
 * scanner detection, and symbolic differentiation treatment for:
 * - AuxiliaryInputSymbol / AuxiliaryInputRef
 * - AuxiliaryOutputSymbol / AuxiliaryOutputRef
 * - AuxiliaryStateSymbol / AuxiliaryStateRef (library-wide semantics)
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Forms/PointEvaluator.h"
#include "Analysis/FormExprScanner.h"

using svmp::FE::Real;
using namespace svmp::FE::forms;

// ============================================================================
//  FormExpr factory methods
// ============================================================================

TEST(AuxiliaryVocabulary, AuxiliaryInputSymbolFactory)
{
    auto expr = FormExpr::auxiliaryInput("Q");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryInputSymbol);
    EXPECT_TRUE(expr.node()->symbolName().has_value());
    EXPECT_EQ(*expr.node()->symbolName(), "Q");
}

TEST(AuxiliaryVocabulary, AuxiliaryInputRefFactory)
{
    auto expr = FormExpr::auxiliaryInputRef(3);
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryInputRef);
    EXPECT_TRUE(expr.node()->slotIndex().has_value());
    EXPECT_EQ(*expr.node()->slotIndex(), 3u);
}

TEST(AuxiliaryVocabulary, AuxiliaryOutputSymbolFactory)
{
    auto expr = FormExpr::auxiliaryOutput("P_out");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryOutputSymbol);
    EXPECT_EQ(*expr.node()->symbolName(), "P_out");
}

TEST(AuxiliaryVocabulary, AuxiliaryOutputRefFactory)
{
    auto expr = FormExpr::auxiliaryOutputRef(0);
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryOutputRef);
    EXPECT_EQ(*expr.node()->slotIndex(), 0u);
}

TEST(AuxiliaryVocabulary, AuxiliaryStateSymbolFactory)
{
    auto expr = FormExpr::auxiliaryState("X");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryStateSymbol);
    EXPECT_EQ(*expr.node()->symbolName(), "X");
}

TEST(AuxiliaryVocabulary, AuxiliaryStateRefFactory)
{
    auto expr = FormExpr::auxiliaryStateRef(2);
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryStateRef);
    EXPECT_EQ(*expr.node()->slotIndex(), 2u);
}

TEST(AuxiliaryVocabulary, EmptyNameThrows)
{
    EXPECT_THROW(FormExpr::auxiliaryInput(""), std::invalid_argument);
    EXPECT_THROW(FormExpr::auxiliaryOutput(""), std::invalid_argument);
    EXPECT_THROW(FormExpr::auxiliaryState(""), std::invalid_argument);
}

// ============================================================================
//  Vocabulary.h helpers
// ============================================================================

TEST(AuxiliaryVocabulary, VocabularyAuxiliaryInput)
{
    auto expr = AuxiliaryInput("flow_rate");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryInputSymbol);
}

TEST(AuxiliaryVocabulary, VocabularyAuxiliaryInputSlot)
{
    auto expr = AuxiliaryInputSlot(5);
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryInputRef);
    EXPECT_EQ(*expr.node()->slotIndex(), 5u);
}

TEST(AuxiliaryVocabulary, VocabularyAuxiliaryOutput)
{
    auto expr = AuxiliaryOutput("P_total");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryOutputSymbol);
}

TEST(AuxiliaryVocabulary, VocabularyAuxiliaryOutputSlot)
{
    auto expr = AuxiliaryOutputSlot(1);
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryOutputRef);
}

TEST(AuxiliaryVocabulary, VocabularyAuxiliaryState)
{
    auto expr = AuxiliaryState("resistance");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryStateSymbol);
}

// ============================================================================
//  PointEvaluator evaluation
// ============================================================================

TEST(AuxiliaryVocabulary, EvalAuxiliaryInputRef)
{
    auto expr = FormExpr::auxiliaryInputRef(0) + FormExpr::auxiliaryInputRef(1);
    std::vector<Real> inputs = {3.0, 7.0};

    PointEvalContext ctx;
    ctx.auxiliary_inputs = inputs;

    Real result = evaluateScalarAt(expr, ctx);
    EXPECT_DOUBLE_EQ(result, 10.0);
}

TEST(AuxiliaryVocabulary, EvalAuxiliaryOutputRef)
{
    auto expr = FormExpr::auxiliaryOutputRef(0) * FormExpr::constant(2.0);
    std::vector<Real> outputs = {5.0};

    PointEvalContext ctx;
    ctx.auxiliary_outputs = outputs;

    Real result = evaluateScalarAt(expr, ctx);
    EXPECT_DOUBLE_EQ(result, 10.0);
}

TEST(AuxiliaryVocabulary, EvalMixedInputOutputState)
{
    // expr = aux_input[0] + aux_output[0] + aux_state[0]
    auto expr = FormExpr::auxiliaryInputRef(0) +
                FormExpr::auxiliaryOutputRef(0) +
                FormExpr::auxiliaryStateRef(0);

    std::vector<Real> inputs = {1.0};
    std::vector<Real> outputs = {2.0};
    std::vector<Real> aux = {3.0};

    PointEvalContext ctx;
    ctx.auxiliary_inputs = inputs;
    ctx.auxiliary_outputs = outputs;
    ctx.coupled_aux = aux;

    Real result = evaluateScalarAt(expr, ctx);
    EXPECT_DOUBLE_EQ(result, 6.0);
}

// ============================================================================
//  Scanner detection
// ============================================================================

TEST(AuxiliaryVocabulary, ScannerDetectsInputSymbols)
{
    auto expr = FormExpr::auxiliaryInput("Q") + FormExpr::auxiliaryInput("P");
    auto scan = svmp::FE::analysis::scanFormExpr(*expr.node());

    ASSERT_EQ(scan.auxiliary_input_names.size(), 2u);
    EXPECT_EQ(scan.auxiliary_input_names[0], "Q");
    EXPECT_EQ(scan.auxiliary_input_names[1], "P");
}

TEST(AuxiliaryVocabulary, ScannerDetectsOutputSymbols)
{
    auto expr = FormExpr::auxiliaryOutput("voltage");
    auto scan = svmp::FE::analysis::scanFormExpr(*expr.node());

    ASSERT_EQ(scan.auxiliary_output_names.size(), 1u);
    EXPECT_EQ(scan.auxiliary_output_names[0], "voltage");
}

TEST(AuxiliaryVocabulary, ScannerDetectsStateSymbols)
{
    auto expr = FormExpr::auxiliaryState("X");
    auto scan = svmp::FE::analysis::scanFormExpr(*expr.node());

    ASSERT_EQ(scan.auxiliary_state_names.size(), 1u);
    EXPECT_EQ(scan.auxiliary_state_names[0], "X");
}

TEST(AuxiliaryVocabulary, ScannerDeduplicates)
{
    auto expr = FormExpr::auxiliaryInput("Q") + FormExpr::auxiliaryInput("Q");
    auto scan = svmp::FE::analysis::scanFormExpr(*expr.node());

    EXPECT_EQ(scan.auxiliary_input_names.size(), 1u);
}

// ============================================================================
//  ToString (stringification)
// ============================================================================

TEST(AuxiliaryVocabulary, ToStringAuxInput)
{
    auto expr = FormExpr::auxiliaryInput("Q");
    auto str = expr.node()->toString();
    EXPECT_NE(str.find("aux_input"), std::string::npos);
    EXPECT_NE(str.find("Q"), std::string::npos);
}

TEST(AuxiliaryVocabulary, ToStringAuxInputRef)
{
    auto expr = FormExpr::auxiliaryInputRef(3);
    auto str = expr.node()->toString();
    EXPECT_NE(str.find("aux_input"), std::string::npos);
    EXPECT_NE(str.find("3"), std::string::npos);
}

TEST(AuxiliaryVocabulary, ToStringAuxOutput)
{
    auto expr = FormExpr::auxiliaryOutput("P_out");
    auto str = expr.node()->toString();
    EXPECT_NE(str.find("aux_output"), std::string::npos);
    EXPECT_NE(str.find("P_out"), std::string::npos);
}

TEST(AuxiliaryVocabulary, ToStringAuxOutputRef)
{
    auto expr = FormExpr::auxiliaryOutputRef(0);
    auto str = expr.node()->toString();
    EXPECT_NE(str.find("aux_output"), std::string::npos);
    EXPECT_NE(str.find("0"), std::string::npos);
}

// ============================================================================
//  Expression composition
// ============================================================================

TEST(AuxiliaryVocabulary, CompositeExpression)
{
    // Simulate an RCR-like expression:
    // P_out = R_p * Q + P_d
    // Using slot-resolved refs for evaluation:
    auto P_out = FormExpr::auxiliaryInputRef(0) * FormExpr::constant(100.0) +
                 FormExpr::auxiliaryStateRef(0);

    std::vector<Real> inputs = {0.5};  // Q = 0.5
    std::vector<Real> aux = {80.0};    // P_d = 80

    PointEvalContext ctx;
    ctx.auxiliary_inputs = inputs;
    ctx.coupled_aux = aux;

    Real result = evaluateScalarAt(P_out, ctx);
    EXPECT_DOUBLE_EQ(result, 130.0);  // 100*0.5 + 80
}

TEST(AuxiliaryVocabulary, InstanceQualifiedAuxiliaryOutput)
{
    // AuxiliaryOutput(instance, name) should produce an AuxiliaryOutputSymbol
    // with "instance/name" as the symbol name.
    auto expr = AuxiliaryOutput("rcr_outlet", "P_out");
    EXPECT_TRUE(expr.isValid());
    EXPECT_EQ(expr.node()->type(), FormExprType::AuxiliaryOutputSymbol);
    auto sym = expr.node()->symbolName();
    ASSERT_TRUE(sym.has_value());
    EXPECT_EQ(std::string(*sym), "rcr_outlet/P_out");
}

TEST(AuxiliaryVocabulary, InstanceQualifiedOutputScansCorrectly)
{
    // Scanner should detect the "instance/name" symbol as an output name.
    auto expr = AuxiliaryOutput("inst1", "val") +
                AuxiliaryOutput("inst2", "val");
    auto result = svmp::FE::analysis::scanFormExpr(*expr.node());

    // Both instance-qualified names should appear.
    EXPECT_GE(result.auxiliary_output_names.size(), 2u);
    bool found_inst1 = false, found_inst2 = false;
    for (const auto& n : result.auxiliary_output_names) {
        if (n == "inst1/val") found_inst1 = true;
        if (n == "inst2/val") found_inst2 = true;
    }
    EXPECT_TRUE(found_inst1);
    EXPECT_TRUE(found_inst2);
}
