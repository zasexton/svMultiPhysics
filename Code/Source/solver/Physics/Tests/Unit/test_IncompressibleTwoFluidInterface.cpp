/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.h"

#include "FE/Forms/FormCompiler.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace ns = svmp::Physics::formulations::navier_stokes;

struct InterfaceExpressions {
  std::shared_ptr<FE::spaces::H1Space> scalar_space;
  std::shared_ptr<FE::spaces::ProductSpace> vector_space;
  FE::forms::FormExpr u_negative;
  FE::forms::FormExpr p_negative;
  FE::forms::FormExpr v_negative;
  FE::forms::FormExpr q_negative;
  FE::forms::FormExpr u_positive;
  FE::forms::FormExpr p_positive;
  FE::forms::FormExpr v_positive;
  FE::forms::FormExpr q_positive;
};

InterfaceExpressions makeExpressions() {
  auto scalar_space =
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
  auto vector_space =
      std::make_shared<FE::spaces::ProductSpace>(scalar_space, 3);
  return InterfaceExpressions{
      .scalar_space = scalar_space,
      .vector_space = vector_space,
      .u_negative = FE::forms::TrialFunction(*vector_space, "u_negative"),
      .p_negative = FE::forms::TrialFunction(*scalar_space, "p_negative"),
      .v_negative = FE::forms::TestFunction(*vector_space, "v_negative"),
      .q_negative = FE::forms::TestFunction(*scalar_space, "q_negative"),
      .u_positive = FE::forms::TrialFunction(*vector_space, "u_positive"),
      .p_positive = FE::forms::TrialFunction(*scalar_space, "p_positive"),
      .v_positive = FE::forms::TestFunction(*vector_space, "v_positive"),
      .q_positive = FE::forms::TestFunction(*scalar_space, "q_positive"),
  };
}

ns::IncompressibleTwoFluidInterfaceParameters makeParameters() {
  return ns::IncompressibleTwoFluidInterfaceParameters{
      .dimension = 3,
      .interface_marker = 71,
      .negative_density = FE::Real{1000.0},
      .positive_density = FE::Real{1.0},
      .negative_viscosity = FE::Real{0.01},
      .positive_viscosity = FE::Real{0.001},
      .nitsche_gamma = FE::Real{24.0},
      .surface_tension = FE::Real{0.072},
      .include_transient_penalty = false,
  };
}

ns::IncompressibleTwoFluidInterfaceForms
build(const ns::IncompressibleTwoFluidInterfaceParameters &parameters) {
  const auto expressions = makeExpressions();
  return ns::buildIncompressibleTwoFluidInterfaceForms(
      expressions.u_negative, expressions.p_negative, expressions.v_negative,
      expressions.q_negative, expressions.u_positive, expressions.p_positive,
      expressions.v_positive, expressions.q_positive, parameters);
}

bool expressionContains(const FE::forms::FormExpr &expression,
                        FE::forms::FormExprType target) {
  if (!expression.isValid()) {
    return false;
  }
  std::vector<const FE::forms::FormExprNode *> pending{expression.node()};
  while (!pending.empty()) {
    const auto *node = pending.back();
    pending.pop_back();
    if (node->type() == target) {
      return true;
    }
    for (const auto *child : node->children()) {
      if (child != nullptr) {
        pending.push_back(child);
      }
    }
  }
  return false;
}

TEST(IncompressibleTwoFluidInterface,
     ViscosityWeightsAreComplementaryAndSideReversalInvariant) {
  const auto parameters = makeParameters();
  const auto weights = ns::incompressibleTwoFluidInterfaceWeights(parameters);
  EXPECT_NEAR(weights.negative_traction + weights.positive_traction,
              FE::Real{1.0},
              FE::Real{4.0} * std::numeric_limits<FE::Real>::epsilon());
  EXPECT_DOUBLE_EQ(weights.negative_complement, weights.positive_traction);
  EXPECT_DOUBLE_EQ(weights.positive_complement, weights.negative_traction);

  auto reversed = parameters;
  std::swap(reversed.negative_density, reversed.positive_density);
  std::swap(reversed.negative_viscosity, reversed.positive_viscosity);
  const auto reversed_weights =
      ns::incompressibleTwoFluidInterfaceWeights(reversed);
  EXPECT_DOUBLE_EQ(reversed_weights.negative_traction,
                   weights.positive_traction);
  EXPECT_DOUBLE_EQ(reversed_weights.positive_traction,
                   weights.negative_traction);
  EXPECT_DOUBLE_EQ(reversed_weights.harmonic_viscosity,
                   weights.harmonic_viscosity);
  EXPECT_DOUBLE_EQ(reversed_weights.harmonic_density, weights.harmonic_density);
}

TEST(IncompressibleTwoFluidInterface,
     WeightsRemainFiniteWithoutOverflowAtExtremeMaterialScales) {
  auto parameters = makeParameters();
  parameters.negative_density = std::numeric_limits<FE::Real>::max();
  parameters.positive_density =
      std::numeric_limits<FE::Real>::max() / FE::Real{2.0};
  parameters.negative_viscosity =
      std::numeric_limits<FE::Real>::max() / FE::Real{4.0};
  parameters.positive_viscosity =
      std::numeric_limits<FE::Real>::max() / FE::Real{2.0};
  const auto weights = ns::incompressibleTwoFluidInterfaceWeights(parameters);
  EXPECT_TRUE(std::isfinite(weights.negative_traction));
  EXPECT_TRUE(std::isfinite(weights.positive_traction));
  EXPECT_TRUE(std::isfinite(weights.harmonic_viscosity));
  EXPECT_TRUE(std::isfinite(weights.harmonic_density));
  EXPECT_NEAR(weights.negative_traction + weights.positive_traction,
              FE::Real{1.0},
              FE::Real{4.0} * std::numeric_limits<FE::Real>::epsilon());
}

TEST(IncompressibleTwoFluidInterface,
     ResidualCompilesAsFourFieldGeneratedInterfaceCoupling) {
  const auto forms = build(makeParameters());
  EXPECT_TRUE(forms.consistency.isValid());
  EXPECT_TRUE(forms.adjoint.isValid());
  EXPECT_TRUE(forms.penalty.isValid());
  EXPECT_TRUE(forms.surface_energy.isValid());
  EXPECT_TRUE(forms.residual.isValid());

  FE::forms::FormCompiler compiler;
  const auto mixed =
      compiler.compileMixed(forms.residual, FE::forms::FormKind::Residual);
  EXPECT_EQ(mixed.numTestFields(), 4u);
  EXPECT_EQ(mixed.numTrialFields(), 4u);
  EXPECT_TRUE(mixed.domainSummary().has_interface_face_terms);
  ASSERT_EQ(mixed.domainSummary().interface_markers.size(), 1u);
  EXPECT_EQ(mixed.domainSummary().interface_markers.front(), 71);
  EXPECT_EQ(mixed.numActiveBlocks(), 12u);
}

TEST(IncompressibleTwoFluidInterface,
     LiteralZeroSurfaceTensionOmitsSurfaceEnergyTree) {
  auto parameters = makeParameters();
  parameters.surface_tension = FE::Real{0.0};
  const auto forms = build(parameters);
  EXPECT_FALSE(forms.surface_energy.isValid());
  EXPECT_TRUE(forms.residual.isValid());
}

TEST(IncompressibleTwoFluidInterface,
     TransientPenaltyRetainsEffectiveStepTerminal) {
  auto parameters = makeParameters();
  parameters.include_transient_penalty = true;
  parameters.surface_tension = FE::Real{0.0};
  const auto forms = build(parameters);
  EXPECT_TRUE(expressionContains(forms.penalty,
                                 FE::forms::FormExprType::EffectiveTimeStep));
}

TEST(IncompressibleTwoFluidInterface, RejectsInvalidParameters) {
  auto parameters = makeParameters();
  parameters.dimension = 1;
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.interface_marker = -1;
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.negative_density = FE::Real{0.0};
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.positive_viscosity = std::numeric_limits<FE::Real>::quiet_NaN();
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.nitsche_gamma = FE::Real{-1.0};
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.surface_tension = FE::Real{-1.0};
  EXPECT_THROW(build(parameters), std::invalid_argument);
}

TEST(IncompressibleTwoFluidInterface, RejectsMissingExpression) {
  const auto expressions = makeExpressions();
  EXPECT_THROW((void)ns::buildIncompressibleTwoFluidInterfaceForms(
                   FE::forms::FormExpr{}, expressions.p_negative,
                   expressions.v_negative, expressions.q_negative,
                   expressions.u_positive, expressions.p_positive,
                   expressions.v_positive, expressions.q_positive,
                   makeParameters()),
               std::invalid_argument);
}

} // namespace
