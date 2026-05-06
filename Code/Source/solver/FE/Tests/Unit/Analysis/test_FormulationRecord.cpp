/**
 * @file test_FormulationRecord.cpp
 * @brief Unit tests for FormulationRecord population and FormExprScanner
 *
 * Tests verify:
 *   - FormExprScanner DAG scanning utility
 *   - FormulationRecord structural correctness
 *   - Residual expr handle lifetime
 *   - Stabilization / time derivative / DG flags
 */

#include <gtest/gtest.h>

#include "Analysis/FormulationRecord.h"
#include "Analysis/FormExprScanner.h"
#include "Analysis/ProblemAnalysisTypes.h"

#include "Forms/ConstitutiveModel.h"
#include "Forms/FormExpr.h"
#include "Forms/AffineAnalysis.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

class MatrixIdentityConstitutiveModel final : public forms::ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, int /*dim*/) const override
    {
        return input;
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       int /*dim*/,
                                       DualWorkspace& /*workspace*/) const override
    {
        return input;
    }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind() const override
    {
        return ValueKind::Matrix;
    }

    [[nodiscard]] OutputSpec outputSpec(std::size_t output_index) const override
    {
        if (output_index != 0u) {
            throw std::invalid_argument("MatrixIdentityConstitutiveModel: output index out of range");
        }
        OutputSpec spec;
        spec.kind = ValueKind::Matrix;
        return spec;
    }
};

std::shared_ptr<spaces::FunctionSpace> scalarH1() {
    return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
}

std::shared_ptr<spaces::FunctionSpace> vectorH1(int dim = 3) {
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    return std::make_shared<spaces::ProductSpace>(base, dim);
}

} // namespace

// ============================================================================
// FormExprScanner — isolated DAG scanning tests
// ============================================================================

TEST(FormExprScanner, GradGrad_NoSpecialFlags) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto expr = inner(grad(u), grad(v));

    auto result = scanFormExpr(*expr.node());

    EXPECT_FALSE(result.has_time_derivative);
    EXPECT_FALSE(result.has_cell_diameter);
    EXPECT_FALSE(result.has_jump);
    EXPECT_FALSE(result.has_average);
    EXPECT_FALSE(result.has_stabilization());
    EXPECT_FALSE(result.has_interior_face_terms());
    EXPECT_TRUE(result.boundary_functional_names.empty());
    EXPECT_TRUE(result.auxiliary_state_names.empty());
}

TEST(FormExprScanner, CellDiameter_Alone) {
    // CellDiameter node appears in expression
    auto space = scalarH1();
    auto v = FormExpr::testFunction(*space, "v");
    auto h = FormExpr::cellDiameter();
    auto expr = h * v;

    auto result = scanFormExpr(*expr.node());

    EXPECT_TRUE(result.has_cell_diameter);
    EXPECT_TRUE(result.has_stabilization());
    EXPECT_FALSE(result.has_time_derivative);
}

TEST(FormExprScanner, CellDiameter_Detected) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto h = FormExpr::cellDiameter();
    auto expr = inner(h * grad(u), grad(v));

    auto result = scanFormExpr(*expr.node());

    EXPECT_TRUE(result.has_cell_diameter);
    EXPECT_TRUE(result.has_stabilization());
}

TEST(FormExprScanner, ActiveDomains_DefaultIsCell) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto expr = inner(u, v);

    auto result = scanFormExpr(*expr.node());
    auto domains = result.activeDomains();

    ASSERT_FALSE(domains.empty());
    EXPECT_EQ(domains[0], DomainKind::Cell);
}

TEST(FormExprScanner, ScanResultConvenience) {
    // Verify convenience methods on an empty result
    FormExprScanResult empty;
    EXPECT_FALSE(empty.has_stabilization());
    EXPECT_FALSE(empty.has_interior_face_terms());

    auto domains = empty.activeDomains();
    ASSERT_EQ(domains.size(), 1u);
    EXPECT_EQ(domains[0], DomainKind::Cell);  // Default when no integral nodes found
}

TEST(FormExprScanner, RuntimeMetadataForParameterCoefficientScale) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    ScalarCoefficient mu_func = [](Real, Real, Real) { return Real(2.0); };
    auto gamma = FormExpr::parameter("gamma");
    auto mu = FormExpr::coefficient("mu", mu_func);
    auto expr = ((gamma * mu / FormExpr::cellDiameter()) * u * v).ds(3);

    auto result = scanFormExpr(*expr.node());

    ASSERT_EQ(result.parameter_usages.size(), 1u);
    EXPECT_EQ(result.parameter_usages[0].name, "gamma");
    EXPECT_EQ(result.parameter_usages[0].domain, DomainKind::Boundary);
    EXPECT_EQ(result.parameter_usages[0].boundary_marker, 3);

    ASSERT_EQ(result.coefficient_usages.size(), 1u);
    EXPECT_EQ(result.coefficient_usages[0].name, "mu");
    EXPECT_EQ(result.coefficient_usages[0].rank, FormCoefficientRank::Scalar);
    EXPECT_EQ(result.coefficient_usages[0].domain, DomainKind::Boundary);

    auto scale_it = std::find_if(
        result.scale_usages.begin(), result.scale_usages.end(),
        [](const FormScaleUsage& usage) {
            return usage.h_power == -1 &&
                   std::find(usage.parameter_names.begin(),
                             usage.parameter_names.end(),
                             "gamma") != usage.parameter_names.end() &&
                   std::find(usage.coefficient_names.begin(),
                             usage.coefficient_names.end(),
                             "mu") != usage.coefficient_names.end();
        });
    ASSERT_NE(scale_it, result.scale_usages.end());
    EXPECT_EQ(scale_it->domain, DomainKind::Boundary);
    EXPECT_EQ(scale_it->boundary_marker, 3);
    EXPECT_TRUE(scale_it->exact_for_analysis);
}

TEST(FormExprScanner, InfersGenericConstitutiveMetadataFromDag) {
    auto space = vectorH1();
    auto u = FormExpr::stateField(7, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto model = std::make_shared<MatrixIdentityConstitutiveModel>();
    auto sigma = FormExpr::constitutive(model, grad(u));
    auto expr = inner(sigma, grad(v)).dx();

    auto result = scanFormExpr(*expr.node());

    ASSERT_EQ(result.constitutive_laws.size(), 1u);
    const auto& law = result.constitutive_laws.front();
    EXPECT_NE(law.name.find("constitutive:type:"), std::string::npos);
    EXPECT_EQ(law.primary_field, 7);
    EXPECT_EQ(law.tensor_rank, "rank2");
    EXPECT_EQ(law.symmetry_class, "");
    EXPECT_EQ(law.model.get(), model.get());
    EXPECT_FALSE(law.state_dependent);
    EXPECT_FALSE(law.time_dependent);
}

// ============================================================================
// FormulationRecord — structural tests
// ============================================================================

TEST(FormulationRecord, DefaultConstruction) {
    FormulationRecord rec;
    EXPECT_TRUE(rec.operator_tag.empty());
    EXPECT_TRUE(rec.active_fields.empty());
    EXPECT_TRUE(rec.active_variables.empty());
    EXPECT_EQ(rec.residual_expr, nullptr);
    EXPECT_FALSE(rec.affine_split_succeeded);
    EXPECT_FALSE(rec.is_mixed);
    EXPECT_FALSE(rec.has_interior_face_terms);
    EXPECT_FALSE(rec.has_time_derivative);
    EXPECT_FALSE(rec.has_stabilization_terms);
    EXPECT_TRUE(rec.active_domains.empty());
    EXPECT_TRUE(rec.block_couplings.empty());
    EXPECT_TRUE(rec.variable_couplings.empty());
    EXPECT_TRUE(rec.boundary_functional_dependencies.empty());
    EXPECT_TRUE(rec.auxiliary_state_dependencies.empty());
}

TEST(FormulationRecord, PopulateManually_Poisson) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.is_mixed = false;
    rec.has_time_derivative = false;
    rec.has_stabilization_terms = false;
    rec.has_interior_face_terms = false;
    rec.active_domains = {DomainKind::Cell};
    rec.block_couplings = {{0, 0}};
    rec.active_variables.push_back(VariableKey::field(0));

    EXPECT_EQ(rec.operator_tag, "equations");
    ASSERT_EQ(rec.active_fields.size(), 1u);
    EXPECT_EQ(rec.active_fields[0], 0);
    EXPECT_FALSE(rec.is_mixed);
    ASSERT_EQ(rec.block_couplings.size(), 1u);
    EXPECT_EQ(rec.block_couplings[0], std::make_pair(FieldId{0}, FieldId{0}));
}

TEST(FormulationRecord, PopulateManually_Stokes) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0, 1};
    rec.is_mixed = true;
    rec.block_couplings = {{0,0}, {0,1}, {1,0}, {1,1}};
    rec.active_variables.push_back(VariableKey::field(0));
    rec.active_variables.push_back(VariableKey::field(1));

    EXPECT_TRUE(rec.is_mixed);
    ASSERT_EQ(rec.block_couplings.size(), 4u);
    ASSERT_EQ(rec.active_variables.size(), 2u);
}

TEST(FormulationRecord, PopulateManually_CoupledBoundary) {
    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.active_variables.push_back(VariableKey::field(0));

    auto bf = VariableKey::named(VariableKind::BoundaryFunctional, "Q_out");
    auto aux = VariableKey::named(VariableKind::AuxiliaryState, "P_d");
    rec.active_variables.push_back(bf);
    rec.active_variables.push_back(aux);
    rec.boundary_functional_dependencies.push_back(bf);
    rec.auxiliary_state_dependencies.push_back(aux);
    rec.variable_couplings.emplace_back(VariableKey::field(0), bf);
    rec.variable_couplings.emplace_back(VariableKey::field(0), aux);

    ASSERT_EQ(rec.active_variables.size(), 3u);
    ASSERT_EQ(rec.boundary_functional_dependencies.size(), 1u);
    ASSERT_EQ(rec.auxiliary_state_dependencies.size(), 1u);
    ASSERT_EQ(rec.variable_couplings.size(), 2u);
    EXPECT_EQ(rec.boundary_functional_dependencies[0].name, "Q_out");
    EXPECT_EQ(rec.auxiliary_state_dependencies[0].name, "P_d");
}

TEST(FormulationRecord, ResidualExprHandle_KeepsNodeAlive) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v));

    FormulationRecord rec;
    rec.residual_expr = residual.nodeShared();

    ASSERT_NE(rec.residual_expr, nullptr);
    EXPECT_EQ(rec.residual_expr->type(), FormExprType::InnerProduct);
}

// ============================================================================
// affine_split_succeeded — via AffineAnalysis
// ============================================================================

TEST(FormulationRecord, AffineSplit_LinearPoisson) {
    // Linear Poisson: ∫ grad(u)·grad(v) dx is affine in TrialFunction
    auto space = scalarH1();
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    forms::AffineResidualOptions opts;
    auto split = forms::trySplitAffineResidual(residual, opts);
    EXPECT_TRUE(split.has_value());
}

TEST(FormulationRecord, AffineSplit_ReactionDiffusion) {
    // Reaction-diffusion: ∫ grad(u)·grad(v) + u*v is also affine
    auto space = scalarH1();
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx() + (u * v).dx();

    auto split = forms::trySplitAffineResidual(residual);
    EXPECT_TRUE(split.has_value());
}

// ============================================================================
// block_residual_exprs
// ============================================================================

TEST(FormulationRecord, BlockResidualExprs_Default) {
    FormulationRecord rec;
    EXPECT_TRUE(rec.block_residual_exprs.empty());
}

TEST(FormulationRecord, BlockResidualExprs_SingleField) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v));

    FormulationRecord rec;
    rec.active_fields = {0};
    rec.block_residual_exprs.push_back(
        {{0, 0}, residual.nodeShared()});

    ASSERT_EQ(rec.block_residual_exprs.size(), 1u);
    EXPECT_EQ(rec.block_residual_exprs[0].first.first, FieldId{0});
    EXPECT_EQ(rec.block_residual_exprs[0].first.second, FieldId{0});
    ASSERT_NE(rec.block_residual_exprs[0].second, nullptr);
}
