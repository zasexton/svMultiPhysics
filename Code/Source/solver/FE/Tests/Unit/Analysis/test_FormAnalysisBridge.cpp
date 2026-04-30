/**
 * @file test_FormAnalysisBridge.cpp
 * @brief Unit tests for public Forms metadata bridge records.
 */

#include <gtest/gtest.h>

#include "Analysis/FormAnalysisBridge.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

std::shared_ptr<spaces::FunctionSpace> scalarH1()
{
    return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
}

const FormTerminalMetadata* findTerminal(
    const std::vector<FormTerminalMetadata>& terminals,
    FormTerminalKind kind)
{
    const auto it = std::find_if(
        terminals.begin(), terminals.end(), [kind](const auto& terminal) {
            return terminal.kind == kind;
        });
    return it == terminals.end() ? nullptr : &*it;
}

const FormBridgeFeatureGate* findGate(
    const std::vector<FormBridgeFeatureGate>& gates,
    FormBridgeFeature feature)
{
    const auto it = std::find_if(
        gates.begin(), gates.end(), [feature](const auto& gate) {
            return gate.feature == feature;
        });
    return it == gates.end() ? nullptr : &*it;
}

} // namespace

TEST(FormAnalysisBridge, ToStringReportsStableNames)
{
    EXPECT_STREQ(toString(FormBridgeFeature::ContributionIdentity),
                 "ContributionIdentity");
    EXPECT_STREQ(toString(FormBridgeFeatureStatus::Partial), "Partial");
    EXPECT_STREQ(toString(FormTerminalKind::BoundaryFunctionalSymbol),
                 "BoundaryFunctionalSymbol");
}

TEST(FormAnalysisBridge, NormalizesBoundaryFunctionalAndBoundaryIntegralRef)
{
    const auto space = scalarH1();
    const auto u = FormExpr::stateField(0, *space, "u");
    const auto v = FormExpr::testFunction(0, *space, "v");
    const auto source = FormExpr::boundaryIntegral(u, 7, "Q_wall");
    const auto ref = FormExpr::boundaryIntegralRef(3);
    const auto residual = (source * v).ds(7) + (ref * v).ds(7);

    FormAnalysisBridgeOptions options;
    options.contribution_name = "coupled_boundary";
    options.origin = "unit_test";

    const auto terminals =
        collectFormTerminalMetadata(*residual.node(), options, "equations");

    const auto* functional =
        findTerminal(terminals, FormTerminalKind::BoundaryFunctionalSymbol);
    ASSERT_NE(functional, nullptr);
    EXPECT_EQ(functional->symbol_name, "Q_wall");
    EXPECT_EQ(functional->domain, DomainKind::Boundary);
    EXPECT_EQ(functional->boundary_marker, 7);
    ASSERT_TRUE(functional->graph_variable.has_value());
    EXPECT_EQ(functional->graph_variable->kind, VariableKind::BoundaryFunctional);
    EXPECT_EQ(functional->graph_variable->name, "Q_wall");

    const auto* integral_ref =
        findTerminal(terminals, FormTerminalKind::BoundaryIntegralRef);
    ASSERT_NE(integral_ref, nullptr);
    ASSERT_TRUE(integral_ref->slot.has_value());
    EXPECT_EQ(*integral_ref->slot, 3u);
    ASSERT_TRUE(integral_ref->graph_variable.has_value());
    EXPECT_EQ(integral_ref->graph_variable->kind, VariableKind::BoundaryFunctional);
    EXPECT_EQ(integral_ref->graph_variable->name, "slot:3");

    const auto* nested_state = findTerminal(terminals, FormTerminalKind::StateField);
    ASSERT_NE(nested_state, nullptr);
    EXPECT_EQ(nested_state->field_id, FieldId{0});
    EXPECT_EQ(nested_state->domain, DomainKind::Boundary);
    EXPECT_EQ(nested_state->boundary_marker, 7);
}

TEST(FormAnalysisBridge, NormalizesTerminalRecordIdentityScopeAndValue)
{
    const auto space = scalarH1();
    const auto u = FormExpr::stateField(0, *space, "u");
    const auto v = FormExpr::testFunction(0, *space, "v");
    const auto residual = (u * v).ds(7);

    FormAnalysisBridgeOptions options;
    options.contribution_name = "coupled_boundary";
    options.origin = "unit_test";
    options.system_name = "wall_system";
    options.owner_participant_name = "wall_participant";

    const auto terminals =
        collectFormTerminalMetadata(*residual.node(), options, "equations");

    const auto* nested_state = findTerminal(terminals, FormTerminalKind::StateField);
    ASSERT_NE(nested_state, nullptr);
    EXPECT_EQ(nested_state->provider, "Forms");
    EXPECT_EQ(nested_state->value_type, FieldType::Scalar);
    EXPECT_EQ(nested_state->value_dimension, 1);
    EXPECT_EQ(nested_state->owner_system_name, "wall_system");
    EXPECT_EQ(nested_state->owner_participant_name, "wall_participant");
    EXPECT_EQ(nested_state->contribution_name, "coupled_boundary");
    EXPECT_EQ(nested_state->origin, "unit_test");
    EXPECT_EQ(nested_state->operator_tag, "equations");
    EXPECT_EQ(nested_state->domain, DomainKind::Boundary);
    EXPECT_EQ(nested_state->boundary_marker, 7);
    ASSERT_TRUE(nested_state->graph_variable.has_value());
    EXPECT_EQ(nested_state->graph_variable->kind, VariableKind::FieldComponent);
    EXPECT_EQ(nested_state->graph_variable->field_id, FieldId{0});
}

TEST(FormAnalysisBridge, ReportsTemporalAndGeometryTerminalCoverage)
{
    const auto space = scalarH1();
    const auto u = FormExpr::stateField(2, *space, "u");
    const auto v = FormExpr::testFunction(2, *space, "v");
    const auto residual =
        (u.dt(1) * v +
         FormExpr::previousSolution(2) * v +
         FormExpr::effectiveTimeStep() * v +
         FormExpr::currentJacobian().det() * v).dx();

    FormulationRecord formulation;
    formulation.operator_tag = "equations";
    formulation.active_fields = {2};
    formulation.residual_expr = residual.nodeShared();

    FormAnalysisBridgeOptions options;
    options.contribution_name = "temporal_geometry";
    options.origin = "unit_test";
    options.system_name = "system";

    const auto metadata = buildFormAnalysisMetadata(formulation, {}, options);

    const auto* derivative =
        findTerminal(metadata.terminals, FormTerminalKind::TimeDerivative);
    ASSERT_NE(derivative, nullptr);
    EXPECT_EQ(derivative->derivative_order, 1);

    const auto* previous =
        findTerminal(metadata.terminals, FormTerminalKind::PreviousSolutionRef);
    ASSERT_NE(previous, nullptr);
    ASSERT_TRUE(previous->history_index.has_value());
    EXPECT_EQ(*previous->history_index, 2);

    EXPECT_NE(findTerminal(metadata.terminals, FormTerminalKind::EffectiveTimeStep),
              nullptr);
    EXPECT_NE(findTerminal(metadata.terminals, FormTerminalKind::CurrentJacobian),
              nullptr);

    const auto* temporal_gate =
        findGate(metadata.feature_gates, FormBridgeFeature::TemporalTerminals);
    ASSERT_NE(temporal_gate, nullptr);
    EXPECT_EQ(temporal_gate->status, FormBridgeFeatureStatus::Partial);

    const auto* geometry_gate =
        findGate(metadata.feature_gates, FormBridgeFeature::GeometryTerminals);
    ASSERT_NE(geometry_gate, nullptr);
    EXPECT_EQ(geometry_gate->status, FormBridgeFeatureStatus::Partial);
}

TEST(FormAnalysisBridge, PreservesGeometrySensitivityProvenanceOptions)
{
    FormulationRecord formulation;
    formulation.operator_tag = "equations";
    formulation.active_fields = {7};

    FormAnalysisBridgeOptions options;
    options.contribution_name = "cut_geometry";
    options.origin = "unit_test";
    options.geometry_sensitivity_provenance.push_back(
        FormGeometrySensitivityProvenanceMetadata{
            .kind = FormGeometrySensitivityProvenanceKind::CutGeometry,
            .provenance_id = "cut:wall:42",
            .construction_policy = "level_set_cut_quadrature",
            .target_kind = "embedded_surface",
            .source_stable_id = 991,
            .parent_entity = 3,
            .parent_geometry_dofs = {31, 32, 33},
            .cut_topology_revision = 17,
            .quadrature_policy_key = 81,
            .visible_to_assembly_paths = {"residual", "jacobian"},
            .location_sensitivity_available = true,
            .jacobian_sensitivity_available = true,
            .measure_sensitivity_available = true,
            .normal_sensitivity_available = true,
            .quadrature_weight_sensitivity_available = true,
            .ad_compatible = true,
            .sensitivity_sample_count = 6,
            .geometry_fields = {7},
        });

    const auto metadata = buildFormAnalysisMetadata(formulation, {}, options);

    ASSERT_EQ(metadata.geometry_sensitivity_provenance.size(), 1u);
    const auto& provenance = metadata.geometry_sensitivity_provenance[0];
    EXPECT_EQ(provenance.kind, FormGeometrySensitivityProvenanceKind::CutGeometry);
    EXPECT_EQ(provenance.provenance_id, "cut:wall:42");
    EXPECT_EQ(provenance.parent_geometry_dofs,
              (std::vector<MeshIndex>{31, 32, 33}));
    EXPECT_EQ(provenance.visible_to_assembly_paths,
              (std::vector<std::string>{"residual", "jacobian"}));
    EXPECT_TRUE(provenance.quadrature_weight_sensitivity_available);
    EXPECT_EQ(provenance.geometry_fields, (std::vector<FieldId>{7}));
}

TEST(FormAnalysisBridge, AdaptsContributionDescriptorsToInstalledBlocks)
{
    const auto space = scalarH1();
    const auto a = FormExpr::stateField(1, *space, "a");
    const auto b = FormExpr::testFunction(0, *space, "b");
    const auto residual = (a * b).dI(11);

    FormulationRecord formulation;
    formulation.operator_tag = "equations";
    formulation.active_fields = {0, 1};
    formulation.residual_expr = residual.nodeShared();
    formulation.variable_couplings.emplace_back(
        VariableKey::field(0),
        VariableKey::named(VariableKind::AuxiliaryInput, "system/input"));

    ContributionDescriptor contribution;
    contribution.origin = "unit_test";
    contribution.operator_tag = "equations";
    contribution.domain = DomainKind::InterfaceFace;
    contribution.interface_marker = 11;
    contribution.test_variables = {VariableKey::field(0)};
    contribution.trial_variables = {VariableKey::field(1)};

    FormAnalysisBridgeOptions options;
    options.contribution_name = "interface_block";
    options.origin = "unit_test";
    options.system_name = "system";

    const auto metadata =
        buildFormAnalysisMetadata(formulation, {contribution}, options);

    EXPECT_EQ(metadata.contribution_name, "interface_block");
    EXPECT_EQ(metadata.origin, "unit_test");
    EXPECT_EQ(metadata.system_name, "system");
    ASSERT_EQ(metadata.installed_fields.size(), 2u);

    ASSERT_EQ(metadata.installed_blocks.size(), 1u);
    EXPECT_EQ(metadata.installed_blocks[0].residual_row, VariableKey::field(0));
    EXPECT_EQ(metadata.installed_blocks[0].dependency, VariableKey::field(1));
    EXPECT_TRUE(metadata.installed_blocks[0].has_matrix);
    ASSERT_EQ(metadata.installed_blocks[0].domains.size(), 1u);
    EXPECT_EQ(metadata.installed_blocks[0].domains[0], DomainKind::InterfaceFace);

    const auto aux_dep = std::find_if(
        metadata.installed_dependencies.begin(),
        metadata.installed_dependencies.end(),
        [](const auto& dependency) {
            return dependency.dependency.kind == VariableKind::AuxiliaryInput;
        });
    ASSERT_NE(aux_dep, metadata.installed_dependencies.end());
    EXPECT_FALSE(aux_dep->contributes_matrix_block);
    EXPECT_TRUE(aux_dep->contributes_vector);

    EXPECT_TRUE(bridgeFeatureAvailable(metadata,
                                       FormBridgeFeature::InstalledBlocks));
    EXPECT_FALSE(bridgeFeatureAvailable(metadata,
                                        FormBridgeFeature::GeometryTerminals));
}
