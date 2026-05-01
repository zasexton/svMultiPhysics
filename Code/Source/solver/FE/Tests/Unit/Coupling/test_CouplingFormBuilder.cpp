#include "Coupling/CouplingFormBuilder.h"

#include "Core/FEException.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* builderSystemToken()
{
    return reinterpret_cast<const systems::FESystem*>(1);
}

CouplingContext makeBuilderContext()
{
    const auto* system = builderSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
    });
    builder.addField({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .field_name = "primary",
        .field_id = 7,
        .space = space,
        .components = 1,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "volume",
        .kind = CouplingRegionKind::Domain,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 12,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "interior",
        .kind = CouplingRegionKind::InteriorFace,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Minus,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "provider_owned",
        .kind = CouplingRegionKind::UserDefined,
    });
    builder.addSharedRegion(SharedRegionRef{
        .name = "shared_interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {{
            .participant_name = "participant",
            .system_name = "system",
            .system = system,
            .region_name = "interface",
            .kind = CouplingRegionKind::InterfaceFace,
            .marker = 17,
            .side = CouplingInterfaceSide::Minus,
        }},
    });
    return builder.build();
}

CouplingContext makeNWayBuilderContext()
{
    const auto* system = builderSystemToken();
    const auto space =
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const std::array<std::string, 3> participants{
        "branch_a",
        "branch_b",
        "branch_c",
    };

    CouplingContextBuilder builder;
    for (std::size_t i = 0; i < participants.size(); ++i) {
        builder.addParticipant({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
        });
        builder.addField({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
            .field_name = "flow",
            .field_id = static_cast<FieldId>(20 + i),
            .space = space,
            .components = 1,
        });
        builder.addRegion({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
            .region_name = "outlet",
            .kind = CouplingRegionKind::Boundary,
            .marker = static_cast<int>(30 + i),
        });
    }
    return builder.build();
}

bool containsFormExprType(const forms::FormExprNode& node,
                          forms::FormExprType type)
{
    if (node.type() == type) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsFormExprType(*child, type)) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(CouplingFormBuilder, BuildsFieldBoundStateAndTestSymbols)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto state = builder.state("participant", "primary", "u");
    const auto test = builder.test("participant", "primary", "w");

    ASSERT_TRUE(state.isValid());
    ASSERT_TRUE(test.isValid());
    EXPECT_EQ(state.node()->type(), forms::FormExprType::StateField);
    EXPECT_EQ(test.node()->type(), forms::FormExprType::TestFunction);
    ASSERT_TRUE(state.node()->fieldId().has_value());
    ASSERT_TRUE(test.node()->fieldId().has_value());
    EXPECT_EQ(*state.node()->fieldId(), 7);
    EXPECT_EQ(*test.node()->fieldId(), 7);
    EXPECT_NE(state.toString().find("u"), std::string::npos);
}

TEST(CouplingFormBuilder, BuildsTemporalTermsThroughFormsVocabulary)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto derivative = builder.timeDerivative("participant", "primary", "u", 2);
    ASSERT_TRUE(derivative.isValid());
    EXPECT_EQ(derivative.node()->type(), forms::FormExprType::TimeDerivative);
    ASSERT_TRUE(derivative.node()->timeDerivativeOrder().has_value());
    EXPECT_EQ(*derivative.node()->timeDerivativeOrder(), 2);

    const auto previous = builder.previousSolution("participant", "primary", 3);
    ASSERT_TRUE(previous.isValid());
    EXPECT_EQ(previous.node()->type(), forms::FormExprType::PreviousSolutionRef);
    ASSERT_TRUE(previous.node()->historyIndex().has_value());
    EXPECT_EQ(*previous.node()->historyIndex(), 3);

    EXPECT_EQ(builder.time().node()->type(), forms::FormExprType::Time);
    EXPECT_EQ(builder.timeStep().node()->type(), forms::FormExprType::TimeStep);
    EXPECT_EQ(builder.effectiveTimeStep().node()->type(), forms::FormExprType::EffectiveTimeStep);
}

TEST(CouplingFormBuilder, PreviousSolutionIsTrialScopedWithoutSourceSymbol)
{
    using PreviousSolutionMember = forms::FormExpr (
        CouplingFormBuilder::*)(std::string_view, std::string_view, int) const;
    static_assert(std::is_invocable_v<PreviousSolutionMember,
                                      const CouplingFormBuilder&,
                                      std::string_view,
                                      std::string_view,
                                      int>);
    static_assert(!std::is_invocable_v<PreviousSolutionMember,
                                       const CouplingFormBuilder&,
                                       std::string_view,
                                       std::string_view,
                                       std::string_view,
                                       int>);

    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto previous = builder.previousSolution("participant", "primary", 2);
    ASSERT_TRUE(previous.isValid());
    EXPECT_EQ(previous.node()->type(), forms::FormExprType::PreviousSolutionRef);
    EXPECT_FALSE(previous.node()->symbolName().has_value());
    ASSERT_TRUE(previous.node()->historyIndex().has_value());
    EXPECT_EQ(*previous.node()->historyIndex(), 2);

    const auto residual =
        (previous * builder.test("participant", "primary", "w")).dx();
    const auto provenance = builder.terminalProvenanceFor(residual);
    ASSERT_EQ(provenance.size(), 1u);
    EXPECT_EQ(provenance[0].kind,
              CouplingFormTerminalProvenanceKind::PreviousSolution);
    ASSERT_TRUE(provenance[0].field.has_value());
    EXPECT_EQ(provenance[0].field->participant_name, "participant");
    EXPECT_EQ(provenance[0].field->field_name, "primary");
    EXPECT_EQ(provenance[0].temporal_quantity,
              CouplingTemporalQuantity::FieldHistoryValue);
    EXPECT_EQ(provenance[0].history_index, 2);
}

TEST(CouplingFormBuilder, BuildsMeshAndGeometryTerminalsThroughFormsVocabulary)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const CouplingGeometryTerminalScope scope{
        .participant_name = "participant",
    };

    auto expect_type = [](const forms::FormExpr& expr,
                          forms::FormExprType type) {
        ASSERT_TRUE(expr.isValid());
        EXPECT_EQ(expr.node()->type(), type);
    };

    expect_type(builder.meshDisplacement(scope), forms::FormExprType::MeshDisplacement);
    expect_type(builder.meshVelocity(scope), forms::FormExprType::MeshVelocity);
    expect_type(builder.meshAcceleration(scope), forms::FormExprType::MeshAcceleration);
    expect_type(builder.previousMeshVelocity(scope),
                forms::FormExprType::PreviousMeshVelocity);
    expect_type(builder.predictedMeshVelocity(scope),
                forms::FormExprType::PredictedMeshVelocity);

    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::MeshDisplacement, scope),
                forms::FormExprType::MeshDisplacement);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::Coordinate, scope),
                forms::FormExprType::Coordinate);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::ReferenceCoordinate, scope),
                forms::FormExprType::ReferenceCoordinate);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CurrentCoordinate, scope),
                forms::FormExprType::CurrentCoordinate);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::PreviousCoordinate, scope),
                forms::FormExprType::PreviousCoordinate);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate, scope),
                forms::FormExprType::ReferencePhysicalCoordinate);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::Jacobian, scope),
                forms::FormExprType::Jacobian);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::JacobianInverse, scope),
                forms::FormExprType::JacobianInverse);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::JacobianDeterminant, scope),
                forms::FormExprType::JacobianDeterminant);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CurrentJacobian, scope),
                forms::FormExprType::CurrentJacobian);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::ReferenceJacobian, scope),
                forms::FormExprType::ReferenceJacobian);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant,
                    scope),
                forms::FormExprType::Determinant);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant,
                    scope),
                forms::FormExprType::Determinant);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::Normal, scope),
                forms::FormExprType::Normal);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CurrentNormal, scope),
                forms::FormExprType::CurrentNormal);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::ReferenceNormal, scope),
                forms::FormExprType::ReferenceNormal);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CurrentMeasure, scope),
                forms::FormExprType::CurrentMeasure);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::ReferenceMeasure, scope),
                forms::FormExprType::ReferenceMeasure);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::SurfaceJacobian, scope),
                forms::FormExprType::SurfaceJacobian);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CellDiameter, scope),
                forms::FormExprType::CellDiameter);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CellVolume, scope),
                forms::FormExprType::CellVolume);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::FacetArea, scope),
                forms::FormExprType::FacetArea);
    expect_type(builder.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CellDomainId, scope),
                forms::FormExprType::CellDomainId);
}

TEST(CouplingFormBuilder, RecordsMeshTemporalProvenanceRoles)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const CouplingGeometryTerminalScope scope{
        .participant_name = "participant",
    };

    const auto mesh_velocity = builder.meshVelocity(scope);
    const auto mesh_acceleration = builder.meshAcceleration(scope);
    const auto previous_velocity = builder.previousMeshVelocity(scope);
    const auto predicted_velocity = builder.predictedMeshVelocity(scope);
    const auto test = builder.test("participant", "primary", "w");
    const auto residual =
        ((forms::dot(mesh_velocity, mesh_acceleration) +
          forms::dot(previous_velocity, predicted_velocity)) * test).dx();

    const auto provenance = builder.terminalProvenanceFor(residual);
    ASSERT_EQ(provenance.size(), 4u);

    auto expect_mesh_temporal =
        [](const CouplingFormTerminalProvenanceDeclaration& declaration,
           std::uint64_t sequence,
           CouplingTemporalQuantity quantity,
           systems::MeshMotionFieldRole role) {
            EXPECT_EQ(declaration.kind,
                      CouplingFormTerminalProvenanceKind::MeshTemporal);
            EXPECT_EQ(declaration.terminal_sequence, sequence);
            EXPECT_EQ(declaration.temporal_quantity, quantity);
            ASSERT_TRUE(declaration.scope.has_value());
            ASSERT_TRUE(declaration.scope->participant_name.has_value());
            EXPECT_EQ(*declaration.scope->participant_name, "participant");
            ASSERT_TRUE(declaration.mesh_motion_role.has_value());
            EXPECT_EQ(*declaration.mesh_motion_role, role);
        };

    expect_mesh_temporal(provenance[0],
                         0,
                         CouplingTemporalQuantity::MeshVelocity,
                         systems::MeshMotionFieldRole::Velocity);
    expect_mesh_temporal(provenance[1],
                         1,
                         CouplingTemporalQuantity::MeshAcceleration,
                         systems::MeshMotionFieldRole::Acceleration);
    expect_mesh_temporal(provenance[2],
                         2,
                         CouplingTemporalQuantity::PreviousMeshVelocity,
                         systems::MeshMotionFieldRole::PreviousVelocity);
    expect_mesh_temporal(provenance[3],
                         3,
                         CouplingTemporalQuantity::PredictedMeshVelocity,
                         systems::MeshMotionFieldRole::PredictedVelocity);
}

TEST(CouplingFormBuilder, AttachesTerminalProvenanceInResidualEncounterOrder)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const CouplingGeometryTerminalScope scope{
        .participant_name = "participant",
    };

    const auto previous = builder.previousSolution("participant", "primary", 2);
    const auto mesh_velocity = builder.meshVelocity(scope);
    const auto current_normal = builder.geometryTerminal(
        CouplingGeometryTerminalQuantity::CurrentNormal, scope);
    const auto test = builder.test("participant", "primary", "w");
    const auto residual =
        ((previous + forms::dot(mesh_velocity, current_normal)) * test).dx();

    const auto provenance = builder.terminalProvenanceFor(residual);
    ASSERT_EQ(provenance.size(), 3u);

    EXPECT_EQ(provenance[0].kind,
              CouplingFormTerminalProvenanceKind::PreviousSolution);
    EXPECT_EQ(provenance[0].terminal_sequence, 0u);
    ASSERT_TRUE(provenance[0].field.has_value());
    EXPECT_EQ(provenance[0].field->participant_name, "participant");
    EXPECT_EQ(provenance[0].field->field_name, "primary");
    EXPECT_EQ(provenance[0].temporal_quantity,
              CouplingTemporalQuantity::FieldHistoryValue);
    EXPECT_EQ(provenance[0].history_index, 2);

    EXPECT_EQ(provenance[1].kind,
              CouplingFormTerminalProvenanceKind::MeshTemporal);
    EXPECT_EQ(provenance[1].terminal_sequence, 1u);
    EXPECT_EQ(provenance[1].temporal_quantity,
              CouplingTemporalQuantity::MeshVelocity);
    ASSERT_TRUE(provenance[1].scope.has_value());
    ASSERT_TRUE(provenance[1].scope->participant_name.has_value());
    EXPECT_EQ(*provenance[1].scope->participant_name, "participant");
    ASSERT_TRUE(provenance[1].mesh_motion_role.has_value());
    EXPECT_EQ(*provenance[1].mesh_motion_role,
              systems::MeshMotionFieldRole::Velocity);

    EXPECT_EQ(provenance[2].kind,
              CouplingFormTerminalProvenanceKind::GeometryTerminal);
    EXPECT_EQ(provenance[2].terminal_sequence, 2u);
    EXPECT_EQ(provenance[2].geometry_quantity,
              CouplingGeometryTerminalQuantity::CurrentNormal);
    ASSERT_TRUE(provenance[2].scope.has_value());
    ASSERT_TRUE(provenance[2].scope->participant_name.has_value());
    EXPECT_EQ(*provenance[2].scope->participant_name, "participant");

    CouplingFormContribution contribution;
    contribution.residual = residual;
    const auto attached =
        builder.attachTerminalProvenance(std::move(contribution));
    ASSERT_EQ(attached.terminal_provenance.size(), provenance.size());
    EXPECT_EQ(attached.terminal_provenance[0].history_index, 2);
    EXPECT_EQ(attached.terminal_provenance[1].temporal_quantity,
              CouplingTemporalQuantity::MeshVelocity);
    EXPECT_EQ(attached.terminal_provenance[2].geometry_quantity,
              CouplingGeometryTerminalQuantity::CurrentNormal);
}

TEST(CouplingFormBuilder, BuildsEquationContributionsWithTerminalProvenance)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const CouplingGeometryTerminalScope scope{
        .participant_name = "participant",
    };

    const auto u = builder.state("participant", "primary", "u");
    const auto w = builder.test("participant", "primary", "w");
    const auto normal = builder.geometryTerminal(
        CouplingGeometryTerminalQuantity::CurrentNormal, scope);

    auto contribution = builder.equationContribution(
        CouplingEquationContributionRequest{
            .contribution_name = "participant.primary.normal_balance",
            .origin = "CouplingFormBuilderTest",
            .residual_field_uses = {
                CouplingFieldUse{
                    .participant_name = "participant",
                    .field_name = "primary",
                },
            },
            .trial_field_uses = {
                CouplingFieldUse{
                    .participant_name = "participant",
                    .field_name = "primary",
                },
            },
            .residual = (forms::dot(u, normal) * w).dx(),
        });

    EXPECT_EQ(contribution.contribution_name,
              "participant.primary.normal_balance");
    EXPECT_EQ(contribution.origin, "CouplingFormBuilderTest");
    EXPECT_EQ(contribution.operator_name, "equations");
    ASSERT_EQ(contribution.field_uses.size(), 1u);
    EXPECT_EQ(contribution.field_uses.front().participant_name, "participant");
    EXPECT_EQ(contribution.field_uses.front().field_name, "primary");
    ASSERT_EQ(contribution.extra_trial_field_uses.size(), 1u);
    EXPECT_EQ(contribution.extra_trial_field_uses.front().participant_name,
              "participant");
    EXPECT_EQ(contribution.extra_trial_field_uses.front().field_name,
              "primary");
    ASSERT_EQ(contribution.terminal_provenance.size(), 1u);
    EXPECT_EQ(contribution.terminal_provenance.front().kind,
              CouplingFormTerminalProvenanceKind::GeometryTerminal);
    EXPECT_EQ(contribution.terminal_provenance.front().geometry_quantity,
              CouplingGeometryTerminalQuantity::CurrentNormal);
}

TEST(CouplingFormBuilder, RejectsTerminalTransformsThatLoseProvenanceIdentity)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto previous = builder.previousSolution("participant", "primary", 1);
    const auto preserved = previous.transformNodes(
        [](const forms::FormExprNode&) -> std::optional<forms::FormExpr> {
            return std::nullopt;
        });
    EXPECT_EQ(preserved.nodeShared(), previous.nodeShared());

    const auto test = builder.test("participant", "primary", "w");
    CouplingFormContribution preserved_contribution;
    preserved_contribution.residual = (preserved * test).dx();
    const auto attached =
        builder.attachTerminalProvenance(std::move(preserved_contribution));
    ASSERT_EQ(attached.terminal_provenance.size(), 1u);
    EXPECT_EQ(attached.terminal_provenance[0].terminal_sequence, 0u);
    EXPECT_EQ(attached.terminal_provenance[0].history_index, 1);

    const auto replaced = previous.transformNodes(
        [](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
            if (node.type() == forms::FormExprType::PreviousSolutionRef) {
                return forms::FormExpr::previousSolution(
                    node.historyIndex().value_or(1));
            }
            return std::nullopt;
        });
    EXPECT_NE(replaced.nodeShared(), previous.nodeShared());

    CouplingFormContribution replaced_contribution;
    replaced_contribution.residual = (replaced * test).dx();
    EXPECT_THROW(static_cast<void>(builder.attachTerminalProvenance(
                     std::move(replaced_contribution))),
                 InvalidArgumentException);
}

TEST(CouplingFormBuilder, RejectsInvalidTemporalRequestsAndUnknownFields)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    EXPECT_THROW(static_cast<void>(builder.timeDerivative("participant", "primary", "u", 0)),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(builder.previousSolution("participant", "primary", 0)),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(builder.state("participant", "missing", "u")),
                 InvalidArgumentException);
}

TEST(CouplingFormBuilder, DelegatesRegionLookupsToContext)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto region = builder.region("participant", "surface");
    EXPECT_EQ(region.marker, 12);
    EXPECT_EQ(region.kind, CouplingRegionKind::Boundary);
}

TEST(CouplingFormBuilder, LowersRegionKindsToFormsMeasures)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const auto integrand =
        builder.state("participant", "primary", "u") *
        builder.test("participant", "primary", "w");

    const auto cell = builder.integrate(integrand, "participant", "volume");
    ASSERT_TRUE(cell.isValid());
    EXPECT_EQ(cell.node()->type(), forms::FormExprType::CellIntegral);

    const auto boundary = builder.integrate(integrand, "participant", "surface");
    ASSERT_TRUE(boundary.isValid());
    EXPECT_EQ(boundary.node()->type(), forms::FormExprType::BoundaryIntegral);
    ASSERT_TRUE(boundary.node()->boundaryMarker().has_value());
    EXPECT_EQ(*boundary.node()->boundaryMarker(), 12);

    const auto interior = builder.integrate(integrand, "participant", "interior");
    ASSERT_TRUE(interior.isValid());
    EXPECT_EQ(interior.node()->type(), forms::FormExprType::InteriorFaceIntegral);

    const auto interface = builder.integrate(integrand, "participant", "interface");
    ASSERT_TRUE(interface.isValid());
    EXPECT_EQ(interface.node()->type(), forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(interface.node()->interfaceMarker().has_value());
    EXPECT_EQ(*interface.node()->interfaceMarker(), 17);

    const auto shared_interface =
        builder.integrateShared(integrand, "shared_interface", "participant");
    ASSERT_TRUE(shared_interface.isValid());
    EXPECT_EQ(shared_interface.node()->type(), forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(shared_interface.node()->interfaceMarker().has_value());
    EXPECT_EQ(*shared_interface.node()->interfaceMarker(), 17);
    const auto shared_region = builder.sharedRegion("shared_interface", "participant");
    EXPECT_EQ(shared_region.marker, 17);
    EXPECT_EQ(shared_region.side, CouplingInterfaceSide::Minus);
}

TEST(CouplingFormBuilder, BuildsSharedInterfaceViewsThroughFormsVocabulary)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto shared = builder.sharedInterface("shared_interface");
    EXPECT_EQ(std::string(shared.name()), "shared_interface");
    EXPECT_EQ(shared.group().participant_regions.size(), 1u);

    const auto side = shared.side("participant");
    EXPECT_EQ(std::string(side.participantName()), "participant");
    EXPECT_EQ(side.region().side, CouplingInterfaceSide::Minus);

    const auto state = side.state("primary", "u");
    const auto test = side.test("primary", "w");
    const auto derivative = side.dt("primary", "udot");
    const auto normal = side.normal();
    const auto normal_component = side.normalComponent(normal);
    const auto normal_projection = side.normalProjection(normal);
    const auto tangential_projection = side.tangentialProjection(normal);

    ASSERT_TRUE(state.isValid());
    ASSERT_TRUE(test.isValid());
    ASSERT_TRUE(derivative.isValid());
    ASSERT_TRUE(normal.isValid());
    ASSERT_TRUE(normal_component.isValid());
    ASSERT_TRUE(normal_projection.isValid());
    ASSERT_TRUE(tangential_projection.isValid());
    EXPECT_TRUE(containsFormExprType(*state.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*test.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*derivative.node(),
                                     forms::FormExprType::TimeDerivative));
    EXPECT_TRUE(containsFormExprType(*normal.node(),
                                     forms::FormExprType::Normal));
    EXPECT_TRUE(containsFormExprType(*normal.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*normal_component.node(),
                                     forms::FormExprType::InnerProduct));
    EXPECT_TRUE(containsFormExprType(*normal_component.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*normal_projection.node(),
                                     forms::FormExprType::InnerProduct));
    EXPECT_TRUE(containsFormExprType(*normal_projection.node(),
                                     forms::FormExprType::Multiply));
    EXPECT_TRUE(containsFormExprType(*tangential_projection.node(),
                                     forms::FormExprType::Subtract));
    EXPECT_TRUE(containsFormExprType(*tangential_projection.node(),
                                     forms::FormExprType::InnerProduct));

    const auto residual =
        shared.integral((state + forms::dot(normal, normal)) * test,
                        "participant");
    ASSERT_TRUE(residual.isValid());
    EXPECT_EQ(residual.node()->type(), forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*residual.node()->interfaceMarker(), 17);

    const auto provenance = builder.terminalProvenanceFor(residual);
    ASSERT_EQ(provenance.size(), 1u);
    EXPECT_EQ(provenance[0].geometry_quantity,
              CouplingGeometryTerminalQuantity::Normal);
    ASSERT_TRUE(provenance[0].scope.has_value());
    ASSERT_TRUE(provenance[0].scope->location.has_value());
    EXPECT_EQ(provenance[0].scope->location->side,
              CouplingInterfaceSide::Minus);
    ASSERT_TRUE(provenance[0].scope->location->shared_region_name.has_value());
    EXPECT_EQ(*provenance[0].scope->location->shared_region_name,
              "shared_interface");
}

TEST(CouplingFormBuilder, RejectsMissingSharedInterfaceViewMappings)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    EXPECT_THROW(static_cast<void>(builder.sharedInterface("missing").group()),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(
                     builder.sharedInterface("shared_interface").side("missing")),
                 InvalidArgumentException);
}

TEST(CouplingFormBuilder, BuildsRegionRelationViewsThroughFormsVocabulary)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    CouplingRegionRelationRequirement requirement{
        .relation_name = "volume_surface_balance",
        .relation_kind = CouplingRegionRelationKind::VolumeBoundaryRelation,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "participant",
                .region_name = "volume",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "participant",
                .region_name = "surface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            },
        },
    };

    const auto relation = builder.regionRelation(std::move(requirement));
    EXPECT_EQ(std::string(relation.name()), "volume_surface_balance");

    const auto volume = relation.endpoint("participant", "volume");
    const auto surface = relation.endpoint("participant", "surface");
    EXPECT_EQ(volume.region().kind, CouplingRegionKind::Domain);
    EXPECT_EQ(surface.region().kind, CouplingRegionKind::Boundary);

    const auto volume_term =
        volume.state("primary", "u") * volume.test("primary", "w");
    const auto surface_normal = surface.normal();
    const auto surface_normal_component =
        surface.normalComponent(surface_normal);
    const auto surface_normal_projection =
        surface.normalProjection(surface_normal);
    const auto surface_tangential_projection =
        surface.tangentialProjection(surface_normal);
    const auto surface_term =
        surface.state("primary", "u_b") * surface.test("primary", "w_b");
    EXPECT_TRUE(containsFormExprType(*surface_normal.node(),
                                     forms::FormExprType::Normal));
    ASSERT_TRUE(surface_normal_component.isValid());
    ASSERT_TRUE(surface_normal_projection.isValid());
    ASSERT_TRUE(surface_tangential_projection.isValid());
    EXPECT_TRUE(containsFormExprType(*surface_normal_component.node(),
                                     forms::FormExprType::InnerProduct));
    EXPECT_TRUE(containsFormExprType(*surface_normal_projection.node(),
                                     forms::FormExprType::Multiply));
    EXPECT_TRUE(containsFormExprType(*surface_normal_projection.node(),
                                     forms::FormExprType::InnerProduct));
    EXPECT_TRUE(containsFormExprType(*surface_tangential_projection.node(),
                                     forms::FormExprType::Subtract));
    EXPECT_TRUE(containsFormExprType(*surface_tangential_projection.node(),
                                     forms::FormExprType::InnerProduct));

    const auto volume_integral = volume.integral(volume_term);
    const auto surface_integral =
        relation.integral(surface_term + forms::dot(surface_normal, surface_normal),
                          "participant",
                          "surface");
    ASSERT_TRUE(volume_integral.isValid());
    ASSERT_TRUE(surface_integral.isValid());
    EXPECT_EQ(volume_integral.node()->type(),
              forms::FormExprType::CellIntegral);
    EXPECT_EQ(surface_integral.node()->type(),
              forms::FormExprType::BoundaryIntegral);
    ASSERT_TRUE(surface_integral.node()->boundaryMarker().has_value());
    EXPECT_EQ(*surface_integral.node()->boundaryMarker(), 12);

    const std::array<forms::FormExpr, 2> terms{
        volume_integral,
        surface_integral,
    };
    const auto summed = relation.sum(terms);
    EXPECT_TRUE(summed.isValid());

    EXPECT_THROW(static_cast<void>(
                     relation.endpoint("participant", "missing")),
                 InvalidArgumentException);
}

TEST(CouplingFormBuilder, BuildsNWayRelationResidualThroughEndpointViews)
{
    const auto context = makeNWayBuilderContext();
    const CouplingFormBuilder builder(context);
    CouplingRegionRelationRequirement requirement{
        .relation_name = "junction_balance",
        .relation_kind = CouplingRegionRelationKind::NWayInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "branch_a",
                .region_name = "outlet",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "branch_b",
                .region_name = "outlet",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "branch_c",
                .region_name = "outlet",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            },
        },
    };

    const auto relation = builder.regionRelation(std::move(requirement));
    const auto branch_a = relation.endpoint("branch_a", "outlet");
    const auto branch_b = relation.endpoint("branch_b", "outlet");
    const auto branch_c = relation.endpoint("branch_c", "outlet");

    const std::array<forms::FormExpr, 3> branch_terms{
        branch_a.integral(branch_a.state("flow", "q_a") *
                          branch_a.test("flow", "w_a")),
        branch_b.integral(branch_b.state("flow", "q_b") *
                          branch_b.test("flow", "w_b")),
        branch_c.integral(branch_c.state("flow", "q_c") *
                          branch_c.test("flow", "w_c")),
    };

    for (std::size_t i = 0; i < branch_terms.size(); ++i) {
        ASSERT_TRUE(branch_terms[i].isValid());
        ASSERT_EQ(branch_terms[i].node()->type(),
                  forms::FormExprType::BoundaryIntegral);
        ASSERT_TRUE(branch_terms[i].node()->boundaryMarker().has_value());
        EXPECT_EQ(*branch_terms[i].node()->boundaryMarker(),
                  static_cast<int>(30 + i));
    }
    EXPECT_TRUE(relation.sum(branch_terms).isValid());
}

TEST(CouplingFormBuilder, AuthorsInterfaceResidualWithSideRestrictions)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const auto state = builder.state("participant", "primary", "u");
    const auto test = builder.test("participant", "primary", "w");

    const auto residual =
        ((state.minus() - state.plus()) * test.minus()).dI(17);

    ASSERT_TRUE(residual.isValid());
    ASSERT_EQ(residual.node()->type(), forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*residual.node()->interfaceMarker(), 17);
    EXPECT_TRUE(
        containsFormExprType(*residual.node(), forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(
        containsFormExprType(*residual.node(), forms::FormExprType::RestrictPlus));
}

TEST(CouplingFormBuilder, RejectsUserDefinedRegionFormsLowering)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const auto integrand =
        builder.state("participant", "primary", "u") *
        builder.test("participant", "primary", "w");

    EXPECT_THROW(static_cast<void>(
                     builder.integrate(integrand, "participant", "provider_owned")),
                 InvalidArgumentException);
}
