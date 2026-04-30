#include "Coupling/CouplingGraph.h"
#include "Coupling/CouplingTemporalRequirements.h"
#include "Systems/FESystem.h"
#include "Systems/SystemState.h"

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

CouplingTemporalRequirement fieldDerivative(int order)
{
    return CouplingTemporalRequirement{
        .quantity = CouplingTemporalQuantity::FieldDerivative,
        .field = CouplingFieldUse{.participant_name = "left", .field_name = "primary"},
        .derivative_order = order,
    };
}

CouplingTemporalRequirement fieldHistory(int index)
{
    return CouplingTemporalRequirement{
        .quantity = CouplingTemporalQuantity::FieldHistoryValue,
        .field = CouplingFieldUse{.participant_name = "left", .field_name = "primary"},
        .history_index = index,
    };
}

CouplingContractDeclaration temporalDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.temporal_requirements.push_back(fieldDerivative(1));
    declaration.temporal_requirements.push_back(fieldHistory(2));
    declaration.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::EffectiveTimeStep,
    });
    return declaration;
}

const systems::FESystem* temporalSystemToken(int token)
{
    return reinterpret_cast<const systems::FESystem*>(token);
}

CouplingContext meshTemporalContext()
{
    const auto* left_system = temporalSystemToken(1);
    const auto* right_system = temporalSystemToken(2);
    const CouplingRegionRef left_interface{
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 7,
        .side = CouplingInterfaceSide::Minus,
    };
    const CouplingRegionRef right_interface{
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 8,
        .side = CouplingInterfaceSide::Plus,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
    });
    builder.addRegion(left_interface);
    builder.addRegion(right_interface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_interface, right_interface},
    });
    return builder.build();
}

CouplingGeometryTerminalScope participantMeshScope(std::string participant_name)
{
    return CouplingGeometryTerminalScope{
        .participant_name = std::move(participant_name),
    };
}

CouplingGeometryTerminalScope sharedInterfaceLocationScope()
{
    return CouplingGeometryTerminalScope{
        .location = CouplingGeometryTerminalLocationDeclaration{
            .region_kind = CouplingRegionKind::InterfaceFace,
            .shared_region_name = "interface",
        },
    };
}

CouplingTemporalRequirement meshTemporalRequirement(
    CouplingTemporalQuantity quantity,
    systems::MeshMotionFieldRole role,
    CouplingGeometryTerminalScope scope)
{
    return CouplingTemporalRequirement{
        .quantity = quantity,
        .mesh_motion_scope = std::move(scope),
        .mesh_motion_role = role,
    };
}

CouplingFormTemporalProvenance meshTemporalSymbol(
    CouplingTemporalQuantity quantity,
    systems::MeshMotionFieldRole role,
    CouplingGeometryTerminalScope scope)
{
    return CouplingFormTemporalProvenance{
        .mesh_motion_scope = std::move(scope),
        .mesh_motion_role = role,
        .quantity = quantity,
    };
}

} // namespace

TEST(CouplingTemporalRequirements, SummarizesFieldAndGlobalTemporalNeeds)
{
    const std::array<CouplingTemporalRequirement, 3> requirements{
        fieldDerivative(1),
        fieldHistory(3),
        CouplingTemporalRequirement{.quantity = CouplingTemporalQuantity::TimeStep},
    };

    const auto summary = summarizeTemporalRequirements(
        std::span<const CouplingTemporalRequirement>(requirements));

    EXPECT_EQ(summary.max_derivative_order, 1);
    EXPECT_EQ(summary.max_history_index, 3);
    EXPECT_TRUE(summary.requires_time_step);
    EXPECT_EQ(summary.field_temporal_requirements.size(), 2u);
}

TEST(CouplingTemporalRequirements, DerivesHistoryDepthFromSystemStateView)
{
    const std::array<Real, 1> previous{1.0};
    const std::array<Real, 1> previous2{0.0};
    systems::SystemStateView state;
    state.u_prev = std::span<const Real>(previous);
    state.u_prev2 = std::span<const Real>(previous2);

    const auto availability = temporalAvailabilityFromSystemState(
        state,
        /*max_derivative_order=*/2);

    EXPECT_EQ(availability.max_derivative_order, 2);
    EXPECT_EQ(availability.history_depth, 2);
    EXPECT_TRUE(availability.provides_time);
    EXPECT_TRUE(availability.provides_time_step);
}

TEST(CouplingTemporalRequirements, RejectsUnsupportedRequiredTemporalNeeds)
{
    const std::array<CouplingTemporalRequirement, 3> requirements{
        fieldDerivative(2),
        fieldHistory(3),
        CouplingTemporalRequirement{.quantity = CouplingTemporalQuantity::EffectiveTimeStep},
    };

    CouplingTemporalAvailability availability;
    availability.max_derivative_order = 1;
    availability.history_depth = 2;
    availability.provides_effective_time_step = false;

    const auto validation = validateTemporalRequirements(
        std::span<const CouplingTemporalRequirement>(requirements),
        availability);

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("derivative requirement exceeds"), std::string::npos);
    EXPECT_NE(text.find("history requirement exceeds"), std::string::npos);
    EXPECT_NE(text.find("effective-time-step symbol is unavailable"),
              std::string::npos);
}

TEST(CouplingTemporalRequirements, OptionalUnsupportedTemporalNeedsDoNotFail)
{
    const std::array<CouplingTemporalRequirement, 2> requirements{
        CouplingTemporalRequirement{
            .quantity = CouplingTemporalQuantity::FieldDerivative,
            .field = CouplingFieldUse{.participant_name = "left", .field_name = "primary"},
            .derivative_order = 2,
            .requirement = CouplingRequirement::Optional,
        },
        CouplingTemporalRequirement{
            .quantity = CouplingTemporalQuantity::FieldHistoryValue,
            .field = CouplingFieldUse{.participant_name = "left", .field_name = "primary"},
            .history_index = 2,
            .requirement = CouplingRequirement::Optional,
        },
    };

    const CouplingTemporalAvailability availability;
    const auto validation = validateTemporalRequirements(
        std::span<const CouplingTemporalRequirement>(requirements),
        availability);

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingTemporalRequirements, CouplingGraphValidatesAggregatedDeclarations)
{
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{temporalDeclaration()};
    const auto declaration_validation = graph.buildDeclarationGraph(
        CouplingContext{},
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(declaration_validation.ok()) << formatDiagnostics(declaration_validation);

    CouplingTemporalAvailability insufficient;
    insufficient.max_derivative_order = 1;
    insufficient.history_depth = 1;
    insufficient.provides_effective_time_step = true;

    const auto validation = graph.validateTemporalRequirements(insufficient);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("history requirement exceeds"),
              std::string::npos);

    CouplingTemporalAvailability sufficient;
    sufficient.max_derivative_order = 1;
    sufficient.history_depth = 2;
    sufficient.provides_effective_time_step = true;
    EXPECT_TRUE(graph.validateTemporalRequirements(sufficient).ok());
}

TEST(CouplingTemporalRequirements, CouplingGraphRequiresInstalledTemporalSymbolsDeclared)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "temporal_form";
    metadata.temporal_symbols.push_back(
        CouplingFormTemporalProvenance{
            .quantity = CouplingTemporalQuantity::TimeStep,
        });
    metadata.temporal_symbols.push_back(
        CouplingFormTemporalProvenance{
            .quantity = CouplingTemporalQuantity::FieldHistoryValue,
            .history_index = 2,
        });

    const auto run_validation = [&]() {
        CouplingGraph graph;
        const std::array<CouplingContractDeclaration, 1> declarations{declaration};
        const std::array<CouplingFormAnalysisMetadata, 1> installed{metadata};
        return graph.buildFinalizedGraph(
            CouplingContext{},
            std::span<const CouplingContractDeclaration>(declarations),
            std::span<const CouplingFormAnalysisMetadata>(installed));
    };

    const auto missing = run_validation();
    EXPECT_FALSE(missing.ok());
    const auto missing_text = formatDiagnostics(missing);
    EXPECT_NE(missing_text.find("time_step"), std::string::npos);
    EXPECT_NE(missing_text.find("field_history_value"), std::string::npos);

    declaration.temporal_requirements.push_back(
        CouplingTemporalRequirement{
            .quantity = CouplingTemporalQuantity::TimeStep,
        });
    const auto partially_declared = run_validation();
    EXPECT_FALSE(partially_declared.ok());
    EXPECT_NE(formatDiagnostics(partially_declared).find("field_history_value"),
              std::string::npos);

    declaration.temporal_requirements.push_back(
        CouplingTemporalRequirement{
            .quantity = CouplingTemporalQuantity::FieldHistoryValue,
            .history_index = 2,
        });
    EXPECT_TRUE(run_validation().ok());
}

TEST(CouplingTemporalRequirements, CouplingGraphValidatesMeshTemporalScopeAndRoles)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "mesh_temporal_contract";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.temporal_requirements.push_back(meshTemporalRequirement(
        CouplingTemporalQuantity::MeshVelocity,
        systems::MeshMotionFieldRole::Velocity,
        participantMeshScope("left")));
    declaration.temporal_requirements.push_back(meshTemporalRequirement(
        CouplingTemporalQuantity::MeshAcceleration,
        systems::MeshMotionFieldRole::Acceleration,
        participantMeshScope("left")));
    declaration.temporal_requirements.push_back(meshTemporalRequirement(
        CouplingTemporalQuantity::PreviousMeshVelocity,
        systems::MeshMotionFieldRole::PreviousVelocity,
        participantMeshScope("left")));
    declaration.temporal_requirements.push_back(meshTemporalRequirement(
        CouplingTemporalQuantity::PredictedMeshVelocity,
        systems::MeshMotionFieldRole::PredictedVelocity,
        participantMeshScope("left")));

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "mesh_temporal_form";
    metadata.temporal_symbols.push_back(meshTemporalSymbol(
        CouplingTemporalQuantity::MeshVelocity,
        systems::MeshMotionFieldRole::Velocity,
        participantMeshScope("left")));
    metadata.temporal_symbols.push_back(meshTemporalSymbol(
        CouplingTemporalQuantity::MeshAcceleration,
        systems::MeshMotionFieldRole::Acceleration,
        participantMeshScope("left")));
    metadata.temporal_symbols.push_back(meshTemporalSymbol(
        CouplingTemporalQuantity::PreviousMeshVelocity,
        systems::MeshMotionFieldRole::PreviousVelocity,
        participantMeshScope("left")));
    metadata.temporal_symbols.push_back(meshTemporalSymbol(
        CouplingTemporalQuantity::PredictedMeshVelocity,
        systems::MeshMotionFieldRole::PredictedVelocity,
        participantMeshScope("left")));

    const auto validate = [&](const CouplingFormAnalysisMetadata& installed_metadata) {
        CouplingGraph graph;
        const std::array<CouplingContractDeclaration, 1> declarations{declaration};
        const std::array<CouplingFormAnalysisMetadata, 1> installed{installed_metadata};
        return graph.buildFinalizedGraph(
            meshTemporalContext(),
            std::span<const CouplingContractDeclaration>(declarations),
            std::span<const CouplingFormAnalysisMetadata>(installed));
    };

    EXPECT_TRUE(validate(metadata).ok());

    auto wrong_role = metadata;
    wrong_role.temporal_symbols[0].mesh_motion_role =
        systems::MeshMotionFieldRole::Acceleration;
    const auto role_validation = validate(wrong_role);
    EXPECT_FALSE(role_validation.ok());
    EXPECT_NE(formatDiagnostics(role_validation).find("mesh_velocity"),
              std::string::npos);

    auto wrong_scope = metadata;
    wrong_scope.temporal_symbols[1].mesh_motion_scope =
        participantMeshScope("right");
    const auto scope_validation = validate(wrong_scope);
    EXPECT_FALSE(scope_validation.ok());
    EXPECT_NE(formatDiagnostics(scope_validation).find("mesh_acceleration"),
              std::string::npos);
}

TEST(CouplingTemporalRequirements, CouplingGraphRejectsAmbiguousSharedRegionMeshTemporalScope)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "mesh_temporal_contract";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.temporal_requirements.push_back(meshTemporalRequirement(
        CouplingTemporalQuantity::MeshVelocity,
        systems::MeshMotionFieldRole::Velocity,
        sharedInterfaceLocationScope()));

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = graph.buildFinalizedGraph(
        meshTemporalContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>{});

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("ambiguous"), std::string::npos);
    EXPECT_NE(text.find("interface"), std::string::npos);
}
