#include "Coupling/CouplingGraph.h"
#include "Coupling/CouplingTemporalRequirements.h"
#include "Systems/SystemState.h"

#include <gtest/gtest.h>

#include <array>
#include <span>
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
