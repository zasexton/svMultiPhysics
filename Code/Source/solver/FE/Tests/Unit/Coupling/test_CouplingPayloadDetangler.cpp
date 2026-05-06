#include "Coupling/CouplingPayloadDetangler.h"

#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* systemToken(std::uintptr_t value)
{
    return reinterpret_cast<const systems::FESystem*>(value);
}

struct DetanglerFixtureData {
    std::shared_ptr<spaces::H1Space> space{
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1)};
    CouplingContext context;
};

DetanglerFixtureData makeDetanglerContext()
{
    DetanglerFixtureData data;
    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "fluid",
        .system_name = "fluid_system",
        .system = systemToken(1u),
    });
    builder.addParticipant({
        .participant_name = "solid",
        .system_name = "solid_system",
        .system = systemToken(2u),
    });
    builder.addField({
        .participant_name = "fluid",
        .system_name = "fluid_system",
        .system = systemToken(1u),
        .field_name = "velocity",
        .field_id = 1,
        .space = data.space,
        .components = 2,
    });
    builder.addField({
        .participant_name = "fluid",
        .system_name = "fluid_system",
        .system = systemToken(1u),
        .field_name = "pressure",
        .field_id = 2,
        .space = data.space,
        .components = 1,
    });
    builder.addField({
        .participant_name = "solid",
        .system_name = "solid_system",
        .system = systemToken(2u),
        .field_name = "displacement",
        .field_id = 3,
        .space = data.space,
        .components = 2,
    });
    data.context = builder.build();
    return data;
}

CouplingPayloadExtractionRequest baseRequest()
{
    return CouplingPayloadExtractionRequest{
        .exchange_name = "fluid_load",
        .contribution_name = "fsi.fsi_interface.fluid_traction_balance",
        .producer_participant_name = "fluid",
        .producer_field_name = "velocity",
        .consumer_participant_name = "solid",
        .consumer_field_name = "displacement",
        .value = vectorValue(2),
        .preferred_kind = CouplingPayloadKind::CoefficientExpression,
        .fallback_policy =
            CouplingPayloadFallbackPolicy::WarnAndUseResidualRecipe,
        .transfer = CouplingTransferDeclaration{
            .kind = CouplingTransferKind::Identity,
        },
    };
}

CouplingPayloadExtractionResult extractOne(
    const CouplingContext& context,
    const CouplingFormContribution& contribution,
    CouplingPayloadExtractionRequest request = baseRequest())
{
    const CouplingPayloadDetangler detangler;
    const std::array<CouplingFormContribution, 1> contributions{contribution};
    const std::array<CouplingPayloadExtractionRequest, 1> requests{request};
    return detangler.extract(context,
                             std::span<const CouplingFormContribution>(
                                 contributions),
                             std::span<const CouplingPayloadExtractionRequest>(
                                 requests),
                             "fsi");
}

} // namespace

TEST(CouplingPayloadDetangler, ExtractsSeparableTractionCoefficient)
{
    const auto fixture = makeDetanglerContext();
    const auto p_f = forms::StateField(2, *fixture.space, "p_f");
    const auto w_s = forms::TestField(3, *fixture.space, "w_s");
    const auto n_f = forms::FormExpr::normal();

    CouplingFormContribution contribution;
    contribution.contribution_name =
        "fsi.fsi_interface.fluid_traction_balance";
    contribution.residual = (-forms::inner(p_f * n_f, w_s)).dI();

    const auto result = extractOne(fixture.context, contribution);
    ASSERT_EQ(result.exchanges.size(), 1u);
    EXPECT_TRUE(result.diagnostics.empty());
    ASSERT_TRUE(result.exchanges.front().extracted_payload.has_value());
    const auto& metadata = *result.exchanges.front().extracted_payload;
    EXPECT_TRUE(metadata.exact);
    EXPECT_EQ(metadata.reason, CouplingPayloadExtractionReason::Exact);
    EXPECT_EQ(metadata.payload_kind, CouplingPayloadKind::CoefficientExpression);
    EXPECT_TRUE(metadata.payload_expression.isValid());
    EXPECT_FALSE(metadata.payload_expression.hasTest());
    ASSERT_TRUE(result.exchanges.front().producer.has_value());
    EXPECT_EQ(result.exchanges.front().producer->kind,
              CouplingEndpointKind::RegionData);
}

TEST(CouplingPayloadDetangler, FallsBackForSymmetricWeakEnforcement)
{
    const auto fixture = makeDetanglerContext();
    const auto u_f = forms::StateField(1, *fixture.space, "u_f");
    const auto w_f = forms::TestField(1, *fixture.space, "w_f");
    const auto u_s = forms::StateField(3, *fixture.space, "u_s");
    const auto w_s = forms::TestField(3, *fixture.space, "w_s");

    CouplingFormContribution contribution;
    contribution.contribution_name =
        "fsi.fsi_interface.fluid_traction_balance";
    contribution.residual =
        (forms::inner(u_f - u_s, w_f) +
         forms::inner(u_s - u_f, w_s)).dI();

    auto request = baseRequest();
    request.fallback_policy =
        CouplingPayloadFallbackPolicy::WarnAndSplitSymmetric;
    const auto result = extractOne(fixture.context, contribution, request);
    ASSERT_EQ(result.exchanges.size(), 1u);
    ASSERT_EQ(result.diagnostics.size(), 1u);
    EXPECT_EQ(result.diagnostics.front().reason,
              CouplingPayloadExtractionReason::SymmetricWeakEnforcement);
    ASSERT_TRUE(result.exchanges.front().extracted_payload.has_value());
    EXPECT_FALSE(result.exchanges.front().extracted_payload->exact);
    EXPECT_EQ(result.exchanges.front().extracted_payload->payload_kind,
              CouplingPayloadKind::ResidualRecipe);
    EXPECT_TRUE(result.exchanges.front()
                    .extracted_payload->payload_expression.isValid());
}

TEST(CouplingPayloadDetangler, FallsBackForStabilizedTraceTerms)
{
    const auto fixture = makeDetanglerContext();
    const auto u_f = forms::StateField(1, *fixture.space, "u_f");
    const auto w_s = forms::TestField(3, *fixture.space, "w_s");

    CouplingFormContribution contribution;
    contribution.contribution_name =
        "fsi.fsi_interface.fluid_traction_balance";
    contribution.residual =
        forms::inner(forms::jump(u_f), forms::jump(w_s)).dI();

    const auto result = extractOne(fixture.context, contribution);
    ASSERT_EQ(result.exchanges.size(), 1u);
    ASSERT_EQ(result.diagnostics.size(), 1u);
    EXPECT_EQ(result.diagnostics.front().reason,
              CouplingPayloadExtractionReason::StabilizedTraceOperator);
    EXPECT_EQ(result.exchanges.front().extracted_payload->payload_kind,
              CouplingPayloadKind::ResidualRecipe);
}

TEST(CouplingPayloadDetangler, ClassifiesConsumerOnlyResidualAsConstraint)
{
    const auto fixture = makeDetanglerContext();
    const auto u_s = forms::StateField(3, *fixture.space, "u_s");
    const auto w_s = forms::TestField(3, *fixture.space, "w_s");

    CouplingFormContribution contribution;
    contribution.contribution_name =
        "fsi.fsi_interface.fluid_traction_balance";
    contribution.residual = forms::inner(u_s, w_s).dI();

    const auto result = extractOne(fixture.context, contribution);
    ASSERT_EQ(result.exchanges.size(), 1u);
    ASSERT_EQ(result.diagnostics.size(), 1u);
    EXPECT_EQ(result.diagnostics.front().reason,
              CouplingPayloadExtractionReason::ConstraintResidualNotLoad);
    EXPECT_EQ(result.exchanges.front().extracted_payload->payload_kind,
              CouplingPayloadKind::ConstraintResidual);
}
