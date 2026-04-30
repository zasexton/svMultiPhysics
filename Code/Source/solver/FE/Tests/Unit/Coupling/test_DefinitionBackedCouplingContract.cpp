#include "Coupling/DefinitionBackedCouplingContract.h"

#include "Coupling/CouplingFormBuilder.h"
#include "CouplingTestHelpers.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

CouplingValueDescriptor scalarDescriptor()
{
    return CouplingValueDescriptor{
        .rank = CouplingValueRank::Scalar,
        .components = 1,
    };
}

CouplingFieldRequirement scalarFieldRequirement(std::string participant,
                                                std::string field)
{
    return CouplingFieldRequirement{
        .field = CouplingFieldUse{
            .participant_name = std::move(participant),
            .field_name = std::move(field),
        },
        .value = scalarDescriptor(),
    };
}

CouplingContext makeDefinitionContext()
{
    const auto left = test::participantBinding("left", 1u);
    const auto right = test::participantBinding("right", 2u);
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const auto left_interface = test::interfaceRegionRef(
        left,
        "interface",
        7,
        CouplingInterfaceSide::Minus,
        100u);
    const auto right_interface = test::interfaceRegionRef(
        right,
        "interface",
        8,
        CouplingInterfaceSide::Plus,
        200u);

    CouplingContextBuilder builder;
    builder.addParticipant(test::participantRef(left));
    builder.addParticipant(test::participantRef(right));
    builder.addField(test::fieldRef(left, "primary", 1, space, 1));
    builder.addField(test::fieldRef(right, "primary", 2, space, 1));
    builder.addRegion(left_interface);
    builder.addRegion(right_interface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .required_participant_names = {"left", "right"},
        .participant_regions = {left_interface, right_interface},
    });
    return builder.build();
}

class FixtureDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "definition_fixture";
    }

protected:
    [[nodiscard]] std::string contractInstanceName() const override
    {
        return "definition_fixture_instance";
    }

    void define(CouplingDefinitionBuilder& builder) const override
    {
        builder.participant("left")
            .participant("right")
            .fieldRequirement(scalarFieldRequirement("left", "primary"))
            .fieldRequirement(scalarFieldRequirement("right", "primary"))
            .sharedInterface(CouplingSharedInterfaceRequirement{
                .shared_region_name = "interface",
                .participant_names = {"left", "right"},
            })
            .monolithic([](const CouplingContext&,
                           const CouplingFormBuilder& forms) {
                const auto gamma = forms.sharedInterface("interface");
                const auto left = gamma.side("left");
                const auto right = gamma.side("right");

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "definition_fixture.primary_balance";
                contribution.origin = "FixtureDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "right",
                    .field_name = "primary",
                }};
                contribution.extra_trial_field_uses = {CouplingFieldUse{
                    .participant_name = "left",
                    .field_name = "primary",
                }};
                contribution.residual = gamma.integral(
                    left.state("primary", "u_left") *
                        right.test("primary", "w_right"),
                    "right");
                return std::vector<CouplingFormContribution>{contribution};
            });

        builder
            .exchange("primary_channel",
                      CouplingFieldUse{
                          .participant_name = "left",
                          .field_name = "primary",
                      },
                      CouplingFieldUse{
                          .participant_name = "right",
                          .field_name = "primary",
                      })
            .sharedInterface("interface")
            .transfer(test::identityTransfer());
    }
};

} // namespace

TEST(DefinitionBackedCouplingContract, CompilesDefinitionToBackendRecords)
{
    const FixtureDefinitionContract contract;

    const auto declaration = contract.declare();

    EXPECT_EQ(declaration.contract_type, "definition_fixture");
    EXPECT_EQ(declaration.contract_name, "definition_fixture_instance");
    EXPECT_EQ(declaration.dependency_declaration_mode,
              CouplingDependencyDeclarationMode::InferFromInstalledForms);
    ASSERT_EQ(declaration.participants.size(), 2u);
    EXPECT_EQ(declaration.participants[0].participant_name, "left");
    EXPECT_EQ(declaration.participants[1].participant_name, "right");
    ASSERT_EQ(declaration.field_requirements.size(), 2u);
    EXPECT_EQ(declaration.field_requirements[0].field.participant_name, "left");
    EXPECT_EQ(declaration.field_requirements[0].value.rank,
              CouplingValueRank::Scalar);
    ASSERT_EQ(declaration.shared_interface_requirements.size(), 1u);
    EXPECT_EQ(declaration.shared_interface_requirements[0].shared_region_name,
              "interface");
    ASSERT_EQ(declaration.partitioned_exchange_declarations.size(), 1u);

    const auto& exchange = declaration.partitioned_exchange_declarations.front();
    EXPECT_EQ(exchange.producer_port.contract_instance_name,
              "definition_fixture_instance");
    EXPECT_EQ(exchange.producer_port.port_name, "primary_channel.producer");
    ASSERT_TRUE(exchange.producer.has_value());
    ASSERT_TRUE(exchange.producer->participant_name.has_value());
    EXPECT_EQ(*exchange.producer->participant_name, "left");
    EXPECT_EQ(exchange.value.rank, CouplingValueRank::Scalar);
    EXPECT_EQ(exchange.value.components, 1);
}

TEST(DefinitionBackedCouplingContract, ReportsSupportedLoweringsFromDefinition)
{
    const FixtureDefinitionContract contract;

    EXPECT_TRUE(contract.supportsMonolithicLowering());
    EXPECT_TRUE(contract.supportsPartitionedLowering());
}

TEST(DefinitionBackedCouplingContract, BuildsMonolithicFormsFromDefinition)
{
    const auto context = makeDefinitionContext();
    const CouplingFormBuilder forms(context);
    const FixtureDefinitionContract contract;

    const auto contributions = contract.buildMonolithicForms(context, forms);

    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions[0].contribution_name,
              "definition_fixture.primary_balance");
    EXPECT_EQ(contributions[0].origin, "FixtureDefinitionContract");
    EXPECT_TRUE(contributions[0].residual.isValid());
    ASSERT_EQ(contributions[0].field_uses.size(), 1u);
    EXPECT_EQ(contributions[0].field_uses[0].participant_name, "right");
    ASSERT_EQ(contributions[0].extra_trial_field_uses.size(), 1u);
    EXPECT_EQ(contributions[0].extra_trial_field_uses[0].participant_name,
              "left");
}

TEST(DefinitionBackedCouplingContract, BuildsPartitionedExchangesFromDefinition)
{
    const auto context = makeDefinitionContext();
    const FixtureDefinitionContract contract;

    const auto exchanges =
        contract.buildPartitionedExchangeDeclarations(context);

    ASSERT_EQ(exchanges.size(), 1u);
    EXPECT_EQ(exchanges.front().producer_port.port_name,
              "primary_channel.producer");
    EXPECT_EQ(exchanges.front().consumer_port.port_name,
              "primary_channel.consumer");
    ASSERT_TRUE(exchanges.front().shared_region_name.has_value());
    EXPECT_EQ(*exchanges.front().shared_region_name, "interface");
    EXPECT_EQ(exchanges.front().transfer.kind, CouplingTransferKind::Identity);
}

TEST(DefinitionBackedCouplingContract, ValidatesThroughCouplingGraph)
{
    const auto context = makeDefinitionContext();
    const FixtureDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));
}
