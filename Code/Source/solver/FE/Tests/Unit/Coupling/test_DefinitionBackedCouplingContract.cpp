#include "Coupling/DefinitionBackedCouplingContract.h"

#include "Coupling/CouplingFormBuilder.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "CouplingTestHelpers.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
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

CouplingContext makeNWayContext()
{
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    CouplingContextBuilder builder;
    for (int i = 0; i < 3; ++i) {
        const auto name = std::string("branch_") + static_cast<char>('a' + i);
        const auto binding = test::participantBinding(
            name,
            11u + static_cast<std::uint64_t>(i));
        builder.addParticipant(test::participantRef(binding));
        builder.addField(test::fieldRef(binding, "primary", 20 + i, space, 1));
        builder.addRegion(test::boundaryRegionRef(binding, "junction", 30 + i));
    }
    return builder.build();
}

CouplingContext makeMixedDimensionalContext()
{
    const auto body = test::participantBinding("body", 21u);
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant(test::participantRef(body));
    builder.addField(test::fieldRef(body, "primary", 41, space, 1));
    builder.addRegion(CouplingRegionRef{
        .participant_name = "body",
        .system_name = body.system_name,
        .system = body.system,
        .region_name = "volume",
        .kind = CouplingRegionKind::Domain,
    });
    builder.addRegion(test::boundaryRegionRef(body, "surface", 42));
    return builder.build();
}

CouplingRegionRelationRequirement nWayRelationRequirement()
{
    return CouplingRegionRelationRequirement{
        .relation_name = "junction_balance",
        .relation_kind = CouplingRegionRelationKind::NWayInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "branch_a",
                .region_name = "junction",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "branch_b",
                .region_name = "junction",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "branch_c",
                .region_name = "junction",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                .enforcement_strategies = {"conservation"},
            },
        },
        .selected_lowering = CouplingRelationLoweringRequest{
            .mode = CouplingMode::Monolithic,
            .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            .enforcement_strategy = "conservation",
        },
        .required_region_kind = CouplingRegionKind::Boundary,
        .require_distinct_participants = true,
    };
}

CouplingRegionRelationRequirement mixedDimensionalRelationRequirement()
{
    return CouplingRegionRelationRequirement{
        .relation_name = "volume_surface_balance",
        .relation_kind = CouplingRegionRelationKind::VolumeBoundaryRelation,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "body",
                .region_name = "volume",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "body",
                .region_name = "surface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                .enforcement_strategies = {"balance"},
            },
        },
        .selected_lowering = CouplingRelationLoweringRequest{
            .mode = CouplingMode::Monolithic,
            .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            .enforcement_strategy = "balance",
        },
    };
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

class NWayDefinitionContract final : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "n_way_definition_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        const auto relation_requirement = nWayRelationRequirement();
        builder.participant("branch_a")
            .participant("branch_b")
            .participant("branch_c")
            .fieldRequirement(scalarFieldRequirement("branch_a", "primary"))
            .fieldRequirement(scalarFieldRequirement("branch_b", "primary"))
            .fieldRequirement(scalarFieldRequirement("branch_c", "primary"))
            .regionRelation(relation_requirement)
            .monolithic([relation_requirement](const CouplingContext&,
                                                const CouplingFormBuilder& forms) {
                const auto relation = forms.regionRelation(relation_requirement);
                std::vector<forms::FormExpr> branch_terms;
                for (const auto& participant :
                     {"branch_a", "branch_b", "branch_c"}) {
                    const auto endpoint = relation.endpoint(participant, "junction");
                    branch_terms.push_back(endpoint.integral(
                        endpoint.state("primary",
                                       std::string("u_") + participant) *
                        endpoint.test("primary",
                                      std::string("w_") + participant)));
                }

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "n_way_definition_fixture.junction_balance";
                contribution.origin = "NWayDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {
                    CouplingFieldUse{
                        .participant_name = "branch_a",
                        .field_name = "primary",
                    },
                    CouplingFieldUse{
                        .participant_name = "branch_b",
                        .field_name = "primary",
                    },
                    CouplingFieldUse{
                        .participant_name = "branch_c",
                        .field_name = "primary",
                    },
                };
                contribution.residual = relation.sum(branch_terms);
                return std::vector<CouplingFormContribution>{contribution};
            });
    }
};

class MixedDimensionalDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "mixed_dimensional_definition_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        const auto relation_requirement = mixedDimensionalRelationRequirement();
        builder.participant("body")
            .fieldRequirement(scalarFieldRequirement("body", "primary"))
            .region(CouplingRegionUse{
                .participant_name = "body",
                .region_name = "volume",
                .required_region_kind = CouplingRegionKind::Domain,
            })
            .region(CouplingRegionUse{
                .participant_name = "body",
                .region_name = "surface",
                .required_region_kind = CouplingRegionKind::Boundary,
            })
            .regionRelation(relation_requirement)
            .monolithic([relation_requirement](const CouplingContext&,
                                                const CouplingFormBuilder& forms) {
                const auto relation = forms.regionRelation(relation_requirement);
                const auto volume = relation.endpoint("body", "volume");
                const auto surface = relation.endpoint("body", "surface");

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "mixed_dimensional_definition_fixture.volume_surface_balance";
                contribution.origin = "MixedDimensionalDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "body",
                    .field_name = "primary",
                }};
                contribution.residual =
                    volume.integral(volume.state("primary", "u") *
                                    volume.test("primary", "w")) +
                    surface.integral(surface.state("primary", "u_b") *
                                     surface.test("primary", "w_b"));
                return std::vector<CouplingFormContribution>{contribution};
            });
    }
};

class PartitionedStrategyDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "partitioned_strategy_definition_fixture";
    }

protected:
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
            .regionRelation(CouplingRegionRelationRequirement{
                .relation_name = "fixed_point_interface",
                .relation_kind = CouplingRegionRelationKind::SidePairedInterface,
                .endpoints = {
                    CouplingRegionEndpointDeclaration{
                        .participant_name = "left",
                        .region_name = "interface",
                        .shared_region_name = "interface",
                    },
                    CouplingRegionEndpointDeclaration{
                        .participant_name = "right",
                        .region_name = "interface",
                        .shared_region_name = "interface",
                    },
                },
                .lowering_capabilities = {
                    CouplingRelationLoweringCapability{
                        .lowering_kind =
                            CouplingRelationLoweringKind::PartitionedExchange,
                        .partitioned_solve_strategies = {
                            CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
                        },
                    },
                },
                .selected_lowering = CouplingRelationLoweringRequest{
                    .mode = CouplingMode::Partitioned,
                    .lowering_kind =
                        CouplingRelationLoweringKind::PartitionedExchange,
                    .partitioned_solve_strategy =
                        CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
                },
                .required_region_kind = CouplingRegionKind::InterfaceFace,
                .require_opposite_sides_for_side_pair = true,
            });

        builder
            .exchange("fixed_point_primary",
                      CouplingFieldUse{
                          .participant_name = "left",
                          .field_name = "primary",
                      },
                      CouplingFieldUse{
                          .participant_name = "right",
                          .field_name = "primary",
                      })
            .sharedInterface("interface")
            .transfer(test::identityTransfer())
            .strategy(CouplingPartitionedStrategyDeclaration{
                .solve_strategy =
                    CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
                .relaxation_strategy =
                    CouplingPartitionedRelaxationStrategy::Constant,
                .convergence_norm =
                    CouplingPartitionedConvergenceNorm::ExchangeIncrement,
                .relaxation_factor = 0.4,
                .max_iterations = 6,
            });
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

TEST(DefinitionBackedCouplingContract, SupportsNWayFormsFixture)
{
    const auto context = makeNWayContext();
    const CouplingFormBuilder forms(context);
    const NWayDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    EXPECT_EQ(declaration.region_relation_requirements.front().relation_kind,
              CouplingRegionRelationKind::NWayInterface);
    ASSERT_TRUE(
        declaration.region_relation_requirements.front().selected_lowering.has_value());
    EXPECT_EQ(declaration.region_relation_requirements.front()
                  .selected_lowering->lowering_kind,
              CouplingRelationLoweringKind::MonolithicForms);

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "n_way_definition_fixture.junction_balance");
    EXPECT_TRUE(contributions.front().residual.isValid());
    ASSERT_EQ(contributions.front().field_uses.size(), 3u);
}

TEST(DefinitionBackedCouplingContract, SupportsMixedDimensionalFormsFixture)
{
    const auto context = makeMixedDimensionalContext();
    const CouplingFormBuilder forms(context);
    const MixedDimensionalDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    EXPECT_EQ(declaration.region_relation_requirements.front().relation_kind,
              CouplingRegionRelationKind::VolumeBoundaryRelation);

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "mixed_dimensional_definition_fixture.volume_surface_balance");
    EXPECT_TRUE(contributions.front().residual.isValid());
}

TEST(DefinitionBackedCouplingContract, SupportsPartitionedStrategyFixture)
{
    const auto context = makeDefinitionContext();
    const PartitionedStrategyDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->mode, CouplingMode::Partitioned);
    ASSERT_TRUE(relation.selected_lowering->partitioned_solve_strategy.has_value());
    EXPECT_EQ(*relation.selected_lowering->partitioned_solve_strategy,
              CouplingPartitionedSolveStrategy::StaggeredFixedPoint);
    ASSERT_EQ(declaration.partitioned_exchange_declarations.size(), 1u);
    EXPECT_EQ(declaration.partitioned_exchange_declarations.front()
                  .strategy.solve_strategy,
              CouplingPartitionedSolveStrategy::StaggeredFixedPoint);
    EXPECT_EQ(declaration.partitioned_exchange_declarations.front()
                  .strategy.max_iterations,
              6);

    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_EQ(plan.exchanges.size(), 1u);
    EXPECT_EQ(plan.exchanges.front().strategy.solve_strategy,
              CouplingPartitionedSolveStrategy::StaggeredFixedPoint);
}
