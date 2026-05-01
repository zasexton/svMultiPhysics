#include "Coupling/DefinitionBackedCouplingContract.h"

#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Coupling/CouplingFormBuilder.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "CouplingTestHelpers.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

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

class SharedAuxiliaryOutputModel final : public systems::AuxiliaryStateModel {
public:
    [[nodiscard]] std::string modelName() const override
    {
        return "SharedAuxiliaryOutputModel";
    }

    [[nodiscard]] int dimension() const override { return 1; }

    [[nodiscard]] systems::AuxiliaryStructuralMetadata structuralMetadata()
        const override
    {
        systems::AuxiliaryStructuralMetadata metadata;
        metadata.variable_kinds = {systems::AuxiliaryVariableKind::Differential};
        return metadata;
    }

    void evaluateResidual(const systems::AuxiliaryLocalContext&,
                          systems::AuxiliaryResidualRequest& request) const override
    {
        if (!request.residual.empty()) {
            request.residual[0] = 0.0;
        }
    }

    [[nodiscard]] int outputCount() const override { return 1; }

    [[nodiscard]] std::vector<std::string> outputNames() const override
    {
        return {"interface_response"};
    }

    void evaluateOutputs(const systems::AuxiliaryLocalContext& ctx,
                         std::span<Real> output) const override
    {
        if (!output.empty()) {
            output[0] = ctx.x.empty() ? 0.0 : ctx.x[0];
        }
    }
};

CouplingValueDescriptor scalarDescriptor()
{
    return CouplingValueDescriptor{
        .rank = CouplingValueRank::Scalar,
        .components = 1,
    };
}

CouplingValueDescriptor vectorDescriptor(int components)
{
    return CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = components,
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

CouplingFieldRequirement vectorFieldRequirement(std::string participant,
                                                std::string field,
                                                int components)
{
    return CouplingFieldRequirement{
        .field = CouplingFieldUse{
            .participant_name = std::move(participant),
            .field_name = std::move(field),
        },
        .value = vectorDescriptor(components),
    };
}

CouplingRegionRef domainRegionRef(const test::ParticipantBinding& binding,
                                  std::string region_name)
{
    return CouplingRegionRef{
        .participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .system = binding.system,
        .region_name = std::move(region_name),
        .kind = CouplingRegionKind::Domain,
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
    builder.addRegion(domainRegionRef(body, "volume"));
    builder.addRegion(test::boundaryRegionRef(body, "surface", 42));
    return builder.build();
}

CouplingContext makeElectroThermalContext(bool include_heat_source = true)
{
    const auto electric = test::participantBinding("electric", 31u);
    const auto thermal = test::participantBinding("thermal", 31u);
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant(test::participantRef(electric));
    builder.addParticipant(test::participantRef(thermal));
    builder.addField(test::fieldRef(electric, "potential", 51, space, 1));
    builder.addField(test::fieldRef(thermal, "temperature", 52, space, 1));
    if (include_heat_source) {
        builder.addField(test::fieldRef(thermal, "heat_source", 53, space, 1));
    }
    builder.addRegion(domainRegionRef(electric, "domain"));
    builder.addRegion(domainRegionRef(thermal, "domain"));
    return builder.build();
}

CouplingContext makeContactContext(bool include_shared_region = true)
{
    const auto master = test::participantBinding("master", 41u);
    const auto slave = test::participantBinding("slave", 42u);
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 2);
    const auto master_contact = test::interfaceRegionRef(
        master,
        "contact",
        17,
        CouplingInterfaceSide::Minus,
        300u);
    const auto slave_contact = test::interfaceRegionRef(
        slave,
        "contact",
        18,
        CouplingInterfaceSide::Plus,
        400u);

    CouplingContextBuilder builder;
    builder.addParticipant(test::participantRef(master));
    builder.addParticipant(test::participantRef(slave));
    builder.addField(test::fieldRef(master, "displacement", 61, space, 2));
    builder.addField(test::fieldRef(slave, "displacement", 62, space, 2));
    builder.addRegion(master_contact);
    builder.addRegion(slave_contact);
    if (include_shared_region) {
        builder.addSharedRegion(SharedRegionRef{
            .name = "contact",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .required_participant_names = {"master", "slave"},
            .participant_regions = {master_contact, slave_contact},
        });
    }
    return builder.build();
}

struct AuxiliaryExchangeContextFixture {
    std::shared_ptr<spaces::H1Space> space{
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1)};
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh{
        std::make_shared<forms::test::SingleTetraMeshAccess>()};
    systems::FESystem left_system{mesh};
    systems::FESystem right_system{mesh};
    systems::FESystem auxiliary_system{mesh};
    FieldId left_field{INVALID_FIELD_ID};
    FieldId right_field{INVALID_FIELD_ID};
    CouplingContext context;

    explicit AuxiliaryExchangeContextFixture(
        systems::AuxiliarySolveMode solve_mode =
            systems::AuxiliarySolveMode::Partitioned)
    {
        left_field = left_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });
        right_field = right_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });

        systems::AuxiliaryInputSpec input;
        input.name = "left_primary";
        input.size = 1;
        input.producer = systems::AuxiliaryInputProducer::DirectUserData;
        auxiliary_system.auxiliaryInputRegistry().registerInput(input);

        auto model = std::make_shared<SharedAuxiliaryOutputModel>();
        auto instance = systems::use(model).name("shared_response").global();
        if (solve_mode == systems::AuxiliarySolveMode::Partitioned) {
            instance.partitioned("ForwardEuler");
        } else {
            instance.monolithic();
        }
        instance.initialize({0.0});
        auxiliary_system.deployAuxiliaryModel(std::move(instance));
        auxiliary_system.finalizeAuxiliaryLayout();

        const test::ParticipantBinding left{
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
        };
        const test::ParticipantBinding right{
            .participant_name = "right",
            .system_name = "right_system",
            .system = &right_system,
        };
        const test::ParticipantBinding shared_auxiliary{
            .participant_name = "shared_auxiliary",
            .system_name = "shared_auxiliary_system",
            .system = &auxiliary_system,
        };
        const auto left_interface = test::interfaceRegionRef(
            left,
            "interface",
            71,
            CouplingInterfaceSide::Minus,
            500u);
        const auto right_interface = test::interfaceRegionRef(
            right,
            "interface",
            72,
            CouplingInterfaceSide::Plus,
            600u);

        CouplingContextBuilder builder;
        builder.addParticipant(test::participantRef(left));
        builder.addParticipant(test::participantRef(right));
        builder.addParticipant(test::participantRef(shared_auxiliary));
        builder.addField(
            test::fieldRef(left, "primary", left_field, space, 1));
        builder.addField(
            test::fieldRef(right, "primary", right_field, space, 1));
        builder.addRegion(left_interface);
        builder.addRegion(right_interface);
        builder.addSharedRegion(SharedRegionRef{
            .name = "interface",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .required_participant_names = {"left", "right"},
            .participant_regions = {left_interface, right_interface},
        });
        context = builder.build();
    }
};

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

CouplingRegionRelationRequirement electroThermalRelationRequirement()
{
    return CouplingRegionRelationRequirement{
        .relation_name = "joule_heating",
        .relation_kind = CouplingRegionRelationKind::EmbeddedRelation,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "electric",
                .region_name = "domain",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "thermal",
                .region_name = "domain",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                .enforcement_strategies = {"source"},
            },
            CouplingRelationLoweringCapability{
                .lowering_kind =
                    CouplingRelationLoweringKind::PartitionedExchange,
                .fidelity = CouplingRelationLoweringFidelity::Lagged,
                .partitioned_solve_strategies = {
                    CouplingPartitionedSolveStrategy::ExplicitLagged,
                },
            },
        },
        .selected_lowering = CouplingRelationLoweringRequest{
            .mode = CouplingMode::Monolithic,
            .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            .enforcement_strategy = "source",
        },
        .required_region_kind = CouplingRegionKind::Domain,
    };
}

CouplingRegionRelationRequirement contactFrictionRelationRequirement()
{
    return CouplingRegionRelationRequirement{
        .relation_name = "contact_friction_interface",
        .relation_kind = CouplingRegionRelationKind::SidePairedInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "master",
                .region_name = "contact",
                .shared_region_name = "contact",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "slave",
                .region_name = "contact",
                .shared_region_name = "contact",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicExpert,
                .enforcement_strategies = {"active_set", "friction"},
            },
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::PartitionedExpert,
                .fidelity = CouplingRelationLoweringFidelity::Lagged,
                .partitioned_solve_strategies = {
                    CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
                },
            },
        },
        .selected_lowering = CouplingRelationLoweringRequest{
            .mode = CouplingMode::Monolithic,
            .lowering_kind = CouplingRelationLoweringKind::MonolithicExpert,
            .expert_fallback_enabled = true,
            .enforcement_strategy = "active_set",
        },
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .require_opposite_sides_for_side_pair = true,
    };
}

CouplingRegionRelationRequirement sidePairedInterfaceRelationRequirement(
    std::string relation_name,
    std::string enforcement_strategy,
    CouplingRelationLoweringKind lowering_kind,
    CouplingMode mode)
{
    return CouplingRegionRelationRequirement{
        .relation_name = std::move(relation_name),
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
                .lowering_kind = lowering_kind,
                .enforcement_strategies = {enforcement_strategy},
            },
        },
        .selected_lowering = CouplingRelationLoweringRequest{
            .mode = mode,
            .lowering_kind = lowering_kind,
            .expert_fallback_enabled =
                isExpertRelationLoweringKind(lowering_kind),
            .enforcement_strategy = enforcement_strategy,
        },
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .require_opposite_sides_for_side_pair = true,
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

class OptionValidatingDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    explicit OptionValidatingDefinitionContract(bool options_valid)
        : options_valid_(options_valid)
    {
    }

    [[nodiscard]] std::string name() const override
    {
        return "option_validating_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        builder.participant("left")
            .fieldRequirement(scalarFieldRequirement("left", "primary"));
    }

    void validateDefinitionOptions(
        const CouplingContext&,
        CouplingValidationResult& result) const override
    {
        if (!options_valid_) {
            result.addError("definition option validation rejected configuration");
        }
    }

private:
    bool options_valid_{true};
};

class InterfacePenaltyGeometryDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "interface_penalty_geometry_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        const auto relation_requirement = sidePairedInterfaceRelationRequirement(
            "penalty_interface",
            "penalty",
            CouplingRelationLoweringKind::MonolithicForms,
            CouplingMode::Monolithic);
        builder.participant("left")
            .participant("right")
            .fieldRequirement(scalarFieldRequirement("left", "primary"))
            .fieldRequirement(scalarFieldRequirement("right", "primary"))
            .sharedInterface(CouplingSharedInterfaceRequirement{
                .shared_region_name = "interface",
                .participant_names = {"left", "right"},
            })
            .regionRelation(relation_requirement)
            .monolithic([relation_requirement](const CouplingContext&,
                                                const CouplingFormBuilder& forms) {
                static_cast<void>(relation_requirement);
                const auto gamma = forms.sharedInterface("interface");
                const auto left = gamma.side("left");
                const auto right = gamma.side("right");
                const auto jump =
                    left.state("primary", "u_left") -
                    right.state("primary", "u_right");
                const auto test = left.test("primary", "w_left");
                const auto current_normal = left.geometryTerminal(
                    CouplingGeometryTerminalQuantity::CurrentNormal);

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "interface_penalty_geometry_fixture.penalty_interface";
                contribution.origin = "InterfacePenaltyGeometryDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "left",
                    .field_name = "primary",
                }};
                contribution.extra_trial_field_uses = {CouplingFieldUse{
                    .participant_name = "right",
                    .field_name = "primary",
                }};
                contribution.residual = gamma.integral(
                    jump * test + forms::dot(current_normal, current_normal) * test,
                    "left");
                contribution =
                    forms.attachTerminalProvenance(std::move(contribution));
                return std::vector<CouplingFormContribution>{contribution};
            });
    }
};

class SharedRegionOnlyEndpointDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "shared_region_endpoint_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        CouplingRegionRelationRequirement relation_requirement{
            .relation_name = "shared_temperature_balance",
            .relation_kind = CouplingRegionRelationKind::SidePairedInterface,
            .endpoints = {
                CouplingRegionEndpointDeclaration{
                    .participant_name = "left",
                    .shared_region_name = "interface",
                },
                CouplingRegionEndpointDeclaration{
                    .participant_name = "right",
                    .shared_region_name = "interface",
                },
            },
            .lowering_capabilities = {
                CouplingRelationLoweringCapability{
                    .lowering_kind =
                        CouplingRelationLoweringKind::MonolithicForms,
                    .enforcement_strategies = {"penalty"},
                },
            },
            .selected_lowering = CouplingRelationLoweringRequest{
                .mode = CouplingMode::Monolithic,
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                .enforcement_strategy = "penalty",
            },
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .require_opposite_sides_for_side_pair = true,
            .require_common_monolithic_system = false,
        };

        builder.participant("left")
            .participant("right")
            .fieldRequirement(scalarFieldRequirement("left", "primary"))
            .fieldRequirement(scalarFieldRequirement("right", "primary"))
            .sharedInterface(CouplingSharedInterfaceRequirement{
                .shared_region_name = "interface",
                .participant_names = {"left", "right"},
            })
            .regionRelation(relation_requirement)
            .monolithic([relation_requirement](const CouplingContext&,
                                                const CouplingFormBuilder& forms) {
                const auto relation = forms.regionRelation(relation_requirement);
                const auto left = relation.endpoint("left");
                const auto right = relation.endpoint("right");

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "shared_region_endpoint_fixture.shared_temperature_balance";
                contribution.origin =
                    "SharedRegionOnlyEndpointDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "left",
                    .field_name = "primary",
                }};
                contribution.extra_trial_field_uses = {CouplingFieldUse{
                    .participant_name = "right",
                    .field_name = "primary",
                }};
                contribution.residual = left.integral(
                    (left.state("primary", "u_left") -
                     right.state("primary", "u_right")) *
                    left.test("primary", "w_left"));
                return std::vector<CouplingFormContribution>{contribution};
            });
    }
};

class MultiplierExpertDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "multiplier_expert_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        const auto space =
            std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
        builder.participant("left")
            .participant("right")
            .fieldRequirement(scalarFieldRequirement("left", "primary"))
            .fieldRequirement(scalarFieldRequirement("right", "primary"))
            .sharedInterface(CouplingSharedInterfaceRequirement{
                .shared_region_name = "interface",
                .participant_names = {"left", "right"},
            })
            .additionalField(CouplingAdditionalFieldDeclaration{
                .field_namespace = CouplingAdditionalFieldNamespace::Contract,
                .namespace_name = name(),
                .system_participant_name = "left",
                .field_name = "lambda",
                .space = space,
                .components = 1,
                .scope = CouplingAdditionalFieldScope::InterfaceFace,
                .shared_region_name = "interface",
            })
            .regionRelation(sidePairedInterfaceRelationRequirement(
                "multiplier_interface",
                "multiplier",
                CouplingRelationLoweringKind::MonolithicExpert,
                CouplingMode::Monolithic));
    }
};

class BoundaryFunctionalDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "boundary_functional_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        const auto relation_requirement = mixedDimensionalRelationRequirement();
        builder.participant("body")
            .fieldRequirement(scalarFieldRequirement("body", "primary"))
            .nonFieldDependency(CouplingNonFieldDependencyRequirement{
                .kind =
                    CouplingNonFieldDependencyRequirementKind::BoundaryFunctional,
                .participant_name = "body",
                .name = "surface_flux",
                .region = CouplingRegionEndpointDeclaration{
                    .participant_name = "body",
                    .region_name = "surface",
                },
                .required_region_kind = CouplingRegionKind::Boundary,
                .require_analysis_variable_key = true,
            })
            .regionRelation(relation_requirement)
            .monolithic([relation_requirement](const CouplingContext&,
                                                const CouplingFormBuilder& forms) {
                const auto relation = forms.regionRelation(relation_requirement);
                const auto surface = relation.endpoint("body", "surface");

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "boundary_functional_fixture.surface_balance";
                contribution.origin = "BoundaryFunctionalDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "body",
                    .field_name = "primary",
                }};
                contribution.residual = surface.integral(
                    surface.state("primary", "u_b") *
                    surface.test("primary", "w_b"));
                return std::vector<CouplingFormContribution>{contribution};
            });
    }
};

class AuxiliaryExchangeDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    explicit AuxiliaryExchangeDefinitionContract(
        CouplingMode mode = CouplingMode::Partitioned)
        : mode_(mode)
    {
    }

    [[nodiscard]] std::string name() const override
    {
        return "auxiliary_exchange_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        CouplingRegionRelationRequirement relation_requirement{
            .relation_name = "shared_auxiliary_relation",
            .relation_kind = CouplingRegionRelationKind::AuxiliaryPDECoupling,
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
                    .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
                    .enforcement_strategies = {"auxiliary_load"},
                },
                CouplingRelationLoweringCapability{
                    .lowering_kind =
                        CouplingRelationLoweringKind::PartitionedExchange,
                    .fidelity = CouplingRelationLoweringFidelity::Lagged,
                    .partitioned_solve_strategies = {
                        CouplingPartitionedSolveStrategy::ExplicitLagged,
                    },
                },
            },
            .selected_lowering = CouplingRelationLoweringRequest{
                .mode = mode_,
                .lowering_kind =
                    mode_ == CouplingMode::Monolithic
                        ? CouplingRelationLoweringKind::MonolithicForms
                        : CouplingRelationLoweringKind::PartitionedExchange,
                .enforcement_strategy =
                    mode_ == CouplingMode::Monolithic ? "auxiliary_load" : "",
                .partitioned_solve_strategy =
                    mode_ == CouplingMode::Partitioned
                        ? std::optional<CouplingPartitionedSolveStrategy>(
                              CouplingPartitionedSolveStrategy::ExplicitLagged)
                        : std::nullopt,
            },
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .require_distinct_participants = true,
        };

        builder.participant("left")
            .participant("right")
            .fieldRequirement(scalarFieldRequirement("left", "primary"))
            .fieldRequirement(scalarFieldRequirement("right", "primary"))
            .sharedInterface(CouplingSharedInterfaceRequirement{
                .shared_region_name = "interface",
                .participant_names = {"left", "right"},
            })
            .nonFieldDependency(CouplingNonFieldDependencyRequirement{
                .kind =
                    CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
                .participant_name = "shared_auxiliary",
                .name = "left_primary",
                .require_analysis_variable_key = true,
            })
            .nonFieldDependency(CouplingNonFieldDependencyRequirement{
                .kind =
                    CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput,
                .participant_name = "shared_auxiliary",
                .name = "interface_response",
                .require_analysis_variable_key = true,
            })
            .regionRelation(relation_requirement);

        if (mode_ == CouplingMode::Monolithic) {
            builder.monolithic([](const CouplingContext&,
                                  const CouplingFormBuilder& forms) {
                const auto gamma = forms.sharedInterface("interface");
                const auto right = gamma.side("right");
                const auto response =
                    forms::AuxiliaryOutput("interface_response");

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "auxiliary_exchange_fixture.interface_response_load";
                contribution.origin = "AuxiliaryExchangeDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "right",
                    .field_name = "primary",
                }};
                contribution.residual = gamma.integral(
                    response * right.test("primary", "w_right"),
                    "right");
                return std::vector<CouplingFormContribution>{contribution};
            });
        }

        if (mode_ != CouplingMode::Partitioned) {
            return;
        }

        builder.exchange(
                   "left_state_to_auxiliary",
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::Field,
                       .participant_name = "left",
                       .endpoint_name = "primary",
                   },
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::AuxiliaryInput,
                       .participant_name = "shared_auxiliary",
                       .endpoint_name = "left_primary",
                   })
            .producerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "interface",
            })
            .value(scalarDescriptor())
            .transfer(test::identityTransfer());
        builder.exchange(
                   "auxiliary_response_to_right",
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::AuxiliaryOutput,
                       .participant_name = "shared_auxiliary",
                       .endpoint_name = "interface_response",
                   },
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::Field,
                       .participant_name = "right",
                       .endpoint_name = "primary",
                   })
            .consumerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "interface",
            })
            .value(scalarDescriptor())
            .transfer(test::identityTransfer());
    }

private:
    CouplingMode mode_{CouplingMode::Partitioned};
};

class AuxiliaryExpertDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "auxiliary_expert_fixture";
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
            .nonFieldDependency(CouplingNonFieldDependencyRequirement{
                .kind =
                    CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput,
                .participant_name = "shared_auxiliary",
                .name = "interface_response",
                .require_analysis_variable_key = true,
            })
            .regionRelation(CouplingRegionRelationRequirement{
                .relation_name = "shared_auxiliary_expert_relation",
                .relation_kind = CouplingRegionRelationKind::AuxiliaryPDECoupling,
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
                            CouplingRelationLoweringKind::MonolithicExpert,
                        .enforcement_strategies = {"auxiliary_load"},
                    },
                },
                .selected_lowering = CouplingRelationLoweringRequest{
                    .mode = CouplingMode::Monolithic,
                    .lowering_kind =
                        CouplingRelationLoweringKind::MonolithicExpert,
                    .expert_fallback_enabled = true,
                    .enforcement_strategy = "auxiliary_load",
                },
                .required_region_kind = CouplingRegionKind::InterfaceFace,
                .require_distinct_participants = true,
            });
    }
};

class ElectroThermalDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "electro_thermal_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        const auto relation_requirement = electroThermalRelationRequirement();
        builder.participant("electric")
            .participant("thermal")
            .fieldRequirement(scalarFieldRequirement("electric", "potential"))
            .fieldRequirement(scalarFieldRequirement("thermal", "temperature"))
            .fieldRequirement(scalarFieldRequirement("thermal", "heat_source"))
            .region(CouplingRegionUse{
                .participant_name = "electric",
                .region_name = "domain",
                .required_region_kind = CouplingRegionKind::Domain,
            })
            .region(CouplingRegionUse{
                .participant_name = "thermal",
                .region_name = "domain",
                .required_region_kind = CouplingRegionKind::Domain,
            })
            .nonFieldDependency(CouplingNonFieldDependencyRequirement{
                .kind = CouplingNonFieldDependencyRequirementKind::Coefficient,
                .participant_name = "electric",
                .name = "conductivity",
            })
            .regionRelation(relation_requirement)
            .monolithic([relation_requirement](const CouplingContext&,
                                                const CouplingFormBuilder& forms) {
                const auto relation = forms.regionRelation(relation_requirement);
                const auto electric = relation.endpoint("electric", "domain");
                const auto thermal = relation.endpoint("thermal", "domain");
                const auto phi = electric.state("potential", "phi");
                const auto theta = thermal.test("temperature", "theta");

                CouplingFormContribution contribution;
                contribution.contribution_name =
                    "electro_thermal_fixture.joule_heat_source";
                contribution.origin = "ElectroThermalDefinitionContract";
                contribution.operator_name = "equations";
                contribution.field_uses = {CouplingFieldUse{
                    .participant_name = "thermal",
                    .field_name = "temperature",
                }};
                contribution.extra_trial_field_uses = {CouplingFieldUse{
                    .participant_name = "electric",
                    .field_name = "potential",
                }};
                contribution.residual =
                    thermal.integral(-forms::inner(forms::grad(phi),
                                                   forms::grad(phi)) *
                                     theta);
                return std::vector<CouplingFormContribution>{contribution};
            });

        builder
            .exchange("temperature_to_electric",
                      CouplingFieldUse{
                          .participant_name = "thermal",
                          .field_name = "temperature",
                      },
                      CouplingFieldUse{
                          .participant_name = "electric",
                          .field_name = "potential",
                      })
            .producerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "thermal",
                .region_name = "domain",
            })
            .consumerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "electric",
                .region_name = "domain",
            })
            .transfer(test::identityTransfer());
        builder
            .exchange("joule_heat_to_thermal",
                      CouplingFieldUse{
                          .participant_name = "electric",
                          .field_name = "potential",
                      },
                      CouplingFieldUse{
                          .participant_name = "thermal",
                          .field_name = "heat_source",
                      })
            .producerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "electric",
                .region_name = "domain",
            })
            .consumerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "thermal",
                .region_name = "domain",
            })
            .transfer(test::identityTransfer());
    }
};

class ContactFrictionDefinitionContract final
    : public DefinitionBackedCouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "contact_friction_fixture";
    }

protected:
    void define(CouplingDefinitionBuilder& builder) const override
    {
        builder.participant("master")
            .participant("slave")
            .fieldRequirement(
                vectorFieldRequirement("master", "displacement", 2))
            .fieldRequirement(vectorFieldRequirement("slave", "displacement", 2))
            .sharedInterface(CouplingSharedInterfaceRequirement{
                .shared_region_name = "contact",
                .participant_names = {"master", "slave"},
            })
            .regionRelation(contactFrictionRelationRequirement());
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

template <typename Contract>
std::string validationFailureText(const Contract& contract,
                                  const CouplingContext& context)
{
    try {
        contract.validate(context);
    } catch (const InvalidArgumentException& e) {
        return e.what();
    }
    return {};
}

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

TEST(DefinitionBackedCouplingContract, ReportsFixtureDiagnosticsThroughCouplingGraph)
{
    const ElectroThermalDefinitionContract electro_thermal;
    const auto missing_field_text = validationFailureText(
        electro_thermal,
        makeElectroThermalContext(/*include_heat_source=*/false));
    EXPECT_NE(missing_field_text.find(
                  "required coupling field-shape field is missing from the context"),
              std::string::npos);
    EXPECT_NE(missing_field_text.find("heat_source"), std::string::npos);

    const ContactFrictionDefinitionContract contact;
    const auto missing_contact_text = validationFailureText(
        contact,
        makeContactContext(/*include_shared_region=*/false));
    EXPECT_NE(missing_contact_text.find(
                  "required shared-interface region is missing from the context"),
              std::string::npos);
    EXPECT_NE(missing_contact_text.find("contact"), std::string::npos);
}

TEST(DefinitionBackedCouplingContract, RunsDefinitionOptionValidationHook)
{
    const auto context = makeDefinitionContext();
    EXPECT_NO_THROW(OptionValidatingDefinitionContract(true).validate(context));

    try {
        OptionValidatingDefinitionContract(false).validate(context);
        FAIL() << "expected definition option validation failure";
    } catch (const InvalidArgumentException& e) {
        EXPECT_NE(std::string(e.what()).find(
                      "definition option validation rejected configuration"),
                  std::string::npos);
    }
}

TEST(DefinitionBackedCouplingContract, SupportsNWayFormsFixture)
{
    const auto context = makeNWayContext();
    const CouplingFormBuilder forms(context);
    const NWayDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    EXPECT_EQ(relation.relation_kind, CouplingRegionRelationKind::NWayInterface);
    ASSERT_EQ(relation.lowering_capabilities.size(), 1u);
    EXPECT_EQ(relation.lowering_capabilities.front().lowering_kind,
              CouplingRelationLoweringKind::MonolithicForms);
    ASSERT_EQ(relation.lowering_capabilities.front().enforcement_strategies.size(),
              1u);
    EXPECT_EQ(relation.lowering_capabilities.front().enforcement_strategies.front(),
              "conservation");
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->lowering_kind,
              CouplingRelationLoweringKind::MonolithicForms);

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "n_way_definition_fixture.junction_balance");
    EXPECT_TRUE(contributions.front().residual.isValid());
    ASSERT_EQ(contributions.front().field_uses.size(), 3u);
}

TEST(DefinitionBackedCouplingContract, SupportsPenaltyGeometryFixture)
{
    const auto context = makeDefinitionContext();
    const CouplingFormBuilder forms(context);
    const InterfacePenaltyGeometryDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->enforcement_strategy, "penalty");

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "interface_penalty_geometry_fixture.penalty_interface");
    EXPECT_TRUE(contributions.front().residual.isValid());
    ASSERT_EQ(contributions.front().terminal_provenance.size(), 1u);
    EXPECT_EQ(contributions.front().terminal_provenance.front().geometry_quantity,
              CouplingGeometryTerminalQuantity::CurrentNormal);
}

TEST(DefinitionBackedCouplingContract, SupportsSharedRegionOnlyRelationEndpoints)
{
    const auto context = makeDefinitionContext();
    const CouplingFormBuilder forms(context);
    const SharedRegionOnlyEndpointDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    ASSERT_EQ(relation.endpoints.size(), 2u);
    EXPECT_TRUE(relation.endpoints[0].region_name.empty());
    ASSERT_TRUE(relation.endpoints[0].shared_region_name.has_value());
    EXPECT_EQ(*relation.endpoints[0].shared_region_name, "interface");

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "shared_region_endpoint_fixture.shared_temperature_balance");
    ASSERT_TRUE(contributions.front().residual.isValid());
    ASSERT_EQ(contributions.front().residual.node()->type(),
              forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(
        contributions.front().residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*contributions.front().residual.node()->interfaceMarker(), 7);
}

TEST(DefinitionBackedCouplingContract, SupportsMultiplierExpertFixture)
{
    const auto context = makeDefinitionContext();
    const MultiplierExpertDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.additional_fields.size(), 1u);
    EXPECT_EQ(declaration.additional_fields.front().field_name, "lambda");
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    ASSERT_EQ(relation.lowering_capabilities.size(), 1u);
    EXPECT_EQ(relation.lowering_capabilities.front().lowering_kind,
              CouplingRelationLoweringKind::MonolithicExpert);
    ASSERT_EQ(relation.lowering_capabilities.front().enforcement_strategies.size(),
              1u);
    EXPECT_EQ(relation.lowering_capabilities.front().enforcement_strategies.front(),
              "multiplier");
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->lowering_kind,
              CouplingRelationLoweringKind::MonolithicExpert);
    EXPECT_TRUE(relation.selected_lowering->expert_fallback_enabled);
    EXPECT_FALSE(contract.supportsMonolithicLowering());
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

TEST(DefinitionBackedCouplingContract, SupportsBoundaryFunctionalFixture)
{
    const auto context = makeMixedDimensionalContext();
    const CouplingFormBuilder forms(context);
    const BoundaryFunctionalDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.non_field_dependencies.size(), 1u);
    EXPECT_EQ(declaration.non_field_dependencies.front().kind,
              CouplingNonFieldDependencyRequirementKind::BoundaryFunctional);
    ASSERT_TRUE(declaration.non_field_dependencies.front().region.has_value());
    EXPECT_EQ(declaration.non_field_dependencies.front().region->region_name,
              "surface");

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "boundary_functional_fixture.surface_balance");
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

TEST(DefinitionBackedCouplingContract, SupportsAuxiliaryExchangeFixture)
{
    const AuxiliaryExchangeDefinitionContract contract;
    const auto declaration = contract.declare();

    EXPECT_TRUE(validateContractDeclarationShape(declaration).ok())
        << formatDiagnostics(validateContractDeclarationShape(declaration));
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    EXPECT_EQ(relation.relation_kind,
              CouplingRegionRelationKind::AuxiliaryPDECoupling);
    ASSERT_EQ(relation.lowering_capabilities.size(), 2u);
    EXPECT_EQ(relation.lowering_capabilities[0].lowering_kind,
              CouplingRelationLoweringKind::MonolithicForms);
    EXPECT_EQ(relation.lowering_capabilities[1].lowering_kind,
              CouplingRelationLoweringKind::PartitionedExchange);
    ASSERT_EQ(relation.lowering_capabilities[1]
                  .partitioned_solve_strategies.size(),
              1u);
    EXPECT_EQ(relation.lowering_capabilities[1]
                  .partitioned_solve_strategies.front(),
              CouplingPartitionedSolveStrategy::ExplicitLagged);
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->mode, CouplingMode::Partitioned);
    ASSERT_EQ(declaration.non_field_dependencies.size(), 2u);
    EXPECT_EQ(declaration.non_field_dependencies[0].kind,
              CouplingNonFieldDependencyRequirementKind::AuxiliaryInput);
    EXPECT_EQ(declaration.non_field_dependencies[1].kind,
              CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput);

    const auto exchanges = contract.buildPartitionedExchangeDeclarations(
        CouplingContext{});
    ASSERT_EQ(exchanges.size(), 2u);
    ASSERT_TRUE(exchanges[0].consumer.has_value());
    EXPECT_EQ(exchanges[0].consumer->kind, CouplingEndpointKind::AuxiliaryInput);
    ASSERT_TRUE(exchanges[1].producer.has_value());
    EXPECT_EQ(exchanges[1].producer->kind,
              CouplingEndpointKind::AuxiliaryOutput);
}

TEST(DefinitionBackedCouplingContract, ValidatesAuxiliaryFixtureThroughPlanGenerator)
{
    AuxiliaryExchangeContextFixture fixture;
    const AuxiliaryExchangeDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(fixture.context));

    const auto exchanges =
        contract.buildPartitionedExchangeDeclarations(fixture.context);
    const std::span<const CouplingExchangeDeclaration> exchange_span(
        exchanges.data(),
        exchanges.size());
    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(fixture.context, exchange_span);
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(fixture.context, exchange_span);
    ASSERT_EQ(plan.exchanges.size(), 2u);
    EXPECT_EQ(plan.exchanges[0].consumer.resolved_kind,
              CouplingEndpointKind::AuxiliaryInput);
    EXPECT_EQ(plan.exchanges[0].consumer.registry_provider,
              "AuxiliaryInputRegistry");
    EXPECT_EQ(plan.exchanges[1].producer.resolved_kind,
              CouplingEndpointKind::AuxiliaryOutput);
    EXPECT_EQ(plan.exchanges[1].producer.registry_provider,
              "AuxiliaryOutputRegistry");
}

TEST(DefinitionBackedCouplingContract, SupportsAuxiliaryMonolithicFixture)
{
    AuxiliaryExchangeContextFixture fixture(
        systems::AuxiliarySolveMode::Monolithic);
    const CouplingFormBuilder forms(fixture.context);
    const AuxiliaryExchangeDefinitionContract contract(CouplingMode::Monolithic);

    EXPECT_NO_THROW(contract.validate(fixture.context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->mode, CouplingMode::Monolithic);
    EXPECT_EQ(relation.selected_lowering->lowering_kind,
              CouplingRelationLoweringKind::MonolithicForms);
    EXPECT_EQ(relation.selected_lowering->enforcement_strategy,
              "auxiliary_load");

    const auto contributions =
        contract.buildMonolithicForms(fixture.context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "auxiliary_exchange_fixture.interface_response_load");
    EXPECT_TRUE(contributions.front().residual.isValid());
    EXPECT_TRUE(contract.supportsMonolithicLowering());
    EXPECT_FALSE(contract.supportsPartitionedLowering());
}

TEST(DefinitionBackedCouplingContract, RejectsAuxiliaryStrategyMismatch)
{
    AuxiliaryExchangeContextFixture partitioned_fixture(
        systems::AuxiliarySolveMode::Partitioned);
    const AuxiliaryExchangeDefinitionContract monolithic_contract(
        CouplingMode::Monolithic);

    const auto mismatch_text = validationFailureText(
        monolithic_contract,
        partitioned_fixture.context);
    EXPECT_NE(mismatch_text.find(
                  "auxiliary dependency solve mode does not match selected coupling strategy"),
              std::string::npos);
    EXPECT_NE(mismatch_text.find("selected=monolithic"), std::string::npos);
    EXPECT_NE(mismatch_text.find("auxiliary=partitioned"), std::string::npos);

    const AuxiliaryExpertDefinitionContract expert_contract;
    EXPECT_NO_THROW(expert_contract.validate(partitioned_fixture.context));
}

TEST(DefinitionBackedCouplingContract, SupportsElectroThermalCapabilityFixture)
{
    const auto context = makeElectroThermalContext();
    const CouplingFormBuilder forms(context);
    const ElectroThermalDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    EXPECT_EQ(relation.relation_kind, CouplingRegionRelationKind::EmbeddedRelation);
    ASSERT_EQ(relation.lowering_capabilities.size(), 2u);
    EXPECT_EQ(relation.lowering_capabilities[0].lowering_kind,
              CouplingRelationLoweringKind::MonolithicForms);
    EXPECT_EQ(relation.lowering_capabilities[0].enforcement_strategies.front(),
              "source");
    EXPECT_EQ(relation.lowering_capabilities[1].lowering_kind,
              CouplingRelationLoweringKind::PartitionedExchange);
    EXPECT_EQ(relation.lowering_capabilities[1].fidelity,
              CouplingRelationLoweringFidelity::Lagged);
    ASSERT_EQ(declaration.non_field_dependencies.size(), 1u);
    EXPECT_EQ(declaration.non_field_dependencies.front().kind,
              CouplingNonFieldDependencyRequirementKind::Coefficient);

    const auto contributions = contract.buildMonolithicForms(context, forms);
    ASSERT_EQ(contributions.size(), 1u);
    EXPECT_EQ(contributions.front().contribution_name,
              "electro_thermal_fixture.joule_heat_source");
    EXPECT_TRUE(contributions.front().residual.isValid());

    const auto exchanges =
        contract.buildPartitionedExchangeDeclarations(context);
    ASSERT_EQ(exchanges.size(), 2u);
    ASSERT_TRUE(exchanges.front().producer_region.has_value());
    EXPECT_EQ(exchanges.front().producer_region->region_name, "domain");

    const std::span<const CouplingExchangeDeclaration> exchange_span(
        exchanges.data(),
        exchanges.size());
    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(context, exchange_span);
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);
    const auto plan = generator.generate(context, exchange_span);
    ASSERT_EQ(plan.exchanges.size(), 2u);
    ASSERT_TRUE(plan.exchanges.front().producer_region.has_value());
    EXPECT_EQ(plan.exchanges.front().producer_region->kind,
              CouplingRegionKind::Domain);
}

TEST(DefinitionBackedCouplingContract, SupportsContactFrictionCapabilityFixture)
{
    const auto context = makeContactContext();
    const ContactFrictionDefinitionContract contract;

    EXPECT_NO_THROW(contract.validate(context));

    const auto declaration = contract.declare();
    ASSERT_EQ(declaration.region_relation_requirements.size(), 1u);
    const auto& relation = declaration.region_relation_requirements.front();
    EXPECT_EQ(relation.relation_kind,
              CouplingRegionRelationKind::SidePairedInterface);
    ASSERT_EQ(relation.lowering_capabilities.size(), 2u);
    EXPECT_EQ(relation.lowering_capabilities[0].lowering_kind,
              CouplingRelationLoweringKind::MonolithicExpert);
    EXPECT_EQ(relation.lowering_capabilities[0].enforcement_strategies[0],
              "active_set");
    EXPECT_EQ(relation.lowering_capabilities[0].enforcement_strategies[1],
              "friction");
    EXPECT_EQ(relation.lowering_capabilities[1].lowering_kind,
              CouplingRelationLoweringKind::PartitionedExpert);
    EXPECT_EQ(relation.lowering_capabilities[1].fidelity,
              CouplingRelationLoweringFidelity::Lagged);
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_TRUE(relation.selected_lowering->expert_fallback_enabled);
    EXPECT_FALSE(contract.supportsMonolithicLowering());
}
