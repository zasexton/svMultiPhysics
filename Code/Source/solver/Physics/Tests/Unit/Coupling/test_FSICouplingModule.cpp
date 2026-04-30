#include "Physics/Coupling/FSICouplingModule.h"

#include <gtest/gtest.h>

#include "Coupling/CouplingDiagnostics.h"
#include "Coupling/CouplingFormBuilder.h"
#include "Coupling/CouplingContext.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "Core/FEException.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svmp;
using namespace svmp::Physics::coupling;
namespace fec = svmp::FE::coupling;
namespace forms = svmp::FE::forms;

namespace {

std::shared_ptr<const FE::spaces::FunctionSpace> multiplierSpace()
{
    return FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Triangle3,
        1,
        3);
}

bool hasField(const fec::CouplingContractDeclaration& declaration,
              const std::string& participant,
              const std::string& field)
{
    return std::any_of(declaration.fields.begin(),
                       declaration.fields.end(),
                       [&](const fec::CouplingFieldUse& use) {
                           return use.participant_name == participant &&
                                  use.field_name == field;
                       });
}

bool hasFieldUse(const std::vector<fec::CouplingFieldUse>& fields,
                 const std::string& participant,
                 const std::string& field)
{
    return std::any_of(fields.begin(),
                       fields.end(),
                       [&](const fec::CouplingFieldUse& use) {
                           return use.participant_name == participant &&
                                  use.field_name == field;
                       });
}

bool containsFormExprType(const forms::FormExprNode& node,
                          forms::FormExprType type)
{
    if (node.type() == type) {
        return true;
    }
    const auto children = node.children();
    return std::any_of(children.begin(),
                       children.end(),
                       [&](const forms::FormExprNode* child) {
                           return child != nullptr &&
                                  containsFormExprType(*child, type);
                       });
}

const fec::CouplingFormContribution* findContribution(
    const std::vector<fec::CouplingFormContribution>& contributions,
    const std::string& name)
{
    const auto it = std::find_if(
        contributions.begin(),
        contributions.end(),
        [&](const fec::CouplingFormContribution& contribution) {
            return contribution.contribution_name == name;
        });
    return it == contributions.end() ? nullptr : &*it;
}

struct FSIFieldComponents {
    int fluid_velocity{3};
    int fluid_pressure{1};
    int solid_displacement{3};
    int solid_velocity{3};
    int mesh_displacement{3};
};

std::shared_ptr<const FE::spaces::FunctionSpace> space(int components)
{
    return FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Triangle3,
        1,
        components);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
svmp::search::LogicalInterfaceRegionId fsiLogicalInterfaceRegion(
    const std::string& participant_name)
{
    const bool fluid = participant_name == "fluid";
    return svmp::search::LogicalInterfaceRegionId{
        .persistent_id = fluid ? "fluid_interface" : "solid_interface",
        .name = fluid ? "fluid_surface" : "solid_surface",
    };
}

svmp::search::InterfaceRevisionSnapshot fsiInterfaceRevisionSnapshot(
    const std::string& participant_name)
{
    const bool fluid = participant_name == "fluid";
    svmp::search::InterfaceRevisionSnapshot snapshot;
    snapshot.configuration = svmp::Configuration::Reference;
    snapshot.geometry_revision = fluid ? 101 : 201;
    snapshot.reference_geometry_revision = fluid ? 102 : 202;
    snapshot.current_geometry_revision = fluid ? 103 : 203;
    snapshot.topology_revision = fluid ? 104 : 204;
    snapshot.ownership_revision = fluid ? 105 : 205;
    snapshot.numbering_revision = fluid ? 106 : 206;
    snapshot.field_layout_revision = fluid ? 107 : 207;
    snapshot.label_revision = fluid ? 108 : 208;
    snapshot.active_configuration_epoch = fluid ? 109 : 209;
    return snapshot;
}
#endif

fec::CouplingFieldRef field(std::string participant_name,
                            std::string system_name,
                            const FE::systems::FESystem* system,
                            std::string field_name,
                            FE::FieldId field_id,
                            int components)
{
    return fec::CouplingFieldRef{
        .participant_name = std::move(participant_name),
        .system_name = std::move(system_name),
        .system = system,
        .field_name = std::move(field_name),
        .field_id = field_id,
        .space = space(components),
        .components = components,
    };
}

fec::CouplingRegionRef interfaceRegion(std::string participant_name,
                                       std::string system_name,
                                       const FE::systems::FESystem* system,
                                       std::string region_name,
                                       int marker,
                                       fec::CouplingInterfaceSide side)
{
    fec::CouplingRegionRef region{
        .participant_name = std::move(participant_name),
        .system_name = std::move(system_name),
        .system = system,
        .region_name = std::move(region_name),
        .kind = fec::CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = side,
    };
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    region.logical_region =
        fsiLogicalInterfaceRegion(region.participant_name);
    region.revision_snapshot =
        fsiInterfaceRevisionSnapshot(region.participant_name);
#endif
    return region;
}

struct FSIContextFixture {
    std::shared_ptr<svmp::Physics::test::SingleTetraMeshAccess> mesh{
        std::make_shared<svmp::Physics::test::SingleTetraMeshAccess>()};
    FE::systems::FESystem fluid_system{mesh};
    FE::systems::FESystem solid_system{mesh};
    FE::systems::FESystem mesh_system{mesh};
    fec::CouplingContext context;

    explicit FSIContextFixture(const FSIFieldComponents& components = {},
                               bool include_mesh = false,
                               bool include_shared_region = true,
                               bool include_fluid_mapping = true,
                               bool include_solid_mapping = true,
                               bool include_mesh_displacement = true,
                               bool use_shared_system = true,
                               bool include_interface_topology = true)
    {
        auto* fluid_system_ref = &fluid_system;
        auto* solid_system_ref = use_shared_system ? &fluid_system : &solid_system;
        auto* mesh_system_ref = use_shared_system ? &fluid_system : &mesh_system;
        const std::string fluid_system_name =
            use_shared_system ? "fsi_system" : "fluid_system";
        const std::string solid_system_name =
            use_shared_system ? "fsi_system" : "solid_system";
        const std::string mesh_system_name =
            use_shared_system ? "fsi_system" : "mesh_system";
        const int fluid_marker = 10;
        const int solid_marker = use_shared_system ? 10 : 20;

        if (include_interface_topology) {
            fluid_system_ref->setInterfaceMesh(
                fluid_marker,
                std::make_shared<const svmp::InterfaceMesh>());
            if (!use_shared_system) {
                solid_system_ref->setInterfaceMesh(
                    solid_marker,
                    std::make_shared<const svmp::InterfaceMesh>());
            }
        }

        auto fluid_region = interfaceRegion("fluid",
                                            fluid_system_name,
                                            fluid_system_ref,
                                            "fluid_interface",
                                            fluid_marker,
                                            fec::CouplingInterfaceSide::Minus);
        auto solid_region = interfaceRegion("solid",
                                            solid_system_name,
                                            solid_system_ref,
                                            "solid_interface",
                                            solid_marker,
                                            fec::CouplingInterfaceSide::Plus);
        std::vector<fec::CouplingRegionRef> shared_regions;
        if (include_fluid_mapping) {
            shared_regions.push_back(fluid_region);
        }
        if (include_solid_mapping) {
            shared_regions.push_back(solid_region);
        }

        fec::CouplingContextBuilder builder;
        builder.addParticipant(fec::CouplingParticipantRef{
                   .participant_name = "fluid",
                   .system_name = fluid_system_name,
                   .system = fluid_system_ref,
               })
            .addParticipant(fec::CouplingParticipantRef{
                .participant_name = "solid",
                .system_name = solid_system_name,
                .system = solid_system_ref,
            })
            .addField(field("fluid",
                            fluid_system_name,
                            fluid_system_ref,
                            "velocity",
                            static_cast<FE::FieldId>(1),
                            components.fluid_velocity))
            .addField(field("fluid",
                            fluid_system_name,
                            fluid_system_ref,
                            "pressure",
                            static_cast<FE::FieldId>(2),
                            components.fluid_pressure))
            .addField(field("solid",
                            solid_system_name,
                            solid_system_ref,
                            "displacement",
                            static_cast<FE::FieldId>(3),
                            components.solid_displacement))
            .addField(field("solid",
                            solid_system_name,
                            solid_system_ref,
                            "velocity",
                            static_cast<FE::FieldId>(4),
                            components.solid_velocity))
            .addRegion(fluid_region)
            .addRegion(solid_region);

        if (include_mesh) {
            builder.addParticipant(fec::CouplingParticipantRef{
                       .participant_name = "mesh",
                       .system_name = mesh_system_name,
                       .system = mesh_system_ref,
                   });
            if (include_mesh_displacement) {
                builder.addField(field("mesh",
                                       mesh_system_name,
                                       mesh_system_ref,
                                       "displacement",
                                       static_cast<FE::FieldId>(5),
                                       components.mesh_displacement));
            }
        }

        if (include_shared_region) {
            builder.addSharedRegion(fec::SharedRegionRef{
                .name = "interface",
                .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
                .participant_regions = std::move(shared_regions),
            });
        }
        context = builder.build();
    }
};

FSICouplingOptions partitionedIdentityOptions()
{
    FSICouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;
    options.solid_to_fluid_transfer.kind = fec::CouplingTransferKind::Identity;
    options.fluid_to_solid_transfer.kind = fec::CouplingTransferKind::Identity;
    return options;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
fec::CouplingInterfaceMapProvenance interfaceMapProvenance(
    const std::string& source_participant,
    const std::string& target_participant)
{
    const auto system_name = [](const std::string& participant) {
        return participant + "_system";
    };
    const auto marker = [](const std::string& participant) {
        return participant == "fluid" ? 10 : 20;
    };

    return fec::CouplingInterfaceMapProvenance{
        .interface_map_name = "fsi_interface_map",
        .interface_entry_name = "interface",
        .interface_search_registry_name = "default_search",
        .source_system_name = system_name(source_participant),
        .target_system_name = system_name(target_participant),
        .source_interface_marker = marker(source_participant),
        .target_interface_marker = marker(target_participant),
        .source_logical_region =
            fsiLogicalInterfaceRegion(source_participant),
        .target_logical_region =
            fsiLogicalInterfaceRegion(target_participant),
        .source_revision_snapshot =
            fsiInterfaceRevisionSnapshot(source_participant),
        .target_revision_snapshot =
            fsiInterfaceRevisionSnapshot(target_participant),
        .source_search_revision_key = 3,
        .target_search_revision_key = 5,
        .map_revision_key = 7,
        .map_state = svmp::search::InterfaceMapState::Committed,
        .operator_state = FE::systems::InterfaceOperatorState::AcceptedTimeStep,
        .accepted_revision_key = 11,
        .trial_revision_key = 13,
        .time = 0.25,
        .time_level_epoch = 17,
    };
}
#endif

fec::CouplingTransferDeclaration interfaceTransfer(
    fec::CouplingTransferKind kind,
    const std::string& source_participant,
    const std::string& target_participant)
{
    fec::CouplingTransferDeclaration transfer;
    transfer.kind = kind;
    transfer.interface_declaration = fec::CouplingInterfaceTransferDeclaration{
        .frame_policy = fec::CouplingInterfaceFramePolicy::SourceToTargetVector,
    };
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    transfer.interface_map =
        interfaceMapProvenance(source_participant, target_participant);
#else
    static_cast<void>(source_participant);
    static_cast<void>(target_participant);
#endif
    return transfer;
}

void expectValidationFailureContaining(const FSICouplingModule& module,
                                       const fec::CouplingContext& context,
                                       const std::string& message)
{
    try {
        module.validate(context);
        FAIL() << "expected validation to fail";
    } catch (const FE::InvalidArgumentException& exception) {
        EXPECT_NE(std::string(exception.what()).find(message), std::string::npos)
            << exception.what();
    }
}

void expectFormBuildFailureContaining(const FSICouplingModule& module,
                                      const fec::CouplingContext& context,
                                      const fec::CouplingFormBuilder& form_builder,
                                      const std::string& message)
{
    try {
        static_cast<void>(module.buildMonolithicForms(context, form_builder));
        FAIL() << "expected form build to fail";
    } catch (const FE::InvalidArgumentException& exception) {
        EXPECT_NE(std::string(exception.what()).find(message), std::string::npos)
            << exception.what();
    }
}

} // namespace

TEST(FSICouplingModule, DeclaresMonolithicRequirements)
{
    FSICouplingOptions options;
    options.mesh_name = "mesh";

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    EXPECT_EQ(declaration.contract_type, "fsi");
    EXPECT_EQ(declaration.contract_name, "fsi");
    ASSERT_EQ(declaration.participants.size(), 3u);
    EXPECT_EQ(declaration.participants[0].participant_name, "fluid");
    EXPECT_EQ(declaration.participants[1].participant_name, "solid");
    EXPECT_EQ(declaration.participants[2].participant_name, "mesh");

    EXPECT_TRUE(hasField(declaration, "fluid", "velocity"));
    EXPECT_TRUE(hasField(declaration, "fluid", "pressure"));
    EXPECT_TRUE(hasField(declaration, "solid", "displacement"));
    EXPECT_TRUE(hasField(declaration, "solid", "velocity"));
    EXPECT_TRUE(hasField(declaration, "mesh", "displacement"));

    ASSERT_EQ(declaration.shared_regions.size(), 1u);
    EXPECT_EQ(declaration.shared_regions[0].shared_region_name, "interface");
    ASSERT_TRUE(declaration.shared_regions[0].required_region_kind.has_value());
    EXPECT_EQ(*declaration.shared_regions[0].required_region_kind,
              fec::CouplingRegionKind::InterfaceFace);

    EXPECT_TRUE(declaration.partitioned_exchange_declarations.empty());
}

TEST(FSICouplingModule, DeclaresMultiplierFieldWhenEnabled)
{
    FSICouplingOptions options;
    options.multiplier.enabled = true;
    options.multiplier.contract_field_namespace = "fsi_interface";
    options.multiplier.system_participant_name = "solid";
    options.multiplier.field_name = "interface_multiplier";
    options.multiplier.space = multiplierSpace();
    options.multiplier.components = 3;

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    ASSERT_EQ(declaration.additional_fields.size(), 1u);
    const auto& field = declaration.additional_fields.front();
    EXPECT_EQ(field.field_namespace, fec::CouplingAdditionalFieldNamespace::Contract);
    EXPECT_EQ(field.namespace_name, "fsi_interface");
    EXPECT_EQ(field.system_participant_name, "solid");
    EXPECT_EQ(field.field_name, "interface_multiplier");
    ASSERT_NE(field.space, nullptr);
    EXPECT_EQ(field.space->value_dimension(), 3);
    EXPECT_EQ(field.components, 3);
    EXPECT_EQ(field.scope, fec::CouplingAdditionalFieldScope::InterfaceFace);
    ASSERT_TRUE(field.shared_region_name.has_value());
    EXPECT_EQ(*field.shared_region_name, "interface");
}

TEST(FSICouplingModule, DeclaresTemporalDerivativeWhenSolidVelocityDerived)
{
    FSICouplingOptions options;
    options.use_solid_displacement_derivative = true;

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    EXPECT_FALSE(hasField(declaration, "solid", "velocity"));
    ASSERT_EQ(declaration.temporal_requirements.size(), 1u);
    const auto& requirement = declaration.temporal_requirements.front();
    EXPECT_EQ(requirement.quantity, fec::CouplingTemporalQuantity::FieldDerivative);
    ASSERT_TRUE(requirement.field.has_value());
    EXPECT_EQ(requirement.field->participant_name, "solid");
    EXPECT_EQ(requirement.field->field_name, "displacement");
    EXPECT_EQ(requirement.derivative_order, 1);
}

TEST(FSICouplingModule, DeclaresPartitionedExchanges)
{
    const auto options = partitionedIdentityOptions();

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    ASSERT_EQ(declaration.partitioned_exchange_declarations.size(), 2u);
    const auto& displacement = declaration.partitioned_exchange_declarations[0];
    EXPECT_EQ(displacement.producer_port.contract_instance_name, "fsi");
    EXPECT_EQ(displacement.producer_port.port_name, "solid_displacement");
    EXPECT_EQ(displacement.consumer_port.port_name, "fluid_displacement");
    EXPECT_EQ(displacement.value.rank, fec::CouplingValueRank::Vector);
    EXPECT_EQ(displacement.value.components, 3);
    ASSERT_TRUE(displacement.producer.has_value());
    EXPECT_EQ(displacement.producer->participant_name, "solid");
    EXPECT_EQ(displacement.producer->endpoint_name, "displacement");
    EXPECT_EQ(displacement.producer->temporal.slot,
              fec::CouplingTemporalSlot::Current);
    ASSERT_TRUE(displacement.consumer.has_value());
    EXPECT_EQ(displacement.consumer->participant_name, "fluid");
    EXPECT_EQ(displacement.consumer->endpoint_name, "velocity");
    EXPECT_EQ(displacement.consumer->temporal.slot,
              fec::CouplingTemporalSlot::Current);
    EXPECT_EQ(displacement.shared_region_name, "interface");
    EXPECT_EQ(displacement.transfer.kind, fec::CouplingTransferKind::Identity);

    const auto& load = declaration.partitioned_exchange_declarations[1];
    EXPECT_EQ(load.producer_port.port_name, "fluid_load");
    EXPECT_EQ(load.consumer_port.port_name, "solid_load");
    ASSERT_TRUE(load.producer.has_value());
    EXPECT_EQ(load.producer->participant_name, "fluid");
    EXPECT_EQ(load.producer->temporal.slot,
              fec::CouplingTemporalSlot::Current);
    ASSERT_TRUE(load.consumer.has_value());
    EXPECT_EQ(load.consumer->participant_name, "solid");
    EXPECT_EQ(load.consumer->temporal.slot,
              fec::CouplingTemporalSlot::Current);

    ASSERT_EQ(declaration.group_hints.size(), 1u);
    EXPECT_EQ(declaration.group_hints[0].name, "fsi_participants");
    EXPECT_EQ(declaration.group_hints[0].participant_names,
              (std::vector<std::string>{"fluid", "solid"}));

    FSIContextFixture fixture(FSIFieldComponents{},
                              false,
                              true,
                              true,
                              true,
                              true,
                              false);
    const auto resolved_exchanges =
        module.buildPartitionedExchangeDeclarations(fixture.context);
    ASSERT_EQ(resolved_exchanges.size(), 2u);
    ASSERT_TRUE(resolved_exchanges[0].producer_region.has_value());
    EXPECT_EQ(resolved_exchanges[0].producer_region->participant_name, "solid");
    EXPECT_EQ(resolved_exchanges[0].producer_region->region_name,
              "solid_interface");
    ASSERT_TRUE(resolved_exchanges[0].consumer_region.has_value());
    EXPECT_EQ(resolved_exchanges[0].consumer_region->participant_name, "fluid");
    EXPECT_EQ(resolved_exchanges[0].consumer_region->region_name,
              "fluid_interface");
    ASSERT_TRUE(resolved_exchanges[1].producer_region.has_value());
    EXPECT_EQ(resolved_exchanges[1].producer_region->participant_name, "fluid");
    ASSERT_TRUE(resolved_exchanges[1].consumer_region.has_value());
    EXPECT_EQ(resolved_exchanges[1].consumer_region->participant_name, "solid");
}

TEST(FSICouplingModule, ValidatesPartitionedIdentityTransfersWithResolvedRegions)
{
    const auto options = partitionedIdentityOptions();
    const FSICouplingModule module(options);
    FSIContextFixture fixture(FSIFieldComponents{},
                              false,
                              true,
                              true,
                              true,
                              true,
                              false);

    EXPECT_NO_THROW(module.validate(fixture.context));

    const auto exchanges =
        module.buildPartitionedExchangeDeclarations(fixture.context);
    const auto declaration = module.declare();
    const fec::PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const fec::CouplingExchangeDeclaration>(exchanges),
        std::span<const fec::CouplingGroupHint>(declaration.group_hints));
    ASSERT_TRUE(validation.ok()) << fec::formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const fec::CouplingExchangeDeclaration>(exchanges),
        std::span<const fec::CouplingGroupHint>(declaration.group_hints));

    ASSERT_EQ(plan.exchanges.size(), 2u);
    EXPECT_EQ(plan.exchanges[0].producer.field_id, static_cast<FE::FieldId>(3));
    EXPECT_EQ(plan.exchanges[0].consumer.field_id, static_cast<FE::FieldId>(1));
    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              fec::CouplingResolvedTemporalBackingKind::SystemStateCurrent);
    EXPECT_EQ(plan.exchanges[0].consumer.temporal.backing,
              fec::CouplingResolvedTemporalBackingKind::SystemStateCurrent);
    ASSERT_TRUE(plan.exchanges[0].producer_region.has_value());
    EXPECT_EQ(plan.exchanges[0].producer_region->region_name,
              "solid_interface");
    ASSERT_TRUE(plan.exchanges[0].consumer_region.has_value());
    EXPECT_EQ(plan.exchanges[0].consumer_region->region_name,
              "fluid_interface");
    ASSERT_EQ(plan.group_hints.size(), 1u);
    EXPECT_EQ(plan.group_hints[0].participant_names,
              (std::vector<std::string>{"fluid", "solid"}));
}

TEST(FSICouplingModule, ResolvesExplicitPartitionedFieldTemporalSlots)
{
    auto options = partitionedIdentityOptions();
    options.partitioned_temporal.solid_displacement_source =
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::History,
            .history_index = 2,
        };
    options.partitioned_temporal.fluid_displacement_target =
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::Predicted,
        };
    options.partitioned_temporal.fluid_load_source =
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::Predicted,
        };
    options.partitioned_temporal.solid_load_target =
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::History,
            .history_index = 1,
        };

    const FSICouplingModule module(options);
    FSIContextFixture fixture(FSIFieldComponents{},
                              false,
                              true,
                              true,
                              true,
                              true,
                              false);

    EXPECT_NO_THROW(module.validate(fixture.context));

    const auto exchanges =
        module.buildPartitionedExchangeDeclarations(fixture.context);
    ASSERT_EQ(exchanges.size(), 2u);
    ASSERT_TRUE(exchanges[0].producer.has_value());
    EXPECT_EQ(exchanges[0].producer->temporal.slot,
              fec::CouplingTemporalSlot::History);
    ASSERT_TRUE(exchanges[0].producer->temporal.history_index.has_value());
    EXPECT_EQ(*exchanges[0].producer->temporal.history_index, 2);
    ASSERT_TRUE(exchanges[0].consumer.has_value());
    EXPECT_EQ(exchanges[0].consumer->temporal.slot,
              fec::CouplingTemporalSlot::Predicted);
    ASSERT_TRUE(exchanges[1].producer.has_value());
    EXPECT_EQ(exchanges[1].producer->temporal.slot,
              fec::CouplingTemporalSlot::Predicted);
    ASSERT_TRUE(exchanges[1].consumer.has_value());
    EXPECT_EQ(exchanges[1].consumer->temporal.slot,
              fec::CouplingTemporalSlot::History);
    ASSERT_TRUE(exchanges[1].consumer->temporal.history_index.has_value());
    EXPECT_EQ(*exchanges[1].consumer->temporal.history_index, 1);

    const auto declaration = module.declare();
    const fec::PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        fixture.context,
        std::span<const fec::CouplingExchangeDeclaration>(exchanges),
        std::span<const fec::CouplingGroupHint>(declaration.group_hints));

    ASSERT_EQ(plan.exchanges.size(), 2u);
    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              fec::CouplingResolvedTemporalBackingKind::SystemStateHistory);
    ASSERT_TRUE(plan.exchanges[0].producer.temporal.storage_index.has_value());
    EXPECT_EQ(*plan.exchanges[0].producer.temporal.storage_index, 1);
    EXPECT_EQ(plan.exchanges[0].consumer.temporal.backing,
              fec::CouplingResolvedTemporalBackingKind::SystemStatePredicted);
    EXPECT_EQ(plan.exchanges[1].producer.temporal.backing,
              fec::CouplingResolvedTemporalBackingKind::SystemStatePredicted);
    EXPECT_EQ(plan.exchanges[1].consumer.temporal.backing,
              fec::CouplingResolvedTemporalBackingKind::SystemStateHistory);
    ASSERT_TRUE(plan.exchanges[1].consumer.temporal.storage_index.has_value());
    EXPECT_EQ(*plan.exchanges[1].consumer.temporal.storage_index, 0);
}

TEST(FSICouplingModule, RejectsUnsupportedPartitionedFieldTemporalSlots)
{
    const std::vector<fec::CouplingTemporalSlotDescriptor> unsupported{
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::Accepted,
        },
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::Stage,
            .stage_index = 0,
        },
        fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::External,
        },
    };

    for (const auto temporal : unsupported) {
        SCOPED_TRACE(fec::toString(temporal.slot));
        auto options = partitionedIdentityOptions();
        options.partitioned_temporal.solid_displacement_source = temporal;
        const FSICouplingModule module(options);
        FSIContextFixture fixture(FSIFieldComponents{},
                                  false,
                                  true,
                                  true,
                                  true,
                                  true,
                                  false);

        expectValidationFailureContaining(
            module,
            fixture.context,
            "field endpoint temporal slot " +
                std::string(fec::toString(temporal.slot)));
    }
}

TEST(FSICouplingModule, ValidatesPartitionedInterfaceTransferKinds)
{
    const std::vector<fec::CouplingTransferKind> transfer_kinds{
        fec::CouplingTransferKind::InterfacePointwiseInterpolation,
        fec::CouplingTransferKind::InterfaceConservativeProjection,
        fec::CouplingTransferKind::InterfaceMortar,
    };

    for (const auto kind : transfer_kinds) {
        SCOPED_TRACE(fec::toString(kind));
        FSICouplingOptions options;
        options.mode = fec::CouplingMode::Partitioned;
        options.solid_to_fluid_transfer =
            interfaceTransfer(kind, "solid", "fluid");
        options.fluid_to_solid_transfer =
            interfaceTransfer(kind, "fluid", "solid");
        const FSICouplingModule module(options);
        FSIContextFixture fixture(FSIFieldComponents{},
                                  false,
                                  true,
                                  true,
                                  true,
                                  true,
                                  false);

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        EXPECT_NO_THROW(module.validate(fixture.context));

        const auto exchanges =
            module.buildPartitionedExchangeDeclarations(fixture.context);
        const auto declaration = module.declare();
        const fec::PartitionedCouplingPlanGenerator generator;
        const auto plan = generator.generate(
            fixture.context,
            std::span<const fec::CouplingExchangeDeclaration>(exchanges),
            std::span<const fec::CouplingGroupHint>(declaration.group_hints));

        ASSERT_EQ(plan.exchanges.size(), 2u);
        EXPECT_EQ(plan.exchanges[0].transfer.kind, kind);
        ASSERT_TRUE(plan.exchanges[0].transfer.interface_options.has_value());
        EXPECT_EQ(plan.exchanges[0].transfer.interface_options->component_count,
                  3u);
        ASSERT_TRUE(plan.exchanges[0].transfer.interface_map.has_value());
        EXPECT_EQ(plan.exchanges[0].transfer.interface_map->source_system_name,
                  "solid_system");
        EXPECT_EQ(plan.exchanges[0].transfer.interface_map->target_system_name,
                  "fluid_system");
        EXPECT_EQ(plan.exchanges[1].transfer.kind, kind);
        ASSERT_TRUE(plan.exchanges[1].transfer.interface_options.has_value());
        ASSERT_TRUE(plan.exchanges[1].transfer.interface_map.has_value());
        EXPECT_EQ(plan.exchanges[1].transfer.interface_map->source_system_name,
                  "fluid_system");
        EXPECT_EQ(plan.exchanges[1].transfer.interface_map->target_system_name,
                  "solid_system");
#else
        expectValidationFailureContaining(
            module,
            fixture.context,
            "interface partitioned transfers require mesh interface support");
#endif
    }
}

TEST(FSICouplingModule, BuildsFormsAuthoredVelocityContinuity)
{
    FSIContextFixture fixture;
    const fec::CouplingFormBuilder form_builder(fixture.context);
    const FSICouplingModule module;

    EXPECT_TRUE(module.supportsMonolithicLowering());
    const auto contributions = module.buildMonolithicForms(fixture.context,
                                                           form_builder);
    ASSERT_EQ(contributions.size(), 2u);
    const auto* contribution =
        findContribution(contributions, "fsi_velocity_continuity");
    ASSERT_NE(contribution, nullptr);
    EXPECT_EQ(contribution->contribution_name, "fsi_velocity_continuity");
    EXPECT_EQ(contribution->origin, "FSICouplingModule");
    EXPECT_EQ(contribution->operator_name, "equations");
    EXPECT_TRUE(hasFieldUse(contribution->field_uses, "fluid", "velocity"));
    EXPECT_TRUE(
        hasFieldUse(contribution->extra_trial_field_uses, "solid", "velocity"));
    EXPECT_TRUE(contribution->terminal_provenance.empty());

    ASSERT_TRUE(contribution->residual.isValid());
    ASSERT_EQ(contribution->residual.node()->type(),
              forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(contribution->residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*contribution->residual.node()->interfaceMarker(), 10);
    EXPECT_TRUE(containsFormExprType(*contribution->residual.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*contribution->residual.node(),
                                     forms::FormExprType::RestrictPlus));
}

TEST(FSICouplingModule, BuildsFormsAuthoredPressureTractionBalance)
{
    FSIContextFixture fixture;
    const fec::CouplingFormBuilder form_builder(fixture.context);
    const FSICouplingModule module;

    const auto contributions = module.buildMonolithicForms(fixture.context,
                                                           form_builder);
    ASSERT_EQ(contributions.size(), 2u);
    const auto* contribution =
        findContribution(contributions, "fsi_pressure_traction_balance");
    ASSERT_NE(contribution, nullptr);
    EXPECT_EQ(contribution->origin, "FSICouplingModule");
    EXPECT_EQ(contribution->operator_name, "equations");
    EXPECT_TRUE(hasFieldUse(contribution->field_uses,
                            "solid",
                            "displacement"));
    EXPECT_TRUE(
        hasFieldUse(contribution->extra_trial_field_uses, "fluid", "pressure"));
    ASSERT_EQ(contribution->terminal_provenance.size(), 1u);
    EXPECT_EQ(contribution->terminal_provenance[0].kind,
              fec::CouplingFormTerminalProvenanceKind::GeometryTerminal);
    EXPECT_EQ(contribution->terminal_provenance[0].geometry_quantity,
              fec::CouplingGeometryTerminalQuantity::Normal);

    ASSERT_TRUE(contribution->residual.isValid());
    ASSERT_EQ(contribution->residual.node()->type(),
              forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(contribution->residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*contribution->residual.node()->interfaceMarker(), 10);
    EXPECT_TRUE(containsFormExprType(*contribution->residual.node(),
                                     forms::FormExprType::Normal));
    EXPECT_TRUE(containsFormExprType(*contribution->residual.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*contribution->residual.node(),
                                     forms::FormExprType::RestrictPlus));
}

TEST(FSICouplingModule, BuildsDisplacementDerivativeVelocityContinuity)
{
    FSIContextFixture fixture;
    const fec::CouplingFormBuilder form_builder(fixture.context);

    FSICouplingOptions options;
    options.use_solid_displacement_derivative = true;
    const FSICouplingModule module(options);

    const auto contributions = module.buildMonolithicForms(fixture.context,
                                                           form_builder);
    ASSERT_EQ(contributions.size(), 2u);
    const auto* contribution =
        findContribution(contributions, "fsi_velocity_continuity");
    ASSERT_NE(contribution, nullptr);
    EXPECT_TRUE(hasFieldUse(contribution->field_uses, "fluid", "velocity"));
    EXPECT_TRUE(hasFieldUse(contribution->extra_trial_field_uses,
                            "solid",
                            "displacement"));
    EXPECT_TRUE(containsFormExprType(*contribution->residual.node(),
                                     forms::FormExprType::TimeDerivative));
}

TEST(FSICouplingModule, RejectsMonolithicFormsWithIncompatibleSystems)
{
    const FSICouplingModule module;
    FSIContextFixture fixture(FSIFieldComponents{},
                              false,
                              true,
                              true,
                              true,
                              true,
                              false);
    expectValidationFailureContaining(
        module,
        fixture.context,
        "FSI monolithic fields must be registered in one compatible FESystem");

    const fec::CouplingFormBuilder form_builder(fixture.context);
    expectFormBuildFailureContaining(
        module,
        fixture.context,
        form_builder,
        "FSI monolithic fields must be registered in one compatible FESystem");
}

TEST(FSICouplingModule, RejectsMonolithicFormsWithoutInterfaceTopology)
{
    const FSICouplingModule module;
    FSIContextFixture fixture(FSIFieldComponents{},
                              false,
                              true,
                              true,
                              true,
                              true,
                              true,
                              false);
    expectValidationFailureContaining(
        module,
        fixture.context,
        "FSI monolithic interface marker is missing registered interface topology");

    const fec::CouplingFormBuilder form_builder(fixture.context);
    expectFormBuildFailureContaining(
        module,
        fixture.context,
        form_builder,
        "FSI monolithic interface marker is missing registered interface topology");
}

TEST(FSICouplingModule, RejectsInvalidOptionsDuringValidation)
{
    FSICouplingOptions options;
    options.interface_components = 0;

    const FSICouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}

TEST(FSICouplingModule, ValidatesFieldComponentCountsAgainstInterfaceDimension)
{
    FSICouplingOptions options;
    options.mesh_name = "mesh";
    const FSICouplingModule module(options);

    FSIContextFixture valid_fixture(FSIFieldComponents{}, true);
    EXPECT_NO_THROW(module.validate(valid_fixture.context));

    auto components = FSIFieldComponents{};
    components.fluid_velocity = 2;
    FSIContextFixture fluid_velocity_fixture(components, true);
    expectValidationFailureContaining(
        module,
        fluid_velocity_fixture.context,
        "FSI fluid velocity field component count must match the interface component count");

    components = FSIFieldComponents{};
    components.fluid_pressure = 2;
    FSIContextFixture fluid_pressure_fixture(components, true);
    expectValidationFailureContaining(
        module,
        fluid_pressure_fixture.context,
        "FSI fluid pressure field component count must be scalar");

    components = FSIFieldComponents{};
    components.solid_displacement = 2;
    FSIContextFixture solid_displacement_fixture(components, true);
    expectValidationFailureContaining(
        module,
        solid_displacement_fixture.context,
        "FSI solid displacement field component count must match the interface component count");

    components = FSIFieldComponents{};
    components.solid_velocity = 2;
    FSIContextFixture solid_velocity_fixture(components, true);
    expectValidationFailureContaining(
        module,
        solid_velocity_fixture.context,
        "FSI solid velocity field component count must match the interface component count");

    components = FSIFieldComponents{};
    components.mesh_displacement = 2;
    FSIContextFixture mesh_displacement_fixture(components, true);
    expectValidationFailureContaining(
        module,
        mesh_displacement_fixture.context,
        "FSI mesh displacement field component count must match the interface component count");
}

TEST(FSICouplingModule, ValidatesInterfaceSharedRegionMappings)
{
    FSICouplingOptions options;
    const FSICouplingModule module(options);

    FSIContextFixture missing_group(FSIFieldComponents{}, false, false);
    expectValidationFailureContaining(
        module,
        missing_group.context,
        "FSI interface shared region is missing");

    FSIContextFixture missing_fluid(FSIFieldComponents{}, false, true, false, true);
    expectValidationFailureContaining(
        module,
        missing_fluid.context,
        "FSI interface shared region must map the fluid participant");

    FSIContextFixture missing_solid(FSIFieldComponents{}, false, true, true, false);
    expectValidationFailureContaining(
        module,
        missing_solid.context,
        "FSI interface shared region must map the solid participant");
}

TEST(FSICouplingModule, RejectsALEWithoutMeshDisplacement)
{
    FSICouplingOptions options;
    options.mesh_name = "mesh";

    {
        auto missing_field_options = options;
        missing_field_options.mesh_displacement_field = std::nullopt;
        const FSICouplingModule module(missing_field_options);
        FSIContextFixture fixture(FSIFieldComponents{}, true);
        expectValidationFailureContaining(
            module,
            fixture.context,
            "FSI ALE mode requires a mesh displacement field");
    }

    {
        const FSICouplingModule module(options);
        FSIContextFixture fixture(FSIFieldComponents{}, false);
        expectValidationFailureContaining(
            module,
            fixture.context,
            "FSI ALE mode requires a registered mesh participant");
    }

    {
        const FSICouplingModule module(options);
        FSIContextFixture fixture(FSIFieldComponents{},
                                  true,
                                  true,
                                  true,
                                  true,
                                  false);
        expectValidationFailureContaining(
            module,
            fixture.context,
            "FSI ALE mode requires a registered mesh displacement field");
    }
}

TEST(FSICouplingModule, RejectsUnconfiguredPartitionedTransfers)
{
    FSICouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;

    const FSICouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}
