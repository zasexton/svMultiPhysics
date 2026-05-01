#include "Physics/Coupling/FSICouplingModule.h"
#include "Physics/Coupling/ThermalInterfaceCouplingModule.h"

#include <gtest/gtest.h>

#include "Coupling/CouplingFormBuilder.h"
#include "Core/FEException.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace svmp;
using namespace svmp::Physics::coupling;
namespace fec = svmp::FE::coupling;
namespace forms = svmp::FE::forms;

namespace {

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

const fec::CouplingRelationLoweringCapability* findCapability(
    const fec::CouplingRegionRelationRequirement& relation,
    fec::CouplingRelationLoweringKind kind)
{
    const auto it = std::find_if(
        relation.lowering_capabilities.begin(),
        relation.lowering_capabilities.end(),
        [kind](const fec::CouplingRelationLoweringCapability& capability) {
            return capability.lowering_kind == kind;
        });
    return it == relation.lowering_capabilities.end() ? nullptr : &*it;
}

std::shared_ptr<const FE::spaces::FunctionSpace> scalarSpace()
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1,
                             FE::ElementType::Triangle3,
                             1,
                             1);
}

fec::CouplingFieldRef thermalField(std::string participant_name,
                                   FE::systems::FESystem* system,
                                   FE::FieldId field_id)
{
    return fec::CouplingFieldRef{
        .participant_name = std::move(participant_name),
        .system_name = "thermal_system",
        .system = system,
        .field_name = "temperature",
        .field_id = field_id,
        .space = scalarSpace(),
        .components = 1,
    };
}

fec::CouplingRegionRef thermalInterfaceRegion(
    std::string participant_name,
    FE::systems::FESystem* system,
    fec::CouplingInterfaceSide side)
{
    return fec::CouplingRegionRef{
        .participant_name = std::move(participant_name),
        .system_name = "thermal_system",
        .system = system,
        .region_name = "wall",
        .kind = fec::CouplingRegionKind::InterfaceFace,
        .marker = 10,
        .side = side,
    };
}

struct ThermalContextFixture {
    std::shared_ptr<svmp::Physics::test::SingleTetraMeshAccess> mesh{
        std::make_shared<svmp::Physics::test::SingleTetraMeshAccess>()};
    FE::systems::FESystem system{mesh};
    fec::CouplingContext context;

    ThermalContextFixture()
    {
        system.setInterfaceMesh(10,
                                std::make_shared<const svmp::InterfaceMesh>());

        auto side_a_region = thermalInterfaceRegion(
            "side_a",
            &system,
            fec::CouplingInterfaceSide::Minus);
        auto side_b_region = thermalInterfaceRegion(
            "side_b",
            &system,
            fec::CouplingInterfaceSide::Plus);

        fec::CouplingContextBuilder builder;
        builder.addParticipant(fec::CouplingParticipantRef{
                   .participant_name = "side_a",
                   .system_name = "thermal_system",
                   .system = &system,
               })
            .addParticipant(fec::CouplingParticipantRef{
                .participant_name = "side_b",
                .system_name = "thermal_system",
                .system = &system,
            })
            .addField(thermalField("side_a", &system, 1))
            .addField(thermalField("side_b", &system, 2))
            .addRegion(side_a_region)
            .addRegion(side_b_region)
            .addSharedRegion(fec::SharedRegionRef{
                .name = "interface",
                .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
                .participant_regions = {side_a_region, side_b_region},
            });
        context = builder.build();
    }
};

} // namespace

TEST(ThermalInterfaceCouplingModule, DeclaresMonolithicTemperatureInterface)
{
    ThermalInterfaceCouplingOptions options;
    options.contract_name = "wall_temperature";
    options.side_a_name = "fluid_thermal";
    options.side_b_name = "solid_thermal";
    options.interface_name = "wall";

    const ThermalInterfaceCouplingModule module(options);
    const auto declaration = module.declare();

    EXPECT_EQ(declaration.contract_type, "thermal_interface");
    EXPECT_EQ(declaration.contract_name, "wall_temperature");
    ASSERT_EQ(declaration.participants.size(), 2u);
    EXPECT_EQ(declaration.participants[0].participant_name, "fluid_thermal");
    EXPECT_EQ(declaration.participants[1].participant_name, "solid_thermal");

    EXPECT_TRUE(hasField(declaration, "fluid_thermal", "temperature"));
    EXPECT_TRUE(hasField(declaration, "solid_thermal", "temperature"));
    EXPECT_FALSE(hasField(declaration, "fluid_thermal", "heat_flux"));
    EXPECT_FALSE(hasField(declaration, "solid_thermal", "heat_flux"));

    ASSERT_EQ(declaration.shared_regions.size(), 1u);
    EXPECT_EQ(declaration.shared_regions[0].shared_region_name, "wall");
    ASSERT_TRUE(declaration.shared_regions[0].required_region_kind.has_value());
    EXPECT_EQ(*declaration.shared_regions[0].required_region_kind,
              fec::CouplingRegionKind::InterfaceFace);
    EXPECT_TRUE(declaration.partitioned_exchange_declarations.empty());
}

TEST(ThermalInterfaceCouplingModule, DeclaresRelationLoweringCapabilities)
{
    const ThermalInterfaceCouplingModule monolithic_module;
    const auto monolithic = monolithic_module.declare();

    ASSERT_EQ(monolithic.region_relation_requirements.size(), 1u);
    const auto& relation = monolithic.region_relation_requirements.front();
    EXPECT_EQ(relation.relation_name, "thermal_interface");
    ASSERT_EQ(relation.endpoints.size(), 2u);
    EXPECT_TRUE(relation.endpoints[0].region_name.empty());
    ASSERT_TRUE(relation.endpoints[0].shared_region_name.has_value());
    EXPECT_EQ(*relation.endpoints[0].shared_region_name, "interface");
    ASSERT_TRUE(relation.selected_lowering.has_value());
    EXPECT_EQ(relation.selected_lowering->lowering_kind,
              fec::CouplingRelationLoweringKind::MonolithicForms);
    EXPECT_EQ(relation.selected_lowering->enforcement_strategy,
              "temperature_continuity_penalty");
    const auto* monolithic_capability = findCapability(
        relation,
        fec::CouplingRelationLoweringKind::MonolithicForms);
    ASSERT_NE(monolithic_capability, nullptr);
    EXPECT_EQ(monolithic_capability->fidelity,
              fec::CouplingRelationLoweringFidelity::Exact);
    const auto* partitioned_capability = findCapability(
        relation,
        fec::CouplingRelationLoweringKind::PartitionedExchange);
    ASSERT_NE(partitioned_capability, nullptr);
    EXPECT_EQ(partitioned_capability->fidelity,
              fec::CouplingRelationLoweringFidelity::Lagged);

    ThermalInterfaceCouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;
    options.temperature_transfer.kind = fec::CouplingTransferKind::Identity;
    options.heat_flux_transfer.kind = fec::CouplingTransferKind::Identity;
    const ThermalInterfaceCouplingModule partitioned_module(options);
    const auto partitioned = partitioned_module.declare();
    ASSERT_EQ(partitioned.region_relation_requirements.size(), 1u);
    ASSERT_TRUE(partitioned.region_relation_requirements.front()
                    .selected_lowering.has_value());
    EXPECT_EQ(partitioned.region_relation_requirements.front()
                  .selected_lowering->lowering_kind,
              fec::CouplingRelationLoweringKind::PartitionedExchange);
}

TEST(ThermalInterfaceCouplingModule, DeclaresPartitionedTemperatureAndHeatFluxExchanges)
{
    ThermalInterfaceCouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;
    options.temperature_transfer.kind = fec::CouplingTransferKind::Identity;
    options.heat_flux_transfer.kind = fec::CouplingTransferKind::Identity;

    const ThermalInterfaceCouplingModule module(options);
    const auto declaration = module.declare();

    EXPECT_TRUE(hasField(declaration, "side_a", "temperature"));
    EXPECT_TRUE(hasField(declaration, "side_b", "temperature"));
    EXPECT_TRUE(hasField(declaration, "side_a", "heat_flux"));
    EXPECT_TRUE(hasField(declaration, "side_b", "heat_flux"));

    ASSERT_EQ(declaration.partitioned_exchange_declarations.size(), 2u);
    const auto& temperature = declaration.partitioned_exchange_declarations[0];
    EXPECT_EQ(temperature.producer_port.contract_instance_name, "thermal_interface");
    EXPECT_EQ(temperature.producer_port.port_name, "side_a_temperature");
    EXPECT_EQ(temperature.consumer_port.port_name, "side_b_temperature");
    EXPECT_EQ(temperature.value.rank, fec::CouplingValueRank::Scalar);
    EXPECT_EQ(temperature.value.components, 1);
    ASSERT_TRUE(temperature.producer.has_value());
    EXPECT_EQ(temperature.producer->participant_name, "side_a");
    EXPECT_EQ(temperature.producer->endpoint_name, "temperature");
    ASSERT_TRUE(temperature.consumer.has_value());
    EXPECT_EQ(temperature.consumer->participant_name, "side_b");
    EXPECT_EQ(temperature.consumer->endpoint_name, "temperature");
    EXPECT_EQ(temperature.transfer.kind, fec::CouplingTransferKind::Identity);

    const auto& heat_flux = declaration.partitioned_exchange_declarations[1];
    EXPECT_EQ(heat_flux.producer_port.port_name, "side_b_heat_flux");
    EXPECT_EQ(heat_flux.consumer_port.port_name, "side_a_heat_flux");
    ASSERT_TRUE(heat_flux.producer.has_value());
    EXPECT_EQ(heat_flux.producer->participant_name, "side_b");
    EXPECT_EQ(heat_flux.producer->endpoint_name, "heat_flux");
    ASSERT_TRUE(heat_flux.consumer.has_value());
    EXPECT_EQ(heat_flux.consumer->participant_name, "side_a");
    EXPECT_EQ(heat_flux.consumer->endpoint_name, "heat_flux");
    EXPECT_EQ(heat_flux.transfer.kind, fec::CouplingTransferKind::Identity);

    ASSERT_EQ(declaration.group_hints.size(), 1u);
    EXPECT_EQ(declaration.group_hints[0].name, "thermal_interface_participants");
    EXPECT_EQ(declaration.group_hints[0].participant_names,
              (std::vector<std::string>{"side_a", "side_b"}));

    const auto built_exchanges =
        module.buildPartitionedExchangeDeclarations(fec::CouplingContext{});
    ASSERT_EQ(built_exchanges.size(), 2u);
    EXPECT_EQ(built_exchanges[0].producer_port.port_name, "side_a_temperature");
    EXPECT_EQ(built_exchanges[1].producer_port.port_name, "side_b_heat_flux");
}

TEST(ThermalInterfaceCouplingModule, BuildsPenaltyTemperatureContinuityForms)
{
    ThermalContextFixture fixture;
    const fec::CouplingFormBuilder form_builder(fixture.context);
    const ThermalInterfaceCouplingModule module;

    EXPECT_TRUE(module.supportsMonolithicLowering());
    const auto contributions = module.buildMonolithicForms(fixture.context,
                                                           form_builder);
    ASSERT_EQ(contributions.size(), 2u);

    const auto* side_a = findContribution(
        contributions,
        "thermal_interface.thermal_interface.temperature_continuity_side_a");
    ASSERT_NE(side_a, nullptr);
    EXPECT_EQ(side_a->origin, "ThermalInterfaceCouplingModule");
    EXPECT_EQ(side_a->operator_name, "equations");
    EXPECT_TRUE(hasFieldUse(side_a->field_uses, "side_a", "temperature"));
    EXPECT_TRUE(
        hasFieldUse(side_a->extra_trial_field_uses, "side_b", "temperature"));
    ASSERT_TRUE(side_a->residual.isValid());
    ASSERT_EQ(side_a->residual.node()->type(),
              forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(side_a->residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*side_a->residual.node()->interfaceMarker(), 10);
    EXPECT_TRUE(containsFormExprType(*side_a->residual.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*side_a->residual.node(),
                                     forms::FormExprType::RestrictPlus));

    const auto* side_b = findContribution(
        contributions,
        "thermal_interface.thermal_interface.temperature_continuity_side_b");
    ASSERT_NE(side_b, nullptr);
    EXPECT_TRUE(hasFieldUse(side_b->field_uses, "side_b", "temperature"));
    EXPECT_TRUE(
        hasFieldUse(side_b->extra_trial_field_uses, "side_a", "temperature"));
    ASSERT_TRUE(side_b->residual.isValid());
    ASSERT_EQ(side_b->residual.node()->type(),
              forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(side_b->residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*side_b->residual.node()->interfaceMarker(), 10);
}

TEST(ThermalInterfaceCouplingModule, RejectsInvalidOptionsDuringValidation)
{
    ThermalInterfaceCouplingOptions options;
    options.side_b_name = "side_a";

    const ThermalInterfaceCouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}

TEST(ThermalInterfaceCouplingModule,
     RejectsUnsupportedMonolithicFormulationThroughRelationCapability)
{
    ThermalContextFixture fixture;
    ThermalInterfaceCouplingOptions options;
    options.formulation =
        ThermalInterfaceFormulation::SymmetricNitscheDiffusion;

    const ThermalInterfaceCouplingModule module(options);
    try {
        module.validate(fixture.context);
        FAIL() << "expected relation capability diagnostic";
    } catch (const FE::InvalidArgumentException& e) {
        const std::string text = e.what();
        EXPECT_NE(text.find("selected relation lowering strategy is unsupported"),
                  std::string::npos);
        EXPECT_NE(text.find("enforcement=symmetric_nitsche_diffusion"),
                  std::string::npos);
        EXPECT_NE(text.find("enforcement=temperature_continuity_penalty"),
                  std::string::npos);
    }
}

TEST(ThermalInterfaceCouplingModule, RejectsUnconfiguredPartitionedTransfers)
{
    ThermalInterfaceCouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;

    const ThermalInterfaceCouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}

TEST(ThermalInterfaceCouplingModule, CoexistsWithFsiDeclarations)
{
    FSICouplingOptions fsi_options;
    fsi_options.contract_name = "mechanical_wall";

    ThermalInterfaceCouplingOptions thermal_options;
    thermal_options.contract_name = "thermal_wall";
    thermal_options.side_a_name = "fluid";
    thermal_options.side_b_name = "solid";

    const FSICouplingModule fsi(fsi_options);
    const ThermalInterfaceCouplingModule thermal(thermal_options);

    const auto fsi_declaration = fsi.declare();
    const auto thermal_declaration = thermal.declare();

    EXPECT_EQ(fsi_declaration.contract_type, "fsi");
    EXPECT_EQ(thermal_declaration.contract_type, "thermal_interface");
    EXPECT_EQ(fsi_declaration.contract_name, "mechanical_wall");
    EXPECT_EQ(thermal_declaration.contract_name, "thermal_wall");
    EXPECT_EQ(thermal_declaration.participants[0].participant_name, "fluid");
    EXPECT_EQ(thermal_declaration.participants[1].participant_name, "solid");
}
