#include "Physics/Coupling/FSICouplingModule.h"

#include "Core/FEException.h"
#include "Spaces/SpaceFactory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

using namespace svmp;
using namespace svmp::Physics::coupling;
namespace fec = svmp::FE::coupling;

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
    FSICouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;
    options.solid_to_fluid_transfer.kind = fec::CouplingTransferKind::Identity;
    options.fluid_to_solid_transfer.kind = fec::CouplingTransferKind::Identity;

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
    ASSERT_TRUE(displacement.consumer.has_value());
    EXPECT_EQ(displacement.consumer->participant_name, "fluid");
    EXPECT_EQ(displacement.consumer->endpoint_name, "velocity");
    EXPECT_EQ(displacement.shared_region_name, "interface");
    EXPECT_EQ(displacement.transfer.kind, fec::CouplingTransferKind::Identity);

    const auto& load = declaration.partitioned_exchange_declarations[1];
    EXPECT_EQ(load.producer_port.port_name, "fluid_load");
    EXPECT_EQ(load.consumer_port.port_name, "solid_load");
    ASSERT_TRUE(load.producer.has_value());
    EXPECT_EQ(load.producer->participant_name, "fluid");
    ASSERT_TRUE(load.consumer.has_value());
    EXPECT_EQ(load.consumer->participant_name, "solid");

    ASSERT_EQ(declaration.group_hints.size(), 1u);
    EXPECT_EQ(declaration.group_hints[0].name, "fsi_participants");
    EXPECT_EQ(declaration.group_hints[0].participant_names,
              (std::vector<std::string>{"fluid", "solid"}));

    EXPECT_TRUE(module.buildPartitionedExchangeDeclarations(fec::CouplingContext{}).empty());
}

TEST(FSICouplingModule, RejectsInvalidOptionsDuringValidation)
{
    FSICouplingOptions options;
    options.interface_components = 0;

    const FSICouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}

TEST(FSICouplingModule, RejectsUnconfiguredPartitionedTransfers)
{
    FSICouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;

    const FSICouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}
