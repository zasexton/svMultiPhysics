#include "Physics/Coupling/FSICouplingModule.h"
#include "Physics/Coupling/ThermalInterfaceCouplingModule.h"

#include "Core/FEException.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace svmp;
using namespace svmp::Physics::coupling;
namespace fec = svmp::FE::coupling;

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

    EXPECT_TRUE(module.buildPartitionedExchangeDeclarations(fec::CouplingContext{}).empty());
}

TEST(ThermalInterfaceCouplingModule, RejectsInvalidOptionsDuringValidation)
{
    ThermalInterfaceCouplingOptions options;
    options.side_b_name = "side_a";

    const ThermalInterfaceCouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
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
