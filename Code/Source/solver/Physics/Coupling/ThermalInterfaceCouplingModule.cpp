/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/ThermalInterfaceCouplingModule.h"

#include "FE/Coupling/CouplingGraph.h"

#include <array>
#include <span>
#include <utility>

namespace svmp {
namespace Physics {
namespace coupling {

namespace {

namespace fec = FE::coupling;

fec::CouplingFieldUse fieldUse(const std::string& participant,
                               const std::string& field)
{
    return fec::CouplingFieldUse{
        .participant_name = participant,
        .field_name = field,
    };
}

fec::CouplingEndpointRef fieldEndpoint(const std::string& participant,
                                       const std::string& field)
{
    return fec::CouplingEndpointRef{
        .kind = fec::CouplingEndpointKind::Field,
        .participant_name = participant,
        .endpoint_name = field,
        .temporal = fec::CouplingTemporalSlotDescriptor{
            .slot = fec::CouplingTemporalSlot::Current,
        },
    };
}

fec::CouplingPortId port(const ThermalInterfaceCouplingOptions& options,
                         std::string port_name)
{
    return fec::CouplingPortId{
        .contract_instance_name = options.contract_name,
        .port_name = std::move(port_name),
    };
}

fec::CouplingValueDescriptor scalarValue(int components)
{
    return fec::CouplingValueDescriptor{
        .rank = components == 1 ? fec::CouplingValueRank::Scalar
                                : fec::CouplingValueRank::Vector,
        .components = components,
    };
}

void appendPartitionedExchangeDeclarations(
    const ThermalInterfaceCouplingOptions& options,
    fec::CouplingContractDeclaration& declaration)
{
    if (options.mode != fec::CouplingMode::Partitioned) {
        return;
    }

    declaration.partitioned_exchange_declarations.push_back(
        fec::CouplingExchangeDeclaration{
            .producer_port = port(options, "side_a_temperature"),
            .consumer_port = port(options, "side_b_temperature"),
            .value = scalarValue(options.temperature_components),
            .producer = fieldEndpoint(options.side_a_name,
                                      options.side_a_temperature_field),
            .consumer = fieldEndpoint(options.side_b_name,
                                      options.side_b_temperature_field),
            .shared_region_name = options.interface_name,
            .transfer = options.temperature_transfer,
        });

    declaration.partitioned_exchange_declarations.push_back(
        fec::CouplingExchangeDeclaration{
            .producer_port = port(options, "side_b_heat_flux"),
            .consumer_port = port(options, "side_a_heat_flux"),
            .value = scalarValue(options.heat_flux_components),
            .producer = fieldEndpoint(options.side_b_name,
                                      options.side_b_heat_flux_field),
            .consumer = fieldEndpoint(options.side_a_name,
                                      options.side_a_heat_flux_field),
            .shared_region_name = options.interface_name,
            .transfer = options.heat_flux_transfer,
        });

    declaration.group_hints.push_back(fec::CouplingGroupHint{
        .name = options.contract_name + "_participants",
        .participant_names = {options.side_a_name, options.side_b_name},
    });
}

fec::CouplingValidationResult validateOptionShape(
    const ThermalInterfaceCouplingOptions& options)
{
    fec::CouplingValidationResult result;
    if (options.contract_name.empty()) {
        result.addError("thermal interface coupling requires a contract instance name");
    }
    if (options.side_a_name.empty() || options.side_b_name.empty()) {
        result.addError("thermal interface coupling requires participant names");
    }
    if (options.side_a_name == options.side_b_name) {
        result.addError("thermal interface coupling participants must be distinct");
    }
    if (options.interface_name.empty()) {
        result.addError("thermal interface coupling requires a shared-region name");
    }
    if (options.side_a_temperature_field.empty() ||
        options.side_b_temperature_field.empty()) {
        result.addError("thermal interface coupling requires temperature field names");
    }
    if (options.temperature_components <= 0 || options.heat_flux_components <= 0) {
        result.addError("thermal interface coupling requires positive component counts");
    }
    if (options.mode == fec::CouplingMode::Partitioned) {
        if (options.side_a_heat_flux_field.empty() ||
            options.side_b_heat_flux_field.empty()) {
            result.addError("thermal interface partitioned mode requires heat-flux field names");
        }
        if (options.temperature_transfer.kind == fec::CouplingTransferKind::Unspecified ||
            options.heat_flux_transfer.kind == fec::CouplingTransferKind::Unspecified) {
            result.addError(
                "thermal interface partitioned mode requires explicit transfer declarations");
        }
    }
    return result;
}

} // namespace

ThermalInterfaceCouplingModule::ThermalInterfaceCouplingModule(
    ThermalInterfaceCouplingOptions options)
    : options_(std::move(options))
{
}

std::string ThermalInterfaceCouplingModule::name() const
{
    return "thermal_interface";
}

fec::CouplingContractDeclaration ThermalInterfaceCouplingModule::declare() const
{
    fec::CouplingContractDeclaration declaration;
    declaration.contract_type = name();
    declaration.contract_name = options_.contract_name;
    declaration.participants.push_back({.participant_name = options_.side_a_name});
    declaration.participants.push_back({.participant_name = options_.side_b_name});

    declaration.fields.push_back(fieldUse(options_.side_a_name,
                                          options_.side_a_temperature_field));
    declaration.fields.push_back(fieldUse(options_.side_b_name,
                                          options_.side_b_temperature_field));
    if (options_.mode == fec::CouplingMode::Partitioned) {
        declaration.fields.push_back(fieldUse(options_.side_a_name,
                                              options_.side_a_heat_flux_field));
        declaration.fields.push_back(fieldUse(options_.side_b_name,
                                              options_.side_b_heat_flux_field));
    }

    declaration.shared_regions.push_back(fec::CouplingSharedRegionUse{
        .shared_region_name = options_.interface_name,
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
    });

    appendPartitionedExchangeDeclarations(options_, declaration);
    return declaration;
}

void ThermalInterfaceCouplingModule::validate(const fec::CouplingContext& ctx) const
{
    auto result = validateOptionShape(options_);
    const auto declaration = declare();
    FE::coupling::CouplingGraph graph;
    const std::array<fec::CouplingContractDeclaration, 1> declarations{declaration};
    result.append(graph.buildDeclarationGraph(
        ctx,
        std::span<const fec::CouplingContractDeclaration>(declarations)));
    throwIfInvalid(result);
}

std::vector<fec::CouplingExchangeDeclaration>
ThermalInterfaceCouplingModule::buildPartitionedExchangeDeclarations(
    const fec::CouplingContext&) const
{
    return {};
}

} // namespace coupling
} // namespace Physics
} // namespace svmp
