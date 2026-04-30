/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/ThermalInterfaceCouplingModule.h"

#include "FE/Coupling/CouplingDefinitionBuilder.h"
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

fec::CouplingValueDescriptor scalarValue(int components)
{
    return fec::CouplingValueDescriptor{
        .rank = components == 1 ? fec::CouplingValueRank::Scalar
                                : fec::CouplingValueRank::Vector,
        .components = components,
    };
}

void declareFieldRequirement(fec::CouplingDefinitionBuilder& builder,
                             fec::CouplingFieldUse field,
                             fec::CouplingValueDescriptor value)
{
    builder.field(field);
    builder.fieldRequirement(fec::CouplingFieldRequirement{
        .field = std::move(field),
        .value = std::move(value),
    });
}

void appendPartitionedExchangeDeclarations(
    const ThermalInterfaceCouplingOptions& options,
    fec::CouplingDefinitionBuilder& builder)
{
    if (options.mode != fec::CouplingMode::Partitioned) {
        return;
    }

    builder
        .exchange("temperature",
                  fieldUse(options.side_a_name,
                           options.side_a_temperature_field),
                  fieldUse(options.side_b_name,
                           options.side_b_temperature_field))
        .producerPort("side_a_temperature")
        .consumerPort("side_b_temperature")
        .sharedInterface(options.interface_name)
        .value(scalarValue(options.temperature_components))
        .transfer(options.temperature_transfer);

    builder
        .exchange("heat_flux",
                  fieldUse(options.side_b_name,
                           options.side_b_heat_flux_field),
                  fieldUse(options.side_a_name,
                           options.side_a_heat_flux_field))
        .producerPort("side_b_heat_flux")
        .consumerPort("side_a_heat_flux")
        .sharedInterface(options.interface_name)
        .value(scalarValue(options.heat_flux_components))
        .transfer(options.heat_flux_transfer);

    builder.group(options.contract_name + "_participants",
                  {options.side_a_name, options.side_b_name});
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

std::string ThermalInterfaceCouplingModule::contractInstanceName() const
{
    return options_.contract_name;
}

void ThermalInterfaceCouplingModule::define(
    fec::CouplingDefinitionBuilder& builder) const
{
    builder.participant(options_.side_a_name);
    builder.participant(options_.side_b_name);

    declareFieldRequirement(builder,
                            fieldUse(options_.side_a_name,
                                     options_.side_a_temperature_field),
                            scalarValue(options_.temperature_components));
    declareFieldRequirement(builder,
                            fieldUse(options_.side_b_name,
                                     options_.side_b_temperature_field),
                            scalarValue(options_.temperature_components));

    if (options_.mode == fec::CouplingMode::Partitioned) {
        declareFieldRequirement(builder,
                                fieldUse(options_.side_a_name,
                                         options_.side_a_heat_flux_field),
                                scalarValue(options_.heat_flux_components));
        declareFieldRequirement(builder,
                                fieldUse(options_.side_b_name,
                                         options_.side_b_heat_flux_field),
                                scalarValue(options_.heat_flux_components));
    }

    builder.sharedRegion(fec::CouplingSharedRegionUse{
        .shared_region_name = options_.interface_name,
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
    });
    builder.sharedInterface(fec::CouplingSharedInterfaceRequirement{
        .shared_region_name = options_.interface_name,
        .participant_names = {options_.side_a_name, options_.side_b_name},
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
        .require_all_participants = true,
        .require_opposite_sides_for_two_participants = true,
        .require_monolithic_topology =
            options_.mode == fec::CouplingMode::Monolithic,
    });

    appendPartitionedExchangeDeclarations(options_, builder);
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

} // namespace coupling
} // namespace Physics
} // namespace svmp
