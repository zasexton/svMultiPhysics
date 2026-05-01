/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/ThermalInterfaceCouplingModule.h"

#include "FE/Coupling/CouplingDefinitionBuilder.h"
#include "FE/Forms/InterfaceConditions.h"

#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace coupling {

namespace {

namespace fec = FE::coupling;
namespace forms = FE::forms;

constexpr const char* kThermalInterfaceRelationName = "thermal_interface";

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

std::string generatedName(const ThermalInterfaceCouplingOptions& options,
                          std::string local_name)
{
    return fec::makeCouplingGeneratedName(fec::CouplingGeneratedNameRequest{
        .contract_name = options.contract_name,
        .relation_name = kThermalInterfaceRelationName,
        .local_name = std::move(local_name),
    });
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

std::string thermalEnforcementStrategy(
    ThermalInterfaceFormulation formulation)
{
    switch (formulation) {
    case ThermalInterfaceFormulation::TemperatureContinuityPenalty:
        return "temperature_continuity_penalty";
    case ThermalInterfaceFormulation::SymmetricNitscheDiffusion:
        return "symmetric_nitsche_diffusion";
    case ThermalInterfaceFormulation::ExplicitFluxBalance:
        return "explicit_flux_balance";
    }
    return "unknown";
}

fec::CouplingRelationLoweringRequest selectedLowering(
    fec::CouplingMode mode,
    std::string enforcement_strategy)
{
    return fec::CouplingRelationLoweringRequest{
        .mode = mode,
        .lowering_kind = mode == fec::CouplingMode::Monolithic
                             ? fec::CouplingRelationLoweringKind::MonolithicForms
                             : fec::CouplingRelationLoweringKind::PartitionedExchange,
        .enforcement_strategy = std::move(enforcement_strategy),
    };
}

fec::CouplingRegionRelationRequirement thermalInterfaceRelation(
    const ThermalInterfaceCouplingOptions& options)
{
    return fec::CouplingRegionRelationRequirement{
        .relation_name = kThermalInterfaceRelationName,
        .relation_kind = fec::CouplingRegionRelationKind::SidePairedInterface,
        .endpoints = {
            fec::CouplingRegionEndpointDeclaration{
                .participant_name = options.side_a_name,
                .shared_region_name = options.interface_name,
            },
            fec::CouplingRegionEndpointDeclaration{
                .participant_name = options.side_b_name,
                .shared_region_name = options.interface_name,
            },
        },
        .lowering_capabilities = {
            fec::CouplingRelationLoweringCapability{
                .lowering_kind =
                    fec::CouplingRelationLoweringKind::MonolithicForms,
                .fidelity = fec::CouplingRelationLoweringFidelity::Exact,
                .enforcement_strategies = {
                    "temperature_continuity_penalty",
                },
            },
            fec::CouplingRelationLoweringCapability{
                .lowering_kind =
                    fec::CouplingRelationLoweringKind::PartitionedExchange,
                .fidelity = fec::CouplingRelationLoweringFidelity::Lagged,
            },
        },
        .selected_lowering = selectedLowering(
            options.mode,
            thermalEnforcementStrategy(options.formulation)),
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
        .require_all_endpoints = true,
        .require_distinct_participants = true,
        .require_opposite_sides_for_side_pair = true,
        .require_common_monolithic_system =
            options.mode == fec::CouplingMode::Monolithic,
        .require_registered_topology =
            options.mode == fec::CouplingMode::Monolithic,
    };
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
        .transfer(options.heat_flux_transfer);

    builder.group(options.contract_name + "_participants",
                  {options.side_a_name, options.side_b_name});
}

std::vector<fec::CouplingFormContribution> buildThermalPenaltyForms(
    const ThermalInterfaceCouplingOptions& options,
    const fec::CouplingContext& ctx,
    const fec::CouplingFormBuilder& form_builder)
{
    static_cast<void>(ctx);

    const auto interface = form_builder.sharedInterface(options.interface_name);
    const auto side_a = interface.side(options.side_a_name);
    const auto side_b = interface.side(options.side_b_name);

    const auto temperature_a =
        side_a.state(options.side_a_temperature_field, "T_a");
    const auto temperature_a_test =
        side_a.test(options.side_a_temperature_field, "w_a");
    const auto temperature_b =
        side_b.state(options.side_b_temperature_field, "T_b");
    const auto temperature_b_test =
        side_b.test(options.side_b_temperature_field, "w_b");
    forms::bc::TraceNitscheOptions penalty_options;
    penalty_options.gamma = options.temperature_penalty;
    penalty_options.scale_with_p = false;
    const auto penalty_terms = forms::interface::scalarContinuityPenaltyTerms(
        temperature_a,
        temperature_a_test,
        temperature_b,
        temperature_b_test,
        forms::FormExpr::constant(1.0),
        penalty_options);

    std::vector<fec::CouplingFormContribution> contributions;

    fec::CouplingFormContribution side_a_residual;
    side_a_residual.contribution_name =
        generatedName(options, "temperature_continuity_side_a");
    side_a_residual.origin = "ThermalInterfaceCouplingModule";
    side_a_residual.operator_name = "equations";
    side_a_residual.field_uses = {fieldUse(options.side_a_name,
                                           options.side_a_temperature_field)};
    side_a_residual.extra_trial_field_uses = {
        fieldUse(options.side_b_name, options.side_b_temperature_field)};
    side_a_residual.residual =
        interface.integral(penalty_terms.first_side, options.side_a_name);
    contributions.push_back(
        form_builder.attachTerminalProvenance(std::move(side_a_residual)));

    fec::CouplingFormContribution side_b_residual;
    side_b_residual.contribution_name =
        generatedName(options, "temperature_continuity_side_b");
    side_b_residual.origin = "ThermalInterfaceCouplingModule";
    side_b_residual.operator_name = "equations";
    side_b_residual.field_uses = {fieldUse(options.side_b_name,
                                           options.side_b_temperature_field)};
    side_b_residual.extra_trial_field_uses = {
        fieldUse(options.side_a_name, options.side_a_temperature_field)};
    side_b_residual.residual =
        interface.integral(penalty_terms.second_side, options.side_b_name);
    contributions.push_back(
        form_builder.attachTerminalProvenance(std::move(side_b_residual)));

    return contributions;
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
    if (!(options.temperature_penalty > 0.0)) {
        result.addError("thermal interface temperature penalty must be positive");
    }
    if (options.mode == fec::CouplingMode::Partitioned) {
        if (options.side_a_heat_flux_field.empty() ||
            options.side_b_heat_flux_field.empty()) {
            result.addError("thermal interface partitioned mode requires heat-flux field names");
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
    builder.regionRelation(thermalInterfaceRelation(options_));

    appendPartitionedExchangeDeclarations(options_, builder);

    if (options_.mode == fec::CouplingMode::Monolithic &&
        options_.formulation ==
            ThermalInterfaceFormulation::TemperatureContinuityPenalty) {
        const auto options = options_;
        builder.monolithic([options](const fec::CouplingContext& ctx,
                                     const fec::CouplingFormBuilder& forms) {
            return buildThermalPenaltyForms(options, ctx, forms);
        });
    }
}

void ThermalInterfaceCouplingModule::validateDefinitionOptions(
    const fec::CouplingContext&,
    fec::CouplingValidationResult& result) const
{
    result.append(validateOptionShape(options_));
}

} // namespace coupling
} // namespace Physics
} // namespace svmp
