/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/FSICouplingModule.h"

#include "Core/FEException.h"
#include "FE/Coupling/CouplingDefinitionBuilder.h"
#include "FE/Coupling/CouplingFormBuilder.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace coupling {

namespace {

namespace fec = FE::coupling;
namespace forms = FE::forms;

constexpr const char* kFSIRelationName = "fsi_interface";
constexpr const char* kFSIOrigin = "FSICouplingModule";

fec::CouplingFieldUse fieldUse(const std::string& participant,
                               const std::string& field)
{
    return fec::CouplingFieldUse{
        .participant_name = participant,
        .field_name = field,
    };
}

fec::CouplingValueDescriptor scalarValue()
{
    return fec::CouplingValueDescriptor{
        .rank = fec::CouplingValueRank::Scalar,
        .components = 1,
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

fec::CouplingValueDescriptor interfaceVectorValue(const FSICouplingOptions& options)
{
    return fec::CouplingValueDescriptor{
        .rank = fec::CouplingValueRank::Vector,
        .components = options.interface_components,
    };
}

std::string generatedName(const FSICouplingOptions& options,
                          std::string local_name)
{
    return fec::makeCouplingGeneratedName(fec::CouplingGeneratedNameRequest{
        .contract_name = options.contract_name,
        .relation_name = kFSIRelationName,
        .local_name = std::move(local_name),
    });
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

fec::CouplingRegionRelationRequirement fsiInterfaceRelation(
    const FSICouplingOptions& options)
{
    return fec::CouplingRegionRelationRequirement{
        .relation_name = kFSIRelationName,
        .relation_kind = fec::CouplingRegionRelationKind::SidePairedInterface,
        .endpoints = {
            fec::CouplingRegionEndpointDeclaration{
                .participant_name = options.fluid_name,
                .shared_region_name = options.interface_name,
            },
            fec::CouplingRegionEndpointDeclaration{
                .participant_name = options.solid_name,
                .shared_region_name = options.interface_name,
            },
        },
        .lowering_capabilities = {
            fec::CouplingRelationLoweringCapability{
                .lowering_kind =
                    fec::CouplingRelationLoweringKind::MonolithicForms,
                .fidelity = fec::CouplingRelationLoweringFidelity::Exact,
                .enforcement_strategies = {"velocity_traction_balance"},
            },
            fec::CouplingRelationLoweringCapability{
                .lowering_kind =
                    fec::CouplingRelationLoweringKind::PartitionedExchange,
                .fidelity = fec::CouplingRelationLoweringFidelity::Lagged,
                .enforcement_strategies = {"velocity_traction_balance"},
                .partitioned_solve_strategies = {
                    fec::CouplingPartitionedSolveStrategy::ExplicitLagged,
                    fec::CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
                },
            },
        },
        .selected_lowering = selectedLowering(
            options.mode,
            "velocity_traction_balance"),
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

struct SolidInterfaceVelocity {
    fec::CouplingFieldUse dependency;
    forms::FormExpr value;
};

SolidInterfaceVelocity solidInterfaceVelocity(
    const FSICouplingOptions& options,
    const fec::CouplingInterfaceSideView& solid_side)
{
    if (options.use_solid_displacement_derivative) {
        return SolidInterfaceVelocity{
            .dependency = fieldUse(options.solid_name,
                                   options.solid_displacement_field),
            .value = solid_side.dt(options.solid_displacement_field, "dt_d_s"),
        };
    }

    FE_THROW_IF(!options.solid_velocity_field.has_value() ||
                    options.solid_velocity_field->empty(),
                FE::InvalidArgumentException,
                "FSI monolithic form requires a solid velocity field or displacement derivative");
    const auto& solid_velocity_field = *options.solid_velocity_field;
    return SolidInterfaceVelocity{
        .dependency = fieldUse(options.solid_name, solid_velocity_field),
        .value = solid_side.state(solid_velocity_field, "v_s"),
    };
}

struct FSIMonolithicEquations {
    fec::CouplingFieldUse solid_velocity_dependency;
    forms::FormExpr velocity_continuity;
    forms::FormExpr pressure_traction_balance;
};

FSIMonolithicEquations fsiMonolithicEquations(
    const FSICouplingOptions& options,
    const fec::CouplingFormBuilder& form_builder)
{
    const auto interface = form_builder.sharedInterface(options.interface_name);
    const auto fluid = interface.side(options.fluid_name);
    const auto solid = interface.side(options.solid_name);

    const auto u_f = fluid.state(options.fluid_velocity_field, "u_f");
    const auto w_f = fluid.test(options.fluid_velocity_field, "w_f");
    auto v_s = solidInterfaceVelocity(options, solid);

    const auto p_f = fluid.state(options.fluid_pressure_field, "p_f");
    const auto w_s = solid.test(options.solid_displacement_field, "w_s");
    const auto n_f = fluid.normal();

    return FSIMonolithicEquations{
        .solid_velocity_dependency = std::move(v_s.dependency),
        .velocity_continuity = interface.integral(
            forms::inner(u_f - v_s.value, w_f),
            options.fluid_name),
        .pressure_traction_balance = interface.integral(
            -forms::inner(p_f * n_f, w_s),
            options.fluid_name),
    };
}

std::optional<fec::CouplingGeometrySensitivityDeclaration>
meshGeometrySensitivity(const FSICouplingOptions& options)
{
    if (!options.mesh_name.has_value() ||
        !options.mesh_displacement_field.has_value()) {
        return std::nullopt;
    }
    return fec::CouplingGeometrySensitivityDeclaration{
        .mode = forms::GeometrySensitivityMode::MeshMotionUnknowns,
        .mesh_motion_field = fieldUse(*options.mesh_name,
                                      *options.mesh_displacement_field),
    };
}

fec::CouplingFormInstallOptionsDeclaration installOptionsWith(
    const std::optional<fec::CouplingGeometrySensitivityDeclaration>&
        geometry_sensitivity)
{
    fec::CouplingFormInstallOptionsDeclaration declaration;
    if (geometry_sensitivity.has_value()) {
        declaration.geometry_sensitivity = *geometry_sensitivity;
    }
    return declaration;
}

void appendMeshGeometryDependency(
    std::vector<fec::CouplingFieldUse>& trial_fields,
    const FSICouplingOptions& options,
    const std::optional<fec::CouplingGeometrySensitivityDeclaration>&
        geometry_sensitivity)
{
    if (!geometry_sensitivity.has_value()) {
        return;
    }
    trial_fields.push_back(
        fieldUse(*options.mesh_name, *options.mesh_displacement_field));
}

void appendPartitionedExchangeDeclarations(
    const FSICouplingOptions& options,
    fec::CouplingDefinitionBuilder& builder)
{
    if (options.mode != fec::CouplingMode::Partitioned) {
        return;
    }

    builder
        .exchange("solid_displacement",
                  fieldUse(options.solid_name,
                           options.solid_displacement_field),
                  fieldUse(options.fluid_name,
                           options.fluid_velocity_field))
        .producerPort("solid_displacement")
        .consumerPort("fluid_displacement")
        .sharedInterface(options.interface_name)
        .transfer(options.solid_to_fluid_transfer)
        .producerTemporal(options.partitioned_temporal
                              .solid_displacement_source)
        .consumerTemporal(options.partitioned_temporal
                              .fluid_displacement_target);

    builder
        .exchange("fluid_load",
                  fieldUse(options.fluid_name,
                           options.fluid_velocity_field),
                  fieldUse(options.solid_name,
                           options.solid_displacement_field))
        .producerPort("fluid_load")
        .consumerPort("solid_load")
        .sharedInterface(options.interface_name)
        .transfer(options.fluid_to_solid_transfer)
        .producerTemporal(options.partitioned_temporal.fluid_load_source)
        .consumerTemporal(options.partitioned_temporal.solid_load_target);

    builder.group(options.contract_name + "_participants",
                  {options.fluid_name, options.solid_name});
}

std::vector<fec::CouplingFormContribution> buildFSIMonolithicForms(
    const FSICouplingOptions& options,
    const fec::CouplingContext& ctx,
    const fec::CouplingFormBuilder& form_builder)
{
    static_cast<void>(ctx);

    auto equations = fsiMonolithicEquations(options, form_builder);
    const auto geometry_sensitivity = meshGeometrySensitivity(options);
    std::vector<fec::CouplingFormContribution> contributions;
    contributions.reserve(2);

    std::vector<fec::CouplingFieldUse> velocity_dependencies{
        equations.solid_velocity_dependency};
    appendMeshGeometryDependency(velocity_dependencies,
                                 options,
                                 geometry_sensitivity);
    contributions.push_back(form_builder.equationContribution(
        fec::CouplingEquationContributionRequest{
            .contribution_name = generatedName(options, "velocity_continuity"),
            .origin = kFSIOrigin,
            .residual_field_uses = {
                fieldUse(options.fluid_name, options.fluid_velocity_field),
            },
            .trial_field_uses = std::move(velocity_dependencies),
            .install_options_declaration =
                installOptionsWith(geometry_sensitivity),
            .residual = std::move(equations.velocity_continuity),
        }));

    std::vector<fec::CouplingFieldUse> traction_dependencies{
        fieldUse(options.fluid_name, options.fluid_pressure_field)};
    appendMeshGeometryDependency(traction_dependencies,
                                 options,
                                 geometry_sensitivity);
    contributions.push_back(form_builder.equationContribution(
        fec::CouplingEquationContributionRequest{
            .contribution_name =
                generatedName(options, "pressure_traction_balance"),
            .origin = kFSIOrigin,
            .residual_field_uses = {
                fieldUse(options.solid_name,
                         options.solid_displacement_field),
            },
            .trial_field_uses = std::move(traction_dependencies),
            .install_options_declaration =
                installOptionsWith(geometry_sensitivity),
            .residual = std::move(equations.pressure_traction_balance),
        }));

    return contributions;
}

fec::CouplingValidationResult validateOptionShape(const FSICouplingOptions& options)
{
    fec::CouplingValidationResult result;
    if (options.contract_name.empty()) {
        result.addError("FSI coupling requires a contract instance name");
    }
    if (options.fluid_name.empty() || options.solid_name.empty()) {
        result.addError("FSI coupling requires fluid and solid participant names");
    }
    if (options.mesh_name.has_value() && options.mesh_name->empty()) {
        result.addError("FSI ALE mode requires a mesh participant name");
    }
    if (options.interface_name.empty()) {
        result.addError("FSI coupling requires an interface shared-region name");
    }
    if (options.interface_components <= 0) {
        result.addError("FSI coupling requires a positive interface component count");
    }
    if (options.multiplier.enabled) {
        if (options.multiplier.field_name.empty()) {
            result.addError("FSI multiplier requires a field name");
        }
        if (options.multiplier.space == nullptr) {
            result.addError("FSI multiplier requires a function space");
        }
        if (options.multiplier.components < 0) {
            result.addError("FSI multiplier component count cannot be negative");
        }
    }
    if (options.mode == fec::CouplingMode::Partitioned) {
        if (options.solid_to_fluid_transfer.kind == fec::CouplingTransferKind::Unspecified ||
            options.fluid_to_solid_transfer.kind == fec::CouplingTransferKind::Unspecified) {
            result.addError("FSI partitioned mode requires explicit transfer declarations");
        }
    }
    return result;
}

void validateALEMeshRequirements(fec::CouplingValidationResult& result,
                                 const fec::CouplingContext& ctx,
                                 const FSICouplingOptions& options)
{
    if (!options.mesh_name.has_value()) {
        return;
    }
    if (!options.mesh_displacement_field.has_value() ||
        options.mesh_displacement_field->empty()) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = *options.mesh_name,
            .message = "FSI ALE mode requires a mesh displacement field",
        });
        return;
    }
    if (!ctx.hasParticipant(*options.mesh_name)) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = *options.mesh_name,
            .message = "FSI ALE mode requires a registered mesh participant",
        });
    }
    if (!ctx.hasField(*options.mesh_name, *options.mesh_displacement_field)) {
        result.add(fec::CouplingDiagnostic{
            .severity = fec::CouplingDiagnosticSeverity::Error,
            .contract_name = options.contract_name,
            .participant_name = *options.mesh_name,
            .field_name = *options.mesh_displacement_field,
            .message = "FSI ALE mode requires a registered mesh displacement field",
        });
    }
}

} // namespace

FSICouplingModule::FSICouplingModule(FSICouplingOptions options)
    : options_(std::move(options))
{
}

std::string FSICouplingModule::name() const
{
    return "fsi";
}

std::string FSICouplingModule::contractInstanceName() const
{
    return options_.contract_name;
}

void FSICouplingModule::define(fec::CouplingDefinitionBuilder& builder) const
{
    builder.participant(options_.fluid_name);
    builder.participant(options_.solid_name);
    if (options_.mesh_name.has_value()) {
        builder.participant(*options_.mesh_name);
    }

    declareFieldRequirement(builder,
                            fieldUse(options_.fluid_name,
                                     options_.fluid_velocity_field),
                            interfaceVectorValue(options_));
    declareFieldRequirement(builder,
                            fieldUse(options_.fluid_name,
                                     options_.fluid_pressure_field),
                            scalarValue());
    declareFieldRequirement(builder,
                            fieldUse(options_.solid_name,
                                     options_.solid_displacement_field),
                            interfaceVectorValue(options_));
    if (!options_.use_solid_displacement_derivative &&
        options_.solid_velocity_field.has_value()) {
        declareFieldRequirement(builder,
                                fieldUse(options_.solid_name,
                                         *options_.solid_velocity_field),
                                interfaceVectorValue(options_));
    }
    if (options_.mesh_name.has_value() &&
        options_.mesh_displacement_field.has_value()) {
        declareFieldRequirement(builder,
                                fieldUse(*options_.mesh_name,
                                         *options_.mesh_displacement_field),
                                interfaceVectorValue(options_));
    }

    builder.sharedRegion(fec::CouplingSharedRegionUse{
        .shared_region_name = options_.interface_name,
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
    });
    builder.sharedInterface(fec::CouplingSharedInterfaceRequirement{
        .shared_region_name = options_.interface_name,
        .participant_names = {options_.fluid_name, options_.solid_name},
        .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
        .require_all_participants = true,
        .require_opposite_sides_for_two_participants = true,
        .require_monolithic_topology =
            options_.mode == fec::CouplingMode::Monolithic,
    });
    builder.regionRelation(fsiInterfaceRelation(options_));

    if (options_.multiplier.enabled) {
        const auto multiplier_namespace =
            options_.multiplier.contract_field_namespace.empty()
                ? options_.contract_name
                : options_.multiplier.contract_field_namespace;
        builder.additionalField(fec::CouplingAdditionalFieldDeclaration{
            .field_namespace = fec::CouplingAdditionalFieldNamespace::Contract,
            .namespace_name = multiplier_namespace,
            .system_participant_name =
                options_.multiplier.system_participant_name.value_or(""),
            .field_name = options_.multiplier.field_name,
            .space = options_.multiplier.space,
            .components = options_.multiplier.components,
            .scope = fec::CouplingAdditionalFieldScope::InterfaceFace,
            .shared_region_name =
                options_.multiplier.shared_region_name.value_or(options_.interface_name),
        });
    }

    appendPartitionedExchangeDeclarations(options_, builder);

    if (options_.mode == fec::CouplingMode::Monolithic) {
        const auto options = options_;
        builder.monolithic([options](const fec::CouplingContext& ctx,
                                     const fec::CouplingFormBuilder& forms) {
            return buildFSIMonolithicForms(options, ctx, forms);
        });
    }
}

void FSICouplingModule::validateDefinitionOptions(
    const fec::CouplingContext& ctx,
    fec::CouplingValidationResult& result) const
{
    result.append(validateOptionShape(options_));
    validateALEMeshRequirements(result, ctx, options_);
}

} // namespace coupling
} // namespace Physics
} // namespace svmp
