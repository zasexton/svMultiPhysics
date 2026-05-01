/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/FSICouplingModule.h"

#include "FE/Coupling/CouplingDefinitionBuilder.h"
#include "FE/Coupling/CouplingFormBuilder.h"

#include <optional>
#include <string>
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
constexpr const char* kFSIEnforcementStrategy = "velocity_traction_balance";

forms::FormExpr solidInterfaceVelocity(
    const FSICouplingOptions& options,
    const fec::CouplingInterfaceSideView& solid_side)
{
    if (options.use_solid_displacement_derivative) {
        return solid_side.dt(options.solid_displacement_field, "dt_d_s");
    }

    return solid_side.requiredState(options.solid_velocity_field, "v_s");
}

std::vector<fec::CouplingFormContribution> fsiFormContributions(
    const FSICouplingOptions& options,
    const fec::CouplingFormBuilder& form_builder)
{
    const auto interface = form_builder.sharedInterface(options.interface_name);
    const auto fluid = interface.side(options.fluid_name);
    const auto solid = interface.side(options.solid_name);

    const auto u_f = fluid.state(options.fluid_velocity_field, "u_f");
    const auto w_f = fluid.test(options.fluid_velocity_field, "w_f");
    const auto v_s = solidInterfaceVelocity(options, solid);

    const auto p_f = fluid.state(options.fluid_pressure_field, "p_f");
    const auto w_s = solid.test(options.solid_displacement_field, "w_s");
    const auto n_f = fluid.normal();

    return form_builder
        .equationSet(fec::CouplingEquationSetRequest{
            .contract_name = options.contract_name,
            .relation_name = kFSIRelationName,
            .origin = kFSIOrigin,
            .geometry_sensitivity = fec::meshMotionGeometrySensitivity(
                options.mesh_name,
                options.mesh_displacement_field),
        })
        .infer({
            fec::CouplingInferredEquationRequest{
                .local_name = "velocity_continuity",
                .residual = interface.integral(
                    forms::inner(u_f - v_s, w_f),
                    options.fluid_name),
            },
            fec::CouplingInferredEquationRequest{
                .local_name = "pressure_traction_balance",
                .residual = interface.integral(
                    -forms::inner(p_f * n_f, w_s),
                    options.fluid_name),
            },
        });
}

fec::CouplingSidePairedPDERequest fsiCouplingRequest(
    const FSICouplingOptions& options,
    fec::CouplingMonolithicFormsCallback monolithic_forms = {})
{
    return fec::CouplingSidePairedPDERequest{
        .interface = fec::sidePairedInterface(
            kFSIRelationName,
            options.interface_name,
            options.fluid_name,
            options.solid_name,
            options.mode,
            kFSIEnforcementStrategy,
            {
                fec::CouplingPartitionedSolveStrategy::ExplicitLagged,
                fec::CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
            }),
        .required_fields = {
            fec::vectorFieldRole(options.fluid_name,
                                 options.fluid_velocity_field,
                                 options.interface_components),
            fec::scalarFieldRole(options.fluid_name,
                                 options.fluid_pressure_field),
            fec::vectorFieldRole(options.solid_name,
                                 options.solid_displacement_field,
                                 options.interface_components),
        },
        .optional_fields = {
            fec::optionalVectorFieldRole(
                options.solid_name,
                options.solid_velocity_field,
                options.interface_components,
                !options.use_solid_displacement_derivative),
            fec::optionalVectorFieldRole(
                options.mesh_name,
                options.mesh_displacement_field,
                options.interface_components,
                options.mesh_name.has_value()),
        },
        .interface_field = fec::contractInterfaceField(
            options.multiplier.field_name,
            options.multiplier.space,
            options.multiplier.shared_region_name.value_or(
                options.interface_name),
            options.multiplier.components,
            options.multiplier.contract_field_namespace,
            options.multiplier.system_participant_name,
            fec::CouplingRequirement::Required,
            options.multiplier.enabled),
        .partitioned_channels = {
            fec::fieldExchange(
                "solid_displacement",
                options.solid_name,
                options.solid_displacement_field,
                options.fluid_name,
                options.fluid_velocity_field,
                options.solid_to_fluid_transfer,
                options.partitioned_temporal.solid_displacement_source,
                options.partitioned_temporal.fluid_displacement_target),
            fec::fieldExchange(
                "fluid_load",
                options.fluid_name,
                options.fluid_velocity_field,
                options.solid_name,
                options.solid_displacement_field,
                options.fluid_to_solid_transfer,
                options.partitioned_temporal.fluid_load_source,
                options.partitioned_temporal.solid_load_target),
        },
        .monolithic_forms = std::move(monolithic_forms),
    };
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
    builder.sidePairedPDECoupling(
        fsiCouplingRequest(
            options_,
            [this](const fec::CouplingContext&,
                   const fec::CouplingFormBuilder& forms) {
                return fsiFormContributions(options_, forms);
            }));
}

} // namespace coupling
} // namespace Physics
} // namespace svmp
