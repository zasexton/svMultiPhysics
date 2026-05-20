/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Coupling/FSICouplingModule.h"

#include "FE/Coupling/CouplingDefinitionBuilder.h"
#include "FE/Coupling/CouplingFormBuilder.h"
#include "FE/Forms/Vocabulary.h"
#include "Core/FEException.h"

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
namespace analysis = FE::analysis;

constexpr const char* kFSIRelationName = "fsi_interface";
constexpr const char* kFSIOrigin = "FSICouplingModule";
constexpr const char* kFSIEnforcementStrategy = "velocity_traction_balance";
constexpr const char* kFSITractionContribution = "fluid_traction_balance";

std::string fsiContributionName(const FSICouplingOptions& options,
                                std::string local_name)
{
    return options.contract_name + "." + kFSIRelationName + "." +
           std::move(local_name);
}

forms::FormExpr solidInterfaceVelocity(
    const FSICouplingOptions& options,
    const fec::CouplingInterfaceSideView& solid_side)
{
    if (options.use_solid_displacement_derivative) {
        return solid_side.dt(options.solid_displacement_field, "dt_d_s");
    }

    return solid_side.requiredState(options.solid_velocity_field, "v_s");
}

struct ViscosityExpression {
    forms::FormExpr value;
    bool structurally_zero{false};
};

std::optional<ViscosityExpression> viscosityFromMetadata(
    const FSICouplingOptions& options,
    const fec::CouplingContext& context,
    const forms::FormExpr& strain_rate)
{
    const auto velocity_field =
        context.field(options.fluid_name, options.fluid_velocity_field);

    const fec::CouplingConstitutiveLawRef* match = nullptr;
    for (const auto& law_ref : context.constitutiveLaws()) {
        if (law_ref.participant_name != options.fluid_name ||
            law_ref.law.role != analysis::ConstitutiveLawRole::DynamicViscosity ||
            law_ref.law.primary_field != velocity_field.field_id) {
            continue;
        }
        if (match != nullptr) {
            throw FE::InvalidArgumentException(
                "FSI coupling found multiple fluid dynamic-viscosity metadata "
                "records for the fluid velocity field");
        }
        match = &law_ref;
    }

    if (match == nullptr) {
        return std::nullopt;
    }

    if (match->law.model) {
        if (match->law.input_measure !=
            analysis::ConstitutiveLawInputMeasure::
                SymmetricGradientSecondInvariant) {
            throw FE::InvalidArgumentException(
                "FSI coupling cannot evaluate fluid dynamic-viscosity metadata "
                "with an unsupported input measure");
        }
        const auto gamma =
            forms::sqrt(forms::FormExpr::constant(2.0) *
                        forms::inner(strain_rate, strain_rate));
        return ViscosityExpression{
            .value = forms::constitutive(match->law.model, gamma).out(0),
        };
    }

    if (!match->law.constant_value_available) {
        throw FE::InvalidArgumentException(
            "FSI coupling fluid dynamic-viscosity metadata has no constant "
            "value or constitutive model");
    }

    return ViscosityExpression{
        .value = forms::FormExpr::constant(match->law.constant_value),
        .structurally_zero = match->law.constant_value == 0.0,
    };
}

ViscosityExpression fallbackFluidViscosity(
    const FSICouplingOptions& options,
    const forms::FormExpr& strain_rate)
{
    if (options.fluid_viscosity_model) {
        const auto gamma =
            forms::sqrt(forms::FormExpr::constant(2.0) *
                        forms::inner(strain_rate, strain_rate));
        return ViscosityExpression{
            .value =
                forms::constitutive(options.fluid_viscosity_model, gamma).out(0),
        };
    }

    return ViscosityExpression{
        .value = forms::FormExpr::constant(options.fluid_viscosity),
        .structurally_zero = options.fluid_viscosity == 0.0,
    };
}

ViscosityExpression fluidViscosity(
    const FSICouplingOptions& options,
    const fec::CouplingContext& context,
    const forms::FormExpr& strain_rate)
{
    if (auto metadata = viscosityFromMetadata(options, context, strain_rate)) {
        return *metadata;
    }
    return fallbackFluidViscosity(options, strain_rate);
}

forms::FormExpr fluidCauchyTraction(
    const FSICouplingOptions& options,
    const fec::CouplingContext& context,
    const forms::FormExpr& u_f,
    const forms::FormExpr& p_f,
    const forms::FormExpr& n_f)
{
    const auto pressure_traction = -p_f * n_f;
    const auto strain_rate = forms::sym(forms::grad(u_f));
    const auto mu = fluidViscosity(options, context, strain_rate);
    if (mu.structurally_zero) {
        return pressure_traction;
    }

    const auto viscous_stress =
        forms::FormExpr::constant(2.0) * mu.value * strain_rate;
    return pressure_traction + viscous_stress * n_f;
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
    const auto t_f = fluidCauchyTraction(
        options,
        form_builder.context(),
        u_f,
        p_f,
        n_f);

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
                .local_name = kFSITractionContribution,
                .residual = interface.integral(
                    forms::inner(t_f, w_s),
                    options.fluid_name),
            },
        });
}

struct FSIDeclaredFields {
    fec::CouplingFieldRoleHandle fluid_velocity;
    fec::CouplingFieldRoleHandle solid_displacement;
};

fec::CouplingSidePairedPDEBuilder fsiInterfaceCoupling(
    fec::CouplingDefinitionBuilder& builder,
    const FSICouplingOptions& options)
{
    return builder.sidePairedPDE(kFSIRelationName)
        .onInterface(options.interface_name)
        .between(options.fluid_name, options.solid_name)
        .mode(options.mode)
        .enforcement(kFSIEnforcementStrategy)
        .partitionedStrategies({
            fec::CouplingPartitionedSolveStrategy::ExplicitLagged,
            fec::CouplingPartitionedSolveStrategy::StaggeredFixedPoint,
        });
}

FSIDeclaredFields declarePrimaryFsiFields(
    fec::CouplingSidePairedPDEBuilder& coupling,
    const FSICouplingOptions& options)
{
    FSIDeclaredFields fields;
    fields.fluid_velocity =
        coupling.vectorField(options.fluid_name,
                             options.fluid_velocity_field,
                             options.interface_components);
    coupling.requiredScalarField(options.fluid_name,
                                 options.fluid_pressure_field);
    fields.solid_displacement =
        coupling.vectorField(options.solid_name,
                             options.solid_displacement_field,
                             options.interface_components);
    return fields;
}

void declareOptionalFsiFields(fec::CouplingSidePairedPDEBuilder& coupling,
                              const FSICouplingOptions& options)
{
    coupling
        .optionalVectorField(options.solid_name,
                             options.solid_velocity_field,
                             options.interface_components)
        .requiredWhen(!options.use_solid_displacement_derivative);

    coupling
        .optionalVectorField(options.mesh_name,
                             options.mesh_displacement_field,
                             options.interface_components)
        .requiredWhen(options.mesh_name.has_value());
}

void declareMultiplierField(fec::CouplingSidePairedPDEBuilder& coupling,
                            const FSICouplingOptions& options)
{
    if (!options.multiplier.enabled) {
        return;
    }

    coupling.interfaceField()
        .name(options.multiplier.field_name)
        .space(options.multiplier.space)
        .components(options.multiplier.components)
        .fieldNamespace(options.multiplier.contract_field_namespace)
        .systemParticipant(options.multiplier.system_participant_name)
        .sharedRegion(options.multiplier.shared_region_name.value_or(
            options.interface_name));
}

void declarePartitionedFsiExchanges(
    fec::CouplingSidePairedPDEBuilder& coupling,
    const FSICouplingOptions& options,
    const FSIDeclaredFields& fields)
{
    coupling.partitionedExchange("solid_displacement")
        .from(fields.solid_displacement)
        .to(fields.fluid_velocity)
        .transfer(options.solid_to_fluid_transfer)
        .producerTemporal(options.partitioned_temporal.solid_displacement_source)
        .consumerTemporal(options.partitioned_temporal.fluid_displacement_target);

    coupling.partitionedPayloadFromForm("fluid_load")
        .contribution(fsiContributionName(options, kFSITractionContribution))
        .from(fields.fluid_velocity)
        .to(fields.solid_displacement)
        .preferred(fec::CouplingPayloadKind::CoefficientExpression)
        .fallback(fec::CouplingPayloadFallbackPolicy::WarnAndUseResidualRecipe)
        .transfer(options.fluid_to_solid_transfer)
        .producerTemporal(options.partitioned_temporal.fluid_load_source)
        .consumerTemporal(options.partitioned_temporal.solid_load_target);
}

void installMonolithicFsiForms(fec::CouplingSidePairedPDEBuilder& coupling,
                               const FSICouplingOptions& options)
{
    const auto form_options = options;
    coupling.monolithicForms(
        [form_options](const fec::CouplingContext&,
                       const fec::CouplingFormBuilder& forms) {
            return fsiFormContributions(form_options, forms);
        });
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
    auto coupling = fsiInterfaceCoupling(builder, options_);
    const auto fields = declarePrimaryFsiFields(coupling, options_);
    declareOptionalFsiFields(coupling, options_);
    declareMultiplierField(coupling, options_);
    declarePartitionedFsiExchanges(coupling, options_, fields);
    installMonolithicFsiForms(coupling, options_);
}

} // namespace coupling
} // namespace Physics
} // namespace svmp
