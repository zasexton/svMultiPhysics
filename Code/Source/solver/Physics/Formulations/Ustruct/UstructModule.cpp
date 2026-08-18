/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/Ustruct/UstructModule.h"

#include "Physics/Formulations/Ustruct/UstructBCFactories.h"
#include "Physics/Formulations/Ustruct/UstructPostProcessing.h"

#include "FE/Forms/FiniteDeformationForms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace ustruct {

UstructModule::UstructModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
    UstructOptions options)
    : displacement_space_(std::move(displacement_space))
    , pressure_space_(std::move(pressure_space))
    , options_(std::move(options))
{
}

void UstructModule::registerOn(FE::systems::FESystem& system) const
{
    if (!displacement_space_) {
        throw std::invalid_argument("UstructModule::registerOn: null displacement_space");
    }
    if (!pressure_space_) {
        throw std::invalid_argument("UstructModule::registerOn: null pressure_space");
    }

    const int dim = displacement_space_->value_dimension();
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument("UstructModule::registerOn: displacement space must have 2 or 3 components");
    }
    if (pressure_space_->value_dimension() != 1) {
        throw std::invalid_argument("UstructModule::registerOn: pressure space must be scalar");
    }
    if (!(options_.density > 0.0)) {
        throw std::invalid_argument("UstructModule::registerOn: density must be > 0");
    }
    if (!options_.deviatoric_pk1_model) {
        throw std::invalid_argument("UstructModule::registerOn: null deviatoric_pk1_model");
    }
    if (options_.use_symbolic_tangent && !options_.deviatoric_pk1_model->inlinable()) {
        throw std::invalid_argument(
            "UstructModule::registerOn: deviatoric_pk1_model must be inlinable when symbolic tangents are enabled");
    }
    if (options_.penalty_parameter < 0.0) {
        throw std::invalid_argument("UstructModule::registerOn: penalty_parameter must be >= 0");
    }
    if (!(options_.kinematic_residual_scale > 0.0) || !std::isfinite(options_.kinematic_residual_scale)) {
        throw std::invalid_argument("UstructModule::registerOn: kinematic_residual_scale must be finite and > 0");
    }

    FE::systems::FieldSpec d_spec;
    d_spec.name = options_.displacement_field_name;
    d_spec.space = displacement_space_;
    d_spec.components = dim;
    const FE::FieldId d_id = system.addField(std::move(d_spec));

    const bool use_time_derivative_terms = options_.enable_time_derivative_terms;
    std::optional<FE::FieldId> v_id;
    if (use_time_derivative_terms) {
        FE::systems::FieldSpec v_spec;
        v_spec.name = options_.velocity_field_name;
        v_spec.space = displacement_space_;
        v_spec.components = dim;
        v_id = system.addField(std::move(v_spec));
    }

    FE::systems::FieldSpec p_spec;
    p_spec.name = options_.pressure_field_name;
    p_spec.space = pressure_space_;
    p_spec.components = 1;
    const FE::FieldId p_id = system.addField(std::move(p_spec));

    system.addOperator("equations");

    using namespace FE::forms;

    const auto d = StateField(d_id, *displacement_space_, options_.displacement_field_name);
    std::optional<FormExpr> v;
    if (v_id) {
        v = StateField(*v_id, *displacement_space_, options_.velocity_field_name);
    }
    const auto p = StateField(p_id, *pressure_space_, options_.pressure_field_name);

    const auto z = TestField(d_id, *displacement_space_, "z");
    std::optional<FormExpr> w;
    if (v_id) {
        w = TestField(*v_id, *displacement_space_, "w");
    }
    const auto q = TestField(p_id, *pressure_space_, "q");

    post::registerPostProcessing(system, d_id, v_id, p_id, *displacement_space_, *pressure_space_, options_);

    const auto kin = finite_deformation::kinematics(d, dim);
    const auto F = kin.F;
    const auto J = kin.J;
    const auto Finv = kin.Finv;

    const auto gradx_p = transpose(Finv) * grad(p);
    const auto gradx_q = transpose(Finv) * grad(q);

    const auto zero = FormExpr::constant(0.0);
    const auto one = FormExpr::constant(1.0);

    std::vector<FormExpr> body_components;
    std::vector<FormExpr> zero_components;
    body_components.reserve(static_cast<std::size_t>(dim));
    zero_components.reserve(static_cast<std::size_t>(dim));
    for (int c = 0; c < dim; ++c) {
        body_components.push_back(FormExpr::constant(options_.body_force[static_cast<std::size_t>(c)]));
        zero_components.push_back(zero);
    }
    const auto body = FormExpr::asVector(std::move(body_components));
    const auto zero_vector = FormExpr::asVector(std::move(zero_components));

    const auto rho0 = FormExpr::constant(options_.density);
    const auto K = FormExpr::constant(options_.penalty_parameter);

    FormExpr rho = rho0;
    FormExpr beta = zero;
    if (options_.penalty_parameter != 0.0 &&
        options_.volumetric_model != VolumetricPenaltyModel::None) {
        switch (options_.volumetric_model) {
        case VolumetricPenaltyModel::Quadratic: {
            const auto denom = K - p;
            rho = rho0 * K / denom;
            beta = one / denom;
            break;
        }
        case VolumetricPenaltyModel::ST91: {
            const auto root = sqrt(p * p + K * K);
            rho = rho0 * (p + root) / K;
            beta = one / root;
            break;
        }
        case VolumetricPenaltyModel::M94: {
            const auto denom = K + p;
            rho = rho0 * denom / K;
            beta = one / denom;
            break;
        }
        case VolumetricPenaltyModel::None:
            break;
        }
    }

    FormExpr tau_m = zero;
    FormExpr tau_c = zero;
    if (options_.enable_stabilization && (options_.ct_m != 0.0 || options_.ct_c != 0.0)) {
        const FE::Real wave_speed = options_.bulk_wave_speed;
        if (!(wave_speed > 0.0) || !std::isfinite(wave_speed)) {
            throw std::invalid_argument("UstructModule::registerOn: invalid stabilization wave speed");
        }

        const auto he = FormExpr::constant(0.5) * FormExpr::cellDiameter();
        const auto c = FormExpr::constant(wave_speed);
        tau_m = FormExpr::constant(options_.ct_m) * (he / c) * (J / rho0);
        tau_c = FormExpr::constant(options_.ct_c) * (he * c) * (rho0 / J);
    }

    FormExpr acceleration_minus_body = zero_vector - body;
    FormExpr r_c = J - rho0 / rho;
    std::optional<FormExpr> kinematic_form;
    if (use_time_derivative_terms) {
        const auto gradx_v = grad(*v) * Finv;
        const auto divx_v = trace(gradx_v);
        acceleration_minus_body = (*v).dt(1) - body;
        r_c = beta * p.dt(1) + divx_v;
        kinematic_form =
            (FormExpr::constant(options_.kinematic_residual_scale) * inner(d.dt(1) - *v, z)).dx();
    }

    const auto r_m = rho * acceleration_minus_body + gradx_p;
    const auto Pdev = FE::forms::constitutive(options_.deviatoric_pk1_model, F).expr();

    const auto momentum_field = use_time_derivative_terms ? *v : d;
    const auto momentum_test = use_time_derivative_terms ? *w : z;
    const FE::FieldId momentum_field_id = use_time_derivative_terms ? *v_id : d_id;
    const auto virtual_div_x = trace(grad(momentum_test) * Finv);

    FormExpr momentum_form =
        (J * rho * inner(acceleration_minus_body, momentum_test)).dx() +
        doubleContraction(Pdev, grad(momentum_test)).dx() +
        (J * (-p) * virtual_div_x).dx() +
        (J * tau_c * r_c * virtual_div_x).dx();

    FormExpr continuity_form =
        (J * q * r_c).dx() +
        (J * tau_m * inner(gradx_q, r_m)).dx();

    for (const auto& bc : options_.follower_pressure) {
        momentum_form = momentum_form + Factories::followerPressureResidual(bc, F, momentum_test);
    }

    FE::systems::BoundaryConditionManager load_bc_manager;
    load_bc_manager.install(options_.traction_neumann, [&](const auto& bc) {
        return Factories::toTractionBC(bc, dim);
    });
    load_bc_manager.install(options_.normal_traction, Factories::toNormalTractionBC);
    load_bc_manager.applyAll(system, momentum_form, momentum_field, momentum_test, momentum_field_id);

    const auto dirichlet_bcs = Factories::prepareDirichletBCs(options_, dim);

    FE::systems::BoundaryConditionManager d_bc_manager;
    d_bc_manager.install(dirichlet_bcs.displacement, [&](const auto& bc) {
        return Factories::toDisplacementEssentialBC(bc, dim, options_.displacement_field_name);
    });
    if (kinematic_form) {
        d_bc_manager.applyAll(system, *kinematic_form, d, z, d_id);
    } else {
        d_bc_manager.applyAll(system, momentum_form, d, z, d_id);
    }

    if (v_id) {
        FE::systems::BoundaryConditionManager v_bc_manager;
        v_bc_manager.install(dirichlet_bcs.velocity, [&](const auto& bc) {
            return Factories::toVelocityEssentialBC(bc, dim, options_.velocity_field_name);
        });
        v_bc_manager.applyAll(system, *v_id);
    }

    FE::systems::BoundaryConditionManager p_bc_manager;
    p_bc_manager.install(options_.pressure_dirichlet, [&](const auto& bc) {
        return Factories::toPressureEssentialBC(bc, options_.pressure_field_name);
    });
    p_bc_manager.applyAll(system, p_id);

    FormExpr residual = momentum_form + continuity_form;
    if (kinematic_form) {
        residual = *kinematic_form + residual;
    }

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.use_symbolic_tangent = options_.use_symbolic_tangent;
    std::vector<FE::FieldId> active_fields{d_id};
    if (v_id) {
        active_fields.push_back(*v_id);
    }
    active_fields.push_back(p_id);

    (void)FE::systems::installFormulation(system, "equations", active_fields, residual, install);
}

} // namespace ustruct
} // namespace formulations
} // namespace Physics
} // namespace svmp
