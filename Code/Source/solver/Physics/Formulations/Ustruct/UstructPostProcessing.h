#ifndef SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_POSTPROCESSING_H
#define SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_POSTPROCESSING_H

/**
 * @file UstructPostProcessing.h
 * @brief Derived result registration helpers for mixed finite-deformation Ustruct.
 */

#include "FE/Forms/FiniteDeformationForms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/PostProcessing/DerivedResultBuilder.h"
#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/Ustruct/UstructModule.h"

#include <optional>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace ustruct {
namespace post {

inline FE::systems::FEQuantityShape symmetricVoigtShape(int dim)
{
    return FE::systems::FEQuantityShape{
        FE::systems::FEQuantityShapeKind::Vector,
        dim == 3 ? 6 : 3,
        dim};
}

inline FE::forms::FormExpr symmetricVoigt(const FE::forms::FormExpr& tensor, int dim)
{
    std::vector<FE::forms::FormExpr> components;
    components.reserve(dim == 3 ? 6u : 3u);
    components.push_back(tensor.component(0, 0));
    components.push_back(tensor.component(1, 1));
    if (dim == 3) {
        components.push_back(tensor.component(2, 2));
        components.push_back(tensor.component(0, 1));
        components.push_back(tensor.component(1, 2));
        components.push_back(tensor.component(2, 0));
    } else {
        components.push_back(tensor.component(0, 1));
    }
    return FE::forms::FormExpr::asVector(std::move(components));
}

inline void addNodePatchResult(FE::systems::FESystem& system,
                               const char* name,
                               FE::systems::FEQuantityShape shape,
                               FE::forms::FormExpr expression,
                               std::vector<FE::FieldId> referenced_fields)
{
    using namespace FE::post;

    system.addDerivedResult(
        DerivedResultBuilder(name)
            .scope(DerivedResultScope::Vertex)
            .policy(DerivedResultPolicy::PatchAverage)
            .shape(shape)
            .expression(std::move(expression))
            .referencedFields(std::move(referenced_fields))
            .build());
}

inline void registerPostProcessing(FE::systems::FESystem& system,
                                   FE::FieldId displacement_id,
                                   std::optional<FE::FieldId> velocity_id,
                                   FE::FieldId pressure_id,
                                   const FE::spaces::FunctionSpace& displacement_space,
                                   const FE::spaces::FunctionSpace& pressure_space,
                                   const UstructOptions& options)
{
    using namespace FE::forms;

    const bool any_requested =
        options.register_def_grad_output ||
        options.register_jacobian_output ||
        options.register_cauchy_stress_output ||
        options.register_divergence_output ||
        options.register_strain_output ||
        options.register_stress_output ||
        options.register_von_mises_stress_output;

    const int dim = displacement_space.value_dimension();
    const auto d = StateField(displacement_id, displacement_space, options.displacement_field_name);
    const auto p = StateField(pressure_id, pressure_space, options.pressure_field_name);
    const std::vector<FE::FieldId> displacement_ref{displacement_id};

    if (!velocity_id.has_value()) {
        std::vector<FormExpr> zero_components;
        zero_components.reserve(static_cast<std::size_t>(dim));
        for (int c = 0; c < dim; ++c) {
            zero_components.push_back(FormExpr::constant(0.0));
        }
        addNodePatchResult(system,
                           options.velocity_field_name.c_str(),
                           FE::systems::FEQuantityShape::vector(dim),
                           FormExpr::asVector(std::move(zero_components)),
                           displacement_ref);
    }

    if (!any_requested) {
        return;
    }

    std::optional<FormExpr> v;
    if (velocity_id.has_value()) {
        v.emplace(StateField(*velocity_id, displacement_space, options.velocity_field_name));
    }

    const auto kin = finite_deformation::kinematics(d, dim);
    const auto F = kin.F;
    const auto J = kin.J;
    const auto Finv = kin.Finv;
    const auto FinvT = kin.FinvT;
    const auto E = kin.green_lagrange;

    const auto Pdev = constitutive(options.deviatoric_pk1_model, F).expr();
    const auto P = Pdev - p * J * FinvT;
    const auto S = Finv * P;
    const auto sigma = (P * transpose(F)) / J;
    const auto sigma_mean = trace(sigma) / FormExpr::constant(static_cast<FE::Real>(dim));
    const auto sigma_dev = sigma - sigma_mean * FormExpr::identity(dim);
    const auto von_mises =
        sqrt(FormExpr::constant(static_cast<FE::Real>(1.5)) * doubleContraction(sigma_dev, sigma_dev));

    const std::vector<FE::FieldId> displacement_pressure_ref{displacement_id, pressure_id};
    std::vector<FE::FieldId> displacement_velocity_ref{displacement_id};
    if (velocity_id.has_value()) {
        displacement_velocity_ref.push_back(*velocity_id);
    }

    if (options.register_def_grad_output) {
        addNodePatchResult(system,
                           "Def_grad",
                           FE::systems::FEQuantityShape::tensor(dim),
                           F,
                           displacement_ref);
    }
    if (options.register_jacobian_output) {
        addNodePatchResult(system,
                           "Jacobian",
                           FE::systems::FEQuantityShape::scalar(),
                           J,
                           displacement_ref);
    }
    if (options.register_divergence_output) {
        const auto divergence = v.has_value()
                                    ? trace(grad(*v) * Finv)
                                    : FormExpr::constant(0.0);
        addNodePatchResult(system,
                           "Divergence",
                           FE::systems::FEQuantityShape::scalar(),
                           divergence,
                           displacement_velocity_ref);
    }
    if (options.register_strain_output) {
        addNodePatchResult(system,
                           "Strain",
                           symmetricVoigtShape(dim),
                           symmetricVoigt(E, dim),
                           displacement_ref);
    }
    if (options.register_stress_output) {
        addNodePatchResult(system,
                           "Stress",
                           symmetricVoigtShape(dim),
                           symmetricVoigt(S, dim),
                           displacement_pressure_ref);
    }
    if (options.register_cauchy_stress_output) {
        addNodePatchResult(system,
                           "Cauchy_stress",
                           symmetricVoigtShape(dim),
                           symmetricVoigt(sigma, dim),
                           displacement_pressure_ref);
    }
    if (options.register_von_mises_stress_output) {
        addNodePatchResult(system,
                           "VonMises_stress",
                           FE::systems::FEQuantityShape::scalar(),
                           von_mises,
                           displacement_pressure_ref);
    }
}

} // namespace post
} // namespace ustruct
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_POSTPROCESSING_H
