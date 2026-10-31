#ifndef SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_BC_FACTORIES_H

/**
 * @file UstructBCFactories.h
 * @brief Boundary-condition factories for the OOP ustruct formulation
 */

#include "Physics/Formulations/Ustruct/UstructModule.h"

#include "FE/Forms/BoundaryCondition.h"
#include "FE/Forms/FiniteDeformationForms.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace ustruct {
namespace Factories {

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::Dirichlet>
toDisplacementEssentialBC(const UstructOptions::DisplacementDirichletBC& bc,
                          int dim,
                          std::string_view symbol)
{
    const int marker =
        FE::forms::bc::boundaryMarkerOrThrow(bc, "ustruct::Factories::toDisplacementEssentialBC");
    return FE::forms::bc::componentDirichlet(
        bc.value, bc.active_components, dim, marker, "ustruct_dirichlet", symbol, "Ustruct displacement Dirichlet");
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::Dirichlet>
toVelocityEssentialBC(const UstructOptions::VelocityDirichletBC& bc,
                      int dim,
                      std::string_view symbol)
{
    const int marker =
        FE::forms::bc::boundaryMarkerOrThrow(bc, "ustruct::Factories::toVelocityEssentialBC");
    return FE::forms::bc::componentDirichlet(
        bc.value, bc.active_components, dim, marker, "ustruct_velocity_dirichlet", symbol, "Ustruct velocity Dirichlet");
}

struct DirichletConditions {
    std::vector<std::unique_ptr<FE::forms::bc::Dirichlet>> displacement{};
    std::vector<std::unique_ptr<FE::forms::bc::Dirichlet>> velocity{};
};

/// The strong conditions of the displacement and of the velocity, the
/// components of each boundary merged (forms::bc::mergeComponents).  With the
/// time-derivative terms the displacement conditions imply velocity
/// conditions (forms::bc::derivedRateConditions); without them a velocity
/// condition holds the velocity at zero.
[[nodiscard]] inline DirichletConditions dirichletConditions(const UstructOptions& options, int dim)
{
    if (!options.enable_time_derivative_terms) {
        for (const auto& bc : options.velocity_dirichlet) {
            for (int c = 0; c < dim; ++c) {
                const auto idx = static_cast<std::size_t>(c);
                if (bc.active_components[idx] && !FE::forms::bc::isZeroConstantScalarValue(bc.value[idx])) {
                    throw std::invalid_argument(
                        "ustruct::Factories::dirichletConditions: quasi-static ustruct disables time-derivative "
                        "terms, so velocity Dirichlet constraints must be constant zero values");
                }
            }
        }
    }
    std::vector<std::unique_ptr<FE::forms::bc::Dirichlet>> displacement;
    std::vector<std::unique_ptr<FE::forms::bc::Dirichlet>> velocity;
    for (const auto& bc : options.displacement_dirichlet) {
        displacement.push_back(toDisplacementEssentialBC(bc, dim, options.displacement_field_name));
    }
    for (const auto& bc : options.velocity_dirichlet) {
        velocity.push_back(toVelocityEssentialBC(bc, dim, options.velocity_field_name));
    }
    DirichletConditions conditions;
    conditions.displacement = FE::forms::bc::mergeComponents(std::move(displacement));
    conditions.velocity = options.enable_time_derivative_terms
                              ? FE::forms::bc::derivedRateConditions(conditions.displacement, std::move(velocity),
                                                                     options.velocity_field_name,
                                                                     "Ustruct velocity Dirichlet")
                              : FE::forms::bc::mergeComponents(std::move(velocity));
    return conditions;
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toPressureEssentialBC(const UstructOptions::PressureDirichletBC& bc,
                      std::string_view symbol)
{
    const int marker =
        FE::forms::bc::boundaryMarkerOrThrow(bc, "ustruct::Factories::toPressureEssentialBC");
    const auto value =
        FE::forms::bc::toScalarExpr(bc.value, FE::forms::bc::markerValueName("ustruct_pressure_dirichlet", marker));
    return std::make_unique<FE::forms::bc::Dirichlet>(marker, value, FE::forms::bc::Strong{}, FE::forms::bc::NaturalFlux{},
                                                      std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toTractionBC(const UstructOptions::TractionNeumannBC& bc,
             int dim)
{
    const int marker =
        FE::forms::bc::boundaryMarkerOrThrow(bc, "ustruct::Factories::toTractionBC");
    const auto traction =
        FE::forms::FormExpr::asVector(
            FE::forms::bc::toVectorExpr(bc.traction, dim, "ustruct_traction", marker));
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, traction);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toNormalTractionBC(const UstructOptions::NormalTractionBC& bc)
{
    const int marker =
        FE::forms::bc::boundaryMarkerOrThrow(bc, "ustruct::Factories::toNormalTractionBC");
    const auto traction =
        FE::forms::bc::toScalarExpr(bc.traction, FE::forms::bc::markerValueName("ustruct_normal_traction", marker));
    return std::make_unique<FE::forms::bc::NaturalBC>(
        marker, traction * FE::forms::FormExpr::normal());
}

[[nodiscard]] inline FE::forms::FormExpr followerPressureExpression(
    const UstructOptions::FollowerPressureBC& bc,
    std::string_view name)
{
    if (bc.ramp.has_value()) {
        const auto& ramp = *bc.ramp;
        if (!(ramp.end_time > ramp.start_time)) {
            throw std::invalid_argument(
                "ustruct::Factories::followerPressureExpression: follower-pressure ramp end_time must be greater than start_time");
        }
        const auto alpha = FE::forms::clamp(
            (FE::forms::FormExpr::time() - FE::forms::FormExpr::constant(ramp.start_time)) /
                FE::forms::FormExpr::constant(ramp.end_time - ramp.start_time),
            FE::forms::FormExpr::constant(0.0),
            FE::forms::FormExpr::constant(1.0));
        return FE::forms::FormExpr::constant(ramp.start_value) +
               alpha * FE::forms::FormExpr::constant(ramp.end_value - ramp.start_value);
    }
    return FE::forms::bc::toScalarExpr(bc.pressure, name);
}

[[nodiscard]] inline FE::forms::FormExpr followerPressureResidual(
    const UstructOptions::FollowerPressureBC& bc,
    const FE::forms::FormExpr& deformation_gradient,
    const FE::forms::FormExpr& test_displacement)
{
    const int marker =
        FE::forms::bc::boundaryMarkerOrThrow(bc, "ustruct::Factories::followerPressureResidual");
    const auto pressure = followerPressureExpression(
        bc, FE::forms::bc::markerValueName("ustruct_follower_pressure", marker));
    const auto nanson =
        FE::forms::finite_deformation::nansonMeasureVector(deformation_gradient, FE::forms::FormExpr::normal());
    return (pressure * FE::forms::inner(nanson, test_displacement)).ds(marker);
}

} // namespace Factories
} // namespace ustruct
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_BC_FACTORIES_H
