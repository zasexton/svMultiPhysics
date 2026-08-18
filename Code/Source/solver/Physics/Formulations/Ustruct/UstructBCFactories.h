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

namespace detail {

template <class BC>
[[nodiscard]] inline BC& findOrAppendByMarker(std::vector<BC>& bcs, int marker)
{
    for (auto& bc : bcs) {
        if (bc.boundary_marker == marker) {
            return bc;
        }
    }

    BC bc{};
    bc.boundary_marker = marker;
    bc.active_components = {false, false, false};
    bcs.push_back(std::move(bc));
    return bcs.back();
}

template <class BC>
inline void mergeComponentBC(std::vector<BC>& merged,
                             const BC& source,
                             int dim,
                             std::string_view description)
{
    auto& target = findOrAppendByMarker(merged, source.boundary_marker);
    for (int c = 0; c < dim; ++c) {
        const auto idx = static_cast<std::size_t>(c);
        if (!source.active_components[idx]) {
            continue;
        }
        if (target.active_components[idx]) {
            throw std::invalid_argument(
                std::string("ustruct::Factories::prepareDirichletBCs: duplicate ") + std::string(description) +
                " constraint for boundary marker " + std::to_string(source.boundary_marker) +
                ", component " + std::to_string(c));
        }
        target.active_components[idx] = true;
        target.value[idx] = source.value[idx];
    }
}

[[nodiscard]] inline bool hasActiveComponent(const UstructOptions::VelocityDirichletBC& bc,
                                             int component)
{
    return bc.active_components[static_cast<std::size_t>(component)];
}

[[nodiscard]] inline std::vector<UstructOptions::DisplacementDirichletBC>
mergeDisplacementDirichletBCs(const std::vector<UstructOptions::DisplacementDirichletBC>& input,
                              int dim)
{
    std::vector<UstructOptions::DisplacementDirichletBC> merged;
    merged.reserve(input.size());
    for (const auto& bc : input) {
        mergeComponentBC(merged, bc, dim, "displacement Dirichlet");
    }
    return merged;
}

[[nodiscard]] inline std::vector<UstructOptions::VelocityDirichletBC>
mergeVelocityDirichletBCs(const std::vector<UstructOptions::VelocityDirichletBC>& input,
                          int dim)
{
    std::vector<UstructOptions::VelocityDirichletBC> merged;
    merged.reserve(input.size());
    for (const auto& bc : input) {
        mergeComponentBC(merged, bc, dim, "velocity Dirichlet");
    }
    return merged;
}

inline void validateQuasiStaticVelocityDirichletBCs(
    const std::vector<UstructOptions::VelocityDirichletBC>& velocity_input,
    int dim)
{
    for (const auto& bc : velocity_input) {
        for (int c = 0; c < dim; ++c) {
            const auto idx = static_cast<std::size_t>(c);
            if (!bc.active_components[idx]) {
                continue;
            }
            if (!FE::forms::bc::isZeroConstantScalarValue(bc.value[idx])) {
                throw std::invalid_argument(
                    "ustruct::Factories::prepareDirichletBCs: quasi-static ustruct disables time-derivative terms, "
                    "so velocity Dirichlet constraints must be constant zero values");
            }
        }
    }
}

[[nodiscard]] inline std::vector<UstructOptions::VelocityDirichletBC>
compatibleVelocityDirichletBCs(
    const std::vector<UstructOptions::VelocityDirichletBC>& velocity_input,
    const std::vector<UstructOptions::DisplacementDirichletBC>& displacement_input,
    int dim)
{
    std::vector<UstructOptions::VelocityDirichletBC> merged;
    merged.reserve(velocity_input.size() + displacement_input.size());

    for (const auto& bc : velocity_input) {
        mergeComponentBC(merged, bc, dim, "velocity Dirichlet");
    }

    for (const auto& disp : displacement_input) {
        auto& velocity = findOrAppendByMarker(merged, disp.boundary_marker);
        for (int c = 0; c < dim; ++c) {
            const auto idx = static_cast<std::size_t>(c);
            if (!disp.active_components[idx]) {
                continue;
            }

            if (hasActiveComponent(velocity, c)) {
                if (FE::forms::bc::isConstantScalarValue(disp.value[idx]) &&
                    !FE::forms::bc::isZeroConstantScalarValue(velocity.value[idx])) {
                    throw std::invalid_argument(
                        "ustruct::Factories::prepareDirichletBCs: fixed displacement Dirichlet constraints require "
                        "zero velocity Dirichlet constraints on the same components");
                }
                continue;
            }

            if (!FE::forms::bc::isConstantScalarValue(disp.value[idx])) {
                throw std::invalid_argument(
                    "ustruct::Factories::prepareDirichletBCs: time- or space-dependent displacement Dirichlet "
                    "constraints require an explicit compatible velocity Dirichlet constraint");
            }

            velocity.active_components[idx] = true;
            velocity.value[idx] = UstructOptions::ScalarValue{FE::Real{0.0}};
        }
    }

    return merged;
}

} // namespace detail

struct PreparedDirichletBCs {
    std::vector<UstructOptions::DisplacementDirichletBC> displacement{};
    std::vector<UstructOptions::VelocityDirichletBC> velocity{};
};

[[nodiscard]] inline PreparedDirichletBCs prepareDirichletBCs(const UstructOptions& options,
                                                             int dim)
{
    PreparedDirichletBCs prepared;
    prepared.displacement = detail::mergeDisplacementDirichletBCs(options.displacement_dirichlet, dim);
    prepared.velocity =
        options.enable_time_derivative_terms
            ? detail::compatibleVelocityDirichletBCs(options.velocity_dirichlet, prepared.displacement, dim)
            : detail::mergeVelocityDirichletBCs(options.velocity_dirichlet, dim);
    if (!options.enable_time_derivative_terms) {
        detail::validateQuasiStaticVelocityDirichletBCs(prepared.velocity, dim);
    }
    return prepared;
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toDisplacementEssentialBC(const UstructOptions::DisplacementDirichletBC& bc,
                          int dim,
                          std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "ustruct::Factories::toDisplacementEssentialBC");
    return FE::forms::bc::vectorComponentEssentialBC(
        bc.value, bc.active_components, dim, marker, "ustruct_dirichlet", symbol, "Ustruct displacement Dirichlet");
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toVelocityEssentialBC(const UstructOptions::VelocityDirichletBC& bc,
                      int dim,
                      std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "ustruct::Factories::toVelocityEssentialBC");
    return FE::forms::bc::vectorComponentEssentialBC(
        bc.value, bc.active_components, dim, marker, "ustruct_velocity_dirichlet", symbol, "Ustruct velocity Dirichlet");
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toPressureEssentialBC(const UstructOptions::PressureDirichletBC& bc,
                      std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "ustruct::Factories::toPressureEssentialBC");
    const auto value =
        FE::forms::bc::toScalarExpr(bc.value, FE::forms::bc::markerValueName("ustruct_pressure_dirichlet", marker));
    return std::make_unique<FE::forms::bc::EssentialBC>(marker, value, std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toTractionBC(const UstructOptions::TractionNeumannBC& bc,
             int dim)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "ustruct::Factories::toTractionBC");
    const auto traction =
        FE::forms::FormExpr::asVector(
            FE::forms::bc::toVectorExpr(bc.traction, dim, "ustruct_traction", marker));
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, traction);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toNormalTractionBC(const UstructOptions::NormalTractionBC& bc)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "ustruct::Factories::toNormalTractionBC");
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
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "ustruct::Factories::followerPressureResidual");
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
