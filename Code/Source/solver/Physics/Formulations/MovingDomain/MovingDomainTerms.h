#ifndef SVMP_PHYSICS_FORMULATIONS_MOVINGDOMAIN_MOVING_DOMAIN_TERMS_H
#define SVMP_PHYSICS_FORMULATIONS_MOVINGDOMAIN_MOVING_DOMAIN_TERMS_H

#include "FE/Forms/Vocabulary.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace moving_domain {

namespace forms = svmp::FE::forms;

[[nodiscard]] inline forms::FormExpr zeroVector(int dim)
{
    std::vector<forms::FormExpr> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        components.push_back(forms::FormExpr::constant(0.0));
    }
    return forms::FormExpr::asVector(std::move(components));
}

[[nodiscard]] inline forms::FormExpr domainVelocityVector(int dim)
{
    std::vector<forms::FormExpr> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        components.push_back(forms::component(forms::domainVelocity(), d));
    }
    return forms::FormExpr::asVector(std::move(components));
}

[[nodiscard]] inline forms::FormExpr domainVelocityOrZero(int dim, bool ale_enabled)
{
    return ale_enabled ? domainVelocityVector(dim) : zeroVector(dim);
}

[[nodiscard]] inline forms::FormExpr relativeConvectiveVelocity(const forms::FormExpr& material_velocity,
                                                               int dim,
                                                               bool ale_enabled)
{
    return ale_enabled ? (material_velocity - domainVelocityVector(dim)) : material_velocity;
}

[[nodiscard]] inline forms::FormExpr movingControlVolumeScalarTransient(const forms::FormExpr& field,
                                                                       const forms::FormExpr& test,
                                                                       const forms::FormExpr& density,
                                                                       int dim,
                                                                       bool enabled)
{
    if (!enabled) {
        return forms::FormExpr::constant(0.0);
    }
    return density * forms::div(domainVelocityVector(dim)) * field * test;
}

[[nodiscard]] inline forms::FormExpr movingControlVolumeVectorTransient(const forms::FormExpr& field,
                                                                       const forms::FormExpr& test,
                                                                       const forms::FormExpr& density,
                                                                       int dim,
                                                                       bool enabled)
{
    if (!enabled) {
        return forms::FormExpr::constant(0.0);
    }
    return density * forms::div(domainVelocityVector(dim)) * forms::inner(field, test);
}

[[nodiscard]] inline forms::FormExpr movingBoundaryKinematicResidual(const forms::FormExpr& physical_velocity,
                                                                    const forms::FormExpr& test_scalar)
{
    return test_scalar * forms::dot(physical_velocity - forms::domainVelocity(),
                                    forms::currentNormal()) *
           forms::currentMeasure();
}

[[nodiscard]] inline forms::FormExpr fsiDisplacementCompatibilityResidual(
    const forms::FormExpr& structural_displacement,
    const forms::FormExpr& test_scalar)
{
    return test_scalar * forms::dot(structural_displacement - forms::meshDisplacement(),
                                    forms::currentNormal()) *
           forms::currentMeasure();
}

[[nodiscard]] inline forms::FormExpr fsiSurfaceTractionPowerResidual(
    const forms::FormExpr& current_traction,
    const forms::FormExpr& interface_velocity_test)
{
    return forms::inner(current_traction, interface_velocity_test) * forms::currentMeasure();
}

[[nodiscard]] inline forms::FormExpr referenceSurfaceMeasureMismatchProbe()
{
    return forms::currentMeasure() - forms::referenceMeasure() +
           forms::dot(forms::currentNormal() - forms::referenceNormal(),
                      forms::currentNormal() - forms::referenceNormal());
}

} // namespace moving_domain
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_MOVINGDOMAIN_MOVING_DOMAIN_TERMS_H
