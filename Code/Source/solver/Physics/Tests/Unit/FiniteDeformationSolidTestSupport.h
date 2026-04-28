/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_TESTS_UNIT_FINITE_DEFORMATION_SOLID_TEST_SUPPORT_H
#define SVMP_PHYSICS_TESTS_UNIT_FINITE_DEFORMATION_SOLID_TEST_SUPPORT_H

#include "FE/Forms/FiniteDeformationForms.h"
#include "FE/Forms/FormExpr.h"

#include <memory>
#include <utility>

namespace svmp {
namespace Physics {
namespace test_support {
namespace solid {

namespace forms = svmp::FE::forms;

struct FiniteDeformationSolidTerms {
    forms::FormExpr displacement{};
    forms::FormExpr test_displacement{};
    forms::FormExpr deformation_gradient{};
    forms::FormExpr first_piola{};
    forms::FormExpr internal_virtual_work{};
};

[[nodiscard]] inline FiniteDeformationSolidTerms totalLagrangianPK1Terms(
    const forms::FormExpr& displacement,
    const forms::FormExpr& test_displacement,
    std::shared_ptr<const forms::ConstitutiveModel> material,
    int dim = 3)
{
    FiniteDeformationSolidTerms out;
    out.displacement = displacement;
    out.test_displacement = test_displacement;
    out.deformation_gradient =
        forms::finite_deformation::deformationGradient(displacement, dim);
    out.first_piola = forms::FormExpr::constitutive(std::move(material),
                                                    out.deformation_gradient);
    out.internal_virtual_work =
        forms::finite_deformation::pk1InternalVirtualWorkDensity(
            out.first_piola, test_displacement);
    return out;
}

[[nodiscard]] inline forms::FormExpr totalLagrangianPK1Residual(
    const forms::FormExpr& displacement,
    const forms::FormExpr& test_displacement,
    std::shared_ptr<const forms::ConstitutiveModel> material,
    int dim = 3)
{
    return totalLagrangianPK1Terms(displacement,
                                   test_displacement,
                                   std::move(material),
                                   dim)
        .internal_virtual_work
        .dx();
}

[[nodiscard]] inline forms::FormExpr followerPressureReferenceResidual(
    const forms::FormExpr& pressure,
    const forms::FormExpr& test_displacement,
    const forms::FormExpr& deformation_gradient,
    int boundary_marker = -1)
{
    const auto nanson =
        forms::finite_deformation::nansonMeasureVector(deformation_gradient,
                                                       forms::FormExpr::normal());
    return -(pressure * forms::inner(nanson, test_displacement)).ds(boundary_marker);
}

[[nodiscard]] inline forms::FormExpr initialStressGeometricStiffnessResidual(
    const forms::FormExpr& initial_stress,
    const forms::FormExpr& trial_displacement,
    const forms::FormExpr& test_displacement)
{
    return forms::finite_deformation::initialStressGeometricStiffnessDensity(
               initial_stress, trial_displacement, test_displacement)
        .dx();
}

} // namespace solid
} // namespace test_support
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_TESTS_UNIT_FINITE_DEFORMATION_SOLID_TEST_SUPPORT_H
