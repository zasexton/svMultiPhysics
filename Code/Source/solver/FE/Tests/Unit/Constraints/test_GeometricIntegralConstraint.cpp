/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constraints/GeometricIntegralConstraint.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <memory>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(GeometricIntegralConstraint, AcceptsProductionSensitivityForNonlinearTangents)
{
    GeometricIntegralConstraintSpec spec{};
    spec.quantity = GeometricIntegralQuantity::EnclosedVolume;
    spec.sensitivity = GeometricConstraintSensitivityPath::Analytic;
    spec.contributes_to_tangent = true;

    EXPECT_TRUE(geometricIntegralConstraintSpecIsValid(spec));

    spec.sensitivity = GeometricConstraintSensitivityPath::Symbolic;
    EXPECT_TRUE(geometricIntegralConstraintSpecIsValid(spec));

    spec.sensitivity = GeometricConstraintSensitivityPath::AD;
    EXPECT_TRUE(geometricIntegralConstraintSpecIsValid(spec));

    spec.sensitivity = GeometricConstraintSensitivityPath::JIT;
    EXPECT_TRUE(geometricIntegralConstraintSpecIsValid(spec));
}

TEST(GeometricIntegralConstraint, RejectsFiniteDifferenceProductionTangents)
{
    GeometricIntegralConstraintSpec spec{};
    spec.quantity = GeometricIntegralQuantity::SurfaceArea;
    spec.boundary_marker = 7;
    spec.sensitivity = GeometricConstraintSensitivityPath::VerificationFiniteDifference;
    spec.contributes_to_tangent = true;

    EXPECT_FALSE(isProductionSensitivity(spec.sensitivity));
    EXPECT_FALSE(geometricIntegralConstraintSpecIsValid(spec));

    spec.contributes_to_tangent = false;
    spec.contributes_to_residual = true;
    EXPECT_TRUE(geometricIntegralConstraintSpecIsValid(spec));
}

TEST(GeometricIntegralConstraint, NamesAndBoundaryRequirementsAreStable)
{
    EXPECT_TRUE(quantityRequiresBoundary(GeometricIntegralQuantity::SurfaceArea));
    EXPECT_TRUE(quantityRequiresBoundary(
        GeometricIntegralQuantity::AverageBoundaryDisplacement));
    EXPECT_FALSE(quantityRequiresBoundary(GeometricIntegralQuantity::EnclosedVolume));

    EXPECT_STREQ(geometricIntegralQuantityName(GeometricIntegralQuantity::CenterOfMass),
                 "CenterOfMass");
    EXPECT_STREQ(geometricConstraintSensitivityPathName(
                     GeometricConstraintSensitivityPath::VerificationFiniteDifference),
                 "VerificationFiniteDifference");
    EXPECT_STREQ(geometricConstraintStateLevelName(
                     GeometricConstraintStateLevel::AcceptedTimeStep),
                 "AcceptedTimeStep");
}

TEST(GeometricIntegralConstraint, FormHelpersBuildSymbolicResidualsForConsistentTangents)
{
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector_space = std::make_shared<spaces::ProductSpace>(scalar_space, 3);

    const FieldId u_id{0};
    const auto u = forms::FormExpr::stateField(u_id, *vector_space, "u");
    const auto F = forms::finite_deformation::deformationGradient(u, 3);

    GeometricIntegralConstraintSpec volume{};
    volume.quantity = GeometricIntegralQuantity::EnclosedVolume;
    volume.target_value = 1.0;
    const auto volume_residual = geometricIntegralResidual(volume, u, F);
    EXPECT_TRUE(volume_residual.isValid());

    GeometricIntegralConstraintSpec area{};
    area.quantity = GeometricIntegralQuantity::SurfaceArea;
    area.boundary_marker = 3;
    area.target_value = 1.0;
    const auto area_residual = geometricIntegralResidual(area, u, F);
    EXPECT_TRUE(area_residual.isValid());

    GeometricIntegralConstraintSpec average_displacement{};
    average_displacement.quantity = GeometricIntegralQuantity::AverageBoundaryDisplacement;
    average_displacement.boundary_marker = 4;
    average_displacement.component = 1;
    average_displacement.target_value = 0.25;
    const auto displacement_residual =
        geometricIntegralResidual(average_displacement, u, F);
    EXPECT_TRUE(displacement_residual.isValid());

    GeometricIntegralConstraintSpec center_of_mass{};
    center_of_mass.quantity = GeometricIntegralQuantity::CenterOfMass;
    center_of_mass.component = 0;
    center_of_mass.target_value = 0.5;
    const auto center_residual = geometricIntegralResidual(center_of_mass, u, F);
    EXPECT_TRUE(center_residual.isValid());

    GeometricIntegralConstraintSpec moment{};
    moment.quantity = GeometricIntegralQuantity::GeometricMoment;
    moment.boundary_marker = 5;
    moment.component = 2;
    moment.moment_order = 2;
    moment.target_value = 0.75;
    const auto moment_residual = geometricIntegralResidual(moment, u, F);
    EXPECT_TRUE(moment_residual.isValid());
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
