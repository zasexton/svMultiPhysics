/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_FINITE_DEFORMATION_KINEMATICS_H
#define SVMP_FE_GEOMETRY_FINITE_DEFORMATION_KINEMATICS_H

/**
 * @file FiniteDeformationKinematics.h
 * @brief Physics-neutral finite-deformation kinematic utilities.
 *
 * The routines in this file define geometric quantities only. Constitutive
 * equations, structural formulations, load stepping, and material updates must
 * remain in Physics or Constitutive layers.
 */

#include "Core/Types.h"
#include "Geometry/FrameAwareTransform.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"

#include <cstdint>

namespace svmp {
namespace FE {
namespace geometry {

enum class KinematicGradientFrame : std::uint8_t {
    Reference,
    Current
};

enum class FiniteDeformationReferencePolicy : std::uint8_t {
    TotalLagrangian,
    UpdatedLagrangian
};

struct FiniteDeformationKinematics {
    math::Matrix<Real, 3, 3> F{};
    math::Matrix<Real, 3, 3> Finv{};
    math::Matrix<Real, 3, 3> FinvT{};
    Real J{1.0};
    math::Matrix<Real, 3, 3> C{};
    math::Matrix<Real, 3, 3> b{};
    math::Matrix<Real, 3, 3> green_lagrange{};
    math::Matrix<Real, 3, 3> almansi{};
    int dim{3};
    FiniteDeformationReferencePolicy reference_policy{
        FiniteDeformationReferencePolicy::TotalLagrangian};
};

struct FiniteDeformationIncrement {
    math::Matrix<Real, 3, 3> dF{};
    math::Matrix<Real, 3, 3> dFinv{};
    math::Matrix<Real, 3, 3> dFinvT{};
    Real dJ{0.0};
    math::Matrix<Real, 3, 3> dC{};
    math::Matrix<Real, 3, 3> db{};
    math::Matrix<Real, 3, 3> dGreenLagrange{};
    math::Matrix<Real, 3, 3> dAlmansi{};
};

struct NansonLinearization {
    math::Vector<Real, 3> oriented_measure_vector_derivative{};
    math::Vector<Real, 3> normal_derivative{};
    Real measure_derivative{0.0};
};

[[nodiscard]] math::Matrix<Real, 3, 3> finiteDeformationIdentity(int dim = 3);

[[nodiscard]] FiniteDeformationKinematics finiteDeformationFromGradient(
    const math::Matrix<Real, 3, 3>& deformation_gradient,
    int dim = 3,
    FiniteDeformationReferencePolicy reference_policy =
        FiniteDeformationReferencePolicy::TotalLagrangian);

[[nodiscard]] FiniteDeformationKinematics finiteDeformationFromDisplacementGradient(
    const math::Matrix<Real, 3, 3>& displacement_gradient_reference,
    int dim = 3,
    FiniteDeformationReferencePolicy reference_policy =
        FiniteDeformationReferencePolicy::TotalLagrangian);

[[nodiscard]] FiniteDeformationKinematics finiteDeformationFromReferenceAndCurrentJacobians(
    const math::Matrix<Real, 3, 3>& current_jacobian,
    const math::Matrix<Real, 3, 3>& reference_inverse_jacobian,
    int dim = 3,
    FiniteDeformationReferencePolicy reference_policy =
        FiniteDeformationReferencePolicy::TotalLagrangian);

[[nodiscard]] math::Matrix<Real, 3, 3> rightCauchyGreen(
    const math::Matrix<Real, 3, 3>& F);

[[nodiscard]] math::Matrix<Real, 3, 3> leftCauchyGreen(
    const math::Matrix<Real, 3, 3>& F);

[[nodiscard]] math::Matrix<Real, 3, 3> greenLagrangeStrain(
    const math::Matrix<Real, 3, 3>& F,
    int dim = 3);

[[nodiscard]] math::Matrix<Real, 3, 3> almansiStrain(
    const math::Matrix<Real, 3, 3>& F,
    int dim = 3);

[[nodiscard]] FiniteDeformationIncrement linearizeFiniteDeformationKinematics(
    const FiniteDeformationKinematics& kinematics,
    const math::Matrix<Real, 3, 3>& dF);

[[nodiscard]] DeformationFrame deformationFrame(
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Vector<Real, 3> currentGradientFromReferenceGradient(
    const math::Vector<Real, 3>& gradient_reference,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Vector<Real, 3> referenceGradientFromCurrentGradient(
    const math::Vector<Real, 3>& gradient_current,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Matrix<Real, 3, 3> currentVectorGradientFromReferenceGradient(
    const math::Matrix<Real, 3, 3>& gradient_reference,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Matrix<Real, 3, 3> referenceVectorGradientFromCurrentGradient(
    const math::Matrix<Real, 3, 3>& gradient_current,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Vector<Real, 3> pushForwardVector(
    FEFieldTransformFamily family,
    const math::Vector<Real, 3>& value_reference,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Vector<Real, 3> pullBackVector(
    FEFieldTransformFamily family,
    const math::Vector<Real, 3>& value_current,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Matrix<Real, 3, 3> pushForwardTensor(
    TensorFrameTransform transform,
    const math::Matrix<Real, 3, 3>& tensor_reference,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] math::Matrix<Real, 3, 3> pullBackTensor(
    TensorFrameTransform transform,
    const math::Matrix<Real, 3, 3>& tensor_current,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] SurfaceMeasureTransform nansonSurfaceTransform(
    const math::Vector<Real, 3>& reference_normal,
    Real reference_measure,
    const FiniteDeformationKinematics& kinematics);

[[nodiscard]] NansonLinearization linearizeNansonSurfaceTransform(
    const math::Vector<Real, 3>& reference_normal,
    Real reference_measure,
    const FiniteDeformationKinematics& kinematics,
    const math::Matrix<Real, 3, 3>& dF);

[[nodiscard]] const char* gradientFrameName(KinematicGradientFrame frame) noexcept;
[[nodiscard]] const char* referencePolicyName(FiniteDeformationReferencePolicy policy) noexcept;

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_FINITE_DEFORMATION_KINEMATICS_H
