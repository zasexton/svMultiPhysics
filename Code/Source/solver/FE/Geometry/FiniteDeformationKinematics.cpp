/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/FiniteDeformationKinematics.h"

#include "Core/FEException.h"

#include <cmath>
#include <string>

namespace svmp {
namespace FE {
namespace geometry {
namespace {

constexpr Real kKinematicsTol = Real(1e-14);

void requireDim(int dim, const char* caller)
{
    FE_THROW_IF(dim < 1 || dim > 3, FEException,
                std::string(caller) + " requires dimension 1, 2, or 3");
}

math::Matrix<Real, 3, 3> identity3()
{
    math::Matrix<Real, 3, 3> I{};
    I(0, 0) = Real(1);
    I(1, 1) = Real(1);
    I(2, 2) = Real(1);
    return I;
}

math::Matrix<Real, 3, 3> effectiveGradient(
    const math::Matrix<Real, 3, 3>& F,
    int dim)
{
    auto out = identity3();
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out(i, j) = F(i, j);
        }
    }
    return out;
}

math::Matrix<Real, 3, 3> effectiveIncrement(
    const math::Matrix<Real, 3, 3>& dF,
    int dim)
{
    math::Matrix<Real, 3, 3> out{};
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out(i, j) = dF(i, j);
        }
    }
    return out;
}

Real traceDim(const math::Matrix<Real, 3, 3>& A, int dim)
{
    Real out = Real(0);
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < n; ++i) {
        out += A(i, i);
    }
    return out;
}

math::Vector<Real, 3> transformVector(
    const math::Matrix<Real, 3, 3>& A,
    const math::Vector<Real, 3>& v,
    int dim)
{
    math::Vector<Real, 3> out{};
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out[i] += A(i, j) * v[j];
        }
    }
    return out;
}

math::Matrix<Real, 3, 3> rightMultiplyDim(
    const math::Matrix<Real, 3, 3>& A,
    const math::Matrix<Real, 3, 3>& B,
    int dim)
{
    math::Matrix<Real, 3, 3> out{};
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            for (std::size_t a = 0; a < n; ++a) {
                out(r, c) += A(r, a) * B(a, c);
            }
        }
    }
    return out;
}

math::Vector<Real, 3> unitOrZero(const math::Vector<Real, 3>& v)
{
    const Real norm = v.norm();
    if (norm <= kKinematicsTol) {
        return math::Vector<Real, 3>{};
    }
    return v / norm;
}

} // namespace

math::Matrix<Real, 3, 3> finiteDeformationIdentity(int dim)
{
    requireDim(dim, "finiteDeformationIdentity");
    return identity3();
}

FiniteDeformationKinematics finiteDeformationFromGradient(
    const math::Matrix<Real, 3, 3>& deformation_gradient,
    int dim,
    FiniteDeformationReferencePolicy reference_policy)
{
    requireDim(dim, "finiteDeformationFromGradient");

    FiniteDeformationKinematics out;
    out.dim = dim;
    out.reference_policy = reference_policy;
    out.F = effectiveGradient(deformation_gradient, dim);
    out.J = out.F.determinant();
    FE_THROW_IF(std::abs(out.J) <= kKinematicsTol, FEException,
                "finiteDeformationFromGradient encountered a singular deformation gradient");
    out.Finv = out.F.inverse();
    out.FinvT = out.Finv.transpose();
    out.C = rightCauchyGreen(out.F);
    out.b = leftCauchyGreen(out.F);
    out.green_lagrange = greenLagrangeStrain(out.F, dim);
    out.almansi = almansiStrain(out.F, dim);
    return out;
}

FiniteDeformationKinematics finiteDeformationFromDisplacementGradient(
    const math::Matrix<Real, 3, 3>& displacement_gradient_reference,
    int dim,
    FiniteDeformationReferencePolicy reference_policy)
{
    requireDim(dim, "finiteDeformationFromDisplacementGradient");
    auto F = identity3();
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            F(i, j) += displacement_gradient_reference(i, j);
        }
    }
    return finiteDeformationFromGradient(F, dim, reference_policy);
}

FiniteDeformationKinematics finiteDeformationFromReferenceAndCurrentJacobians(
    const math::Matrix<Real, 3, 3>& current_jacobian,
    const math::Matrix<Real, 3, 3>& reference_inverse_jacobian,
    int dim,
    FiniteDeformationReferencePolicy reference_policy)
{
    requireDim(dim, "finiteDeformationFromReferenceAndCurrentJacobians");
    const math::Matrix<Real, 3, 3> F = current_jacobian * reference_inverse_jacobian;
    return finiteDeformationFromGradient(F, dim, reference_policy);
}

math::Matrix<Real, 3, 3> rightCauchyGreen(const math::Matrix<Real, 3, 3>& F)
{
    return F.transpose() * F;
}

math::Matrix<Real, 3, 3> leftCauchyGreen(const math::Matrix<Real, 3, 3>& F)
{
    return F * F.transpose();
}

math::Matrix<Real, 3, 3> greenLagrangeStrain(
    const math::Matrix<Real, 3, 3>& F,
    int dim)
{
    requireDim(dim, "greenLagrangeStrain");
    return (rightCauchyGreen(effectiveGradient(F, dim)) - identity3()) * Real(0.5);
}

math::Matrix<Real, 3, 3> almansiStrain(
    const math::Matrix<Real, 3, 3>& F,
    int dim)
{
    requireDim(dim, "almansiStrain");
    const auto b_inv = leftCauchyGreen(effectiveGradient(F, dim)).inverse();
    return (identity3() - b_inv) * Real(0.5);
}

FiniteDeformationIncrement linearizeFiniteDeformationKinematics(
    const FiniteDeformationKinematics& kinematics,
    const math::Matrix<Real, 3, 3>& dF_input)
{
    requireDim(kinematics.dim, "linearizeFiniteDeformationKinematics");

    FiniteDeformationIncrement out;
    out.dF = effectiveIncrement(dF_input, kinematics.dim);
    const math::Matrix<Real, 3, 3> Finv_dF = kinematics.Finv * out.dF;
    out.dJ = kinematics.J * traceDim(Finv_dF, kinematics.dim);
    out.dFinv = (kinematics.Finv * out.dF * kinematics.Finv) * Real(-1);
    out.dFinvT = out.dFinv.transpose();
    out.dC = out.dF.transpose() * kinematics.F + kinematics.F.transpose() * out.dF;
    out.db = out.dF * kinematics.F.transpose() + kinematics.F * out.dF.transpose();
    out.dGreenLagrange = out.dC * Real(0.5);

    const auto b_inv = kinematics.b.inverse();
    out.dAlmansi = (b_inv * out.db * b_inv) * Real(0.5);
    return out;
}

DeformationFrame deformationFrame(const FiniteDeformationKinematics& kinematics)
{
    DeformationFrame frame;
    frame.J = kinematics.F;
    frame.Jinv = kinematics.Finv;
    frame.detJ = kinematics.J;
    frame.dim = kinematics.dim;
    return frame;
}

math::Vector<Real, 3> currentGradientFromReferenceGradient(
    const math::Vector<Real, 3>& gradient_reference,
    const FiniteDeformationKinematics& kinematics)
{
    return transformVector(kinematics.FinvT, gradient_reference, kinematics.dim);
}

math::Vector<Real, 3> referenceGradientFromCurrentGradient(
    const math::Vector<Real, 3>& gradient_current,
    const FiniteDeformationKinematics& kinematics)
{
    return transformVector(kinematics.F.transpose(), gradient_current, kinematics.dim);
}

math::Matrix<Real, 3, 3> currentVectorGradientFromReferenceGradient(
    const math::Matrix<Real, 3, 3>& gradient_reference,
    const FiniteDeformationKinematics& kinematics)
{
    return rightMultiplyDim(gradient_reference, kinematics.Finv, kinematics.dim);
}

math::Matrix<Real, 3, 3> referenceVectorGradientFromCurrentGradient(
    const math::Matrix<Real, 3, 3>& gradient_current,
    const FiniteDeformationKinematics& kinematics)
{
    return rightMultiplyDim(gradient_current, kinematics.F, kinematics.dim);
}

math::Vector<Real, 3> pushForwardVector(
    FEFieldTransformFamily family,
    const math::Vector<Real, 3>& value_reference,
    const FiniteDeformationKinematics& kinematics)
{
    return FrameAwareTransform::pushForwardValue(family, value_reference,
                                                 deformationFrame(kinematics));
}

math::Vector<Real, 3> pullBackVector(
    FEFieldTransformFamily family,
    const math::Vector<Real, 3>& value_current,
    const FiniteDeformationKinematics& kinematics)
{
    return FrameAwareTransform::pullBackValue(family, value_current,
                                              deformationFrame(kinematics));
}

math::Matrix<Real, 3, 3> pushForwardTensor(
    TensorFrameTransform transform,
    const math::Matrix<Real, 3, 3>& tensor_reference,
    const FiniteDeformationKinematics& kinematics)
{
    return FrameAwareTransform::pushForwardTensor(transform, tensor_reference,
                                                  deformationFrame(kinematics));
}

math::Matrix<Real, 3, 3> pullBackTensor(
    TensorFrameTransform transform,
    const math::Matrix<Real, 3, 3>& tensor_current,
    const FiniteDeformationKinematics& kinematics)
{
    return FrameAwareTransform::pullBackTensor(transform, tensor_current,
                                               deformationFrame(kinematics));
}

SurfaceMeasureTransform nansonSurfaceTransform(
    const math::Vector<Real, 3>& reference_normal,
    Real reference_measure,
    const FiniteDeformationKinematics& kinematics)
{
    return FrameAwareTransform::nansonSurfaceTransform(reference_normal,
                                                       reference_measure,
                                                       deformationFrame(kinematics));
}

NansonLinearization linearizeNansonSurfaceTransform(
    const math::Vector<Real, 3>& reference_normal,
    Real reference_measure,
    const FiniteDeformationKinematics& kinematics,
    const math::Matrix<Real, 3, 3>& dF)
{
    const auto inc = linearizeFiniteDeformationKinematics(kinematics, dF);
    const auto current = nansonSurfaceTransform(reference_normal, reference_measure,
                                                kinematics);

    NansonLinearization out;
    const auto finvT_n = transformVector(kinematics.FinvT, reference_normal,
                                         kinematics.dim);
    const auto dfinvT_n = transformVector(inc.dFinvT, reference_normal,
                                          kinematics.dim);
    out.oriented_measure_vector_derivative =
        (finvT_n * inc.dJ + dfinvT_n * kinematics.J) * reference_measure;

    const Real measure = current.measure;
    if (measure > kKinematicsTol) {
        out.measure_derivative =
            current.normal.dot(out.oriented_measure_vector_derivative);
        out.normal_derivative =
            (out.oriented_measure_vector_derivative -
             current.normal * out.measure_derivative) / measure;
        out.normal_derivative = out.normal_derivative -
            current.normal * current.normal.dot(out.normal_derivative);
    } else {
        out.normal_derivative = unitOrZero(out.oriented_measure_vector_derivative);
    }
    return out;
}

const char* gradientFrameName(KinematicGradientFrame frame) noexcept
{
    switch (frame) {
        case KinematicGradientFrame::Reference: return "Reference";
        case KinematicGradientFrame::Current: return "Current";
    }
    return "Unknown";
}

const char* referencePolicyName(FiniteDeformationReferencePolicy policy) noexcept
{
    switch (policy) {
        case FiniteDeformationReferencePolicy::TotalLagrangian:
            return "TotalLagrangian";
        case FiniteDeformationReferencePolicy::UpdatedLagrangian:
            return "UpdatedLagrangian";
    }
    return "Unknown";
}

} // namespace geometry
} // namespace FE
} // namespace svmp
