/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/FrameAwareTransform.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {
namespace {

constexpr Real kTransformTol = Real(1e-14);

[[nodiscard]] math::Matrix<Real, 3, 3> identityMatrix() noexcept
{
    math::Matrix<Real, 3, 3> out{};
    out(0, 0) = Real(1);
    out(1, 1) = Real(1);
    out(2, 2) = Real(1);
    return out;
}

[[nodiscard]] math::Vector<Real, 3> unitOrZero(const math::Vector<Real, 3>& v) noexcept
{
    const Real n = v.norm();
    if (n <= kTransformTol) {
        return math::Vector<Real, 3>{};
    }
    return v / n;
}

[[nodiscard]] bool nonzero(const math::Vector<Real, 3>& v) noexcept
{
    return v.norm() > kTransformTol;
}

[[nodiscard]] math::Vector<Real, 3> vectorTransform(
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

[[nodiscard]] math::Matrix<Real, 3, 3> scaledIdentityOrInverse(
    const math::Matrix<Real, 3, 3>& J,
    int dim)
{
    math::Matrix<Real, 3, 3> effective = identityMatrix();
    const auto n = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            effective(i, j) = J(i, j);
        }
    }
    return effective.inverse();
}

} // namespace

DeformationFrame DeformationFrame::fromJacobian(const math::Matrix<Real, 3, 3>& jacobian,
                                                int dimension)
{
    FE_THROW_IF(dimension < 1 || dimension > 3, FEException,
                "DeformationFrame::fromJacobian requires dimension 1, 2, or 3");

    DeformationFrame frame;
    frame.J = identityMatrix();
    frame.dim = dimension;
    const auto n = static_cast<std::size_t>(dimension);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            frame.J(i, j) = jacobian(i, j);
        }
    }
    frame.detJ = frame.J.determinant();
    FE_THROW_IF(std::abs(frame.detJ) <= kTransformTol, FEException,
                "DeformationFrame::fromJacobian encountered a singular geometry transform");
    frame.Jinv = scaledIdentityOrInverse(frame.J, dimension);
    return frame;
}

FieldTransformSemantics FrameAwareTransform::semantics(FEFieldTransformFamily family) noexcept
{
    switch (family) {
        case FEFieldTransformFamily::H1Scalar:
            return {family, true, false, false, false, false,
                    "H1 scalar values are evaluated in the requested configuration; scalar values are unchanged by frame transforms."};
        case FEFieldTransformFamily::H1Vector:
            return {family, false, true, false, false, false,
                    "H1 vector component fields keep their physical components in the requested configuration; no Piola scaling is applied."};
        case FEFieldTransformFamily::L2Scalar:
            return {family, true, false, false, false, false,
                    "L2 scalar values are scalar density/field values; integration weights own geometry scaling."};
        case FEFieldTransformFamily::L2Vector:
            return {family, false, true, false, false, false,
                    "L2 vector component fields keep their physical components; no continuity trace transform is implied."};
        case FEFieldTransformFamily::HDiv:
            return {family, false, true, true, false, true,
                    "H(div) values use the contravariant Piola transform and preserve normal flux."};
        case FEFieldTransformFamily::HCurl:
            return {family, false, true, false, true, true,
                    "H(curl) values use the covariant Piola transform and preserve tangential circulation."};
        case FEFieldTransformFamily::SurfaceVector:
            return {family, false, true, false, false, false,
                    "Surface vector values are resolved with shared surface-frame projectors."};
        case FEFieldTransformFamily::ShellDirector:
            return {family, false, true, false, false, false,
                    "Shell directors are pushed forward as geometric directions and renormalized."};
        case FEFieldTransformFamily::Rank2Tensor:
            return {family, false, false, false, false, false,
                    "Rank-2 tensor transforms must declare covariant, contravariant, or Piola semantics explicitly."};
    }
    return {family, false, false, false, false, false, ""};
}

const char* FrameAwareTransform::familyName(FEFieldTransformFamily family) noexcept
{
    switch (family) {
        case FEFieldTransformFamily::H1Scalar: return "H1Scalar";
        case FEFieldTransformFamily::H1Vector: return "H1Vector";
        case FEFieldTransformFamily::L2Scalar: return "L2Scalar";
        case FEFieldTransformFamily::L2Vector: return "L2Vector";
        case FEFieldTransformFamily::HDiv: return "HDiv";
        case FEFieldTransformFamily::HCurl: return "HCurl";
        case FEFieldTransformFamily::SurfaceVector: return "SurfaceVector";
        case FEFieldTransformFamily::ShellDirector: return "ShellDirector";
        case FEFieldTransformFamily::Rank2Tensor: return "Rank2Tensor";
    }
    return "Unknown";
}

const char* FrameAwareTransform::componentName(DirectionalComponent component) noexcept
{
    switch (component) {
        case DirectionalComponent::Full: return "Full";
        case DirectionalComponent::Normal: return "Normal";
        case DirectionalComponent::Tangential: return "Tangential";
        case DirectionalComponent::SurfaceTangent0: return "SurfaceTangent0";
        case DirectionalComponent::SurfaceTangent1: return "SurfaceTangent1";
        case DirectionalComponent::ShellDirector: return "ShellDirector";
        case DirectionalComponent::InterfaceNormal: return "InterfaceNormal";
        case DirectionalComponent::InterfaceTangential: return "InterfaceTangential";
    }
    return "Unknown";
}

Real FrameAwareTransform::pushForwardScalar(Real value) noexcept
{
    return value;
}

Real FrameAwareTransform::pullBackScalar(Real value) noexcept
{
    return value;
}

math::Vector<Real, 3> FrameAwareTransform::pushForwardValue(
    FEFieldTransformFamily family,
    const math::Vector<Real, 3>& value_reference,
    const DeformationFrame& frame)
{
    switch (family) {
        case FEFieldTransformFamily::HDiv:
            return hdivPushForward(value_reference, frame);
        case FEFieldTransformFamily::HCurl:
            return hcurlPushForward(value_reference, frame);
        case FEFieldTransformFamily::ShellDirector:
            return pushForwardShellDirector(value_reference, frame);
        case FEFieldTransformFamily::H1Scalar:
        case FEFieldTransformFamily::L2Scalar:
        case FEFieldTransformFamily::Rank2Tensor:
            FE_THROW(FEException, "FrameAwareTransform::pushForwardValue called for non-vector family");
        case FEFieldTransformFamily::H1Vector:
        case FEFieldTransformFamily::L2Vector:
        case FEFieldTransformFamily::SurfaceVector:
            return value_reference;
    }
    return value_reference;
}

math::Vector<Real, 3> FrameAwareTransform::pullBackValue(
    FEFieldTransformFamily family,
    const math::Vector<Real, 3>& value_current,
    const DeformationFrame& frame)
{
    switch (family) {
        case FEFieldTransformFamily::HDiv:
            return hdivPullBack(value_current, frame);
        case FEFieldTransformFamily::HCurl:
            return hcurlPullBack(value_current, frame);
        case FEFieldTransformFamily::ShellDirector:
            return unitOrZero(vectorTransform(frame.Jinv, value_current, frame.dim));
        case FEFieldTransformFamily::H1Scalar:
        case FEFieldTransformFamily::L2Scalar:
        case FEFieldTransformFamily::Rank2Tensor:
            FE_THROW(FEException, "FrameAwareTransform::pullBackValue called for non-vector family");
        case FEFieldTransformFamily::H1Vector:
        case FEFieldTransformFamily::L2Vector:
        case FEFieldTransformFamily::SurfaceVector:
            return value_current;
    }
    return value_current;
}

math::Vector<Real, 3> FrameAwareTransform::hdivPushForward(
    const math::Vector<Real, 3>& value_reference,
    const DeformationFrame& frame)
{
    FE_THROW_IF(std::abs(frame.detJ) <= kTransformTol, FEException,
                "H(div) Piola transform requires nonzero determinant");
    return vectorTransform(frame.J, value_reference, frame.dim) * (Real(1) / frame.detJ);
}

math::Vector<Real, 3> FrameAwareTransform::hdivPullBack(
    const math::Vector<Real, 3>& value_current,
    const DeformationFrame& frame)
{
    return vectorTransform(frame.Jinv, value_current, frame.dim) * frame.detJ;
}

math::Vector<Real, 3> FrameAwareTransform::hcurlPushForward(
    const math::Vector<Real, 3>& value_reference,
    const DeformationFrame& frame)
{
    return vectorTransform(frame.Jinv.transpose(), value_reference, frame.dim);
}

math::Vector<Real, 3> FrameAwareTransform::hcurlPullBack(
    const math::Vector<Real, 3>& value_current,
    const DeformationFrame& frame)
{
    return vectorTransform(frame.J.transpose(), value_current, frame.dim);
}

math::Matrix<Real, 3, 3> FrameAwareTransform::pushForwardTensor(
    TensorFrameTransform transform,
    const math::Matrix<Real, 3, 3>& tensor_reference,
    const DeformationFrame& frame)
{
    switch (transform) {
        case TensorFrameTransform::Identity:
            return tensor_reference;
        case TensorFrameTransform::Covariant:
            return frame.Jinv.transpose() * tensor_reference * frame.Jinv;
        case TensorFrameTransform::Contravariant:
            return frame.J * tensor_reference * frame.J.transpose();
        case TensorFrameTransform::Piola:
            FE_THROW_IF(std::abs(frame.detJ) <= kTransformTol, FEException,
                        "Piola tensor transform requires nonzero determinant");
            return (frame.J * tensor_reference * frame.J.transpose()) * (Real(1) / frame.detJ);
        case TensorFrameTransform::InversePiola:
            return (frame.Jinv * tensor_reference * frame.Jinv.transpose()) * frame.detJ;
    }
    return tensor_reference;
}

math::Matrix<Real, 3, 3> FrameAwareTransform::pullBackTensor(
    TensorFrameTransform transform,
    const math::Matrix<Real, 3, 3>& tensor_current,
    const DeformationFrame& frame)
{
    switch (transform) {
        case TensorFrameTransform::Identity:
            return tensor_current;
        case TensorFrameTransform::Covariant:
            return frame.J.transpose() * tensor_current * frame.J;
        case TensorFrameTransform::Contravariant:
            return frame.Jinv * tensor_current * frame.Jinv.transpose();
        case TensorFrameTransform::Piola:
            return (frame.Jinv * tensor_current * frame.Jinv.transpose()) * frame.detJ;
        case TensorFrameTransform::InversePiola:
            FE_THROW_IF(std::abs(frame.detJ) <= kTransformTol, FEException,
                        "Inverse-Piola tensor pullback requires nonzero determinant");
            return (frame.J * tensor_current * frame.J.transpose()) * (Real(1) / frame.detJ);
    }
    return tensor_current;
}

SurfaceMeasureTransform FrameAwareTransform::nansonSurfaceTransform(
    const math::Vector<Real, 3>& reference_normal,
    Real reference_measure,
    const DeformationFrame& frame)
{
    const math::Vector<Real, 3> measure_vector =
        vectorTransform(frame.Jinv.transpose(), reference_normal, frame.dim) *
        (frame.detJ * reference_measure);
    const Real measure = measure_vector.norm();
    SurfaceMeasureTransform out;
    out.oriented_measure_vector = measure_vector;
    out.measure = measure;
    out.normal = (measure > kTransformTol) ? measure_vector / measure : math::Vector<Real, 3>{};
    return out;
}

math::Vector<Real, 3> FrameAwareTransform::pushForwardShellDirector(
    const math::Vector<Real, 3>& reference_director,
    const DeformationFrame& frame)
{
    return unitOrZero(vectorTransform(frame.J, reference_director, frame.dim));
}

OrthonormalFrame FrameAwareTransform::surfaceFrame(
    const math::Vector<Real, 3>& normal,
    const math::Vector<Real, 3>& tangent_hint,
    const math::Vector<Real, 3>& director_hint)
{
    OrthonormalFrame frame;
    frame.normal = unitOrZero(normal);
    if (!nonzero(frame.normal)) {
        return frame;
    }

    math::Vector<Real, 3> tangent =
        tangent_hint - frame.normal * frame.normal.dot(tangent_hint);
    if (!nonzero(tangent)) {
        tangent = math::Vector<Real, 3>{
            math::Vector<Real, 3>{Real(1), Real(0), Real(0)} -
            frame.normal * frame.normal[0]};
    }
    if (!nonzero(tangent)) {
        tangent = math::Vector<Real, 3>{
            math::Vector<Real, 3>{Real(0), Real(1), Real(0)} -
            frame.normal * frame.normal[1]};
    }

    frame.tangent0 = unitOrZero(tangent);
    frame.tangent1 = unitOrZero(frame.normal.cross(frame.tangent0));
    if (!nonzero(frame.tangent1)) {
        return frame;
    }

    frame.director = nonzero(director_hint) ? unitOrZero(director_hint) : frame.normal;
    frame.valid = true;
    return frame;
}

math::Matrix<Real, 3, 3> FrameAwareTransform::directionProjector(
    const math::Vector<Real, 3>& direction)
{
    const auto d = unitOrZero(direction);
    math::Matrix<Real, 3, 3> out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            out(i, j) = d[i] * d[j];
        }
    }
    return out;
}

math::Matrix<Real, 3, 3> FrameAwareTransform::normalProjector(
    const math::Vector<Real, 3>& normal)
{
    return directionProjector(normal);
}

math::Matrix<Real, 3, 3> FrameAwareTransform::tangentialProjector(
    const math::Vector<Real, 3>& normal)
{
    return identityMatrix() - normalProjector(normal);
}

math::Vector<Real, 3> FrameAwareTransform::normalComponent(
    const math::Vector<Real, 3>& value,
    const math::Vector<Real, 3>& normal)
{
    return normalProjector(normal) * value;
}

Real FrameAwareTransform::normalScalarComponent(
    const math::Vector<Real, 3>& value,
    const math::Vector<Real, 3>& normal)
{
    return value.dot(unitOrZero(normal));
}

math::Vector<Real, 3> FrameAwareTransform::tangentialComponent(
    const math::Vector<Real, 3>& value,
    const math::Vector<Real, 3>& normal)
{
    return tangentialProjector(normal) * value;
}

DirectionalTransform FrameAwareTransform::transformDirectionalComponent(
    DirectionalComponent component,
    const math::Vector<Real, 3>& reference_direction,
    const DeformationFrame& frame,
    const math::Vector<Real, 3>& tangent_hint)
{
    DirectionalTransform out;
    out.component = component;
    out.semantic_name = componentName(component);

    switch (component) {
        case DirectionalComponent::Full:
            out.direction = math::Vector<Real, 3>{};
            out.projector = identityMatrix();
            out.valid = true;
            return out;
        case DirectionalComponent::Normal:
        case DirectionalComponent::InterfaceNormal:
            out.direction = nansonSurfaceTransform(reference_direction, Real(1), frame).normal;
            out.projector = normalProjector(out.direction);
            out.valid = nonzero(out.direction);
            return out;
        case DirectionalComponent::Tangential:
        case DirectionalComponent::InterfaceTangential:
            out.direction = nansonSurfaceTransform(reference_direction, Real(1), frame).normal;
            out.projector = tangentialProjector(out.direction);
            out.valid = nonzero(out.direction);
            return out;
        case DirectionalComponent::SurfaceTangent0:
        case DirectionalComponent::SurfaceTangent1: {
            const auto nanson = nansonSurfaceTransform(reference_direction, Real(1), frame);
            const auto surface = surfaceFrame(nanson.normal, vectorTransform(frame.J, tangent_hint, frame.dim));
            out.direction = (component == DirectionalComponent::SurfaceTangent0)
                                ? surface.tangent0
                                : surface.tangent1;
            out.projector = directionProjector(out.direction);
            out.valid = surface.valid;
            return out;
        }
        case DirectionalComponent::ShellDirector:
            out.direction = pushForwardShellDirector(reference_direction, frame);
            out.projector = directionProjector(out.direction);
            out.valid = nonzero(out.direction);
            return out;
    }

    return out;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
