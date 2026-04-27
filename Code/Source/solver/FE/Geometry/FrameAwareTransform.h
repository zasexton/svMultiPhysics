/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_FRAME_AWARE_TRANSFORM_H
#define SVMP_FE_GEOMETRY_FRAME_AWARE_TRANSFORM_H

/**
 * @file FrameAwareTransform.h
 * @brief Physics-agnostic FE transforms for moving-domain field families.
 *
 * This module is the shared FE vocabulary for frame-aware movement of values,
 * traces, and directional constraints.  Physics modules should call these
 * helpers instead of embedding local Piola or normal/tangent transform rules.
 */

#include "Core/Types.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace geometry {

enum class FEFieldTransformFamily : std::uint8_t {
    H1Scalar,
    H1Vector,
    L2Scalar,
    L2Vector,
    HDiv,
    HCurl,
    SurfaceVector,
    ShellDirector,
    Rank2Tensor
};

enum class TensorFrameTransform : std::uint8_t {
    Identity,
    Covariant,
    Contravariant,
    Piola,
    InversePiola
};

enum class DirectionalComponent : std::uint8_t {
    Full,
    Normal,
    Tangential,
    SurfaceTangent0,
    SurfaceTangent1,
    ShellDirector,
    InterfaceNormal,
    InterfaceTangential
};

struct FieldTransformSemantics {
    FEFieldTransformFamily family{FEFieldTransformFamily::H1Scalar};
    bool scalar_value{false};
    bool component_value{false};
    bool preserves_normal_flux{false};
    bool preserves_tangential_circulation{false};
    bool uses_piola{false};
    const char* description{""};
};

struct DeformationFrame {
    math::Matrix<Real, 3, 3> J{};
    math::Matrix<Real, 3, 3> Jinv{};
    Real detJ{1.0};
    int dim{3};

    [[nodiscard]] static DeformationFrame fromJacobian(const math::Matrix<Real, 3, 3>& jacobian,
                                                       int dimension = 3);
};

struct OrthonormalFrame {
    math::Vector<Real, 3> normal{};
    math::Vector<Real, 3> tangent0{};
    math::Vector<Real, 3> tangent1{};
    math::Vector<Real, 3> director{};
    bool valid{false};
};

struct SurfaceMeasureTransform {
    math::Vector<Real, 3> normal{};
    math::Vector<Real, 3> oriented_measure_vector{};
    Real measure{0.0};
};

struct DirectionalTransform {
    DirectionalComponent component{DirectionalComponent::Full};
    math::Vector<Real, 3> direction{};
    math::Matrix<Real, 3, 3> projector{};
    bool valid{false};
    std::string semantic_name{};
};

class FrameAwareTransform {
public:
    [[nodiscard]] static FieldTransformSemantics semantics(FEFieldTransformFamily family) noexcept;
    [[nodiscard]] static const char* familyName(FEFieldTransformFamily family) noexcept;
    [[nodiscard]] static const char* componentName(DirectionalComponent component) noexcept;

    [[nodiscard]] static Real pushForwardScalar(Real value) noexcept;
    [[nodiscard]] static Real pullBackScalar(Real value) noexcept;

    [[nodiscard]] static math::Vector<Real, 3> pushForwardValue(
        FEFieldTransformFamily family,
        const math::Vector<Real, 3>& value_reference,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Vector<Real, 3> pullBackValue(
        FEFieldTransformFamily family,
        const math::Vector<Real, 3>& value_current,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Vector<Real, 3> hdivPushForward(
        const math::Vector<Real, 3>& value_reference,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Vector<Real, 3> hdivPullBack(
        const math::Vector<Real, 3>& value_current,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Vector<Real, 3> hcurlPushForward(
        const math::Vector<Real, 3>& value_reference,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Vector<Real, 3> hcurlPullBack(
        const math::Vector<Real, 3>& value_current,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Matrix<Real, 3, 3> pushForwardTensor(
        TensorFrameTransform transform,
        const math::Matrix<Real, 3, 3>& tensor_reference,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Matrix<Real, 3, 3> pullBackTensor(
        TensorFrameTransform transform,
        const math::Matrix<Real, 3, 3>& tensor_current,
        const DeformationFrame& frame);

    [[nodiscard]] static SurfaceMeasureTransform nansonSurfaceTransform(
        const math::Vector<Real, 3>& reference_normal,
        Real reference_measure,
        const DeformationFrame& frame);

    [[nodiscard]] static math::Vector<Real, 3> pushForwardShellDirector(
        const math::Vector<Real, 3>& reference_director,
        const DeformationFrame& frame);

    [[nodiscard]] static OrthonormalFrame surfaceFrame(
        const math::Vector<Real, 3>& normal,
        const math::Vector<Real, 3>& tangent_hint = math::Vector<Real, 3>{Real(1), Real(0), Real(0)},
        const math::Vector<Real, 3>& director_hint = math::Vector<Real, 3>{});

    [[nodiscard]] static math::Matrix<Real, 3, 3> directionProjector(
        const math::Vector<Real, 3>& direction);

    [[nodiscard]] static math::Matrix<Real, 3, 3> normalProjector(
        const math::Vector<Real, 3>& normal);

    [[nodiscard]] static math::Matrix<Real, 3, 3> tangentialProjector(
        const math::Vector<Real, 3>& normal);

    [[nodiscard]] static math::Vector<Real, 3> normalComponent(
        const math::Vector<Real, 3>& value,
        const math::Vector<Real, 3>& normal);

    [[nodiscard]] static Real normalScalarComponent(
        const math::Vector<Real, 3>& value,
        const math::Vector<Real, 3>& normal);

    [[nodiscard]] static math::Vector<Real, 3> tangentialComponent(
        const math::Vector<Real, 3>& value,
        const math::Vector<Real, 3>& normal);

    [[nodiscard]] static DirectionalTransform transformDirectionalComponent(
        DirectionalComponent component,
        const math::Vector<Real, 3>& reference_direction,
        const DeformationFrame& frame,
        const math::Vector<Real, 3>& tangent_hint = math::Vector<Real, 3>{Real(1), Real(0), Real(0)});
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_FRAME_AWARE_TRANSFORM_H
