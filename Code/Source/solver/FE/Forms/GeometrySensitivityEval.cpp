/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/GeometrySensitivityEval.h"

#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>

namespace svmp {
namespace FE {
namespace forms {
namespace geometry_sensitivity {
namespace {

[[nodiscard]] std::array<Real, 3> matrixVector(
    const assembly::AssemblyContext::Matrix3x3& A,
    const std::array<Real, 3>& x) noexcept
{
    std::array<Real, 3> out{0.0, 0.0, 0.0};
    for (std::size_t r = 0; r < 3u; ++r) {
        for (std::size_t c = 0; c < 3u; ++c) {
            out[r] += A[r][c] * x[c];
        }
    }
    return out;
}

[[nodiscard]] Real dot3(const std::array<Real, 3>& a,
                        const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] std::array<Real, 3> cross3(const std::array<Real, 3>& a,
                                         const std::array<Real, 3>& b) noexcept
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

[[nodiscard]] std::array<Real, 3> column3(
    const assembly::AssemblyContext::Matrix3x3& A,
    int column) noexcept
{
    const auto c = static_cast<std::size_t>(column);
    return {A[0][c], A[1][c], A[2][c]};
}

[[nodiscard]] LocalIndex dofsPerComponentOrThrow(const assembly::AssemblyContext& ctx,
                                                 const char* label)
{
    const int vd = ctx.trialValueDimension();
    FE_THROW_IF(vd <= 0 || vd > 3, InvalidArgumentException,
                std::string("Forms: ") + label +
                    " requires a 1..3 dimensional mesh-motion trial field");
    const LocalIndex n_trial = ctx.numTrialDofs();
    FE_THROW_IF((n_trial % static_cast<LocalIndex>(vd)) != 0, InvalidArgumentException,
                std::string("Forms: ") + label +
                    " trial DOF count is not divisible by value dimension");
    return static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
}

[[nodiscard]] int scalarComponentForTrialDofOrThrow(const assembly::AssemblyContext& ctx,
                                                    LocalIndex j,
                                                    const char* label)
{
    const int vd = ctx.trialValueDimension();
    const LocalIndex dofs_per_component = dofsPerComponentOrThrow(ctx, label);
    const int component = static_cast<int>(j / dofs_per_component);
    FE_THROW_IF(component < 0 || component >= vd, InvalidArgumentException,
                std::string("Forms: ") + label + " trial component is out of range");
    return component;
}

[[nodiscard]] Real traceProduct(
    const assembly::AssemblyContext::Matrix3x3& A,
    const assembly::AssemblyContext::Matrix3x3& B,
    int dim) noexcept
{
    Real value = 0.0;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            value += A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                     B[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
    }
    return value;
}

} // namespace

std::array<Real, 3> trialGeometryVectorSeed(
    const assembly::AssemblyContext& ctx,
    LocalIndex j,
    LocalIndex q)
{
    if (ctx.trialUsesVectorBasis()) {
        return ctx.trialBasisVectorValue(j, q);
    }

    const int component =
        scalarComponentForTrialDofOrThrow(ctx, j, "geometry sensitivity");

    std::array<Real, 3> out{0.0, 0.0, 0.0};
    out[static_cast<std::size_t>(component)] = ctx.trialBasisValue(j, q);
    return out;
}

assembly::AssemblyContext::Matrix3x3 trialGeometryJacobianSeed(
    const assembly::AssemblyContext& ctx,
    LocalIndex j,
    LocalIndex q)
{
    FE_THROW_IF(ctx.trialUsesVectorBasis(), InvalidArgumentException,
                "Forms: current-geometry sensitivity requires scalar-component H1/Product mesh-motion trials");

    const int component =
        scalarComponentForTrialDofOrThrow(ctx, j, "geometry sensitivity");

    assembly::AssemblyContext::Matrix3x3 out{};
    const auto grad_ref = ctx.trialReferenceGradient(j, q);
    for (int xi = 0; xi < ctx.dimension(); ++xi) {
        out[static_cast<std::size_t>(component)][static_cast<std::size_t>(xi)] =
            grad_ref[static_cast<std::size_t>(xi)];
    }
    return out;
}

Real currentTimeDerivativeCoefficient(
    const assembly::AssemblyContext& ctx,
    int order)
{
    const auto* time_ctx = ctx.timeIntegrationContext();
    FE_THROW_IF(time_ctx == nullptr, InvalidArgumentException,
                "Forms: coupled mesh-velocity sensitivity requires a time-integration context");
    const auto* stencil = time_ctx->stencil(order);
    FE_THROW_IF(stencil == nullptr, InvalidArgumentException,
                "Forms: coupled mesh-velocity sensitivity requires an active time-derivative stencil");
    return stencil->coeff(0);
}

assembly::AssemblyContext::Matrix3x3 trialMeshVelocityJacobianSeed(
    const assembly::AssemblyContext& ctx,
    LocalIndex j,
    LocalIndex q)
{
    if (ctx.trialUsesVectorBasis()) {
        return ctx.trialBasisVectorJacobian(j, q);
    }

    const int component =
        scalarComponentForTrialDofOrThrow(ctx, j, "mesh-velocity sensitivity");

    assembly::AssemblyContext::Matrix3x3 out{};
    const auto grad = ctx.trialPhysicalGradient(j, q);
    for (int d = 0; d < ctx.dimension(); ++d) {
        out[static_cast<std::size_t>(component)][static_cast<std::size_t>(d)] =
            grad[static_cast<std::size_t>(d)];
    }
    return out;
}

Real currentCellMeasureDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j)
{
    const auto dJ = trialGeometryJacobianSeed(ctx, j, q);
    return ctx.currentMeasure(q) *
        traceProduct(ctx.currentInverseJacobian(q), dJ, ctx.dimension());
}

std::array<Real, 3> surfaceTangentDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j,
    int tangent_column)
{
    FE_THROW_IF(ctx.trialUsesVectorBasis(), InvalidArgumentException,
                "Forms: current-face geometry sensitivity requires scalar-component H1/Product mesh-motion trials");

    const int component =
        scalarComponentForTrialDofOrThrow(ctx, j, "geometry sensitivity");

    const auto tangent = column3(ctx.surfaceJacobian(q), tangent_column);
    const auto dxi_ds = matrixVector(ctx.currentInverseJacobian(q), tangent);
    const auto grad_ref = ctx.trialReferenceGradient(j, q);
    const Real dN_ds = dot3(grad_ref, dxi_ds);

    std::array<Real, 3> out{0.0, 0.0, 0.0};
    out[static_cast<std::size_t>(component)] = dN_ds;
    return out;
}

Real currentFaceMeasureDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j)
{
    const int dim = ctx.dimension();
    if (dim == 3) {
        const auto t0 = column3(ctx.surfaceJacobian(q), 0);
        const auto t1 = column3(ctx.surfaceJacobian(q), 1);
        const auto dt0 = surfaceTangentDerivative(ctx, q, j, 0);
        const auto dt1 = surfaceTangentDerivative(ctx, q, j, 1);
        const auto d_area_vec_a = cross3(dt0, t1);
        const auto d_area_vec_b = cross3(t0, dt1);
        const auto normal = ctx.currentNormal(q);
        return dot3(normal, {d_area_vec_a[0] + d_area_vec_b[0],
                             d_area_vec_a[1] + d_area_vec_b[1],
                             d_area_vec_a[2] + d_area_vec_b[2]});
    }
    if (dim == 2) {
        const auto t0 = column3(ctx.surfaceJacobian(q), 0);
        const auto dt0 = surfaceTangentDerivative(ctx, q, j, 0);
        const Real measure = ctx.currentMeasure(q);
        if (!(measure > 0.0)) {
            return 0.0;
        }
        return dot3(t0, dt0) / measure;
    }
    return 0.0;
}

std::array<Real, 3> currentFaceNormalDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j)
{
    const int dim = ctx.dimension();
    if (dim == 3) {
        const auto t0 = column3(ctx.surfaceJacobian(q), 0);
        const auto t1 = column3(ctx.surfaceJacobian(q), 1);
        const auto dt0 = surfaceTangentDerivative(ctx, q, j, 0);
        const auto dt1 = surfaceTangentDerivative(ctx, q, j, 1);
        const auto d_area_vec_a = cross3(dt0, t1);
        const auto d_area_vec_b = cross3(t0, dt1);
        const std::array<Real, 3> d_area_vec{
            d_area_vec_a[0] + d_area_vec_b[0],
            d_area_vec_a[1] + d_area_vec_b[1],
            d_area_vec_a[2] + d_area_vec_b[2]};
        const auto normal = ctx.currentNormal(q);
        const Real normal_part = dot3(normal, d_area_vec);
        std::array<Real, 3> out{};
        const Real measure = ctx.currentMeasure(q);
        if (!(measure > 0.0)) {
            return out;
        }
        for (std::size_t d = 0; d < 3u; ++d) {
            out[d] = (d_area_vec[d] - normal[d] * normal_part) / measure;
        }
        return out;
    }
    if (dim == 2) {
        const auto t0 = column3(ctx.surfaceJacobian(q), 0);
        const auto dt0 = surfaceTangentDerivative(ctx, q, j, 0);
        const Real measure = ctx.currentMeasure(q);
        std::array<Real, 3> out{};
        if (!(measure > 0.0)) {
            return out;
        }
        const std::array<Real, 3> unit_t{t0[0] / measure, t0[1] / measure, 0.0};
        const Real dmeasure = dot3(unit_t, dt0);
        const std::array<Real, 3> dunit_t{
            (dt0[0] - unit_t[0] * dmeasure) / measure,
            (dt0[1] - unit_t[1] * dmeasure) / measure,
            0.0};
        out[0] = -dunit_t[1];
        out[1] = dunit_t[0];
        return out;
    }
    return {0.0, 0.0, 0.0};
}

Real currentMeasureDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j)
{
    if (ctx.contextType() == assembly::ContextType::Cell) {
        return currentCellMeasureDerivative(ctx, q, j);
    }
    return currentFaceMeasureDerivative(ctx, q, j);
}

Real integrationWeightDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j)
{
    const Real measure = ctx.currentMeasure(q);
    if (!(measure > 0.0)) {
        return 0.0;
    }
    return ctx.integrationWeight(q) * currentMeasureDerivative(ctx, q, j) / measure;
}

} // namespace geometry_sensitivity
} // namespace forms
} // namespace FE
} // namespace svmp
