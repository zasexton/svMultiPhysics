#ifndef SVMP_FE_FORMS_GEOMETRY_SENSITIVITY_EVAL_H
#define SVMP_FE_FORMS_GEOMETRY_SENSITIVITY_EVAL_H

/**
 * @file GeometrySensitivityEval.h
 * @brief Shared moving-geometry derivative formulas for FE/Forms kernels.
 */

#include "Assembly/AssemblyContext.h"

#include <array>

namespace svmp {
namespace FE {
namespace forms {
namespace geometry_sensitivity {

[[nodiscard]] std::array<Real, 3> trialGeometryVectorSeed(
    const assembly::AssemblyContext& ctx,
    LocalIndex j,
    LocalIndex q);

[[nodiscard]] assembly::AssemblyContext::Matrix3x3 trialGeometryJacobianSeed(
    const assembly::AssemblyContext& ctx,
    LocalIndex j,
    LocalIndex q);

[[nodiscard]] Real currentTimeDerivativeCoefficient(
    const assembly::AssemblyContext& ctx,
    int order);

[[nodiscard]] assembly::AssemblyContext::Matrix3x3 trialMeshVelocityJacobianSeed(
    const assembly::AssemblyContext& ctx,
    LocalIndex j,
    LocalIndex q);

[[nodiscard]] Real currentCellMeasureDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j);

[[nodiscard]] std::array<Real, 3> surfaceTangentDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j,
    int tangent_column);

[[nodiscard]] Real currentFaceMeasureDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j);

[[nodiscard]] std::array<Real, 3> currentFaceNormalDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j);

[[nodiscard]] Real currentMeasureDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j);

[[nodiscard]] Real integrationWeightDerivative(
    const assembly::AssemblyContext& ctx,
    LocalIndex q,
    LocalIndex j);

} // namespace geometry_sensitivity
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_GEOMETRY_SENSITIVITY_EVAL_H
