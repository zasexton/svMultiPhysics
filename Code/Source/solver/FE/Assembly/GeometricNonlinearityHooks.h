/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_GEOMETRIC_NONLINEARITY_HOOKS_H
#define SVMP_FE_ASSEMBLY_GEOMETRIC_NONLINEARITY_HOOKS_H

/**
 * @file GeometricNonlinearityHooks.h
 * @brief Physics-neutral assembly contribution labels for nonlinear geometry.
 */

#include <cstdint>

namespace svmp {
namespace FE {
namespace assembly {

enum class GeometricNonlinearContributionKind : std::uint8_t {
    MaterialStiffness,
    InitialStressStiffness,
    FollowerLoadSensitivity,
    GeometricIntegralConstraint,
    FrameTransformSensitivity
};

struct GeometricNonlinearContributionPolicy {
    GeometricNonlinearContributionKind kind{
        GeometricNonlinearContributionKind::MaterialStiffness};
    bool contributes_to_residual{true};
    bool contributes_to_tangent{true};
    bool tangent_is_consistent{true};
    bool may_be_lagged{false};
};

[[nodiscard]] constexpr bool requiresConsistentGeometryTangent(
    GeometricNonlinearContributionKind kind) noexcept
{
    return kind == GeometricNonlinearContributionKind::InitialStressStiffness ||
           kind == GeometricNonlinearContributionKind::FollowerLoadSensitivity ||
           kind == GeometricNonlinearContributionKind::GeometricIntegralConstraint ||
           kind == GeometricNonlinearContributionKind::FrameTransformSensitivity;
}

[[nodiscard]] constexpr bool geometricContributionPolicyIsValid(
    const GeometricNonlinearContributionPolicy& policy) noexcept
{
    if (!policy.contributes_to_residual && !policy.contributes_to_tangent) {
        return false;
    }
    if (requiresConsistentGeometryTangent(policy.kind) &&
        policy.contributes_to_tangent &&
        !policy.tangent_is_consistent) {
        return false;
    }
    return true;
}

[[nodiscard]] constexpr const char* geometricNonlinearContributionKindName(
    GeometricNonlinearContributionKind kind) noexcept
{
    switch (kind) {
        case GeometricNonlinearContributionKind::MaterialStiffness:
            return "MaterialStiffness";
        case GeometricNonlinearContributionKind::InitialStressStiffness:
            return "InitialStressStiffness";
        case GeometricNonlinearContributionKind::FollowerLoadSensitivity:
            return "FollowerLoadSensitivity";
        case GeometricNonlinearContributionKind::GeometricIntegralConstraint:
            return "GeometricIntegralConstraint";
        case GeometricNonlinearContributionKind::FrameTransformSensitivity:
            return "FrameTransformSensitivity";
    }
    return "Unknown";
}

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_GEOMETRIC_NONLINEARITY_HOOKS_H
