/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BOUNDARY_PREPARATION_H
#define SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BOUNDARY_PREPARATION_H

#include "Physics/Formulations/MeshMotion/MeshMotionBCFactories.h"

#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::Physics::formulations::mesh_motion::BoundaryPreparation {

/// Normal ownership preflight and prescribed targets; preparing this value does
/// not register fields, publish declarations, or sample coefficient callbacks.
struct PreparedNormalConstraints {
    std::optional<FE::FieldId> candidate_displacement;
    std::set<int> markers;
    std::vector<FE::forms::FormExpr> targets;
};

[[nodiscard]] inline PreparedNormalConstraints prepareNormalConstraints(
    const FE::systems::FESystem& system,
    std::string_view field_name,
    bool bind_as_mesh_displacement,
    int dim,
    std::span<const NormalConstraintBC> constraints,
    const std::string& context)
{
    PreparedNormalConstraints prepared;
    prepared.candidate_displacement = system.meshMotionField(
        FE::systems::MeshMotionFieldRole::Displacement);
    if (!constraints.empty() && !prepared.candidate_displacement.has_value() &&
        !bind_as_mesh_displacement) {
        throw std::invalid_argument(
            context + ": normal constraints require the target field to be bound as mesh displacement");
    }
    if (!prepared.candidate_displacement.has_value()) {
        const auto named = system.findFieldByName(std::string(field_name));
        if (named != FE::INVALID_FIELD_ID) {
            prepared.candidate_displacement = named;
        }
    }
    prepared.targets.reserve(constraints.size());
    for (const auto& bc : constraints) {
        const int marker = FE::forms::bc::boundaryMarkerOrThrow(
            bc, context + " normal_constraint");
        if (!prepared.markers.insert(marker).second) {
            throw std::invalid_argument(
                context + ": duplicate normal constraint marker " + std::to_string(marker));
        }
        if (bc.quantity != NormalConstraintQuantity::Displacement &&
            bc.quantity != NormalConstraintQuantity::Velocity) {
            throw std::invalid_argument(context + ": unknown normal constraint quantity");
        }
        if (!Factories::isFinitePositiveScalarLiteral(bc.penalty)) {
            throw std::invalid_argument(
                context + ": normal constraint penalty must be a finite positive literal");
        }
        if (bc.quantity == NormalConstraintQuantity::Velocity &&
            !Factories::isFinitePositiveScalarLiteral(bc.velocity_time_scale)) {
            throw std::invalid_argument(
                context + ": normal constraint velocity time scale must be a finite positive literal");
        }
        auto target = Factories::effectiveNormalConstraintTarget(
            bc, dim, context + " normal_constraint");
        const auto* target_node = target.node();
        if (target_node == nullptr || target.hasTest() ||
            FE::forms::bc::hasForbiddenPrescribedValueDependency(*target_node)) {
            throw std::invalid_argument(
                context + ": normal constraint target must be independent of FE state and variational geometry");
        }
        prepared.targets.push_back(std::move(target));
        if (!prepared.candidate_displacement.has_value()) {
            continue;
        }
        const auto existing = std::find_if(
            system.meshNormalBoundaryConstraints().begin(),
            system.meshNormalBoundaryConstraints().end(),
            [&](const auto& declaration) {
                return declaration.mesh_displacement_field == *prepared.candidate_displacement &&
                       declaration.boundary_marker == marker;
            });
        if (existing != system.meshNormalBoundaryConstraints().end()) {
            throw FE::InvalidArgumentException(
                context + ": boundary " + std::to_string(marker) +
                " already has normal mesh-motion owner '" + existing->owner_component + "'");
        }
    }
    return prepared;
}

/// Construct pending declarations only after displacement binding succeeds.
/// The caller publishes them after installation and binds their consumers.
[[nodiscard]] inline std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
pendingNormalDeclarations(
    FE::FieldId displacement,
    std::span<const NormalConstraintBC> constraints,
    PreparedNormalConstraints& prepared,
    const std::string& owner,
    const std::string& context)
{
    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration> declarations;
    declarations.reserve(constraints.size());
    for (std::size_t index = 0u; index < constraints.size(); ++index) {
        const auto& bc = constraints[index];
        const int marker = FE::forms::bc::boundaryMarkerOrThrow(
            bc, context + " normal_constraint");
        declarations.push_back(FE::systems::MeshNormalBoundaryConstraintDeclaration{
            .mesh_displacement_field = displacement,
            .boundary_marker = marker,
            .quantity = FE::systems::MeshNormalBoundaryQuantity::DisplacementTrace,
            .target_kind = bc.quantity == NormalConstraintQuantity::Displacement
                               ? FE::systems::MeshNormalBoundaryTargetKind::PrescribedDisplacement
                               : FE::systems::MeshNormalBoundaryTargetKind::TimeScaledPrescribedVelocity,
            .target_expression = std::move(prepared.targets[index]),
            .enforcement_kind = FE::analysis::EnforcementKind::WeakPenalty,
            .related_velocity_field = FE::INVALID_FIELD_ID,
            .owner_component = owner,
        });
    }
    return declarations;
}

/// Publication is deliberately sequential: an earlier policy survives a later
/// invalid policy or duplicate owner, as does the resolved displacement binding.
inline void declareTangentialPolicies(
    FE::systems::FESystem& system,
    FE::FieldId displacement,
    std::span<const TangentialPolicyBC> policies,
    const std::string& owner,
    const std::string& context)
{
    for (const auto& bc : policies) {
        const auto system_policy = [&]() {
            switch (bc.policy) {
            case TangentialMeshPolicy::Free:
                return FE::systems::MeshTangentialBoundaryPolicy::Free;
            case TangentialMeshPolicy::SmoothingOnly:
                return FE::systems::MeshTangentialBoundaryPolicy::SmoothingOnly;
            case TangentialMeshPolicy::Prescribed:
                return FE::systems::MeshTangentialBoundaryPolicy::Prescribed;
            }
            throw std::invalid_argument(context + ": unknown tangential policy");
        }();
        system.declareMeshTangentialBoundaryPolicy(
            FE::systems::MeshTangentialBoundaryPolicyDeclaration{
                .mesh_displacement_field = displacement,
                .boundary_marker = bc.boundary_marker,
                .policy = system_policy,
                .owner_component = owner,
            });
    }
}

/// Both smoothing models translate boundaries in this order. The caller keeps
/// applyAll explicit at the original residual/constraint application point.
template <class Options>
inline void populateBoundaryConditions(
    FE::systems::BoundaryConditionManager& manager,
    const Options& options,
    int dim,
    const std::string& context)
{
    manager.install(options.natural, [&](const auto& bc) {
        return Factories::toVectorNaturalBC(bc, dim, context + " natural");
    });
    manager.install(options.robin, [&](const auto& bc) {
        return Factories::toVectorRobinBC(bc, dim, context + " robin");
    });
    manager.install(options.normal_constraint, [&](const auto& bc) {
        return Factories::toNormalConstraintBC(bc, dim, context + " normal_constraint");
    });
    for (const auto& bc : options.tangential_policy) {
        switch (bc.policy) {
        case TangentialMeshPolicy::Free:
        case TangentialMeshPolicy::SmoothingOnly:
            break;
        case TangentialMeshPolicy::Prescribed:
            manager.add(Factories::toTangentialConstraintBC(
                bc, dim, context + " tangential_policy"));
            break;
        }
    }
    manager.install(options.dirichlet, [&](const auto& bc) {
        return Factories::toVectorEssentialBC(bc, dim, context + " dirichlet", "d_mesh");
    });
}

} // namespace svmp::Physics::formulations::mesh_motion::BoundaryPreparation

#endif // SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BOUNDARY_PREPARATION_H
