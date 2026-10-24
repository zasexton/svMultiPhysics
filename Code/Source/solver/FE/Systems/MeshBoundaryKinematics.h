/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_MESH_BOUNDARY_KINEMATICS_H
#define SVMP_FE_SYSTEMS_MESH_BOUNDARY_KINEMATICS_H

/**
 * @file MeshBoundaryKinematics.h
 * @brief Declarations of the kinematic rows of a moving (fitted ALE)
 *        boundary: the mesh-normal constraints with their consumers and
 *        stage measurements, the coupled mesh displacement they act on, and
 *        the analysis descriptors of the normal and tangential mesh rows.
 *
 * A formulation states which boundaries carry which relation (data); these
 * functions make the FE-system declarations in a fixed order.
 */

#include "Core/Types.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/// How one mesh-normal constraint is consumed and measured.
struct MeshNormalKinematicBinding {
    /// Descriptor source of the mesh row that consumes the constraint.
    std::string mesh_row_source{};
    /// Descriptor source of the related fluid-normal row.
    std::string fluid_row_source{};
    /// Prescribed tangential mesh velocity measured with the constraint.
    std::optional<std::array<Real, 3>> prescribed_tangential_mesh_velocity{};
};

/**
 * @brief Declare mesh-normal constraints, bind their consumers and register
 *        their operator-stage measurements.
 *
 * For each declaration, in order: @p binding_for(boundary marker) (which
 * may throw), then FESystem::declareMeshNormalBoundaryConstraint, the
 * consumer binding on @p operator_tag, and the stage-measurement
 * registration (MeshNormalConstraintStageMeasurements).
 */
void declareMeshNormalKinematics(
    FESystem& system,
    const std::string& operator_tag,
    std::vector<MeshNormalBoundaryConstraintDeclaration> declarations,
    const std::function<MeshNormalKinematicBinding(int boundary_marker)>&
        binding_for);

/**
 * @brief The record of the coupled mesh displacement unknown of
 *        @p ale_binding.
 *
 * Throws std::invalid_argument with @p missing_unknown_message without the
 * unknown, and with @p missing_space_message when it has no space.
 */
[[nodiscard]] const FieldRecord& requireMeshDisplacementRecord(
    const FESystem& system,
    const ALEBinding& ale_binding,
    const std::string& missing_unknown_message,
    const std::string& missing_space_message);

/// The mesh rows of one moving boundary, for its analysis descriptors.
struct MeshBoundaryRows {
    int boundary_marker{-1};
    FieldId mesh_displacement_field{INVALID_FIELD_ID};
    /// The fluid velocity related to the normal row (normal relation only).
    FieldId velocity_field{INVALID_FIELD_ID};
    /// The mesh-normal kinematic relation is installed.
    bool normal_relation{false};
    /// The tangential mesh velocity is prescribed (penalized).
    bool tangential_prescribed{false};
    MeshTangentialBoundaryPolicy tangential_policy{
        MeshTangentialBoundaryPolicy::Free};
    std::string mesh_normal_source{};
    std::string fluid_normal_source{};
    std::string tangential_source{};
    /// Message of the std::invalid_argument thrown when the normal relation
    /// has no velocity field.
    std::string missing_velocity_message{};
};

/**
 * @brief Add the analysis descriptors of the mesh rows of one boundary on
 *        @p operator_tag and bind its tangential-policy consumer.
 *
 * Without installed rows only the tangential descriptor (weak consistent)
 * and the consumer are declared.  With rows: the normal mesh and fluid
 * descriptors when the normal relation is installed, then the tangential
 * descriptor (weak penalty when prescribed) and the consumer.
 */
void declareMeshBoundaryRowDescriptors(
    FESystem& system,
    const std::string& operator_tag,
    const MeshBoundaryRows& rows,
    bool rows_installed);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_MESH_BOUNDARY_KINEMATICS_H
