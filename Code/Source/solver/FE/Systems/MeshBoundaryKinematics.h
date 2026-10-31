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
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"

#include <array>
#include <functional>
#include <optional>
#include <set>
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

/// One prepared relation and the consumer/measurement data that belong to it.
struct MeshNormalKinematicDeclaration {
    MeshNormalBoundaryConstraintDeclaration constraint;
    MeshNormalKinematicBinding binding;
};

/// Declare, bind, and register the measurement of each prepared item in order.
/// Earlier items remain registered if a later declaration fails.
void declareMeshNormalKinematics(
    FESystem& system,
    const std::string& operator_tag,
    std::vector<MeshNormalKinematicDeclaration> declarations);

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

// ---------------------------------------------------------------------------
// The normal-rate boundary relation (dt d - r).n = 0
// ---------------------------------------------------------------------------
//
// One boundary relation couples a field d to a related field r on a boundary
// Gamma: the normal trace of the rate of d equals that of r (for a moving
// domain, the mesh velocity follows the fluid normal velocity).  Its owner
// declares it (MeshNormalBoundaryConstraintDeclaration with
// requires_mesh_flux_consistency and the Nitsche constant gamma_N) and installs
// its Nitsche row on the rows of d,
//     int_Gamma (gamma_N / h_n) ((dt d - r).n)(psi.n);
// the operator that owns the bulk rows of d, with flux kappa grad(q), removes
// its own normal flux on Gamma,
//     - int_Gamma (kappa (grad(q) n).n)(psi.n),
// so that the boundary equation reduces to the relation.  The relation acts on
// the rate, so q must be dt(d), and the combined row is coercive for
// gamma_N > 2 kappa (P1 trace inverse inequality, h_n = 2|T|/|F|).  Each term
// stays in the kernel of the operator that states it.

/// The declaration of the relation (dt d - r).n = 0 on @p boundary_marker,
/// r the field @p related_field (form @p related), with Nitsche constant
/// @p nitsche_gamma, owned by @p owner_component.
[[nodiscard]] MeshNormalBoundaryConstraintDeclaration normalRateRelation(FieldId d_field,
                                                                         int boundary_marker,
                                                                         FieldId related_field,
                                                                         const forms::FormExpr& related,
                                                                         Real nitsche_gamma,
                                                                         std::string owner_component);

/// The Nitsche row of the relation (dt d - r).n = 0 on @p boundary, with
/// @p normal the current normal n (currentNormal(), shared with the other
/// boundary terms of the install) and the normal cell size h_n.
[[nodiscard]] forms::FormExpr normalRateRelationRow(const forms::FormExpr& d,
                                                    const forms::FormExpr& related,
                                                    const forms::FormExpr& psi,
                                                    Real nitsche_gamma,
                                                    const forms::FormExpr& normal,
                                                    const forms::BoundaryDomain& boundary);

/// The operator of d that consumes the normal-rate relations on d.
struct NormalRateRelationConsumer {
    /// The field d; unset before the operator registers it (every relation applies).
    std::optional<FieldId> field{};
    /// Markers where the operator states a normal constraint of its own.
    std::set<int> own_normal_markers{};
    /// The operator's flux acts on the rate, kappa grad(dt d).
    bool flux_of_rate{false};
    /// kappa, when it is a literal.
    std::optional<Real> literal_kappa{};
    /// Prefix of the error messages.
    std::string context{};
    /// What the author changes so that the flux acts on the rate.
    std::string rate_flux_hint{};
};

/**
 * @brief Check that @p consumer can take the normal-rate relations on its
 *        field, before it changes the system.
 *
 * Throws std::invalid_argument for a relation on a marker with a normal
 * constraint of the consumer, for a flux that does not act on the rate, and
 * without a literal kappa with gamma_N > 2 kappa.
 */
void requireNormalRateRelationsConsumable(const FESystem& system,
                                          const NormalRateRelationConsumer& consumer);

/**
 * @brief @p residual minus the consistency term of the flux kappa grad(q) on
 *        each marker with a normal-rate relation on @p field:
 *        R - sum_Gamma ((kappa (grad(q) n).n)(psi.n)).ds(Gamma)
 *
 * @p flux_gradient is grad(q); the markers are taken in the order of
 * FESystem::meshNormalBoundaryConstraints(), each once.
 */
[[nodiscard]] forms::FormExpr addNormalRateRelationConsistency(const FESystem& system,
                                                               FieldId field,
                                                               forms::FormExpr residual,
                                                               const forms::FormExpr& kappa,
                                                               const forms::FormExpr& flux_gradient,
                                                               const forms::FormExpr& psi);

/// The markers with a normal-rate relation on @p field (on any field when
/// unset), in the order of FESystem::meshNormalBoundaryConstraints(), each once.
[[nodiscard]] std::vector<int> normalRateRelationMarkers(const FESystem& system,
                                                         std::optional<FieldId> field = std::nullopt);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_MESH_BOUNDARY_KINEMATICS_H
