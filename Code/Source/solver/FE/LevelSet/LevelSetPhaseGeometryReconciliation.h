/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_LEVELSETPHASEGEOMETRYRECONCILIATION_H
#define SVMP_FE_LEVELSET_LEVELSETPHASEGEOMETRYRECONCILIATION_H

/**
 * @file LevelSetPhaseGeometryReconciliation.h
 * @brief Moves a P1 level set until its cut geometry carries given nodal
 *        liquid moments.
 *
 * After a conservative transport stage the liquid moments that the cut
 * geometry projects onto the phase graph differ from the transported ones.
 * reconcileConservativePhaseGeometry corrects the level set by
 * sensitivity-based steps on those moments (solveLevelSetP1PhaseGeometry-
 * Correction) with a backtracking line search, within a cumulative
 * interface-displacement bound and a sign margin that keeps the cut topology,
 * refreshing the cut geometry through a transaction for every trial.
 * Accepted wall-contact parents can be held fixed.
 */

#include "Core/Types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

struct ConservativePhaseRequest;
struct ConservativePhaseWorkflowHost;
class ConservativePhaseGeometryTransaction;

/// Nodal residual target - current: largest magnitude, Euclidean norm, sum.
// Moved from Application/Core/ApplicationDriver.cpp (lines 24046-24050 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.
struct ConservativePhaseMomentMismatch {
  svmp::FE::Real maximum_nodal_residual{0.0};
  svmp::FE::Real residual_norm{0.0};
  svmp::FE::Real total_residual{0.0};
};

/// The residual target - current of two nodal moment vectors; throws on
/// different sizes or a non-finite residual.  Rank-local.
[[nodiscard]] ConservativePhaseMomentMismatch conservativePhaseMomentMismatch(
    std::span<const svmp::FE::Real> current,
    std::span<const svmp::FE::Real> target);

/// Outcome and accounting of reconcileConservativePhaseGeometry.
// Moved from Application/Core/ApplicationDriver.cpp (lines 24077-24097 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.
struct ConservativePhaseGeometryReconciliationResult {
  bool success{false};
  bool target_reached{false};
  bool limited_by_displacement{false};
  bool limited_by_topology{false};
  int iterations{0};
  int line_search_evaluations{0};
  int geometry_refresh_requests{0};
  int geometry_rebuilds{0};
  int rejected_geometry_trials{0};
  std::size_t contact_protected_nodes{0u};
  svmp::FE::Real allowed_interface_displacement{0.0};
  svmp::FE::Real accumulated_interface_displacement_bound{0.0};
  svmp::FE::Real initial_residual_norm{0.0};
  svmp::FE::Real final_residual_norm{0.0};
  svmp::FE::Real maximum_final_nodal_residual{0.0};
  svmp::FE::Real final_total_residual{0.0};
  svmp::FE::Real maximum_removed_contact_increment{0.0};
  std::string last_rejected_trial_diagnostic{};
  std::string diagnostic{};
};

/**
 * @brief Corrects the level-set slice of @p candidate until the projected
 *        nodal liquid moments match @p target_liquid_phase_mass.
 *
 * Uses host.fe_system.  Every trial refreshes the cut geometry through
 * @p transaction; on return the transaction holds the geometry of the
 * returned @p candidate.  Increments at nodes flagged in
 * @p contact_protected_nodes (graph size, or empty) are removed.  Collective
 * over the FE system's active communicator.
 */
[[nodiscard]] ConservativePhaseGeometryReconciliationResult
reconcileConservativePhaseGeometry(
    const ConservativePhaseWorkflowHost& host,
    ConservativePhaseRequest& request,
    std::span<const svmp::FE::Real> target_liquid_phase_mass,
    std::vector<svmp::FE::Real>& candidate,
    ConservativePhaseGeometryTransaction& transaction,
    std::span<const std::uint8_t> contact_protected_nodes = {});

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSETPHASEGEOMETRYRECONCILIATION_H
