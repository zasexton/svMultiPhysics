#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Accepted-step reconciliation of a transported P1 level set with the
 *        kinematic interface flux of its own transport velocity.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

/**
 * @brief Outcome of one kinematic reconciliation.
 *
 * Volumes are the sharp P1 cut measures of {phi < isovalue}.  All values are
 * communicator-global and identical on every rank.
 */
struct LevelSetKinematicReconciliationResult {
    bool success{false};
    // True when at least one coefficient changed.
    bool applied{false};
    bool converged{false};
    int iterations{0};
    std::size_t previous_interface_cells{0u};
    std::size_t current_interface_cells{0u};
    std::size_t corrected_dofs{0u};
    // Nodes whose correction would have changed their sign class (negative,
    // within tolerance of zero, positive) or moved them more than halfway
    // toward the isovalue.  They move halfway; the rest of their share of the
    // volume change goes to the other nodes of their cells (redistributed)
    // or, if those cannot take it, is dropped (skipped).
    std::size_t sign_preserving_redistributed_dofs{0u};
    std::size_t sign_preserving_skipped_dofs{0u};
    // Nodes left at their transported value because the correction would
    // have changed the class of an adjacent cut: its fragment collapsed
    // (measure at most the tolerance), nearly tangent (at most its square
    // root) or regular, or a side's volume rule pruned for a volume fraction
    // below CutIntegrationContext::minGeneratedCutVolumeFraction().
    std::size_t degeneracy_frozen_dofs{0u};
    Real previous_negative_volume{0.0};
    Real transported_negative_volume{0.0};
    Real reconciled_negative_volume{0.0};
    // int_Gamma w . n ds at the previous and at the reconciled endpoint.
    Real previous_interface_flux{0.0};
    Real current_interface_flux{0.0};
    // dt (previous_interface_flux + current_interface_flux) / 2.
    Real kinematic_volume_change{0.0};
    // Volume change minus kinematic_volume_change before and after.
    Real transported_volume_error{0.0};
    Real reconciled_volume_error{0.0};
    Real max_abs_correction{0.0};
    std::string diagnostic{};
};

/**
 * @brief Reconcile a transported P1 level set with the kinematic interface
 *        flux of its transport velocity.
 *
 * For a P1 level set phi on simplices the liquid measure
 * V(phi) = |{phi < c}| has the exact nodal derivative dV/dphi_i = -g_i,
 *     g_i(phi) = int_{Gamma(phi)} N_i / |grad phi| ds,
 * and the kinematic interface flux of a velocity w splits into nodal parts
 *     F(phi, w) = int_{Gamma(phi)} w . n ds = sum_i f_i,
 *     f_i(phi, w) = int_{Gamma(phi)} N_i (w . grad phi) / |grad phi| ds.
 * A Galerkin transport step phi_p -> phi_t satisfies the kinematic condition
 * phi_t + w . grad phi = 0 only in the L2(Omega) sense, so the change of V
 * differs from the trapezoidal interface flux dt (F_p + F_t) / 2.  The
 * difference concentrates where the transport velocity varies on the mesh
 * scale, for example at a moving contact line.
 *
 * With the transported rate d = (phi_t - phi_p) / dt held fixed, the
 * reconciled endpoint phi solves the lumped interface-kinematic equations
 *     phi_i = phi_t,i - dt r_i(phi) / gbar_i(phi),
 *     r_i = ((M_p + M(phi)) d)_i / 2 + (f_{p,i} + f_i(phi, w_t)) / 2,
 *     gbar_i = (g_{p,i} + g_i(phi)) / 2,
 *     M_{ij}(phi) = int_{Gamma(phi)} N_i N_j / |grad phi| ds,
 * on every node that carries interface measure; the endpoint geometry is
 * updated by fixed-point iteration.  The row sums of (M_p + M(phi)) / 2 are
 * gbar, so summing the equations gives
 *     -gbar . (phi - phi_p) = dt (F_p + F(phi, w_t)) / 2:
 * the trapezoidal volume change equals the trapezoidal kinematic flux.  Each
 * nodal change depends only on the interface inside the node's support;
 * there is no global shift, no multiplier and no numerical parameter.  The
 * target is the flux of the transport velocity itself, so the measure is
 * conserved exactly when, and only as far as, that velocity is discretely
 * divergence free on the liquid.  No node changes its sign class or moves
 * toward the isovalue by more than half of its transported distance; the
 * limited part of a node's share of the volume change is moved to the other
 * nodes of its cells, weighted so that the same lumped volume moves.  Nodes of
 * a cell whose cut would change its class (the fragment degeneracy of the
 * generated-interface builder, or the pruning of a side's volume rule by the
 * geometry snapshot) keep their transported value.  The cut topology of the
 * accepted state is therefore preserved.
 *
 * Supported cells are affine Triangle3 (2D) and Tetra4 (3D) with one scalar
 * level-set DOF per vertex.  The velocity field must provide at least one DOF
 * per spatial component at every vertex.  Coefficient spans are complete
 * (replicated) field slices indexed by the field DOF numbering.  On a
 * distributed mesh only owned cells contribute and the nodal sums are reduced
 * on the level-set DOF communicator.
 *
 * @param isovalue    level-set value of the interface
 * @param tolerance   sign-class tolerance (values within it count as zero)
 * @param dt          accepted time-step size
 */
[[nodiscard]] LevelSetKinematicReconciliationResult
reconcileLevelSetWithKinematicFlux(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs,
    Real isovalue,
    Real tolerance,
    Real dt,
    std::span<const Real> previous_level_set,
    std::span<const Real> transported_level_set,
    std::span<const Real> previous_velocity,
    std::span<const Real> transported_velocity,
    std::vector<Real>& reconciled_level_set);

} // namespace svmp::FE::level_set
