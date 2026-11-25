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
    // A converged correction whose sharp volume attains the kinematic target.
    // On failure the caller's endpoint vector is unchanged.
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
    // or, if its connected cut-cell patch cannot take it, remains unplaced.
    std::size_t sign_preserving_redistributed_dofs{0u};
    std::size_t sign_preserving_skipped_dofs{0u};
    // Nodes left at their transported value because the correction would
    // have changed the class of an adjacent cut: its fragment collapsed
    // (measure at most the tolerance), nearly tangent (at most its square
    // root) or regular, with the measure taken on the cell's reference
    // element as the generated-interface builder takes it, or a side's
    // volume rule pruned for a volume fraction below
    // CutIntegrationContext::minGeneratedCutVolumeFraction().  The volume
    // they would have moved goes to the eligible nodes of their cells like
    // the limited part of a share (counted as redistributed or skipped),
    // except where that would change the class of a cell.
    std::size_t degeneracy_frozen_dofs{0u};
    // The frozen DOFs themselves (ascending field DOF numbers, identical on
    // every rank).
    std::vector<std::size_t> degeneracy_frozen_dof_indices{};
    Real previous_negative_volume{0.0};
    Real transported_negative_volume{0.0};
    Real reconciled_negative_volume{0.0};
    // int_Gamma w . n ds at the previous and at the reconciled endpoint.
    Real previous_interface_flux{0.0};
    Real current_interface_flux{0.0};
    // dt (previous_interface_flux + current_interface_flux) / 2.
    Real kinematic_volume_change{0.0};
    // Each endpoint volume change minus the trapezoidal flux on its own geometry.
    Real transported_volume_error{0.0};
    Real reconciled_volume_error{0.0};
    Real max_abs_correction{0.0};
    std::string diagnostic{};
    // Sharp-volume acceptance includes roundoff, the scaled solve tolerance,
    // and the trapezoidal time error of a pointwise kinematically exact trace.
    Real volume_tolerance{};
    Real transported_interface_flux{};
    bool volume_balance_satisfied{false};
    // A finite candidate exhausted the nonlinear correction or could not
    // attain the physical budget subject to the transported cut topology.
    bool retryable_failure{false};
};

/// Physical endpoint of a candidate representation, measured with its actual
/// transport velocity. Measuring never performs a level-set correction.
struct LevelSetKinematicEndpoint {
    bool success{false};
    Real negative_volume{};
    Real interface_flux{};
    std::string diagnostic;
};

[[nodiscard]] LevelSetKinematicEndpoint measureLevelSetKinematicEndpoint(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs,
    const spaces::FunctionSpace& velocity_space,
    Real isovalue,
    Real tolerance,
    std::span<const Real> level_set,
    std::span<const Real> velocity);

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
 * With the transported rate d = (phi_t - phi_p) / dt held fixed, form the
 * lumped interface residual r_i = ((M_p + M(phi)) d + f_p + f(phi,w_t))_i / 2,
 * M_ij = int_Gamma N_i N_j / |grad phi|, and gbar_i = (g_p + g(phi))_i / 2.
 * On each simplex K let E_K = V_K(phi) - V_K(phi_p) + gbar_K . (phi-phi_p).
 * Its exact swept-volume defect E_K is distributed among the same interface
 * test nodes with weights gbar_K,i / sum_j gbar_K,j. The local update is
 * gbar_i (phi_i - phi_t,i) = -dt r_i + sum_K E_K gbar_K,i / sum_j gbar_K,j.
 * Summing these equations gives the physical trapezoidal normal flux for
 * the exact sharp-volume change, including flow through an open domain.
 * On a trace which already satisfies d + w.grad(phi) = 0 pointwise, use the
 * unmodified kinematics and retain its temporal quadrature remainder in the
 * attainable volume tolerance. This preserves an exactly transported plane.
 * No node changes its sign class or moves toward the isovalue by more than
 * half of its transported distance. The limited part of a nodal share is
 * moved to the other nodes of its cells using their lumped weights. Nodes of
 * a cell whose cut would change its class (the fragment degeneracy of the
 * generated-interface builder, measured on the reference element, or the
 * pruning of a side's volume rule by the geometry snapshot) keep their
 * transported value; the volume they would have moved goes to the eligible
 * nodes of their cells in the same way, except into a cell whose class that
 * would change. If those nodes are blocked, the unused share follows successive
 * connected cut-cell rings to available trace nodes. Pinned interface vertices
 * retain their coefficients and contribute their complete flux in this allocation.
 * The cut topology of the accepted state is preserved.
 *
 * Supported cells are affine Triangle3 (2D) and Tetra4 (3D) with one scalar
 * level-set DOF per vertex and no other level-set DOFs. Corner-only linear
 * velocities can be used without a space. Pass the velocity FE space for a
 * higher-order field so its full trace and quadrature degree can be used.
 * Coefficient spans are complete
 * (replicated) field slices indexed by the field DOF numbering.  On a
 * distributed mesh only owned cells contribute and the nodal sums are reduced
 * on the level-set DOF communicator.
 *
 * @param isovalue    level-set value of the interface
 * @param tolerance   sign-class tolerance (values within it count as zero)
 * @param dt          accepted time-step size
 * @param velocity_space space associated with the supplied velocity DOFs
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
    std::vector<Real>& reconciled_level_set,
    const spaces::FunctionSpace* velocity_space = nullptr);

} // namespace svmp::FE::level_set
