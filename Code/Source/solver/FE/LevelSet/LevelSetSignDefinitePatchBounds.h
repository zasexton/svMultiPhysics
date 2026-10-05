#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Accepted-step one-ring bounds of a transported P1 level set on
 *        vertices whose whole patch lies on one side of the interface.
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
 * @brief Outcome of one sign-definite patch bound.
 *
 * All counts and values are communicator-global and identical on every rank.
 */
struct LevelSetSignDefinitePatchBoundsResult {
    bool success{false};
    // True when at least one coefficient changed.
    bool applied{false};
    // Nodes whose patch is sign definite: every node of every cell around the
    // node is in one strict sign class at the previous state and, apart from
    // the node itself, at the candidate.
    std::size_t sign_definite_dofs{0u};
    // Sign-definite nodes whose candidate value left the range of their
    // patch at the previous state and was moved back to its nearest end.
    std::size_t bounded_dofs{0u};
    // Bounded nodes whose candidate value had left the sign class of their
    // patch, i.e. would have created a spurious interface.
    std::size_t sign_changes_prevented{0u};
    Real max_abs_correction{0.0};
    std::string diagnostic{};
};

/**
 * @brief Bound a transported P1 level set by the one-ring range of the
 *        previous state on every sign-definite patch.
 *
 * The exact solution of phi_t + w . grad phi = 0 is constant along the
 * characteristics, so after a step whose one-ring Courant number is at most
 * one the value at a node is a value of the previous P1 field inside the
 * node's patch:
 *     min_{patch} phi_p <= phi_i <= max_{patch} phi_p.
 * A Galerkin step does not have this local maximum principle: the consistent
 * mass matrix spreads the rate of a node onto its neighbours with negative
 * weights, and at a stagnation point of a diverging transport velocity the
 * central Galerkin stencil is anti-diffusive.  Next to a contact line, where
 * the transport velocity varies on the mesh scale and the transported field is
 * thin, a wall node whose patch lies entirely in one phase can therefore drift
 * to the isovalue and cross it, creating a spurious interface.
 *
 * This function restores the bound on exactly those nodes whose patch lies
 * in one phase: a node is sign definite when every node of every cell
 * containing it is in the same strict sign class (negative or positive, with
 * values within @p tolerance of the isovalue counting as on the interface) at
 * the previous state and, the node itself excepted, at the candidate.  Its
 * candidate value is clamped into [min, max] of the previous values over its
 * patch.  Every other node keeps its candidate value.
 *
 * Because the bounds of a sign-definite node lie in one strict sign class and
 * its neighbours keep that class, no cell around a changed node is cut before
 * or after the change, except when the candidate value had itself crossed the
 * isovalue: the interface, the contact line, the liquid measure and every
 * cut cell are untouched.  There is no numerical parameter.  The bound is
 * exact for one-ring Courant numbers up to one wherever the characteristic
 * starts inside the mesh, i.e. away from inflow boundaries, so it must not
 * be used with level-set inflow boundaries.  For larger steps it only
 * restricts values away from the interface.
 *
 * Supported cells are affine Triangle3 (2D) and Tetra4 (3D) with one scalar
 * level-set DOF per vertex.  Coefficient spans are complete (replicated)
 * field slices indexed by the field DOF numbering.  On a distributed mesh
 * only owned cells contribute and the patch data are reduced on the
 * level-set DOF communicator, so the result does not depend on the
 * partition.
 *
 * @param isovalue    level-set value of the interface
 * @param tolerance   sign-class tolerance (values within it count as zero)
 */
[[nodiscard]] LevelSetSignDefinitePatchBoundsResult
boundLevelSetOnSignDefinitePatches(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    Real isovalue,
    Real tolerance,
    std::span<const Real> previous_level_set,
    std::span<const Real> candidate_level_set,
    std::vector<Real>& bounded_level_set);

} // namespace svmp::FE::level_set
