/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_ORIENTATIONMANAGER_H
#define SVMP_FE_SPACES_ORIENTATIONMANAGER_H

/**
 * @file OrientationManager.h
 * @brief Edge/face orientation helpers for vector element assembly
 *
 * This header provides utilities for determining edge and face orientations
 * and for applying the resulting sign/permutation to DOF vectors. It is
 * independent of the Mesh module and operates purely on local/global vertex
 * index sets.
 */

#include "Core/Types.h"
#include <array>
#include <vector>
#include <algorithm>

namespace svmp {
namespace FE {
namespace spaces {

class OrientationManager {
public:
    using Sign = int;

    /// Orientation of a face relative to a reference
    struct FaceOrientation {
        int rotation{0};       ///< Cyclic rotation (0..n-1)
        bool reflection{false};///< Whether orientation is flipped
        Sign sign{+1};         ///< +1 or -1 for normal/tangent sign
        std::vector<int> vertex_perm; ///< global[i] = local[vertex_perm[i]]
        std::vector<int> perm; ///< Optional explicit permutation of DOFs
    };

    // ---------------------------------------------------------------------
    // Edge orientation
    // ---------------------------------------------------------------------

    /**
     * @brief Simple edge orientation based on vertex ordering
     *
     * Returns +1 if v0 < v1 (canonical direction), -1 otherwise.
     */
    static Sign edge_orientation(int v0, int v1) {
        return (v0 <= v1) ? +1 : -1;
    }

    /**
     * @brief Edge orientation relative to a reference direction
     *
     * Returns +1 if (local_v0, local_v1) matches (ref_v0, ref_v1),
     * -1 if it matches the reversed ordering, throws otherwise.
     */
    static Sign edge_orientation(int local_v0, int local_v1,
                                 int ref_v0, int ref_v1);

    /**
     * @brief Apply edge orientation to DOFs
     *
     * When swap_vertex_dofs is true, the first two entries are treated
     * as vertex DOFs and swapped if sign < 0, while interior DOFs are
     * multiplied by sign. When false, all DOFs are multiplied by sign.
     */
    static std::vector<Real> orient_edge_dofs(
        const std::vector<Real>& edge_dofs,
        Sign sign,
        bool swap_vertex_dofs);

    /**
     * @brief Orient H(curl) edge DOFs for an oriented edge
     *
     * For Nédélec-type edge DOFs (moments of tangential component), a reversed
     * edge orientation flips the tangent direction and reverses the ordering
     * of higher-order edge modes. This helper applies:
     *   - sign * reverse(edge_dofs)  when sign < 0
     *   - identity                  when sign > 0
     */
    static std::vector<Real> orient_hcurl_edge_dofs(
        const std::vector<Real>& edge_dofs,
        Sign sign);

    // ---------------------------------------------------------------------
    // Face orientation for triangles and quads
    // ---------------------------------------------------------------------

    /// Orientation for triangle faces
    static FaceOrientation triangle_face_orientation(
        const std::array<int, 3>& local,
        const std::array<int, 3>& global);

    /// Orientation for quadrilateral faces
    static FaceOrientation quad_face_orientation(
        const std::array<int, 4>& local,
        const std::array<int, 4>& global);

    // ---------------------------------------------------------------------
    // Permutation utilities
    // ---------------------------------------------------------------------

    /// Compute the sign (+1/-1) of a permutation expressed in image form
    static Sign permutation_sign(const std::vector<int>& perm);

    /// Apply permutation perm (new[i] = old[perm[i]])
    static std::vector<Real> apply_permutation(
        const std::vector<Real>& values,
        const std::vector<int>& perm);

    /// Compose permutations p1 ∘ p2 (apply p2 then p1)
    static std::vector<int> compose_permutations(
        const std::vector<int>& p1,
        const std::vector<int>& p2);

    /// Invert permutation
    static std::vector<int> invert_permutation(
        const std::vector<int>& perm);

    /// Check if permutation is even
    static bool is_even_permutation(const std::vector<int>& perm) {
        return permutation_sign(perm) > 0;
    }

    // ---------------------------------------------------------------------
    // Canonical ordering
    // ---------------------------------------------------------------------

    /// Canonical ordering indices for triangle vertices (sorted by value)
    static std::array<int, 3> canonical_ordering(const std::array<int, 3>& v);

    /// Canonical ordering indices for quad vertices (sorted by value)
    static std::array<int, 4> canonical_ordering(const std::array<int, 4>& v);

    // ---------------------------------------------------------------------
    // Face DOF orientation helpers (minimal implementation)
    // ---------------------------------------------------------------------

    /// Orient Pk triangle face DOFs according to FaceOrientation
    static std::vector<Real> orient_triangle_face_dofs(
        const std::vector<Real>& face_dofs,
        const FaceOrientation& orientation,
        int poly_order);

    /// Orient Qk quadrilateral face DOFs according to FaceOrientation
    static std::vector<Real> orient_quad_face_dofs(
        const std::vector<Real>& face_dofs,
        const FaceOrientation& orientation,
        int poly_order);

    /**
     * @brief Orient H(curl) triangle-face tangential DOFs (Nédélec 1st kind, simplex faces)
     *
     * This orients face-interior DOFs on a triangular face for higher-order
     * Nédélec (H(curl)) bases on simplex elements (e.g., Tetra). The face
     * DOFs are assumed to be ordered as:
     *   [ u-directed block (k*(k+1)/2), v-directed block (k*(k+1)/2) ]
     * where k is the polynomial order of the volume element. Each block
     * corresponds to tangential moments against a scalar P_{k-1} basis on
     * the face in the reference (u,v) coordinate system.
     *
     * Note: Unlike quad faces, triangle symmetries can mix the two tangential
     * directions. This routine applies the appropriate 2×2 transformation
     * induced by the vertex permutation, along with the scalar basis
     * permutation for P_{k-1}.
     */
    static std::vector<Real> orient_hcurl_triangle_face_dofs(
        const std::vector<Real>& face_dofs,
        const FaceOrientation& orientation,
        int poly_order);

    /**
     * @brief Orient H(curl) quad-face tangential DOFs (Nédélec 1st kind, tensor-product)
     *
     * This orients face-interior DOFs on a quadrilateral face for higher-order
     * Nédélec (H(curl)) bases on tensor-product elements (e.g., Hex). The face
     * DOFs are assumed to be ordered as:
     *   [ u-directed block (k*(k+1)), w-directed block (k*(k+1)) ]
     * where k is the polynomial order and the blocks correspond to the
     * Q_{k-1,k} and Q_{k,k-1} tangential moment sets, respectively.
     */
    static std::vector<Real> orient_hcurl_quad_face_dofs(
        const std::vector<Real>& face_dofs,
        const FaceOrientation& orientation,
        int poly_order);
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_ORIENTATIONMANAGER_H
