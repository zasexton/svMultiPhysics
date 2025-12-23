/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_SYMMETRICTETRAHEDRONQUADRATURE_H
#define SVMP_FE_QUADRATURE_SYMMETRICTETRAHEDRONQUADRATURE_H

/**
 * @file SymmetricTetrahedronQuadrature.h
 * @brief High-efficiency symmetric quadrature rules for tetrahedra
 *
 * Provides tabulated symmetric quadrature rules that use fewer points than
 * tensor-product rules while maintaining the same polynomial exactness.
 * Rules are based on Keast (1986), Shunn-Ham (2012), and Witherden-Vincent (2015).
 *
 * Reference coordinates: tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 * Reference measure: 1/6
 */

#include "QuadratureRule.h"
#include <array>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Symmetric quadrature rules for tetrahedra with optimal point counts
 *
 * These rules exploit the 24-fold symmetry of the reference tetrahedron to
 * minimize the number of quadrature points while maintaining full polynomial
 * exactness. Much more efficient than Duffy-transformed tensor product rules.
 *
 * Notes on high orders:
 * - Orders 1-8 use tabulated Keast rules (compact point sets).
 * - Orders 9-14 are accepted for API completeness but currently fall back to
 *   `TetrahedronQuadrature` to guarantee correctness.
 */
class SymmetricTetrahedronQuadrature : public QuadratureRule {
public:
    /**
     * @brief Construct a symmetric tetrahedron quadrature rule
     * @param requested_order Desired polynomial exactness (1-14 supported)
     *
     * The actual order achieved may be higher than requested.
     * Throws if order > 14 (use TetrahedronQuadrature for higher orders).
     */
    explicit SymmetricTetrahedronQuadrature(int requested_order);

    /**
     * @brief Maximum supported polynomial order
     */
    static constexpr int max_order() { return 14; }

private:
    static constexpr int max_tabulated_order() { return 8; }

    void initialize_order_1();   // 1 point
    void initialize_order_2();   // 4 points
    void initialize_order_3();   // 5 points
    void initialize_order_4();   // 11 points
    void initialize_order_5();   // 15 points
    void initialize_order_6();   // 24 points
    void initialize_order_7();   // 31 points
    void initialize_order_8();   // 45 points (Keast rule 10)

    /**
     * @brief Add 1 centroid-symmetric point (barycentric (1/4,1/4,1/4,1/4))
     */
    void add_1_point(Real weight);

    /**
     * @brief Add 4 symmetric points from one distinct barycentric coordinate
     * Points: (a,b,b,b) and permutations where 3b = 1-a
     */
    void add_4_symmetric_points(Real a, Real weight);

    /**
     * @brief Add 6 symmetric points from two distinct barycentric coordinates
     * Points: (a,a,b,b) and permutations where 2a + 2b = 1
     */
    void add_6_symmetric_points(Real a, Real weight);

    /**
     * @brief Add 12 symmetric points from three distinct barycentric coordinates
     * Points: (a,a,b,c) and permutations where 2a + b + c = 1
     */
    void add_12_symmetric_points(Real a, Real b, Real weight);

    /**
     * @brief Add 24 symmetric points from four distinct barycentric coordinates
     * Points: (a,b,c,d) and all permutations where a + b + c + d = 1
     */
    void add_24_symmetric_points(Real a, Real b, Real c, Real weight);

    std::vector<QuadPoint> pts_;
    std::vector<Real> wts_;
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_SYMMETRICTETRAHEDRONQUADRATURE_H
