/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_SYMMETRICTRIANGLEQUADRATURE_H
#define SVMP_FE_QUADRATURE_SYMMETRICTRIANGLEQUADRATURE_H

/**
 * @file SymmetricTriangleQuadrature.h
 * @brief High-efficiency symmetric quadrature rules for triangles
 *
 * Provides tabulated symmetric quadrature rules that use fewer points than
 * tensor-product rules while maintaining the same polynomial exactness.
 * Rules are based on Dunavant (1985) and Wandzura-Xiao (2003) formulas.
 *
 * Reference coordinates: triangle with vertices (0,0), (1,0), (0,1)
 * Reference measure: 0.5
 */

#include "QuadratureRule.h"
#include <array>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Symmetric quadrature rules for triangles with optimal point counts
 *
 * These rules exploit the 3-fold symmetry of the reference triangle to
 * minimize the number of quadrature points while maintaining full polynomial
 * exactness. Much more efficient than Duffy-transformed tensor product rules.
 */
class SymmetricTriangleQuadrature : public QuadratureRule {
public:
    /**
     * @brief Construct a symmetric triangle quadrature rule
     * @param requested_order Desired polynomial exactness (1-20 supported)
     *
     * The actual order achieved may be higher than requested.
     * Throws if order > 20 (use TriangleQuadrature for higher orders).
     */
    explicit SymmetricTriangleQuadrature(int requested_order);

    /**
     * @brief Maximum supported polynomial order
     */
    static constexpr int max_order() { return 20; }

private:
    void initialize_order_1();   // 1 point, order 1
    void initialize_order_2();   // 3 points, order 2
    void initialize_order_3();   // 4 points, order 3
    void initialize_order_4();   // 6 points, order 4
    void initialize_order_5();   // 7 points, order 5
    void initialize_order_6();   // 12 points, order 6
    void initialize_order_7();   // 13 points, order 7
    void initialize_order_8();   // 16 points, order 8
    void initialize_order_9();   // 19 points, order 9
    void initialize_order_10();  // 25 points, order 10
    void initialize_order_11();  // 27 points, order 11
    void initialize_order_12();  // 33 points, order 12
    void initialize_order_13();  // 37 points, order 13
    void initialize_order_14();  // 42 points, order 14
    void initialize_order_15();  // 48 points, order 15
    void initialize_order_16();  // 52 points, order 16
    void initialize_order_17();  // 61 points, order 17
    void initialize_order_18();  // 70 points, order 18
    void initialize_order_19();  // 73 points, order 19
    void initialize_order_20();  // 79 points, order 20

    /**
     * @brief Add a centroid-symmetric point (1 point from barycentric coords)
     */
    void add_centroid_point(Real weight);

    /**
     * @brief Add 3 symmetric points from one barycentric coordinate set
     * @param a First barycentric coordinate (b = a, c = 1-2a)
     */
    void add_3_symmetric_points(Real a, Real weight);

    /**
     * @brief Add 6 symmetric points from two distinct barycentric coordinates
     * @param a First barycentric coordinate
     * @param b Second barycentric coordinate (c = 1-a-b)
     */
    void add_6_symmetric_points(Real a, Real b, Real weight);

    std::vector<QuadPoint> pts_;
    std::vector<Real> wts_;
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_SYMMETRICTRIANGLEQUADRATURE_H
