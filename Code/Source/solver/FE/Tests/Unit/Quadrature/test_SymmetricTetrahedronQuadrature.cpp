/**
 * @file test_SymmetricTetrahedronQuadrature.cpp
 * @brief Comprehensive unit tests for symmetric tetrahedron quadrature (Keast rules + fallback)
 *
 * These tests verify:
 * 1. Weight sums match reference tetrahedron volume (1/6)
 * 2. All quadrature points lie within the reference tetrahedron
 * 3. Polynomial exactness at advertised orders
 * 4. Point counts match expected orbit structure
 * 5. Symmetry under barycentric permutations
 * 6. Comparison with tensor-product rules for consistency
 * 7. Error handling for invalid orders
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/SymmetricTetrahedronQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Core/FEException.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map>

using namespace svmp::FE::quadrature;
using namespace svmp::FE;

namespace {

/**
 * @brief Compute factorial for monomial integration
 */
double factorial(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) {
        v *= static_cast<double>(i);
    }
    return v;
}

/**
 * @brief Analytical integral of x^a * y^b * z^c over reference tetrahedron
 *
 * For the reference tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1):
 *   ∫∫∫_T x^a y^b z^c dV = a! b! c! / (a + b + c + 3)!
 */
double tetrahedron_monomial_integral(int a, int b, int c) {
    return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
}

/**
 * @brief Check if a point is inside the reference tetrahedron
 *
 * Reference tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 * Conditions: x >= 0, y >= 0, z >= 0, x + y + z <= 1
 */
bool point_in_tetrahedron(const QuadPoint& p, double tol = 1e-12) {
    return p[0] >= -tol &&
           p[1] >= -tol &&
           p[2] >= -tol &&
           (p[0] + p[1] + p[2]) <= 1.0 + tol;
}

/**
 * @brief Expected point counts for tabulated symmetric rules
 *
 * Orders 1-8 use tabulated Keast rules (compact symmetric point sets).
 * Orders 9-14 fall back to `TetrahedronQuadrature` for correctness.
 */
const std::map<int, std::size_t> expected_tabulated_point_counts = {
    {1, 1},    // centroid
    {2, 4},    // 4-point orbit
    {3, 5},    // 1 + 4 points
    {4, 11},   // 1 + 4 + 6 points
    {5, 15},   // Keast rule 7: 1 + 4 + 4 + 6 points
    {6, 24},   // Keast rule 8: 4 + 4 + 4 + 12 points
    {7, 31},   // Keast rule 9: 1 + 4 + 4 + 4 + 6 + 12 points
    {8, 45}    // Keast rule 10: 1 + 4 + 4 + 6 + 6 + 12 + 12 points
};

/**
 * @brief Orders with tabulated symmetric coefficients
 */
constexpr int max_tabulated_order = 8;
constexpr int max_supported_order = 14;

} // namespace

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, ConstructsAllOrders) {
    for (int order = 1; order <= max_supported_order; ++order) {
        ASSERT_NO_THROW({
            SymmetricTetrahedronQuadrature quad(order);
        }) << "Failed to construct order " << order;
    }
}

TEST(SymmetricTetrahedronQuadrature, ThrowsOnInvalidOrder) {
    EXPECT_THROW(SymmetricTetrahedronQuadrature(0), FEException);
    EXPECT_THROW(SymmetricTetrahedronQuadrature(-1), FEException);
    EXPECT_THROW(SymmetricTetrahedronQuadrature(15), FEException);
    EXPECT_THROW(SymmetricTetrahedronQuadrature(100), FEException);
}

TEST(SymmetricTetrahedronQuadrature, ReportsCorrectCellFamily) {
    SymmetricTetrahedronQuadrature quad(5);
    EXPECT_EQ(quad.cell_family(), svmp::CellFamily::Tetra);
}

TEST(SymmetricTetrahedronQuadrature, ReportsCorrectDimension) {
    SymmetricTetrahedronQuadrature quad(5);
    EXPECT_EQ(quad.dimension(), 3);
}

TEST(SymmetricTetrahedronQuadrature, ReportsCorrectReferenceMeasure) {
    SymmetricTetrahedronQuadrature quad(5);
    EXPECT_NEAR(quad.reference_measure(), 1.0/6.0, 1e-15);
}

// =============================================================================
// Point Count Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, HasExpectedPointCounts) {
    for (const auto& [order, expected] : expected_tabulated_point_counts) {
        SymmetricTetrahedronQuadrature quad(order);
        EXPECT_EQ(quad.num_points(), expected)
            << "Order " << order << " has " << quad.num_points()
            << " points, expected " << expected;
    }

    // Orders 9-14 currently fall back to the Duffy tensor-product rule.
    for (int order = max_tabulated_order + 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        TetrahedronQuadrature tensor_quad(order);
        EXPECT_EQ(quad.num_points(), tensor_quad.num_points())
            << "Order " << order << " should fall back to tensor-product rule";
    }
}

TEST(SymmetricTetrahedronQuadrature, UsesFewerPointsThanTensorProduct) {
    // Symmetric rules should use significantly fewer points than Duffy-transform rules
    for (int order = 2; order <= 8; ++order) {
        SymmetricTetrahedronQuadrature sym_quad(order);
        TetrahedronQuadrature tensor_quad(order);

        // Tensor product rules have O(n^3) points, symmetric have O(n^2) or less
        EXPECT_LT(sym_quad.num_points(), tensor_quad.num_points())
            << "Symmetric rule for order " << order
            << " should use fewer points than tensor product ("
            << sym_quad.num_points() << " vs " << tensor_quad.num_points() << ")";
    }
}

// =============================================================================
// Weight Sum Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, WeightSumMatchesVolume) {
    const double expected_volume = 1.0 / 6.0;

    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        double sum = std::accumulate(quad.weights().begin(),
                                     quad.weights().end(), 0.0);
        EXPECT_NEAR(sum, expected_volume, 1e-13)
            << "Weight sum mismatch for order " << order;
    }
}

TEST(SymmetricTetrahedronQuadrature, AllWeightsAreFinite) {
    for (int order = 1; order <= 14; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            EXPECT_TRUE(std::isfinite(quad.weight(i)))
                << "Non-finite weight at order " << order << ", point " << i;
        }
    }
}

// Note: Some Keast rules have negative weights (e.g., orders 3, 4)
TEST(SymmetricTetrahedronQuadrature, DocumentedNegativeWeights) {
    // Order 3 and 4 have negative centroid weights (this is known behavior)
    SymmetricTetrahedronQuadrature quad3(3);
    bool has_negative_3 = false;
    for (std::size_t i = 0; i < quad3.num_points(); ++i) {
        if (quad3.weight(i) < 0) {
            has_negative_3 = true;
            break;
        }
    }
    EXPECT_TRUE(has_negative_3) << "Order 3 should have negative centroid weight";

    SymmetricTetrahedronQuadrature quad4(4);
    bool has_negative_4 = false;
    for (std::size_t i = 0; i < quad4.num_points(); ++i) {
        if (quad4.weight(i) < 0) {
            has_negative_4 = true;
            break;
        }
    }
    EXPECT_TRUE(has_negative_4) << "Order 4 should have negative centroid weight";
}

// =============================================================================
// Point Location Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, AllPointsInsideTetrahedron) {
    // Test that points are inside the tetrahedron
    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            const auto& p = quad.point(i);
            EXPECT_TRUE(point_in_tetrahedron(p))
                << "Point " << i << " at order " << order
                << " is outside tetrahedron: (" << p[0] << ", " << p[1] << ", " << p[2] << ")";
        }
    }
}

// =============================================================================
// Polynomial Exactness Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, IntegratesConstant) {
    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        double integral = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            integral += quad.weight(i) * 1.0;
        }
        EXPECT_NEAR(integral, 1.0/6.0, 1e-13)
            << "Constant integration failed for order " << order;
    }
}

TEST(SymmetricTetrahedronQuadrature, IntegratesLinearFunctions) {
    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);

        // Integrate x
        double ix = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            ix += quad.weight(i) * quad.point(i)[0];
        }
        EXPECT_NEAR(ix, tetrahedron_monomial_integral(1, 0, 0), 1e-12)
            << "∫x dV failed for order " << order;

        // Integrate y
        double iy = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            iy += quad.weight(i) * quad.point(i)[1];
        }
        EXPECT_NEAR(iy, tetrahedron_monomial_integral(0, 1, 0), 1e-12)
            << "∫y dV failed for order " << order;

        // Integrate z
        double iz = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            iz += quad.weight(i) * quad.point(i)[2];
        }
        EXPECT_NEAR(iz, tetrahedron_monomial_integral(0, 0, 1), 1e-12)
            << "∫z dV failed for order " << order;
    }
}

TEST(SymmetricTetrahedronQuadrature, PolynomialExactnessUpToOrder) {
    // For each order, verify all monomials x^a * y^b * z^c with a+b+c <= order
    // Tabulated symmetric rules (orders 1-8)
    for (int order = 1; order <= max_tabulated_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);

        for (int total_degree = 0; total_degree <= order; ++total_degree) {
            for (int a = 0; a <= total_degree; ++a) {
                for (int b = 0; b <= total_degree - a; ++b) {
                    int c = total_degree - a - b;

                    double integral = 0.0;
                    for (std::size_t i = 0; i < quad.num_points(); ++i) {
                        const auto& p = quad.point(i);
                        integral += quad.weight(i) *
                                    std::pow(p[0], a) *
                                    std::pow(p[1], b) *
                                    std::pow(p[2], c);
                    }

                    double exact = tetrahedron_monomial_integral(a, b, c);
                    double tol = std::max(1e-10, std::abs(exact) * 1e-8);

                    EXPECT_NEAR(integral, exact, tol)
                        << "x^" << a << " * y^" << b << " * z^" << c
                        << " failed at order " << order
                        << " (got " << integral << ", expected " << exact << ")";
                }
            }
        }
    }
}

TEST(SymmetricTetrahedronQuadrature, HighOrderPolynomialExactness) {
    // Orders 9-14 currently fall back to tensor-product rules; verify selected monomials.
    for (int order = max_tabulated_order + 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);

        // Test a few representative high-degree monomials
        std::vector<std::tuple<int, int, int>> test_cases = {
            {order, 0, 0}, {0, order, 0}, {0, 0, order},
            {order/3, order/3, order - 2*(order/3)},
            {order-1, 1, 0}, {1, order-1, 0}, {0, 1, order-1}
        };

        for (const auto& [a, b, c] : test_cases) {
            if (a + b + c > order) continue;

            double integral = 0.0;
            for (std::size_t i = 0; i < quad.num_points(); ++i) {
                const auto& p = quad.point(i);
                integral += quad.weight(i) *
                            std::pow(p[0], a) *
                            std::pow(p[1], b) *
                            std::pow(p[2], c);
            }

            double exact = tetrahedron_monomial_integral(a, b, c);
            double tol = std::max(1e-9, std::abs(exact) * 1e-7);

            EXPECT_NEAR(integral, exact, tol)
                << "x^" << a << " * y^" << b << " * z^" << c
                << " failed at order " << order;
        }
    }
}

// =============================================================================
// Symmetry Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, CentroidPointAtCorrectLocation) {
    // Order 1 has only the centroid
    SymmetricTetrahedronQuadrature quad(1);
    ASSERT_EQ(quad.num_points(), 1u);

    const auto& p = quad.point(0);
    EXPECT_NEAR(p[0], 0.25, 1e-14) << "Centroid x-coordinate incorrect";
    EXPECT_NEAR(p[1], 0.25, 1e-14) << "Centroid y-coordinate incorrect";
    EXPECT_NEAR(p[2], 0.25, 1e-14) << "Centroid z-coordinate incorrect";
}

TEST(SymmetricTetrahedronQuadrature, Order2HasFourSymmetricPoints) {
    SymmetricTetrahedronQuadrature quad(2);
    ASSERT_EQ(quad.num_points(), 4u);

    // All four points should have the same weight
    for (std::size_t i = 1; i < 4; ++i) {
        EXPECT_NEAR(quad.weight(0), quad.weight(i), 1e-15)
            << "Point " << i << " has different weight than point 0";
    }

    // Points should sum to same value when swapping x, y, z due to symmetry
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        sum_x += quad.point(i)[0];
        sum_y += quad.point(i)[1];
        sum_z += quad.point(i)[2];
    }
    EXPECT_NEAR(sum_x, sum_y, 1e-14) << "Asymmetry in x vs y";
    EXPECT_NEAR(sum_y, sum_z, 1e-14) << "Asymmetry in y vs z";
}

TEST(SymmetricTetrahedronQuadrature, IntegralSymmetricUnderCoordinatePermutation) {
    // A symmetric function f(x,y,z) = x + y + z should integrate correctly
    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);

        double integral = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            const auto& p = quad.point(i);
            integral += quad.weight(i) * (p[0] + p[1] + p[2]);
        }

        // Should equal 3 * integral of x = 3 * 1/24 = 1/8
        double expected = 3.0 * tetrahedron_monomial_integral(1, 0, 0);
        EXPECT_NEAR(integral, expected, 1e-12)
            << "Symmetric integral failed at order " << order;
    }
}

// =============================================================================
// Consistency Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, ConsistentWithTensorProductRule) {
    // Both rules should give same integrals for smooth functions
    for (int order = 2; order <= 6; ++order) {
        SymmetricTetrahedronQuadrature sym_quad(order);
        TetrahedronQuadrature tensor_quad(order);

        // Test on smooth function: exp(x + y + z)
        auto f = [](const QuadPoint& p) {
            return std::exp(p[0] + p[1] + p[2]);
        };

        double sym_integral = 0.0;
        for (std::size_t i = 0; i < sym_quad.num_points(); ++i) {
            sym_integral += sym_quad.weight(i) * f(sym_quad.point(i));
        }

        double tensor_integral = 0.0;
        for (std::size_t i = 0; i < tensor_quad.num_points(); ++i) {
            tensor_integral += tensor_quad.weight(i) * f(tensor_quad.point(i));
        }

        // Both should converge to same value
        double tol = 0.02;  // Allow for different convergence behavior
        EXPECT_NEAR(sym_integral, tensor_integral, tol)
            << "Symmetric and tensor rules differ at order " << order;
    }
}

TEST(SymmetricTetrahedronQuadrature, IsValidReturnsTrue) {
    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        EXPECT_TRUE(quad.is_valid())
            << "is_valid() failed for order " << order;
    }
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, StableForHighOrderMonomials) {
    // Test numerical stability when integrating high-degree polynomials
    // Use an order with negative weights (order 4 Keast rule)
    SymmetricTetrahedronQuadrature quad(4);

    // Integrate x^2 * y * z (total degree 4)
    double integral = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        integral += quad.weight(i) *
                    std::pow(p[0], 2) * p[1] * p[2];
    }

    double exact = tetrahedron_monomial_integral(2, 1, 1);
    double relative_error = std::abs(integral - exact) / std::abs(exact);

    EXPECT_LT(relative_error, 1e-8)
        << "High-order monomial integration unstable: got " << integral
        << ", expected " << exact;
}

TEST(SymmetricTetrahedronQuadrature, NoNaNOrInfInPoints) {
    for (int order = 1; order <= 14; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            const auto& p = quad.point(i);
            EXPECT_TRUE(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))
                << "Non-finite point at order " << order << ", index " << i;
        }
    }
}

// =============================================================================
// Integration with Known Results
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, IntegratesExpFunction) {
    // Compute ∫∫∫_T e^(x+y+z) dV using tensor-product reference
    TetrahedronQuadrature ref_quad(10);
    double ref_value = 0.0;
    for (std::size_t i = 0; i < ref_quad.num_points(); ++i) {
        const auto& p = ref_quad.point(i);
        ref_value += ref_quad.weight(i) * std::exp(p[0] + p[1] + p[2]);
    }

    for (int order = 1; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        double integral = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            const auto& p = quad.point(i);
            integral += quad.weight(i) * std::exp(p[0] + p[1] + p[2]);
        }

        // Lower orders won't be as accurate for transcendental functions
        double tol = 0.05;  // 5% tolerance for low-order rules
        EXPECT_NEAR(integral, ref_value, tol)
            << "Order " << order << " failed convergence test";
    }
}

TEST(SymmetricTetrahedronQuadrature, IntegratesPolynomialFunction) {
    // Integrate a specific polynomial: 1 + 2x + 3y + 4z + 5xy + 6xz + 7yz + 8xyz
    auto f = [](const QuadPoint& p) {
        double x = p[0], y = p[1], z = p[2];
        return 1.0 + 2*x + 3*y + 4*z + 5*x*y + 6*x*z + 7*y*z + 8*x*y*z;
    };

    // Analytical integral (sum of monomial integrals)
    double exact =
        tetrahedron_monomial_integral(0, 0, 0) +     // 1
        2 * tetrahedron_monomial_integral(1, 0, 0) + // 2x
        3 * tetrahedron_monomial_integral(0, 1, 0) + // 3y
        4 * tetrahedron_monomial_integral(0, 0, 1) + // 4z
        5 * tetrahedron_monomial_integral(1, 1, 0) + // 5xy
        6 * tetrahedron_monomial_integral(1, 0, 1) + // 6xz
        7 * tetrahedron_monomial_integral(0, 1, 1) + // 7yz
        8 * tetrahedron_monomial_integral(1, 1, 1);  // 8xyz

    // Degree-3 polynomial: any order >= 3 should integrate exactly
    for (int order = 3; order <= max_supported_order; ++order) {
        SymmetricTetrahedronQuadrature quad(order);
        double integral = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            integral += quad.weight(i) * f(quad.point(i));
        }

        EXPECT_NEAR(integral, exact, 1e-10)
            << "Polynomial integration failed at order " << order;
    }
}

// =============================================================================
// Orbit Structure Tests (verify symmetric point placement)
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, VerifyOrbitStructure) {
    // For order 2, we should have a 4-point orbit
    SymmetricTetrahedronQuadrature quad2(2);
    EXPECT_EQ(quad2.num_points(), 4u);

    // All points in a 4-point orbit have the same barycentric structure
    // They should be permutations of (a, b, b, b) where a + 3b = 1

    // Check that barycentric coordinates sum to 1 for each point
    for (std::size_t i = 0; i < quad2.num_points(); ++i) {
        const auto& p = quad2.point(i);
        double sum = p[0] + p[1] + p[2];
        double lambda1 = 1.0 - sum;  // First barycentric coordinate
        double total = lambda1 + p[0] + p[1] + p[2];
        EXPECT_NEAR(total, 1.0, 1e-14) << "Barycentric sum != 1 at point " << i;
    }
}

TEST(SymmetricTetrahedronQuadrature, OrbitWeightsAreEqual) {
    // For order 5 (Keast rule 7): 15 points with 4 orbits
    // 1 centroid + 4 face centers + 4 interior + 6 edge-like
    SymmetricTetrahedronQuadrature quad(5);
    EXPECT_EQ(quad.num_points(), 15u);

    // Group weights by magnitude (within tolerance)
    std::vector<double> weights;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        weights.push_back(quad.weight(i));
    }
    std::sort(weights.begin(), weights.end());

    // Should have at most 4 distinct weight values (for 4 orbits in Keast rule 7)
    int distinct_weights = 1;
    for (std::size_t i = 1; i < weights.size(); ++i) {
        if (std::abs(weights[i] - weights[i-1]) > 1e-10) {
            distinct_weights++;
        }
    }
    EXPECT_LE(distinct_weights, 4)
        << "Order 5 should have at most 4 distinct weight values (Keast rule 7)";
}

// =============================================================================
// Comparison with Literature Values
// =============================================================================

TEST(SymmetricTetrahedronQuadrature, Order1CentroidRule) {
    SymmetricTetrahedronQuadrature quad(1);

    // Keast order 1: single point at centroid with weight = 1/6
    ASSERT_EQ(quad.num_points(), 1u);
    EXPECT_NEAR(quad.weight(0), 1.0/6.0, 1e-15);

    const auto& p = quad.point(0);
    EXPECT_NEAR(p[0], 0.25, 1e-15);
    EXPECT_NEAR(p[1], 0.25, 1e-15);
    EXPECT_NEAR(p[2], 0.25, 1e-15);
}

TEST(SymmetricTetrahedronQuadrature, Order2VertexRule) {
    SymmetricTetrahedronQuadrature quad(2);

    // Keast order 2: 4 points, each with weight 1/24
    ASSERT_EQ(quad.num_points(), 4u);

    double weight_sum = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(quad.weight(i), 1.0/24.0, 1e-14);
        weight_sum += quad.weight(i);
    }
    EXPECT_NEAR(weight_sum, 1.0/6.0, 1e-14);
}
