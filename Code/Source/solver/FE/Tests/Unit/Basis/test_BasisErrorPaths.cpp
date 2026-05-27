/**
 * @file test_BasisErrorPaths.cpp
 * @brief Tests for error handling in basis function constructors and factory
 */

#include <gtest/gtest.h>

#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/BasisExceptions.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/BernsteinBasis.h"
#include "FE/Basis/HermiteBasis.h"
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/ModalTransform.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/SpectralBasis.h"
#include "FE/Basis/VectorBasis.h"

using namespace svmp::FE;
using namespace svmp::FE::basis;

namespace {

class MalformedModalBasis : public BasisFunction {
public:
    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }

    void evaluate_values(const math::Vector<Real, 3>&,
                         std::vector<Real>& values) const override
    {
        values.assign(1u, Real(1));
    }
};

class CompleteFallbackBasis : public BasisFunction {
public:
    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Triangle3; }
    int dimension() const noexcept override { return 2; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override
    {
        values.resize(size());
        values[0] = Real(1) + xi[0];
        values[1] = Real(2) + xi[1];
    }

    void evaluate_gradients(const math::Vector<Real, 3>&,
                            std::vector<Gradient>& gradients) const override
    {
        gradients.assign(size(), Gradient{});
        gradients[0][0] = Real(1);
        gradients[1][1] = Real(1);
    }

    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override
    {
        hessians.assign(size(), Hessian{});
        for (std::size_t d = 0; d < hessians.size(); ++d) {
            for (std::size_t r = 0; r < 3u; ++r) {
                for (std::size_t c = 0; c < 3u; ++c) {
                    hessians[d](r, c) = Real(100) * static_cast<Real>(d + 1u) +
                                        Real(10) * static_cast<Real>(r) +
                                        static_cast<Real>(c) + xi[2];
                }
            }
        }
    }
};

} // namespace

// =============================================================================
// LagrangeBasis error paths
// =============================================================================

TEST(BasisErrorPaths, LagrangeUnknownElementThrows) {
    EXPECT_THROW(LagrangeBasis(ElementType::Unknown, 1), BasisElementCompatibilityException);
}

TEST(BasisErrorPaths, LagrangeNegativeOrderThrows) {
    EXPECT_THROW(LagrangeBasis(ElementType::Line2, -1), BasisConfigurationException);
}

// =============================================================================
// HierarchicalBasis error paths
// =============================================================================

TEST(BasisErrorPaths, HierarchicalUnknownElementThrows) {
    EXPECT_THROW(HierarchicalBasis(ElementType::Unknown, 2), BasisElementCompatibilityException);
}

// =============================================================================
// BernsteinBasis error paths
// =============================================================================

TEST(BasisErrorPaths, BernsteinUnknownElementThrows) {
    EXPECT_THROW(BernsteinBasis(ElementType::Unknown, 2), BasisElementCompatibilityException);
}

// =============================================================================
// SpectralBasis error paths
// =============================================================================

TEST(BasisErrorPaths, SpectralOnWedgeConstructs) {
    EXPECT_NO_THROW(SpectralBasis(ElementType::Wedge6, 2));
}

TEST(BasisErrorPaths, SpectralOnPyramidConstructs) {
    EXPECT_NO_THROW(SpectralBasis(ElementType::Pyramid5, 2));
}

// =============================================================================
// SerendipityBasis error paths
// =============================================================================

TEST(BasisErrorPaths, SerendipityUnsupportedElementThrows) {
    // Serendipity on tetrahedra is not supported
    EXPECT_THROW(SerendipityBasis(ElementType::Tetra4, 2), BasisElementCompatibilityException);
}

TEST(BasisErrorPaths, SerendipityHigherOrderRequestsThrow) {
    EXPECT_NO_THROW(SerendipityBasis(ElementType::Quad4, 3));
    EXPECT_THROW(SerendipityBasis(ElementType::Hex8, 3), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Wedge15, 3), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Pyramid13, 3), NotImplementedException);
}

TEST(BasisErrorPaths, SerendipityGeometryModeDoesNotUnlockHigherOrderHybridFamilies) {
    EXPECT_THROW(SerendipityBasis(ElementType::Hex20, 3, true), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Wedge15, 3, true), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Pyramid13, 3, true), NotImplementedException);
}

// =============================================================================
// HermiteBasis error paths
// =============================================================================

TEST(BasisErrorPaths, HermiteOnTriangleThrows) {
    EXPECT_THROW(HermiteBasis(ElementType::Triangle3, 3), BasisElementCompatibilityException);
}

TEST(BasisErrorPaths, HermiteOnTetraThrows) {
    EXPECT_THROW(HermiteBasis(ElementType::Tetra4, 3), BasisElementCompatibilityException);
}

TEST(BasisErrorPaths, HermiteWrongOrderThrows) {
    // Hermite currently only supports cubic (order 3)
    EXPECT_THROW(HermiteBasis(ElementType::Line2, 2), NotImplementedException);
}

TEST(BasisErrorPaths, HermiteHigherOddOrderStillThrows) {
    EXPECT_THROW(HermiteBasis(ElementType::Quad4, 5), NotImplementedException);
    EXPECT_THROW(HermiteBasis(ElementType::Hex8, 5), NotImplementedException);
}

// =============================================================================
// BSplineBasis error paths
// =============================================================================

TEST(BasisErrorPaths, BSplineEmptyKnotsThrows) {
    EXPECT_THROW(BSplineBasis(2, {}), BasisConfigurationException);
}

TEST(BasisErrorPaths, BSplineTooFewKnotsThrows) {
    // Need at least degree+2 knots
    EXPECT_THROW(BSplineBasis(3, {0.0, 0.5, 1.0}), BasisConfigurationException);
}

// =============================================================================
// BasisFactory error paths
// =============================================================================

TEST(BasisErrorPaths, FactoryUnsupportedBasisTypeThrows) {
    BasisRequest req{ElementType::Line2, BasisType::Custom, 1, Continuity::C0, FieldType::Scalar};
    req.custom_id = "missing";
    EXPECT_THROW((void)basis_factory::create(req), BasisConfigurationException);
}

TEST(BasisErrorPaths, FactoryC1VectorFieldThrows) {
    BasisRequest req{ElementType::Line2, BasisType::Lagrange, 3, Continuity::C1, FieldType::Vector};
    EXPECT_THROW((void)basis_factory::create(req), BasisConfigurationException);
}

TEST(BasisErrorPaths, FactoryHCurlWithBernsteinThrows) {
    BasisRequest req{ElementType::Quad4, BasisType::Bernstein, 0, Continuity::H_curl, FieldType::Vector};
    EXPECT_THROW((void)basis_factory::create(req), BasisConfigurationException);
}

TEST(BasisErrorPaths, NodeOrderingInvalidNodeThrows) {
    EXPECT_THROW((void)ReferenceNodeLayout::get_node_coords(ElementType::Quad8, 99u), BasisNodeOrderingException);
}

TEST(BasisErrorPaths, VectorBasisScalarEvaluationThrows) {
    RaviartThomasBasis basis(ElementType::Triangle3, 0);
    const math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0.0)};
    std::vector<Real> values;
    EXPECT_THROW(basis.evaluate_values(xi, values), BasisEvaluationException);
}

TEST(BasisErrorPaths, RaviartThomasNegativeOrderThrowsInvalidArgument) {
    try {
        (void)RaviartThomasBasis(ElementType::Triangle3, -1);
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::InvalidArgument);
    }
}

TEST(BasisErrorPaths, NedelecNegativeOrderThrowsInvalidArgument) {
    try {
        (void)NedelecBasis(ElementType::Triangle3, -1);
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::InvalidArgument);
    }
}

TEST(BasisErrorPaths, HybridVectorHigherOrdersConstruct) {
    EXPECT_NO_THROW((void)RaviartThomasBasis(ElementType::Wedge6, 3));
    EXPECT_NO_THROW((void)RaviartThomasBasis(ElementType::Pyramid5, 3));
    EXPECT_NO_THROW((void)NedelecBasis(ElementType::Wedge6, 3));
    EXPECT_NO_THROW((void)NedelecBasis(ElementType::Pyramid5, 3));
}

TEST(BasisErrorPaths, BDMOrderZeroThrowsInvalidArgument) {
    EXPECT_THROW(BDMBasis(ElementType::Triangle3, 0), BasisConfigurationException);
    EXPECT_THROW(BDMBasis(ElementType::Tetra4, 0), BasisConfigurationException);
}

TEST(BasisErrorPaths, BDMQuadHigherOrderStillThrows) {
    EXPECT_THROW(BDMBasis(ElementType::Quad4, 2), NotImplementedException);
}

TEST(BasisErrorPaths, BDMUnsupportedTopologyThrows) {
    EXPECT_THROW(BDMBasis(ElementType::Hex8, 1), BasisElementCompatibilityException);
    EXPECT_THROW(BDMBasis(ElementType::Wedge6, 1), BasisElementCompatibilityException);
    EXPECT_THROW(BDMBasis(ElementType::Pyramid5, 1), BasisElementCompatibilityException);
}

TEST(BasisErrorPaths, ModalTransformMalformedModalBasisThrowsConstructionException) {
    MalformedModalBasis modal_basis;
    LagrangeBasis nodal_basis(ElementType::Line2, 1);
    EXPECT_THROW((void)ModalTransform(modal_basis, nodal_basis), BasisConstructionException);
}

TEST(BasisErrorPaths, BasisFunctionDefaultsThrowForMissingDerivatives) {
    MalformedModalBasis basis;
    const math::Vector<Real, 3> xi{Real(0), Real(0), Real(0)};
    std::vector<Gradient> gradients;
    std::vector<Hessian> hessians;

    EXPECT_THROW(basis.evaluate_gradients(xi, gradients), BasisEvaluationException);
    EXPECT_THROW(basis.evaluate_hessians(xi, hessians), BasisEvaluationException);
}

TEST(BasisErrorPaths, BasisFunctionFallbackWritesFlatAndStridedLayouts) {
    CompleteFallbackBasis basis;
    const std::vector<math::Vector<Real, 3>> points = {
        {Real(0.25), Real(0.5), Real(-0.25)},
        {Real(-0.5), Real(0.75), Real(0.125)}
    };

    std::vector<Real> flat_values(basis.size());
    std::vector<Real> flat_gradients(basis.size() * 3u);
    std::vector<Real> flat_hessians(basis.size() * 9u);
    basis.evaluate_values_to(points.front(), flat_values.data());
    basis.evaluate_gradients_to(points.front(), flat_gradients.data());
    basis.evaluate_hessians_to(points.front(), flat_hessians.data());

    std::vector<Real> expected_values;
    std::vector<Gradient> expected_gradients;
    std::vector<Hessian> expected_hessians;
    basis.evaluate_all(points.front(), expected_values, expected_gradients, expected_hessians);
    for (std::size_t d = 0; d < basis.size(); ++d) {
        EXPECT_EQ(flat_values[d], expected_values[d]);
        for (std::size_t c = 0; c < 3u; ++c) {
            EXPECT_EQ(flat_gradients[d * 3u + c], expected_gradients[d][c]);
        }
        for (std::size_t rc = 0; rc < 9u; ++rc) {
            EXPECT_EQ(flat_hessians[d * 9u + rc], expected_hessians[d].data()[rc]);
        }
    }

    constexpr std::size_t output_stride = 3u;
    std::vector<Real> values(basis.size() * output_stride, Real(-99));
    std::vector<Real> gradients(basis.size() * 3u * output_stride, Real(-99));
    std::vector<Real> hessians(basis.size() * 9u * output_stride, Real(-99));
    basis.evaluate_at_quadrature_points_strided(
        points, output_stride, values.data(), gradients.data(), hessians.data());

    for (std::size_t q = 0; q < points.size(); ++q) {
        basis.evaluate_all(points[q], expected_values, expected_gradients, expected_hessians);
        for (std::size_t d = 0; d < basis.size(); ++d) {
            EXPECT_EQ(values[d * output_stride + q], expected_values[d]);
            for (std::size_t c = 0; c < 3u; ++c) {
                EXPECT_EQ(gradients[(d * 3u + c) * output_stride + q],
                          expected_gradients[d][c]);
            }
            for (std::size_t rc = 0; rc < 9u; ++rc) {
                EXPECT_EQ(hessians[(d * 9u + rc) * output_stride + q],
                          expected_hessians[d].data()[rc]);
            }
        }
    }

    for (std::size_t d = 0; d < basis.size(); ++d) {
        EXPECT_EQ(values[d * output_stride + 2u], Real(-99));
    }
}
