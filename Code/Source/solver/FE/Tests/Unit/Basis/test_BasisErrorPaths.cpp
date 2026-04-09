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
    EXPECT_THROW(SerendipityBasis(ElementType::Quad4, 3), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Hex8, 3), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Wedge15, 3), NotImplementedException);
    EXPECT_THROW(SerendipityBasis(ElementType::Pyramid13, 3), NotImplementedException);
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
    EXPECT_THROW(BasisFactory::create(req), BasisConfigurationException);
}

TEST(BasisErrorPaths, FactoryC1VectorFieldThrows) {
    BasisRequest req{ElementType::Line2, BasisType::Lagrange, 3, Continuity::C1, FieldType::Vector};
    EXPECT_THROW(BasisFactory::create(req), BasisConfigurationException);
}

TEST(BasisErrorPaths, FactoryHCurlWithBernsteinThrows) {
    BasisRequest req{ElementType::Quad4, BasisType::Bernstein, 0, Continuity::H_curl, FieldType::Vector};
    EXPECT_THROW(BasisFactory::create(req), BasisConfigurationException);
}

TEST(BasisErrorPaths, NodeOrderingInvalidNodeThrows) {
    EXPECT_THROW((void)NodeOrdering::get_node_coords(ElementType::Quad8, 99u), BasisNodeOrderingException);
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

TEST(BasisErrorPaths, BDMHigherOrderThrowsNotImplemented) {
    EXPECT_THROW(BDMBasis(ElementType::Triangle3, 2), NotImplementedException);
    EXPECT_THROW(BDMBasis(ElementType::Quad4, 2), NotImplementedException);
}

TEST(BasisErrorPaths, BDMUnsupportedTopologyThrows) {
    EXPECT_THROW(BDMBasis(ElementType::Tetra4, 1), BasisElementCompatibilityException);
    EXPECT_THROW(BDMBasis(ElementType::Hex8, 1), BasisElementCompatibilityException);
}

TEST(BasisErrorPaths, ModalTransformMalformedModalBasisThrowsConstructionException) {
    MalformedModalBasis modal_basis;
    LagrangeBasis nodal_basis(ElementType::Line2, 1);
    EXPECT_THROW((void)ModalTransform(modal_basis, nodal_basis), BasisConstructionException);
}
