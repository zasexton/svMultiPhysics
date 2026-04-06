/**
 * @file test_BasisErrorPaths.cpp
 * @brief Tests for error handling in basis function constructors and factory
 */

#include <gtest/gtest.h>

#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/BernsteinBasis.h"
#include "FE/Basis/HermiteBasis.h"
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/SpectralBasis.h"
#include "FE/Core/FEException.h"

using namespace svmp::FE;
using namespace svmp::FE::basis;

// =============================================================================
// LagrangeBasis error paths
// =============================================================================

TEST(BasisErrorPaths, LagrangeUnknownElementThrows) {
    EXPECT_THROW(LagrangeBasis(ElementType::Unknown, 1), FEException);
}

TEST(BasisErrorPaths, LagrangeNegativeOrderThrows) {
    EXPECT_THROW(LagrangeBasis(ElementType::Line2, -1), FEException);
}

// =============================================================================
// HierarchicalBasis error paths
// =============================================================================

TEST(BasisErrorPaths, HierarchicalUnknownElementThrows) {
    EXPECT_THROW(HierarchicalBasis(ElementType::Unknown, 2), FEException);
}

// =============================================================================
// BernsteinBasis error paths
// =============================================================================

TEST(BasisErrorPaths, BernsteinUnsupportedElementThrows) {
    // Bernstein on Tetra4 is not supported
    EXPECT_THROW(BernsteinBasis(ElementType::Tetra4, 2), FEException);
}

// =============================================================================
// SpectralBasis error paths
// =============================================================================

TEST(BasisErrorPaths, SpectralOnWedgeThrows) {
    EXPECT_THROW(SpectralBasis(ElementType::Wedge6, 2), FEException);
}

TEST(BasisErrorPaths, SpectralOnPyramidThrows) {
    EXPECT_THROW(SpectralBasis(ElementType::Pyramid5, 2), FEException);
}

// =============================================================================
// SerendipityBasis error paths
// =============================================================================

TEST(BasisErrorPaths, SerendipityUnsupportedElementThrows) {
    // Serendipity on tetrahedra is not supported
    EXPECT_THROW(SerendipityBasis(ElementType::Tetra4, 2), FEException);
}

// =============================================================================
// HermiteBasis error paths
// =============================================================================

TEST(BasisErrorPaths, HermiteOnTriangleThrows) {
    EXPECT_THROW(HermiteBasis(ElementType::Triangle3, 3), FEException);
}

TEST(BasisErrorPaths, HermiteOnTetraThrows) {
    EXPECT_THROW(HermiteBasis(ElementType::Tetra4, 3), FEException);
}

TEST(BasisErrorPaths, HermiteWrongOrderThrows) {
    // Hermite currently only supports cubic (order 3)
    EXPECT_THROW(HermiteBasis(ElementType::Line2, 2), FEException);
}

// =============================================================================
// BSplineBasis error paths
// =============================================================================

TEST(BasisErrorPaths, BSplineEmptyKnotsThrows) {
    EXPECT_THROW(BSplineBasis(2, {}), FEException);
}

TEST(BasisErrorPaths, BSplineTooFewKnotsThrows) {
    // Need at least degree+2 knots
    EXPECT_THROW(BSplineBasis(3, {0.0, 0.5, 1.0}), FEException);
}

// =============================================================================
// BasisFactory error paths
// =============================================================================

TEST(BasisErrorPaths, FactoryUnsupportedBasisTypeThrows) {
    BasisRequest req{ElementType::Line2, BasisType::Custom, 1, Continuity::C0, FieldType::Scalar};
    EXPECT_THROW(BasisFactory::create(req), FEException);
}

TEST(BasisErrorPaths, FactoryC1VectorFieldThrows) {
    BasisRequest req{ElementType::Line2, BasisType::Lagrange, 3, Continuity::C1, FieldType::Vector};
    EXPECT_THROW(BasisFactory::create(req), FEException);
}

TEST(BasisErrorPaths, FactoryHCurlWithBernsteinThrows) {
    BasisRequest req{ElementType::Quad4, BasisType::Bernstein, 0, Continuity::H_curl, FieldType::Vector};
    EXPECT_THROW(BasisFactory::create(req), FEException);
}
