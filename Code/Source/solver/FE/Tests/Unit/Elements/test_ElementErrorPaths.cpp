/**
 * @file test_ElementErrorPaths.cpp
 * @brief Unit tests for error paths and edge cases in the Elements module
 *
 * Tests cover:
 * - Invalid arguments to constructors (null pointers, negative orders, invalid types)
 * - Factory validation and error handling
 * - Validator detection of inverted/degenerate elements
 * - Cache batch operations with edge cases
 * - SpectralElement order enforcement
 * - High-order element node count validation
 */

#include <gtest/gtest.h>

#include "FE/Elements/ElementFactory.h"
#include "FE/Elements/ElementCache.h"
#include "FE/Elements/ElementValidator.h"
#include "FE/Elements/LagrangeElement.h"
#include "FE/Elements/DiscontinuousElement.h"
#include "FE/Elements/VectorElement.h"
#include "FE/Elements/SpectralElement.h"
#include "FE/Elements/IsogeometricElement.h"
#include "FE/Elements/MixedElement.h"
#include "FE/Elements/CompositeElement.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Elements/ElementTransform.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Core/FEException.h"

using namespace svmp::FE;
using namespace svmp::FE::elements;

//------------------------------------------------------------------------------
// ElementFactory Error Path Tests
//------------------------------------------------------------------------------

TEST(ElementFactoryErrors, ThrowsOnUnknownElementType) {
    ElementRequest req;
    req.element_type = ElementType::Unknown;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Scalar;
    req.order        = 1;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

TEST(ElementFactoryErrors, ThrowsOnNegativeOrder) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Scalar;
    req.order        = -1;
    req.continuity   = Continuity::C0;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

TEST(ElementFactoryErrors, ThrowsOnDGWithVectorField) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Vector;  // Invalid for DG
    req.continuity   = Continuity::L2;
    req.order        = 1;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

TEST(ElementFactoryErrors, ThrowsOnHdivWithScalarField) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Scalar;  // Invalid for H(div)
    req.continuity   = Continuity::H_div;
    req.order        = 0;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

TEST(ElementFactoryErrors, ThrowsOnLagrangeWithVectorField) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Vector;  // Invalid for scalar Lagrange
    req.continuity   = Continuity::C0;
    req.order        = 1;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

TEST(ElementFactoryErrors, ThrowsOnSpectralWithVectorField) {
    ElementRequest req;
    req.element_type = ElementType::Line2;
    req.basis_type   = BasisType::Spectral;
    req.field_type   = FieldType::Vector;  // Invalid for Spectral
    req.order        = 2;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

TEST(ElementFactoryErrors, ThrowsOnUnsupportedBasisType) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type   = BasisType::Hierarchical;  // Not supported by factory
    req.field_type   = FieldType::Scalar;
    req.order        = 1;

    EXPECT_THROW(ElementFactory::create(req), FEException);
}

//------------------------------------------------------------------------------
// LagrangeElement Error Path Tests
//------------------------------------------------------------------------------

TEST(LagrangeElementErrors, ThrowsOnNegativeOrder) {
    EXPECT_THROW(
        LagrangeElement(ElementType::Quad4, -1),
        FEException
    );
}

TEST(LagrangeElementErrors, ThrowsOnVectorField) {
    EXPECT_THROW(
        LagrangeElement(ElementType::Quad4, 1, FieldType::Vector),
        FEException
    );
}

//------------------------------------------------------------------------------
// SpectralElement Error Path Tests
//------------------------------------------------------------------------------

TEST(SpectralElementErrors, ThrowsOnOrderZero) {
    EXPECT_THROW(
        SpectralElement(ElementType::Line2, 0),
        FEException
    );
}

TEST(SpectralElementErrors, ThrowsOnNegativeOrder) {
    EXPECT_THROW(
        SpectralElement(ElementType::Quad4, -1),
        FEException
    );
}

TEST(SpectralElementErrors, ThrowsOnVectorField) {
    EXPECT_THROW(
        SpectralElement(ElementType::Line2, 2, FieldType::Vector),
        FEException
    );
}

TEST(SpectralElementErrors, AcceptsMinimumOrder) {
    // Order 1 should be accepted (minimum for spectral)
    EXPECT_NO_THROW(SpectralElement(ElementType::Line2, 1));
}

//------------------------------------------------------------------------------
// VectorElement Error Path Tests
//------------------------------------------------------------------------------

TEST(VectorElementErrors, ThrowsOnC0Continuity) {
    EXPECT_THROW(
        VectorElement(ElementType::Quad4, 0, Continuity::C0),
        FEException
    );
}

TEST(VectorElementErrors, ThrowsOnL2Continuity) {
    EXPECT_THROW(
        VectorElement(ElementType::Quad4, 0, Continuity::L2),
        FEException
    );
}

TEST(VectorElementErrors, ThrowsOnNegativeOrder) {
    EXPECT_THROW(
        VectorElement(ElementType::Quad4, -1, Continuity::H_div),
        FEException
    );
}

TEST(VectorElementErrors, AcceptsHdiv) {
    EXPECT_NO_THROW(VectorElement(ElementType::Quad4, 0, Continuity::H_div));
}

TEST(VectorElementErrors, AcceptsHcurl) {
    EXPECT_NO_THROW(VectorElement(ElementType::Tetra4, 0, Continuity::H_curl));
}

//------------------------------------------------------------------------------
// IsogeometricElement Error Path Tests
//------------------------------------------------------------------------------

TEST(IsogeometricElementErrors, ThrowsOnNullBasis) {
    auto quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_THROW(
        IsogeometricElement(nullptr, quad, FieldType::Scalar, Continuity::C0),
        FEException
    );
}

TEST(IsogeometricElementErrors, ThrowsOnNullQuadrature) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);

    EXPECT_THROW(
        IsogeometricElement(basis, nullptr, FieldType::Scalar, Continuity::C0),
        FEException
    );
}

TEST(IsogeometricElementErrors, AcceptsValidInputs) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_NO_THROW(
        IsogeometricElement(basis, quad, FieldType::Scalar, Continuity::C0)
    );
}

TEST(IsogeometricElementErrors, ThrowsOnVectorFieldWithScalarBasis) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_THROW(
        IsogeometricElement(basis, quad, FieldType::Vector, Continuity::C0),
        FEException
    );
}

TEST(IsogeometricElementErrors, ThrowsOnScalarFieldWithVectorBasis) {
    auto basis = std::make_shared<basis::RaviartThomasBasis>(ElementType::Quad4, 0);
    auto quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_THROW(
        IsogeometricElement(basis, quad, FieldType::Scalar, Continuity::H_div),
        FEException
    );
}

TEST(IsogeometricElementErrors, AcceptsVectorBasisWithMatchingMetadata) {
    auto basis = std::make_shared<basis::RaviartThomasBasis>(ElementType::Quad4, 0);
    auto quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_NO_THROW(
        IsogeometricElement(basis, quad, FieldType::Vector, Continuity::H_div)
    );
}

//------------------------------------------------------------------------------
// MixedElement Error Path Tests
//------------------------------------------------------------------------------

TEST(MixedElementErrors, ThrowsOnEmptySubElements) {
    std::vector<MixedSubElement> empty_subs;
    EXPECT_THROW(MixedElement{empty_subs}, FEException);
}

TEST(MixedElementErrors, ThrowsOnNullFirstSubElement) {
    std::vector<MixedSubElement> subs = {
        {nullptr, 0}
    };
    EXPECT_THROW(MixedElement{subs}, FEException);
}

TEST(MixedElementErrors, ThrowsOnNullSubsequentSubElement) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);
    std::vector<MixedSubElement> subs = {
        {e1, 0},
        {nullptr, 1}
    };
    EXPECT_THROW(MixedElement{subs}, FEException);
}

TEST(MixedElementErrors, ThrowsOnIncompatibleElementTypes) {
    auto tri = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);
    auto quad = std::make_shared<LagrangeElement>(ElementType::Quad4, 1);
    std::vector<MixedSubElement> subs = {
        {tri, 0},
        {quad, 1}  // Different element type
    };
    EXPECT_THROW(MixedElement{subs}, FEException);
}

TEST(MixedElementErrors, AcceptsCompatibleSubElements) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 2);
    std::vector<MixedSubElement> subs = {
        {e1, 0},
        {e2, 1}
    };
    EXPECT_NO_THROW(MixedElement{subs});
}

//------------------------------------------------------------------------------
// CompositeElement Error Path Tests
//------------------------------------------------------------------------------

TEST(CompositeElementErrors, ThrowsOnEmptyComponents) {
    std::vector<std::shared_ptr<Element>> empty_comps;
    EXPECT_THROW(CompositeElement{empty_comps}, FEException);
}

TEST(CompositeElementErrors, ThrowsOnNullFirstComponent) {
    std::vector<std::shared_ptr<Element>> comps = {nullptr};
    EXPECT_THROW(CompositeElement{comps}, FEException);
}

TEST(CompositeElementErrors, ThrowsOnNullSubsequentComponent) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Quad4, 1);
    std::vector<std::shared_ptr<Element>> comps = {e1, nullptr};
    EXPECT_THROW(CompositeElement{comps}, FEException);
}

TEST(CompositeElementErrors, ThrowsOnIncompatibleComponents) {
    auto tri = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);
    auto tet = std::make_shared<LagrangeElement>(ElementType::Tetra4, 1);
    std::vector<std::shared_ptr<Element>> comps = {tri, tet};
    EXPECT_THROW(CompositeElement{comps}, FEException);
}

TEST(CompositeElementErrors, AcceptsCompatibleComponents) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Quad4, 1);
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Quad4, 2);
    std::vector<std::shared_ptr<Element>> comps = {e1, e2};
    EXPECT_NO_THROW(CompositeElement{comps});
}

//------------------------------------------------------------------------------
// ElementValidator Tests (including inverted element detection)
//------------------------------------------------------------------------------

TEST(ElementValidatorTests, DetectsInvertedElement) {
    // Use a 3D volumetric element so that det(J) preserves orientation.
    // (For 2D surface mappings in 3D, the library uses a full 3x3 frame Jacobian
    // with a constructed orthonormal complement, so det(J) represents an area
    // element and is non-negative by construction.)
    LagrangeElement elem(ElementType::Tetra4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 1);

    // Swap two vertices to invert the reference tetrahedron orientation.
    auto nodes = basis->nodes();
    std::swap(nodes[1], nodes[2]);

    geometry::IsoparametricMapping mapping(basis, nodes);
    ElementQuality q = ElementValidator::validate(elem, mapping);

    EXPECT_FALSE(q.positive_jacobian);
    EXPECT_LT(q.min_detJ, 0.0);
}

TEST(ElementValidatorTests, AcceptsWellFormedElement) {
    LagrangeElement elem(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);

    // Use standard reference quad nodes
    auto nodes = basis->nodes();

    geometry::IsoparametricMapping mapping(basis, nodes);
    ElementQuality q = ElementValidator::validate(elem, mapping);

    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_GT(q.min_detJ, 0.0);
}

TEST(ElementValidatorTests, ReportsConditionNumber) {
    LagrangeElement elem(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);

    // Get nodes and distort them to create a highly elongated quad
    auto nodes = basis->nodes();
    // Scale y to be very small
    for (auto& n : nodes) {
        n[1] *= Real(0.01);  // Very thin element
    }

    geometry::IsoparametricMapping mapping(basis, nodes);
    ElementQuality q = ElementValidator::validate(elem, mapping);

    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_GT(q.max_condition_number, 1.0);  // Distorted elements have high condition number
}

TEST(ElementValidatorTests, AcceptsWellFormedTetrahedron) {
    // Identity Tetra4 mapping on reference simplex
    LagrangeElement elem(ElementType::Tetra4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 1);

    // Use reference tetra nodes
    auto nodes = basis->nodes();
    geometry::IsoparametricMapping mapping(basis, nodes);

    ElementQuality q = ElementValidator::validate(elem, mapping);
    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_GT(q.min_detJ, 0.0);
}

//------------------------------------------------------------------------------
// ElementCache Batch Tests
//------------------------------------------------------------------------------

TEST(ElementCacheBatchTests, ThrowsOnSizeMismatch) {
    ElementCache& cache = ElementCache::instance();
    cache.clear();

    LagrangeElement elem1(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    geometry::IsoparametricMapping mapping(basis, nodes);

    // Mismatched sizes
    std::vector<const Element*> elements = {&elem1};
    std::vector<const geometry::GeometryMapping*> mappings;  // Empty

    EXPECT_THROW(cache.get_batch(elements, mappings), FEException);
}

TEST(ElementCacheBatchTests, HandlesNullPointers) {
    ElementCache& cache = ElementCache::instance();
    cache.clear();

    LagrangeElement elem1(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    geometry::IsoparametricMapping mapping(basis, nodes);

    std::vector<const Element*> elements = {&elem1, nullptr};
    std::vector<const geometry::GeometryMapping*> mappings = {&mapping, nullptr};

    auto results = cache.get_batch(elements, mappings);
    ASSERT_EQ(results.size(), 2u);

    // First should be valid
    EXPECT_NE(results[0].basis, nullptr);
    EXPECT_NE(results[0].jacobian, nullptr);

    // Second should be null (from null input)
    EXPECT_EQ(results[1].basis, nullptr);
    EXPECT_EQ(results[1].jacobian, nullptr);
}

TEST(ElementCacheBatchTests, BatchReturnsValidEntries) {
    ElementCache& cache = ElementCache::instance();
    cache.clear();

    LagrangeElement elem1(ElementType::Quad4, 1);
    LagrangeElement elem2(ElementType::Quad4, 2);

    auto basis1 = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto basis2 = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);

    geometry::IsoparametricMapping mapping1(basis1, basis1->nodes());
    geometry::IsoparametricMapping mapping2(basis2, basis2->nodes());

    std::vector<const Element*> elements = {&elem1, &elem2};
    std::vector<const geometry::GeometryMapping*> mappings = {&mapping1, &mapping2};

    BatchEvaluationHints hints;
    hints.batch_size = 2;

    auto batch_entries = cache.get_batch(elements, mappings, hints);
    ASSERT_EQ(batch_entries.size(), 2u);

    // Compare with individual get() results
    auto e1 = cache.get(elem1, mapping1);
    auto e2 = cache.get(elem2, mapping2);

    EXPECT_EQ(batch_entries[0].basis, e1.basis);
    EXPECT_EQ(batch_entries[0].jacobian, e1.jacobian);
    EXPECT_EQ(batch_entries[1].basis, e2.basis);
    EXPECT_EQ(batch_entries[1].jacobian, e2.jacobian);
}

TEST(ElementCacheBatchTests, BatchReturnsCorrectSize) {
    ElementCache& cache = ElementCache::instance();
    cache.clear();

    LagrangeElement elem1(ElementType::Quad4, 1);
    LagrangeElement elem2(ElementType::Quad4, 2);
    auto basis1 = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto basis2 = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);
    geometry::IsoparametricMapping mapping1(basis1, basis1->nodes());
    geometry::IsoparametricMapping mapping2(basis2, basis2->nodes());

    std::vector<const Element*> elements = {&elem1, &elem2};
    std::vector<const geometry::GeometryMapping*> mappings = {&mapping1, &mapping2};

    BatchEvaluationHints hints;
    hints.batch_size = 2;
    hints.prefetch = true;

    auto results = cache.get_batch(elements, mappings, hints);
    ASSERT_EQ(results.size(), 2u);

    for (const auto& entry : results) {
        EXPECT_NE(entry.basis, nullptr);
        EXPECT_NE(entry.jacobian, nullptr);
    }
}

TEST(ElementCacheBatchTests, OptimalSimdWidthIsPositive) {
    int width = ElementCache::optimal_simd_width();
    EXPECT_GE(width, 1);
    // Should be power of 2 for valid SIMD widths
    EXPECT_TRUE(width == 1 || width == 4 || width == 8 || width == 16);
}

//------------------------------------------------------------------------------
// High-Order Element Node Count Tests
//------------------------------------------------------------------------------

TEST(HighOrderElementTests, Tet10HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Tetra10);
    EXPECT_EQ(ref.num_nodes(), 10u);
}

TEST(HighOrderElementTests, Hex27HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Hex27);
    EXPECT_EQ(ref.num_nodes(), 27u);
}

TEST(HighOrderElementTests, Wedge15HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Wedge15);
    EXPECT_EQ(ref.num_nodes(), 15u);
}

TEST(HighOrderElementTests, Pyramid14HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Pyramid14);
    EXPECT_EQ(ref.num_nodes(), 14u);
}

TEST(HighOrderElementTests, Triangle6HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Triangle6);
    EXPECT_EQ(ref.num_nodes(), 6u);
}

TEST(HighOrderElementTests, Quad9HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Quad9);
    EXPECT_EQ(ref.num_nodes(), 9u);
}

//------------------------------------------------------------------------------
// Cross-Element Type Factory Consistency Tests
//------------------------------------------------------------------------------

TEST(FactoryConsistencyTests, AllLinearTypesCreateSuccessfully) {
    const std::vector<ElementType> linear_types = {
        ElementType::Line2,
        ElementType::Triangle3,
        ElementType::Quad4,
        ElementType::Tetra4,
        ElementType::Hex8,
        ElementType::Wedge6,
        ElementType::Pyramid5
    };

    ElementRequest req;
    req.basis_type = BasisType::Lagrange;
    req.field_type = FieldType::Scalar;
    req.continuity = Continuity::C0;
    req.order = 1;

    for (ElementType type : linear_types) {
        req.element_type = type;
        auto elem = ElementFactory::create(req);
        ASSERT_TRUE(elem) << "Failed to create element for type " << static_cast<int>(type);
        EXPECT_EQ(elem->element_type(), type);
        EXPECT_EQ(elem->dimension(), element_dimension(type));
    }
}

TEST(FactoryConsistencyTests, OrderZeroLagrangeCreatesSuccessfully) {
    ElementRequest req;
    req.element_type = ElementType::Triangle3;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Scalar;
    req.continuity   = Continuity::C0;
    req.order        = 0;  // Piecewise constant

    auto elem = ElementFactory::create(req);
    ASSERT_TRUE(elem);
    EXPECT_EQ(elem->polynomial_order(), 0);
}

//------------------------------------------------------------------------------
// IsogeometricElement Compatibility Tests
//------------------------------------------------------------------------------

TEST(IsogeometricElementErrors, ThrowsOnDimensionMismatch) {
    // 2D basis (Quad4) with 1D quadrature (Line)
    auto basis_2d = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto quad_1d = std::make_shared<quadrature::GaussQuadrature1D>(2);  // 1D Gauss

    EXPECT_THROW(
        IsogeometricElement(basis_2d, quad_1d, FieldType::Scalar, Continuity::C0),
        FEException
    );
}

TEST(IsogeometricElementErrors, ThrowsOnCellFamilyMismatch) {
    // Triangle basis with Quad quadrature (same dimension, different family)
    auto basis_tri = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 1);
    auto quad_quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_THROW(
        IsogeometricElement(basis_tri, quad_quad, FieldType::Scalar, Continuity::C0),
        FEException
    );
}

TEST(IsogeometricElementErrors, AcceptsMatchingBasisAndQuadrature) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto quad = std::make_shared<quadrature::QuadrilateralQuadrature>(2);

    EXPECT_NO_THROW(
        IsogeometricElement(basis, quad, FieldType::Scalar, Continuity::C0)
    );
}

//------------------------------------------------------------------------------
// ElementCache::clear() Verification Tests
//------------------------------------------------------------------------------

TEST(ElementCacheTests, ClearRemovesAllEntries) {
    ElementCache& cache = ElementCache::instance();
    cache.clear();

    // Pre-populate with some entries
    LagrangeElement elem1(ElementType::Quad4, 1);
    LagrangeElement elem2(ElementType::Triangle3, 1);
    auto basis1 = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto basis2 = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 1);
    geometry::IsoparametricMapping mapping1(basis1, basis1->nodes());
    geometry::IsoparametricMapping mapping2(basis2, basis2->nodes());

    // Populate cache
    cache.get(elem1, mapping1);
    cache.get(elem2, mapping2);
    EXPECT_GT(cache.size(), 0u);

    // Clear and verify
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}

TEST(ElementCacheTests, CacheReturnsValidEntriesAfterClear) {
    ElementCache& cache = ElementCache::instance();
    cache.clear();

    LagrangeElement elem(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    geometry::IsoparametricMapping mapping(basis, basis->nodes());

    // Get entry after clear - should compute fresh
    auto entry = cache.get(elem, mapping);
    EXPECT_NE(entry.basis, nullptr);
    EXPECT_NE(entry.jacobian, nullptr);
}

//------------------------------------------------------------------------------
// ElementTransform Tests
//------------------------------------------------------------------------------

TEST(ElementTransformTests, GradientsToPhysicalProducesCorrectDimension) {
    // Create a simple Quad4 element and mapping
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    geometry::IsoparametricMapping mapping(basis, nodes);

    // Evaluate reference gradients at center (0,0)
    math::Vector<Real, 3> xi{Real(0), Real(0), Real(0)};
    std::vector<basis::Gradient> grads_ref(basis->size());
    basis->evaluate_gradients(xi, grads_ref);

    // Transform to physical
    std::vector<math::Vector<Real, 3>> grads_phys;
    ElementTransform::gradients_to_physical(mapping, xi, grads_ref, grads_phys);

    ASSERT_EQ(grads_phys.size(), basis->size());

    // For reference quad on [-1,1]^2 with identity mapping,
    // physical gradients should equal reference gradients
    for (std::size_t i = 0; i < grads_ref.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(grads_phys[i][0]),
                    static_cast<double>(grads_ref[i][0]), 1e-10);
        EXPECT_NEAR(static_cast<double>(grads_phys[i][1]),
                    static_cast<double>(grads_ref[i][1]), 1e-10);
    }
}

TEST(ElementTransformTests, GradientsScaleWithStretchedElement) {
    // Create a stretched Quad4 (2x1 rectangle instead of 2x2)
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();

    // Scale y coordinates by 0.5 to get [-1,1] x [-0.5,0.5]
    for (auto& n : nodes) {
        n[1] *= Real(0.5);
    }

    geometry::IsoparametricMapping mapping(basis, nodes);

    math::Vector<Real, 3> xi{Real(0), Real(0), Real(0)};
    std::vector<basis::Gradient> grads_ref(basis->size());
    basis->evaluate_gradients(xi, grads_ref);

    std::vector<math::Vector<Real, 3>> grads_phys;
    ElementTransform::gradients_to_physical(mapping, xi, grads_ref, grads_phys);

    // Y-gradients should be scaled by 2 (inverse of 0.5 stretch)
    for (std::size_t i = 0; i < grads_ref.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(grads_phys[i][0]),
                    static_cast<double>(grads_ref[i][0]), 1e-10);
        EXPECT_NEAR(static_cast<double>(grads_phys[i][1]),
                    static_cast<double>(grads_ref[i][1]) * 2.0, 1e-10);
    }
}

//------------------------------------------------------------------------------
// Mixed/Composite DOF Aggregation Tests
//------------------------------------------------------------------------------

TEST(MixedElementDOFTests, TotalDOFsEqualsSumOfSubElements) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);  // 3 DOFs
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 2);  // 6 DOFs

    std::vector<MixedSubElement> subs = {
        {e1, 0},
        {e2, 1}
    };

    MixedElement mixed(subs);

    std::size_t expected_dofs = e1->num_dofs() + e2->num_dofs();
    EXPECT_EQ(mixed.num_dofs(), expected_dofs);
}

TEST(MixedElementDOFTests, FieldIdsArePreserved) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Quad4, 1);
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Quad4, 1);

    std::vector<MixedSubElement> subs = {
        {e1, 42},  // Field ID 42
        {e2, 99}   // Field ID 99
    };

    MixedElement mixed(subs);

    const auto& retrieved = mixed.sub_elements();
    ASSERT_EQ(retrieved.size(), 2u);
    EXPECT_EQ(retrieved[0].field_id, 42);
    EXPECT_EQ(retrieved[1].field_id, 99);

    // Metadata: Mixed field type, custom continuity, order = max of suborders
    EXPECT_EQ(mixed.field_type(), FieldType::Mixed);
    EXPECT_EQ(mixed.continuity(), Continuity::Custom);
    EXPECT_EQ(mixed.polynomial_order(),
              std::max(e1->polynomial_order(), e2->polynomial_order()));
}

TEST(CompositeElementDOFTests, TotalDOFsEqualsSumOfComponents) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Quad4, 1);  // 4 DOFs
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Quad4, 2);  // 9 DOFs

    std::vector<std::shared_ptr<Element>> comps = {e1, e2};
    CompositeElement composite(comps);

    std::size_t expected_dofs = e1->num_dofs() + e2->num_dofs();
    EXPECT_EQ(composite.num_dofs(), expected_dofs);
}

TEST(CompositeElementDOFTests, ComponentsAccessible) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Hex8, 1);
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Hex8, 2);

    std::vector<std::shared_ptr<Element>> comps = {e1, e2};
    CompositeElement composite(comps);

    const auto& retrieved = composite.components();
    ASSERT_EQ(retrieved.size(), 2u);
    EXPECT_EQ(retrieved[0]->num_nodes(), 8u);
    EXPECT_EQ(retrieved[1]->num_nodes(), 27u);  // Hex8 order 2 -> Hex27 (full Lagrange)

    // Metadata: Mixed field type, custom continuity, order = max of component orders
    EXPECT_EQ(composite.field_type(), FieldType::Mixed);
    EXPECT_EQ(composite.continuity(), Continuity::Custom);
    EXPECT_EQ(composite.polynomial_order(),
              std::max(e1->polynomial_order(), e2->polynomial_order()));
}

//------------------------------------------------------------------------------
// Additional High-Order Element Tests
//------------------------------------------------------------------------------

TEST(HighOrderElementTests, Pyramid13HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Pyramid13);
    EXPECT_EQ(ref.num_nodes(), 13u);
}

TEST(HighOrderElementTests, Hex20HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Hex20);
    EXPECT_EQ(ref.num_nodes(), 20u);
}

TEST(HighOrderElementTests, Quad8HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Quad8);
    EXPECT_EQ(ref.num_nodes(), 8u);
}

TEST(HighOrderElementTests, Line3HasCorrectNodeCount) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Line3);
    EXPECT_EQ(ref.num_nodes(), 3u);
}

TEST(HighOrderElementTests, LagrangeElementHighOrderQuad) {
    LagrangeElement elem(ElementType::Quad9, 2);
    EXPECT_EQ(elem.num_nodes(), 9u);
    EXPECT_EQ(elem.polynomial_order(), 2);
}

TEST(HighOrderElementTests, LagrangeElementHighOrderTet) {
    LagrangeElement elem(ElementType::Tetra10, 2);
    EXPECT_EQ(elem.num_nodes(), 10u);
    EXPECT_EQ(elem.polynomial_order(), 2);
}
