/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_ELEMENTTRANSFORM_H
#define SVMP_FE_ELEMENTS_ELEMENTTRANSFORM_H

/**
 * @file ElementTransform.h
 * @brief Helpers for transforming basis data between reference and physical space
 *
 * This module also provides facet trace evaluation utilities for H(div) and H(curl)
 * conforming elements. These are essential for verifying interface continuity of
 * vector-valued finite element spaces across shared facets.
 */

#include "Geometry/GeometryMapping.h"
#include "Geometry/PushForward.h"
#include "Basis/BasisFunction.h"

#include <array>

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Result of computing a facet normal or tangent at a point
 */
struct FacetFrame {
    math::Vector<Real, 3> normal{};    ///< Outward unit normal (for faces)
    math::Vector<Real, 3> tangent1{};  ///< First tangent direction
    math::Vector<Real, 3> tangent2{};  ///< Second tangent direction (3D faces only)
    Real jacobian_det{Real(1)};        ///< Facet Jacobian determinant (surface measure scale)
};

/**
 * @brief Utility functions for element-level transformations
 *
 * This class provides thin wrappers around `geometry::PushForward` to
 * transform basis gradients and vector-valued basis functions between
 * reference and physical coordinates.
 *
 * Additionally, it provides facet trace evaluation utilities for verifying
 * H(div) (normal continuity) and H(curl) (tangential continuity) conformity.
 */
class ElementTransform {
public:
    /**
     * @brief Transform scalar basis gradients from reference to physical space
     *
     * @param mapping   Geometry mapping for the current element
     * @param xi        Reference coordinates of the evaluation point
     * @param grads_ref Gradients in reference coordinates (per basis function)
     * @param grads_phys Output physical gradients (resized as needed)
     */
    static void gradients_to_physical(const geometry::GeometryMapping& mapping,
                                      const math::Vector<Real, 3>& xi,
                                      const std::vector<basis::Gradient>& grads_ref,
                                      std::vector<math::Vector<Real, 3>>& grads_phys);

    /**
     * @brief H(div) Piola transform for vector-valued basis functions
     *
     * @param mapping Geometry mapping
     * @param xi      Reference coordinate
     * @param v_ref   Reference-space vector values
     * @param v_phys  Output physical vectors
     */
    static void hdiv_vectors_to_physical(const geometry::GeometryMapping& mapping,
                                         const math::Vector<Real, 3>& xi,
                                         const std::vector<math::Vector<Real, 3>>& v_ref,
                                         std::vector<math::Vector<Real, 3>>& v_phys);

    /**
     * @brief H(curl) Piola transform for vector-valued basis functions
     */
    static void hcurl_vectors_to_physical(const geometry::GeometryMapping& mapping,
                                          const math::Vector<Real, 3>& xi,
                                          const std::vector<math::Vector<Real, 3>>& v_ref,
                                          std::vector<math::Vector<Real, 3>>& v_phys);

    // =========================================================================
    // Facet Frame and Trace Evaluation
    // =========================================================================

    /**
     * @brief Compute facet frame (normal and tangent vectors) at a reference point
     *
     * For 2D elements (triangles, quads), the facet is an edge, and this computes
     * the outward unit normal and edge tangent at the given point.
     *
     * For 3D elements (tets, hexes, wedges, pyramids), the facet is a face, and
     * this computes the outward unit normal and two tangent vectors spanning
     * the face tangent plane.
     *
     * @param mapping       Geometry mapping for the element
     * @param xi            Reference coordinates on the facet (must lie on facet)
     * @param facet_id      Local facet index (0 to num_facets-1)
     * @param element_type  Type of the element (for topology lookup)
     * @return FacetFrame containing normal, tangents, and facet Jacobian determinant
     */
    static FacetFrame compute_facet_frame(const geometry::GeometryMapping& mapping,
                                          const math::Vector<Real, 3>& xi,
                                          int facet_id,
                                          ElementType element_type);

    /**
     * @brief Compute H(div) normal trace: n . v at a facet point
     *
     * Given physical-space vector values (after Piola transform), computes
     * the normal component n . v for each basis function at the facet point.
     *
     * @param v_phys        Physical-space vector values (one per basis function)
     * @param normal        Unit outward normal at the evaluation point
     * @return Vector of normal traces (one per basis function)
     */
    static std::vector<Real> hdiv_normal_trace(
        const std::vector<math::Vector<Real, 3>>& v_phys,
        const math::Vector<Real, 3>& normal);

    /**
     * @brief Compute H(curl) tangential trace: t x v at a facet point (2D: t . v)
     *
     * For 3D elements, returns n x v (the tangential component in the face plane).
     * For 2D elements, returns the scalar t . v (tangent component along edge).
     *
     * @param v_phys        Physical-space vector values (one per basis function)
     * @param frame         Facet frame with normal and tangent vectors
     * @param dim           Element dimension (2 or 3)
     * @return For 3D: vector of tangential traces (each is a 3D vector); for 2D: scalars
     */
    static std::vector<math::Vector<Real, 3>> hcurl_tangential_trace_3d(
        const std::vector<math::Vector<Real, 3>>& v_phys,
        const FacetFrame& frame);

    static std::vector<Real> hcurl_tangential_trace_2d(
        const std::vector<math::Vector<Real, 3>>& v_phys,
        const FacetFrame& frame);

    /**
     * @brief Get reference coordinates of facet vertices for an element type
     *
     * Returns the local vertex indices and reference coordinates for the
     * specified facet. Used to parameterize points on the facet for trace
     * evaluation.
     *
     * @param element_type  Element type
     * @param facet_id      Local facet index
     * @return Pair of (vertex indices, reference coordinates)
     */
    static std::pair<std::vector<LocalIndex>, std::vector<math::Vector<Real, 3>>>
    facet_vertices(ElementType element_type, int facet_id);

    /**
     * @brief Map facet-local coordinates to element reference coordinates
     *
     * Given coordinates on the reference facet (parameterized by 0-1 for edges,
     * or barycentric/tensor-product for faces), returns the corresponding
     * coordinates in the element's reference space.
     *
     * @param element_type  Element type
     * @param facet_id      Local facet index
     * @param facet_coords  Coordinates on the reference facet
     * @return Coordinates in element reference space
     */
    static math::Vector<Real, 3> facet_to_reference(
        ElementType element_type,
        int facet_id,
        const math::Vector<Real, 3>& facet_coords);

    /**
     * @brief Get the outward reference normal direction for a facet
     *
     * Returns the outward-pointing normal direction in reference space for
     * the specified facet of the given element type. This is the "canonical"
     * normal before any mapping to physical space.
     *
     * @param element_type  Element type
     * @param facet_id      Local facet index
     * @return Reference-space outward normal (not necessarily unit length)
     */
    static math::Vector<Real, 3> reference_facet_normal(
        ElementType element_type,
        int facet_id);
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_ELEMENTTRANSFORM_H

