/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_HCURLSPACE_H
#define SVMP_FE_SPACES_HCURLSPACE_H

/**
 * @file HCurlSpace.h
 * @brief H(curl)-conforming vector-valued function space
 *
 * H(curl) spaces (Nedelec elements) have continuous tangential components
 * across element interfaces. The tangential trace n×(v×n) is well-defined
 * and continuous on element faces.
 */

#include "Spaces/FunctionSpace.h"
#include "Spaces/VectorComponentExtractor.h"
#include "Spaces/OrientationManager.h"
#include "Elements/VectorElement.h"

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief H(curl)-conforming space built from Nedelec-type vector elements
 *
 * This space is intended for electromagnetics and other problems requiring
 * tangential continuity. It is constructed from @ref elements::VectorElement
 * with Continuity::H_curl and provides vector-valued fields.
 *
 * Key properties:
 *  - Tangential component is continuous across faces
 *  - DOFs are associated with edges (and faces/interior for higher order)
 *  - Edge DOFs require orientation correction for global conformity
 */
class HCurlSpace : public FunctionSpace {
public:
    using Vec3 = math::Vector<Real, 3>;

    HCurlSpace(ElementType element_type,
               int order);

    SpaceType space_type() const noexcept override { return SpaceType::HCurl; }
    FieldType field_type() const noexcept override { return FieldType::Vector; }
    Continuity continuity() const noexcept override { return Continuity::H_curl; }

    int value_dimension() const noexcept override { return dimension_; }
    int topological_dimension() const noexcept override { return dimension_; }
    int polynomial_order() const noexcept override { return order_; }
    ElementType element_type() const noexcept override { return element_type_; }

    const elements::Element& element() const noexcept override { return *element_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return element_; }

    // =========================================================================
    // Tangential Trace Operations
    // =========================================================================

    /**
     * @brief Compute tangential trace on a face: n×(v×n)
     *
     * The tangential trace is the projection of the vector field onto
     * the tangent plane of the face. This is the trace operator for
     * H(curl) spaces and should be continuous across element interfaces.
     *
     * @param dof_values Element DOF values
     * @param eval_points Points on face (in reference coordinates)
     * @param face_normal Outward unit normal on face
     * @return Tangential trace vectors at evaluation points
     */
    std::vector<Vec3> tangential_trace(
        const std::vector<Real>& dof_values,
        const std::vector<Vec3>& eval_points,
        const Vec3& face_normal) const;

    /**
     * @brief Apply edge orientation correction to DOF values
     *
     * For global H(curl) conformity, edge DOFs must be oriented
     * consistently across all elements sharing an edge.
     *
     * @param edge_dofs DOF values on an edge
     * @param orientation Orientation sign from OrientationManager
     * @return Oriented DOF values
     */
    static std::vector<Real> apply_edge_orientation(
        const std::vector<Real>& edge_dofs,
        OrientationManager::Sign orientation);

    /**
     * @brief Apply face orientation correction to H(curl) face-interior DOFs
     *
     * For higher-order tensor-product Nédélec elements (e.g., Hex), face DOFs
     * require permutation/sign handling under face rotations/reflections.
     */
    static std::vector<Real> apply_face_orientation(
        ElementType face_type,
        const std::vector<Real>& face_dofs,
        const OrientationManager::FaceOrientation& orientation,
        int face_poly_order);

private:
    ElementType element_type_;
    int order_;
    int dimension_;
    std::shared_ptr<elements::Element> element_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_HCURLSPACE_H
