/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_HDIVSPACE_H
#define SVMP_FE_SPACES_HDIVSPACE_H

/**
 * @file HDivSpace.h
 * @brief H(div)-conforming vector-valued function space
 *
 * H(div) spaces (Raviart-Thomas, BDM elements) have continuous normal
 * components across element interfaces. The normal trace v·n is well-defined
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
 * @brief H(div)-conforming space built from Raviart-Thomas / BDM elements
 *
 * This space is intended for mixed formulations and flux-based methods
 * requiring normal continuity across element faces. It is constructed from
 * @ref elements::VectorElement with Continuity::H_div.
 *
 * Key properties:
 *  - Normal component is continuous across faces
 *  - DOFs are associated with faces (and interior for higher order)
 *  - Face DOFs require orientation correction for global conformity
 */
class HDivSpace : public FunctionSpace {
public:
    using Vec3 = math::Vector<Real, 3>;

    HDivSpace(ElementType element_type,
              int order);

    SpaceType space_type() const noexcept override { return SpaceType::HDiv; }
    FieldType field_type() const noexcept override { return FieldType::Vector; }
    Continuity continuity() const noexcept override { return Continuity::H_div; }

    int value_dimension() const noexcept override { return dimension_; }
    int topological_dimension() const noexcept override { return dimension_; }
    int polynomial_order() const noexcept override { return order_; }
    ElementType element_type() const noexcept override { return element_type_; }

    const elements::Element& element() const noexcept override { return *element_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return element_; }

    // =========================================================================
    // Normal Trace Operations
    // =========================================================================

    /**
     * @brief Compute normal trace on a face: v·n
     *
     * The normal trace is the projection of the vector field onto
     * the normal direction. This is the trace operator for H(div)
     * spaces and should be continuous across element interfaces.
     *
     * @param dof_values Element DOF values
     * @param eval_points Points on face (in reference coordinates)
     * @param face_normal Outward unit normal on face
     * @return Normal trace scalars at evaluation points
     */
    std::vector<Real> normal_trace(
        const std::vector<Real>& dof_values,
        const std::vector<Vec3>& eval_points,
        const Vec3& face_normal) const;

    /**
     * @brief Apply face orientation correction to face DOF values
     *
     * For global H(div) conformity, face DOFs must be oriented consistently
     * across all elements sharing a face (including permutation for higher
     * orders and sign flips for reversed face orientation).
     *
     * @param face_type Face element type (Triangle3 or Quad4)
     * @param face_dofs DOF values on the face (in local face ordering)
     * @param orientation Face orientation descriptor
     * @param face_poly_order Polynomial order for the face DOF layout
     * @return Oriented DOF values in global face ordering
     */
    static std::vector<Real> apply_face_orientation(
        ElementType face_type,
        const std::vector<Real>& face_dofs,
        const OrientationManager::FaceOrientation& orientation,
        int face_poly_order);

    /**
     * @brief Apply edge orientation correction to H(div) edge DOFs (2D)
     *
     * For Raviart–Thomas elements in 2D, edge-normal flux moments require a
     * signed reversal under an oriented edge.
     */
    static std::vector<Real> apply_edge_orientation(
        const std::vector<Real>& edge_dofs,
        OrientationManager::Sign orientation) {
        return OrientationManager::orient_hcurl_edge_dofs(edge_dofs, orientation);
    }

private:
    ElementType element_type_;
    int order_;
    int dimension_;
    std::shared_ptr<elements::Element> element_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_HDIVSPACE_H
