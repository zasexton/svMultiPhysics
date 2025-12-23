/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_TRACESPACE_H
#define SVMP_FE_SPACES_TRACESPACE_H

/**
 * @file TraceSpace.h
 * @brief Trace spaces on element boundaries/interfaces
 *
 * TraceSpace is a wrapper around a volume FunctionSpace that represents
 * its restriction to a particular element face.
 *
 * In addition to face DOF extraction/scatter, TraceSpace provides a true
 * (dim-1)-dimensional "prototype element" on the face:
 *  - face `ElementType` + polynomial order
 *  - face basis + quadrature for integration on the face
 *  - a reference-face coordinate system embedded in the volume reference element
 *
 * TraceSpace also provides lifting operators (face → volume injection) for
 * nodal scalar spaces.
 *
 * TraceSpace does not depend on any Mesh data structures; face indices
 * are provided by higher level modules that understand the mesh topology.
 */

#include "Spaces/FunctionSpace.h"
#include "Spaces/FaceRestriction.h"
#include <memory>

namespace svmp {
namespace FE {
namespace geometry {
class GeometryMapping;
}
namespace spaces {

class TraceSpace : public FunctionSpace {
public:
    /// Construct trace space from a volume space and local face id
    TraceSpace(std::shared_ptr<FunctionSpace> volume_space,
               int face_id);

    SpaceType space_type() const noexcept override { return SpaceType::Trace; }
    FieldType field_type() const noexcept override { return volume_space_->field_type(); }
    Continuity continuity() const noexcept override { return volume_space_->continuity(); }

    int value_dimension() const noexcept override { return volume_space_->value_dimension(); }

    /// Trace lives on a (dim-1)-dimensional manifold
    int topological_dimension() const noexcept override {
        const int dim = volume_space_->topological_dimension();
        return dim > 0 ? dim - 1 : 0;
    }

    int polynomial_order() const noexcept override { return volume_space_->polynomial_order(); }
    ElementType element_type() const noexcept override { return face_element_type_; }

    const elements::Element& element() const noexcept override { return *face_element_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override {
        return face_element_;
    }

    /// Local face identifier on the reference element
    int face_id() const noexcept { return face_id_; }

    /// Access the underlying volume space
    const FunctionSpace& volume_space() const noexcept { return *volume_space_; }
    std::shared_ptr<FunctionSpace> volume_space_ptr() const noexcept { return volume_space_; }

    /// Number of DOFs on this face
    std::size_t dofs_per_element() const noexcept override;

    /// Evaluate on the reference face using face coefficients (in face-basis order).
    Value evaluate(const Value& xi,
                   const std::vector<Real>& coefficients) const override;

    /// Face-space interpolation. Uses nodal interpolation for nodal bases and
    /// falls back to the default FunctionSpace L² projection otherwise.
    void interpolate(const ValueFunction& function,
                     std::vector<Real>& coefficients) const override;

    // =========================================================================
    // Face DOF Operations
    // =========================================================================

    /**
     * @brief Get volume-element local DOF indices corresponding to this face
     *
     * Indices are returned in the same order as TraceSpace coefficients
     * (i.e., the face prototype basis ordering).
     *
     * @return Vector of local DOF indices
     */
    std::vector<int> face_dof_indices() const;

    /**
     * @brief Restrict element DOF values to this face
     *
     * Extracts the subset of element DOF values that correspond to
     * DOFs on this face, returning coefficients in face-basis order.
     *
     * @param element_values All DOF values on the element
     * @return DOF values restricted to this face
     */
    std::vector<Real> restrict(const std::vector<Real>& element_values) const;

    /**
     * @brief Scatter face DOF values to element
     *
     * Adds face DOF values (in face-basis order) to the corresponding
     * positions in the volume element DOF vector. Does not zero out
     * element_values first.
     *
     * @param face_values DOF values on this face
     * @param[in,out] element_values Element DOF vector to scatter into
     */
    void scatter(const std::vector<Real>& face_values,
                 std::vector<Real>& element_values) const;

    /**
     * @brief Lift face coefficients into a volume coefficient vector
     *
     * Returns a volume-sized coefficient vector with face-associated DOFs
     * filled from @p face_coefficients and all other entries set to zero.
     *
     * @param face_coefficients Face coefficients in face-basis order
     * @return Volume-sized coefficient vector suitable for evaluation/assembly
     */
    std::vector<Real> lift(const std::vector<Real>& face_coefficients) const;

    /**
     * @brief Embed a face reference point into the volume reference element
     *
     * Maps coordinates on the reference face to coordinates in the volume
     * reference element, using the face's local reference coordinate system.
     *
     * @param xi_face Face reference coordinates (unused components ignored)
     * @return Volume reference coordinates lying on the selected face
     */
    Value embed_face_point(const Value& xi_face) const;

    /**
     * @brief Evaluate via the volume space at a volume reference point on the face
     *
     * Convenience utility for workflows that operate in volume reference
     * coordinates but store coefficients in face space ordering.
     *
     * @param xi_volume Volume reference coordinates (must lie on this face)
     * @param face_coefficients Face coefficients in face-basis order
     * @return Field value evaluated by the volume space at xi_volume
     */
    Value evaluate_from_face(const Value& xi_volume,
                             const std::vector<Real>& face_coefficients) const;

    /**
     * @brief Access the face restriction operator
     */
    const FaceRestriction& face_restriction() const;

private:
    std::shared_ptr<FunctionSpace> volume_space_;
    int face_id_;
    std::shared_ptr<const FaceRestriction> restriction_;

    ElementType face_element_type_{ElementType::Unknown};
    std::shared_ptr<elements::Element> face_element_;
    std::shared_ptr<const geometry::GeometryMapping> face_embedding_;
    std::vector<int> face_to_volume_dof_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_TRACESPACE_H
