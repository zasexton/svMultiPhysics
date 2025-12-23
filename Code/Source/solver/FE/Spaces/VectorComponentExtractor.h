/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_VECTORCOMPONENTEXTRACTOR_H
#define SVMP_FE_SPACES_VECTORCOMPONENTEXTRACTOR_H

/**
 * @file VectorComponentExtractor.h
 * @brief Utility for extracting normal and tangential components from vector fields
 *
 * This module provides static utilities for decomposing vector fields into
 * normal and tangential components relative to a specified direction or surface.
 * These operations are essential for:
 *  - H(div) spaces: normal trace continuity (v·n)
 *  - H(curl) spaces: tangential trace continuity (n×v×n)
 *  - Boundary condition enforcement
 *  - Flux computations in DG methods
 */

#include "Core/Types.h"
#include "Math/Vector.h"
#include <vector>
#include <array>
#include <cmath>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Static utility class for extracting vector components
 *
 * Provides methods to decompose vector fields into normal and tangential
 * components relative to a given direction. All methods are stateless and
 * operate on arrays of vector values at quadrature points.
 */
class VectorComponentExtractor {
public:
    using Vec3 = math::Vector<Real, 3>;
    using Vec2 = math::Vector<Real, 2>;

    /**
     * @brief Extract normal component: v·n (scalar)
     *
     * Computes the projection of each vector onto the normal direction.
     *
     * @param vector_field Vector values at points [npts]
     * @param normals Unit normal vectors at points [npts]
     * @return Scalar normal components [npts]
     */
    static std::vector<Real> normal_component(
        const std::vector<Vec3>& vector_field,
        const std::vector<Vec3>& normals);

    /**
     * @brief Extract normal component with single normal
     *
     * Uses the same normal for all points (common case for flat faces).
     *
     * @param vector_field Vector values at points [npts]
     * @param normal Single unit normal vector
     * @return Scalar normal components [npts]
     */
    static std::vector<Real> normal_component(
        const std::vector<Vec3>& vector_field,
        const Vec3& normal);

    /**
     * @brief Extract tangential component: v - (v·n)n
     *
     * Computes the component of each vector perpendicular to the normal.
     *
     * @param vector_field Vector values at points [npts]
     * @param normals Unit normal vectors at points [npts]
     * @return Tangential vectors [npts]
     */
    static std::vector<Vec3> tangential_component(
        const std::vector<Vec3>& vector_field,
        const std::vector<Vec3>& normals);

    /**
     * @brief Extract tangential component with single normal
     *
     * @param vector_field Vector values at points [npts]
     * @param normal Single unit normal vector
     * @return Tangential vectors [npts]
     */
    static std::vector<Vec3> tangential_component(
        const std::vector<Vec3>& vector_field,
        const Vec3& normal);

    /**
     * @brief Extract tangential component in 2D: v - (v·n)n as scalar cross
     *
     * In 2D, the tangential component can be represented as a scalar
     * (the component along the tangent direction t = (-n_y, n_x)).
     *
     * @param vector_field 2D vector values at points [npts]
     * @param normals 2D unit normal vectors at points [npts]
     * @return Scalar tangential components [npts]
     */
    static std::vector<Real> tangential_component_2d(
        const std::vector<Vec2>& vector_field,
        const std::vector<Vec2>& normals);

    /**
     * @brief Extract tangential component in 2D with single normal
     *
     * @param vector_field 2D vector values at points [npts]
     * @param normal Single 2D unit normal vector
     * @return Scalar tangential components [npts]
     */
    static std::vector<Real> tangential_component_2d(
        const std::vector<Vec2>& vector_field,
        const Vec2& normal);

    /**
     * @brief Project vectors onto tangent plane
     *
     * Projects 3D vectors onto a tangent plane defined by two tangent vectors.
     * Returns the 2D coordinates in the tangent basis.
     *
     * @param vector_field 3D vector values at points [npts]
     * @param tangent1 First tangent direction (should be unit)
     * @param tangent2 Second tangent direction (should be unit and orthogonal to tangent1)
     * @return 2D components in tangent basis [npts]
     */
    static std::vector<Vec2> project_to_tangent_plane(
        const std::vector<Vec3>& vector_field,
        const Vec3& tangent1,
        const Vec3& tangent2);

    /**
     * @brief Compute tangent basis from normal
     *
     * Given a unit normal, constructs an orthonormal tangent basis.
     * Uses a robust algorithm that handles normals aligned with any axis.
     *
     * @param normal Unit normal vector
     * @param[out] tangent1 First tangent vector
     * @param[out] tangent2 Second tangent vector
     */
    static void compute_tangent_basis(
        const Vec3& normal,
        Vec3& tangent1,
        Vec3& tangent2);

    /**
     * @brief Compute tangent from 2D normal
     *
     * Returns the tangent vector rotated 90 degrees counter-clockwise from normal.
     *
     * @param normal 2D unit normal vector
     * @return Tangent vector
     */
    static Vec2 compute_tangent_2d(const Vec2& normal);

    /**
     * @brief Verify orthogonal decomposition: ||v||² = (v·n)² + ||v_t||²
     *
     * Checks that the decomposition satisfies the Pythagorean identity
     * within a specified tolerance.
     *
     * @param v Original vector
     * @param normal Unit normal
     * @param tol Tolerance for the check
     * @return true if decomposition is valid
     */
    static bool verify_decomposition(
        const Vec3& v,
        const Vec3& normal,
        Real tol = Real(1e-10));

    /**
     * @brief Compute the normal trace for H(div) spaces: v·n
     *
     * This is equivalent to normal_component but named for clarity
     * when used with H(div) conforming spaces.
     *
     * @param vector_field Vector values at face quadrature points
     * @param face_normal Outward unit normal on the face
     * @return Normal trace values
     */
    static std::vector<Real> hdiv_normal_trace(
        const std::vector<Vec3>& vector_field,
        const Vec3& face_normal);

    /**
     * @brief Compute the tangential trace for H(curl) spaces: n×(v×n)
     *
     * For H(curl) spaces, the tangential trace is the projection onto
     * the tangent plane, which can be computed as n×(v×n) = v - (v·n)n.
     *
     * @param vector_field Vector values at face quadrature points
     * @param face_normal Outward unit normal on the face
     * @return Tangential trace vectors
     */
    static std::vector<Vec3> hcurl_tangential_trace(
        const std::vector<Vec3>& vector_field,
        const Vec3& face_normal);
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_VECTORCOMPONENTEXTRACTOR_H
