/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/VectorComponentExtractor.h"
#include "Core/FEException.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace spaces {

std::vector<Real> VectorComponentExtractor::normal_component(
    const std::vector<Vec3>& vector_field,
    const std::vector<Vec3>& normals) {

    FE_CHECK_ARG(vector_field.size() == normals.size(),
        "Vector field and normals must have same size");

    std::vector<Real> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        result[i] = vector_field[i].dot(normals[i]);
    }
    return result;
}

std::vector<Real> VectorComponentExtractor::normal_component(
    const std::vector<Vec3>& vector_field,
    const Vec3& normal) {

    std::vector<Real> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        result[i] = vector_field[i].dot(normal);
    }
    return result;
}

std::vector<VectorComponentExtractor::Vec3> VectorComponentExtractor::tangential_component(
    const std::vector<Vec3>& vector_field,
    const std::vector<Vec3>& normals) {

    FE_CHECK_ARG(vector_field.size() == normals.size(),
        "Vector field and normals must have same size");

    std::vector<Vec3> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        const Real vn = vector_field[i].dot(normals[i]);
        result[i] = vector_field[i] - vn * normals[i];
    }
    return result;
}

std::vector<VectorComponentExtractor::Vec3> VectorComponentExtractor::tangential_component(
    const std::vector<Vec3>& vector_field,
    const Vec3& normal) {

    std::vector<Vec3> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        const Real vn = vector_field[i].dot(normal);
        result[i] = vector_field[i] - vn * normal;
    }
    return result;
}

std::vector<Real> VectorComponentExtractor::tangential_component_2d(
    const std::vector<Vec2>& vector_field,
    const std::vector<Vec2>& normals) {

    FE_CHECK_ARG(vector_field.size() == normals.size(),
        "Vector field and normals must have same size");

    std::vector<Real> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        // Tangent is 90 degrees CCW from normal: t = (-n_y, n_x)
        // v·t = v_x * (-n_y) + v_y * n_x = v_y * n_x - v_x * n_y
        result[i] = vector_field[i][1] * normals[i][0] - vector_field[i][0] * normals[i][1];
    }
    return result;
}

std::vector<Real> VectorComponentExtractor::tangential_component_2d(
    const std::vector<Vec2>& vector_field,
    const Vec2& normal) {

    std::vector<Real> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        result[i] = vector_field[i][1] * normal[0] - vector_field[i][0] * normal[1];
    }
    return result;
}

std::vector<VectorComponentExtractor::Vec2> VectorComponentExtractor::project_to_tangent_plane(
    const std::vector<Vec3>& vector_field,
    const Vec3& tangent1,
    const Vec3& tangent2) {

    std::vector<Vec2> result(vector_field.size());
    for (std::size_t i = 0; i < vector_field.size(); ++i) {
        result[i][0] = vector_field[i].dot(tangent1);
        result[i][1] = vector_field[i].dot(tangent2);
    }
    return result;
}

void VectorComponentExtractor::compute_tangent_basis(
    const Vec3& normal,
    Vec3& tangent1,
    Vec3& tangent2) {

    // Find the component of the normal with smallest absolute value
    // This ensures we pick a direction that's not nearly parallel to the normal
    const Real abs_x = std::abs(normal[0]);
    const Real abs_y = std::abs(normal[1]);
    const Real abs_z = std::abs(normal[2]);

    Vec3 helper;
    if (abs_x <= abs_y && abs_x <= abs_z) {
        // x is smallest, use x-axis
        helper = Vec3{Real(1), Real(0), Real(0)};
    } else if (abs_y <= abs_z) {
        // y is smallest, use y-axis
        helper = Vec3{Real(0), Real(1), Real(0)};
    } else {
        // z is smallest, use z-axis
        helper = Vec3{Real(0), Real(0), Real(1)};
    }

    // First tangent: helper × normal (perpendicular to normal)
    tangent1 = math::cross(helper, normal);
    tangent1.normalize();

    // Second tangent: normal × tangent1 (perpendicular to both)
    tangent2 = math::cross(normal, tangent1);
    tangent2.normalize();
}

VectorComponentExtractor::Vec2 VectorComponentExtractor::compute_tangent_2d(const Vec2& normal) {
    // Rotate 90 degrees counter-clockwise
    return Vec2{-normal[1], normal[0]};
}

bool VectorComponentExtractor::verify_decomposition(
    const Vec3& v,
    const Vec3& normal,
    Real tol) {

    const Real vn = v.dot(normal);
    Vec3 vt = v - vn * normal;

    const Real v_norm_sq = v.norm_squared();
    const Real vn_sq = vn * vn;
    const Real vt_norm_sq = vt.norm_squared();

    // Check Pythagorean identity: ||v||² = (v·n)² + ||v_t||²
    return std::abs(v_norm_sq - (vn_sq + vt_norm_sq)) < tol;
}

std::vector<Real> VectorComponentExtractor::hdiv_normal_trace(
    const std::vector<Vec3>& vector_field,
    const Vec3& face_normal) {
    return normal_component(vector_field, face_normal);
}

std::vector<VectorComponentExtractor::Vec3> VectorComponentExtractor::hcurl_tangential_trace(
    const std::vector<Vec3>& vector_field,
    const Vec3& face_normal) {
    return tangential_component(vector_field, face_normal);
}

} // namespace spaces
} // namespace FE
} // namespace svmp
