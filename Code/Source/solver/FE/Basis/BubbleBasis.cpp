/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BubbleBasis.h"

namespace svmp {
namespace FE {
namespace basis {

namespace {

bool is_line(ElementType t) {
    return t == ElementType::Line2 || t == ElementType::Line3;
}

bool is_tri(ElementType t) {
    return t == ElementType::Triangle3 || t == ElementType::Triangle6;
}

bool is_tet(ElementType t) {
    return t == ElementType::Tetra4 || t == ElementType::Tetra10;
}

bool is_quad(ElementType t) {
    return t == ElementType::Quad4 || t == ElementType::Quad8 || t == ElementType::Quad9;
}

bool is_hex(ElementType t) {
    return t == ElementType::Hex8 || t == ElementType::Hex20 || t == ElementType::Hex27;
}

bool is_wedge(ElementType t) {
    return t == ElementType::Wedge6 || t == ElementType::Wedge15 || t == ElementType::Wedge18;
}

bool is_pyramid(ElementType t) {
    return t == ElementType::Pyramid5 || t == ElementType::Pyramid13 || t == ElementType::Pyramid14;
}

} // namespace

BubbleBasis::BubbleBasis(ElementType type) : element_type_(type) {
    if (is_line(type)) {
        dimension_ = 1;
        order_ = 2;
    } else if (is_tri(type)) {
        dimension_ = 2;
        order_ = 3;
    } else if (is_tet(type)) {
        dimension_ = 3;
        order_ = 4;
    } else if (is_quad(type)) {
        dimension_ = 2;
        order_ = 2;
    } else if (is_hex(type)) {
        dimension_ = 3;
        order_ = 2;
    } else if (is_wedge(type)) {
        dimension_ = 3;
        order_ = 5;
    } else if (is_pyramid(type)) {
        dimension_ = 3;
        order_ = 5;
    } else {
        throw BasisElementCompatibilityException("BubbleBasis: unsupported element type",
                                                 __FILE__, __LINE__, __func__);
    }
}

void BubbleBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                  std::vector<Real>& values) const {
    values.resize(1);

    if (is_line(element_type_)) {
        const Real x = xi[0];
        values[0] = Real(1) - x * x;
    } else if (is_tri(element_type_)) {
        // Barycentric: L0 = 1 - xi - eta, L1 = xi, L2 = eta
        const Real L0 = Real(1) - xi[0] - xi[1];
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        values[0] = Real(27) * L0 * L1 * L2;
    } else if (is_tet(element_type_)) {
        const Real L0 = Real(1) - xi[0] - xi[1] - xi[2];
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        const Real L3 = xi[2];
        values[0] = Real(256) * L0 * L1 * L2 * L3;
    } else if (is_quad(element_type_)) {
        values[0] = (Real(1) - xi[0] * xi[0]) * (Real(1) - xi[1] * xi[1]);
    } else if (is_wedge(element_type_)) {
        const Real L0 = Real(1) - xi[0] - xi[1];
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        const Real z = xi[2];
        values[0] = Real(27) * L0 * L1 * L2 * (Real(1) - z * z);
    } else if (is_pyramid(element_type_)) {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        const Real one_minus_z = Real(1) - z;
        const Real ax = one_minus_z * one_minus_z - x * x;
        const Real ay = one_minus_z * one_minus_z - y * y;
        values[0] = (Real(3125) / Real(256)) * z * ax * ay;
    } else {
        // Hex
        values[0] = (Real(1) - xi[0] * xi[0]) *
                    (Real(1) - xi[1] * xi[1]) *
                    (Real(1) - xi[2] * xi[2]);
    }
}

void BubbleBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                     std::vector<Gradient>& gradients) const {
    gradients.resize(1);
    Gradient g{};

    if (is_line(element_type_)) {
        g[0] = -Real(2) * xi[0];
    } else if (is_tri(element_type_)) {
        const Real L0 = Real(1) - xi[0] - xi[1];
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        // db/dxi  = 27 * (dL0/dxi * L1 * L2 + L0 * dL1/dxi * L2 + L0 * L1 * dL2/dxi)
        // dL0/dxi = -1, dL1/dxi = 1, dL2/dxi = 0
        g[0] = Real(27) * (-L1 * L2 + L0 * L2);
        // dL0/deta = -1, dL1/deta = 0, dL2/deta = 1
        g[1] = Real(27) * (-L1 * L2 + L0 * L1);
    } else if (is_tet(element_type_)) {
        const Real L0 = Real(1) - xi[0] - xi[1] - xi[2];
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        const Real L3 = xi[2];
        // dL0/dxi = -1, dL1/dxi = 1, dL2/dxi = 0, dL3/dxi = 0
        g[0] = Real(256) * (-L1 * L2 * L3 + L0 * L2 * L3);
        g[1] = Real(256) * (-L1 * L2 * L3 + L0 * L1 * L3);
        g[2] = Real(256) * (-L1 * L2 * L3 + L0 * L1 * L2);
    } else if (is_quad(element_type_)) {
        const Real x = xi[0], y = xi[1];
        g[0] = -Real(2) * x * (Real(1) - y * y);
        g[1] = (Real(1) - x * x) * (-Real(2) * y);
    } else if (is_wedge(element_type_)) {
        const Real L0 = Real(1) - xi[0] - xi[1];
        const Real L1 = xi[0];
        const Real L2 = xi[1];
        const Real z = xi[2];
        const Real tri = Real(27) * L0 * L1 * L2;
        const Real z_factor = Real(1) - z * z;

        const Real dtri_dxi = Real(27) * (-L1 * L2 + L0 * L2);
        const Real dtri_deta = Real(27) * (-L1 * L2 + L0 * L1);

        g[0] = dtri_dxi * z_factor;
        g[1] = dtri_deta * z_factor;
        g[2] = tri * (Real(-2) * z);
    } else if (is_pyramid(element_type_)) {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        const Real s = Real(1) - z;
        const Real ax = s * s - x * x;
        const Real ay = s * s - y * y;
        const Real scale = Real(3125) / Real(256);

        g[0] = scale * z * (Real(-2) * x) * ay;
        g[1] = scale * z * ax * (Real(-2) * y);
        g[2] = scale * (ax * ay - Real(2) * z * s * (ax + ay));
    } else {
        // Hex
        const Real x = xi[0], y = xi[1], z = xi[2];
        const Real mx = Real(1) - x * x;
        const Real my = Real(1) - y * y;
        const Real mz = Real(1) - z * z;
        g[0] = -Real(2) * x * my * mz;
        g[1] = mx * (-Real(2) * y) * mz;
        g[2] = mx * my * (-Real(2) * z);
    }

    gradients[0] = g;
}

} // namespace basis
} // namespace FE
} // namespace svmp
