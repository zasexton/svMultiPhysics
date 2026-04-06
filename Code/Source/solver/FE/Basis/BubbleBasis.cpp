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

} // namespace

BubbleBasis::BubbleBasis(ElementType type) : element_type_(type) {
    if (is_tri(type)) {
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
    } else {
        throw FEException("BubbleBasis: unsupported element type",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

void BubbleBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                  std::vector<Real>& values) const {
    values.resize(1);

    if (is_tri(element_type_)) {
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

    if (is_tri(element_type_)) {
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
