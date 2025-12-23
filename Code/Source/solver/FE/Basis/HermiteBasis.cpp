/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "HermiteBasis.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace basis {

namespace {

inline void hermite_1d(const Real s,
                       Real& H1, Real& H2, Real& H3, Real& H4,
                       Real& dH1_ds, Real& dH2_ds, Real& dH3_ds, Real& dH4_ds) {
    // Map reference coordinate s ∈ [-1,1] to t ∈ [0,1]
    const Real t  = Real(0.5) * (s + Real(1));
    const Real t2 = t * t;
    const Real t3 = t2 * t;

    // Standard cubic Hermite basis in t (value and slope with respect to t)
    const Real h1 = Real(1) - Real(3) * t2 + Real(2) * t3;  // value at left
    const Real h2 = t - Real(2) * t2 + t3;                  // slope at left
    const Real h3 = Real(3) * t2 - Real(2) * t3;            // value at right
    const Real h4 = -t2 + t3;                               // slope at right

    // Derivatives w.r.t t
    const Real dh1_dt = -Real(6) * t + Real(6) * t2;
    const Real dh2_dt = Real(1) - Real(4) * t + Real(3) * t2;
    const Real dh3_dt = Real(6) * t - Real(6) * t2;
    const Real dh4_dt = -Real(2) * t + Real(3) * t2;

    const Real dt_ds = Real(0.5);

    // Scale slope modes so that derivatives are taken with respect to s,
    // i.e., dH2/ds = 1 at the left node and dH4/ds = 1 at the right node.
    H1 = h1;
    H3 = h3;
    H2 = Real(2) * h2;
    H4 = Real(2) * h4;

    dH1_ds = dh1_dt * dt_ds;
    dH3_ds = dh3_dt * dt_ds;
    dH2_ds = Real(2) * dh2_dt * dt_ds;
    dH4_ds = Real(2) * dh4_dt * dt_ds;
}

} // namespace

HermiteBasis::HermiteBasis(ElementType element_type,
                           int order)
    : element_type_(element_type),
      dimension_(0),
      order_(order),
      size_(0) {
    if (order_ != 3) {
        throw FEException("HermiteBasis currently supports cubic order (3) only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Supported Hermite configurations:
    //  - 1D cubic Hermite on Line2 (4 DOFs)
    //  - 2D bicubic Hermite on Quad4 (16 DOFs)
    if (element_type_ == ElementType::Line2) {
        dimension_ = 1;
        size_ = 4;
    } else if (element_type_ == ElementType::Quad4) {
        dimension_ = 2;
        size_ = 16;
    } else {
        throw FEException("HermiteBasis currently supports Line2 and Quad4 only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

void HermiteBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                   std::vector<Real>& values) const {
    values.resize(size_);

    if (dimension_ == 1) {
        Real H1, H2, H3, H4;
        Real dH1, dH2, dH3, dH4;
        hermite_1d(xi[0], H1, H2, H3, H4, dH1, dH2, dH3, dH4);

        values[0] = H1;
        values[1] = H3;
        values[2] = H2;
        values[3] = H4;
        return;
    }

    if (dimension_ == 2) {
        Real H1x, H2x, H3x, H4x;
        Real dH1x, dH2x, dH3x, dH4x;
        Real H1y, H2y, H3y, H4y;
        Real dH1y, dH2y, dH3y, dH4y;

        hermite_1d(xi[0], H1x, H2x, H3x, H4x, dH1x, dH2x, dH3x, dH4x);
        hermite_1d(xi[1], H1y, H2y, H3y, H4y, dH1y, dH2y, dH3y, dH4y);

        auto set_corner = [&](int corner,
                              Real Vx, Real Sx,
                              Real Vy, Real Sy) {
            const std::size_t base = static_cast<std::size_t>(4 * corner);
            values[base + 0] = Vx * Vy;  // value DOF
            values[base + 1] = Sx * Vy;  // d/dx DOF
            values[base + 2] = Vx * Sy;  // d/dy DOF
            values[base + 3] = Sx * Sy;  // d2/(dx dy) DOF
        };

        // Corner 0: (-1, -1)  -> left/bottom
        set_corner(0, H1x, H2x, H1y, H2y);
        // Corner 1: (+1, -1)  -> right/bottom
        set_corner(1, H3x, H4x, H1y, H2y);
        // Corner 2: (+1, +1)  -> right/top
        set_corner(2, H3x, H4x, H3y, H4y);
        // Corner 3: (-1, +1)  -> left/top
        set_corner(3, H1x, H2x, H3y, H4y);
        return;
    }

    throw FEException("HermiteBasis::evaluate_values: unsupported dimension",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
}

void HermiteBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                      std::vector<Gradient>& gradients) const {
    gradients.resize(size_);

    if (dimension_ == 1) {
        Real H1, H2, H3, H4;
        Real dH1, dH2, dH3, dH4;
        hermite_1d(xi[0], H1, H2, H3, H4, dH1, dH2, dH3, dH4);

        gradients[0] = Gradient{};
        gradients[1] = Gradient{};
        gradients[2] = Gradient{};
        gradients[3] = Gradient{};

        gradients[0][0] = dH1;
        gradients[1][0] = dH3;
        gradients[2][0] = dH2;
        gradients[3][0] = dH4;
        return;
    }

    if (dimension_ == 2) {
        Real H1x, H2x, H3x, H4x;
        Real dH1x, dH2x, dH3x, dH4x;
        Real H1y, H2y, H3y, H4y;
        Real dH1y, dH2y, dH3y, dH4y;

        hermite_1d(xi[0], H1x, H2x, H3x, H4x, dH1x, dH2x, dH3x, dH4x);
        hermite_1d(xi[1], H1y, H2y, H3y, H4y, dH1y, dH2y, dH3y, dH4y);

        auto set_corner = [&](int corner,
                              Real Vx, Real dVx,
                              Real Sx, Real dSx,
                              Real Vy, Real dVy,
                              Real Sy, Real dSy) {
            const std::size_t base = static_cast<std::size_t>(4 * corner);

            // Value DOF: N = Vx * Vy
            gradients[base + 0] = Gradient{};
            gradients[base + 0][0] = dVx * Vy;
            gradients[base + 0][1] = Vx * dVy;

            // d/dx DOF: N = Sx * Vy
            gradients[base + 1] = Gradient{};
            gradients[base + 1][0] = dSx * Vy;
            gradients[base + 1][1] = Sx * dVy;

            // d/dy DOF: N = Vx * Sy
            gradients[base + 2] = Gradient{};
            gradients[base + 2][0] = dVx * Sy;
            gradients[base + 2][1] = Vx * dSy;

            // d2/(dx dy) DOF: N = Sx * Sy
            gradients[base + 3] = Gradient{};
            gradients[base + 3][0] = dSx * Sy;
            gradients[base + 3][1] = Sx * dSy;
        };

        // Corner 0: (-1, -1)  -> left/bottom
        set_corner(0,
                   H1x, dH1x,
                   H2x, dH2x,
                   H1y, dH1y,
                   H2y, dH2y);
        // Corner 1: (+1, -1)  -> right/bottom
        set_corner(1,
                   H3x, dH3x,
                   H4x, dH4x,
                   H1y, dH1y,
                   H2y, dH2y);
        // Corner 2: (+1, +1)  -> right/top
        set_corner(2,
                   H3x, dH3x,
                   H4x, dH4x,
                   H3y, dH3y,
                   H4y, dH4y);
        // Corner 3: (-1, +1)  -> left/top
        set_corner(3,
                   H1x, dH1x,
                   H2x, dH2x,
                   H3y, dH3y,
                   H4y, dH4y);
        return;
    }

    throw FEException("HermiteBasis::evaluate_gradients: unsupported dimension",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
}

} // namespace basis
} // namespace FE
} // namespace svmp
