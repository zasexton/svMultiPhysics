/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "HermiteBasis.h"

#include "Basis/BasisExceptions.h"

namespace svmp {
namespace FE {
namespace basis {

namespace {

inline void hermite_1d(const Real s,
                       Real& H1, Real& H2, Real& H3, Real& H4,
                       Real& dH1_ds, Real& dH2_ds, Real& dH3_ds, Real& dH4_ds,
                       Real& ddH1_ds2, Real& ddH2_ds2, Real& ddH3_ds2, Real& ddH4_ds2) {
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
    const Real ddh1_dt2 = -Real(6) + Real(12) * t;
    const Real ddh2_dt2 = -Real(4) + Real(6) * t;
    const Real ddh3_dt2 = Real(6) - Real(12) * t;
    const Real ddh4_dt2 = -Real(2) + Real(6) * t;

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

    const Real dt_ds_sq = dt_ds * dt_ds;
    ddH1_ds2 = ddh1_dt2 * dt_ds_sq;
    ddH3_ds2 = ddh3_dt2 * dt_ds_sq;
    ddH2_ds2 = Real(2) * ddh2_dt2 * dt_ds_sq;
    ddH4_ds2 = Real(2) * ddh4_dt2 * dt_ds_sq;
}

} // namespace

HermiteBasis::HermiteBasis(ElementType element_type,
                           int order)
    : element_type_(element_type),
      dimension_(0),
      order_(order),
      size_(0) {
    if (order_ != 3) {
        throw NotImplementedException("HermiteBasis is intentionally limited to cubic order (3)",
                                      __FILE__, __LINE__, __func__);
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
    } else if (element_type_ == ElementType::Hex8) {
        dimension_ = 3;
        size_ = 64;
    } else {
        throw BasisElementCompatibilityException("HermiteBasis is intentionally limited to Line2, Quad4, and Hex8",
                                                 __FILE__, __LINE__, __func__);
    }
}

void HermiteBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                   std::vector<Real>& values) const {
    values.resize(size_);

    if (dimension_ == 1) {
        Real H1, H2, H3, H4;
        Real dH1, dH2, dH3, dH4, ddH1, ddH2, ddH3, ddH4;
        hermite_1d(xi[0], H1, H2, H3, H4, dH1, dH2, dH3, dH4, ddH1, ddH2, ddH3, ddH4);

        values[0] = H1;
        values[1] = H3;
        values[2] = H2;
        values[3] = H4;
        return;
    }

    if (dimension_ == 2) {
        Real H1x, H2x, H3x, H4x;
        Real dH1x, dH2x, dH3x, dH4x, ddH1x, ddH2x, ddH3x, ddH4x;
        Real H1y, H2y, H3y, H4y;
        Real dH1y, dH2y, dH3y, dH4y, ddH1y, ddH2y, ddH3y, ddH4y;

        hermite_1d(xi[0], H1x, H2x, H3x, H4x, dH1x, dH2x, dH3x, dH4x,
                   ddH1x, ddH2x, ddH3x, ddH4x);
        hermite_1d(xi[1], H1y, H2y, H3y, H4y, dH1y, dH2y, dH3y, dH4y,
                   ddH1y, ddH2y, ddH3y, ddH4y);

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

    if (dimension_ == 3) {
        Real Hx[4], dHx[4];
        Real ddHx[4];
        Real Hy[4], dHy[4], ddHy[4];
        Real Hz[4], dHz[4], ddHz[4];
        hermite_1d(xi[0], Hx[0], Hx[1], Hx[2], Hx[3], dHx[0], dHx[1], dHx[2], dHx[3],
                   ddHx[0], ddHx[1], ddHx[2], ddHx[3]);
        hermite_1d(xi[1], Hy[0], Hy[1], Hy[2], Hy[3], dHy[0], dHy[1], dHy[2], dHy[3],
                   ddHy[0], ddHy[1], ddHy[2], ddHy[3]);
        hermite_1d(xi[2], Hz[0], Hz[1], Hz[2], Hz[3], dHz[0], dHz[1], dHz[2], dHz[3],
                   ddHz[0], ddHz[1], ddHz[2], ddHz[3]);

        // Hex8 corner ordering (VTK): 0(-,-,-), 1(+,-,-), 2(+,+,-), 3(-,+,-),
        //                             4(-,-,+), 5(+,-,+), 6(+,+,+), 7(-,+,+)
        // For each corner: (Vx, Sx) = (H1, H2) for left, (H3, H4) for right
        const int cx[] = {0, 2, 2, 0, 0, 2, 2, 0}; // index into Hx: 0=H1(left), 2=H3(right)
        const int cy[] = {0, 0, 2, 2, 0, 0, 2, 2};
        const int cz[] = {0, 0, 0, 0, 2, 2, 2, 2};

        for (int c = 0; c < 8; ++c) {
            const std::size_t base = static_cast<std::size_t>(8 * c);
            const int ix = cx[c], iy = cy[c], iz = cz[c];
            // Value (Vx), slope (Sx=Vx+1) indices into Hx/Hy/Hz arrays
            const Real Vx = Hx[ix], Sx = Hx[ix + 1];
            const Real Vy = Hy[iy], Sy = Hy[iy + 1];
            const Real Vz = Hz[iz], Sz = Hz[iz + 1];

            values[base + 0] = Vx * Vy * Vz;  // value
            values[base + 1] = Sx * Vy * Vz;  // d/dx
            values[base + 2] = Vx * Sy * Vz;  // d/dy
            values[base + 3] = Vx * Vy * Sz;  // d/dz
            values[base + 4] = Sx * Sy * Vz;  // d2/(dx dy)
            values[base + 5] = Sx * Vy * Sz;  // d2/(dx dz)
            values[base + 6] = Vx * Sy * Sz;  // d2/(dy dz)
            values[base + 7] = Sx * Sy * Sz;  // d3/(dx dy dz)
        }
        return;
    }

    throw BasisEvaluationException("HermiteBasis::evaluate_values: unsupported dimension",
                                   __FILE__, __LINE__, __func__);
}

void HermiteBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                     std::vector<Hessian>& hessians) const {
    hessians.resize(size_);

    if (dimension_ == 1) {
        Real H1, H2, H3, H4;
        Real dH1, dH2, dH3, dH4;
        Real ddH1, ddH2, ddH3, ddH4;
        hermite_1d(xi[0], H1, H2, H3, H4, dH1, dH2, dH3, dH4, ddH1, ddH2, ddH3, ddH4);

        for (auto& H : hessians) {
            H = Hessian{};
        }
        hessians[0](0, 0) = ddH1;
        hessians[1](0, 0) = ddH3;
        hessians[2](0, 0) = ddH2;
        hessians[3](0, 0) = ddH4;
        return;
    }

    if (dimension_ == 2) {
        Real Hx[4], dHx[4], ddHx[4];
        Real Hy[4], dHy[4], ddHy[4];
        hermite_1d(xi[0], Hx[0], Hx[1], Hx[2], Hx[3], dHx[0], dHx[1], dHx[2], dHx[3],
                   ddHx[0], ddHx[1], ddHx[2], ddHx[3]);
        hermite_1d(xi[1], Hy[0], Hy[1], Hy[2], Hy[3], dHy[0], dHy[1], dHy[2], dHy[3],
                   ddHy[0], ddHy[1], ddHy[2], ddHy[3]);

        auto set_corner = [&](int corner,
                              Real Vx, Real dVx, Real ddVx,
                              Real Sx, Real dSx, Real ddSx,
                              Real Vy, Real dVy, Real ddVy,
                              Real Sy, Real dSy, Real ddSy) {
            const std::size_t base = static_cast<std::size_t>(4 * corner);

            auto fill = [&](std::size_t offset,
                            Real ax, Real dax, Real ddax,
                            Real ay, Real day, Real dday) {
                Hessian H{};
                H(0, 0) = ddax * ay;
                H(0, 1) = dax * day;
                H(1, 0) = H(0, 1);
                H(1, 1) = ax * dday;
                hessians[base + offset] = H;
            };

            fill(0, Vx, dVx, ddVx, Vy, dVy, ddVy);
            fill(1, Sx, dSx, ddSx, Vy, dVy, ddVy);
            fill(2, Vx, dVx, ddVx, Sy, dSy, ddSy);
            fill(3, Sx, dSx, ddSx, Sy, dSy, ddSy);
        };

        set_corner(0, Hx[0], dHx[0], ddHx[0], Hx[1], dHx[1], ddHx[1],
                   Hy[0], dHy[0], ddHy[0], Hy[1], dHy[1], ddHy[1]);
        set_corner(1, Hx[2], dHx[2], ddHx[2], Hx[3], dHx[3], ddHx[3],
                   Hy[0], dHy[0], ddHy[0], Hy[1], dHy[1], ddHy[1]);
        set_corner(2, Hx[2], dHx[2], ddHx[2], Hx[3], dHx[3], ddHx[3],
                   Hy[2], dHy[2], ddHy[2], Hy[3], dHy[3], ddHy[3]);
        set_corner(3, Hx[0], dHx[0], ddHx[0], Hx[1], dHx[1], ddHx[1],
                   Hy[2], dHy[2], ddHy[2], Hy[3], dHy[3], ddHy[3]);
        return;
    }

    if (dimension_ == 3) {
        Real Hx[4], dHx[4], ddHx[4];
        Real Hy[4], dHy[4], ddHy[4];
        Real Hz[4], dHz[4], ddHz[4];
        hermite_1d(xi[0], Hx[0], Hx[1], Hx[2], Hx[3], dHx[0], dHx[1], dHx[2], dHx[3],
                   ddHx[0], ddHx[1], ddHx[2], ddHx[3]);
        hermite_1d(xi[1], Hy[0], Hy[1], Hy[2], Hy[3], dHy[0], dHy[1], dHy[2], dHy[3],
                   ddHy[0], ddHy[1], ddHy[2], ddHy[3]);
        hermite_1d(xi[2], Hz[0], Hz[1], Hz[2], Hz[3], dHz[0], dHz[1], dHz[2], dHz[3],
                   ddHz[0], ddHz[1], ddHz[2], ddHz[3]);

        const int cx[] = {0, 2, 2, 0, 0, 2, 2, 0};
        const int cy[] = {0, 0, 2, 2, 0, 0, 2, 2};
        const int cz[] = {0, 0, 0, 0, 2, 2, 2, 2};

        auto fill = [&](std::size_t idx,
                        Real ax, Real dax, Real ddax,
                        Real ay, Real day, Real dday,
                        Real az, Real daz, Real ddaz) {
            Hessian H{};
            H(0, 0) = ddax * ay * az;
            H(1, 1) = ax * dday * az;
            H(2, 2) = ax * ay * ddaz;
            H(0, 1) = dax * day * az;
            H(1, 0) = H(0, 1);
            H(0, 2) = dax * ay * daz;
            H(2, 0) = H(0, 2);
            H(1, 2) = ax * day * daz;
            H(2, 1) = H(1, 2);
            hessians[idx] = H;
        };

        for (int c = 0; c < 8; ++c) {
            const std::size_t base = static_cast<std::size_t>(8 * c);
            const int ix = cx[c];
            const int iy = cy[c];
            const int iz = cz[c];
            fill(base + 0, Hx[ix], dHx[ix], ddHx[ix], Hy[iy], dHy[iy], ddHy[iy], Hz[iz], dHz[iz], ddHz[iz]);
            fill(base + 1, Hx[ix + 1], dHx[ix + 1], ddHx[ix + 1], Hy[iy], dHy[iy], ddHy[iy], Hz[iz], dHz[iz], ddHz[iz]);
            fill(base + 2, Hx[ix], dHx[ix], ddHx[ix], Hy[iy + 1], dHy[iy + 1], ddHy[iy + 1], Hz[iz], dHz[iz], ddHz[iz]);
            fill(base + 3, Hx[ix], dHx[ix], ddHx[ix], Hy[iy], dHy[iy], ddHy[iy], Hz[iz + 1], dHz[iz + 1], ddHz[iz + 1]);
            fill(base + 4, Hx[ix + 1], dHx[ix + 1], ddHx[ix + 1], Hy[iy + 1], dHy[iy + 1], ddHy[iy + 1], Hz[iz], dHz[iz], ddHz[iz]);
            fill(base + 5, Hx[ix + 1], dHx[ix + 1], ddHx[ix + 1], Hy[iy], dHy[iy], ddHy[iy], Hz[iz + 1], dHz[iz + 1], ddHz[iz + 1]);
            fill(base + 6, Hx[ix], dHx[ix], ddHx[ix], Hy[iy + 1], dHy[iy + 1], ddHy[iy + 1], Hz[iz + 1], dHz[iz + 1], ddHz[iz + 1]);
            fill(base + 7, Hx[ix + 1], dHx[ix + 1], ddHx[ix + 1], Hy[iy + 1], dHy[iy + 1], ddHy[iy + 1], Hz[iz + 1], dHz[iz + 1], ddHz[iz + 1]);
        }
        return;
    }

    throw BasisEvaluationException("HermiteBasis::evaluate_hessians: unsupported dimension",
                                   __FILE__, __LINE__, __func__);
}

void HermiteBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                      std::vector<Gradient>& gradients) const {
    gradients.resize(size_);

    if (dimension_ == 1) {
        Real H1, H2, H3, H4;
        Real dH1, dH2, dH3, dH4, ddH1, ddH2, ddH3, ddH4;
        hermite_1d(xi[0], H1, H2, H3, H4, dH1, dH2, dH3, dH4, ddH1, ddH2, ddH3, ddH4);

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
        Real dH1x, dH2x, dH3x, dH4x, ddH1x, ddH2x, ddH3x, ddH4x;
        Real H1y, H2y, H3y, H4y;
        Real dH1y, dH2y, dH3y, dH4y, ddH1y, ddH2y, ddH3y, ddH4y;

        hermite_1d(xi[0], H1x, H2x, H3x, H4x, dH1x, dH2x, dH3x, dH4x,
                   ddH1x, ddH2x, ddH3x, ddH4x);
        hermite_1d(xi[1], H1y, H2y, H3y, H4y, dH1y, dH2y, dH3y, dH4y,
                   ddH1y, ddH2y, ddH3y, ddH4y);

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

    if (dimension_ == 3) {
        Real Hx[4], dHx[4], ddHx[4];
        Real Hy[4], dHy[4], ddHy[4];
        Real Hz[4], dHz[4], ddHz[4];
        hermite_1d(xi[0], Hx[0], Hx[1], Hx[2], Hx[3], dHx[0], dHx[1], dHx[2], dHx[3],
                   ddHx[0], ddHx[1], ddHx[2], ddHx[3]);
        hermite_1d(xi[1], Hy[0], Hy[1], Hy[2], Hy[3], dHy[0], dHy[1], dHy[2], dHy[3],
                   ddHy[0], ddHy[1], ddHy[2], ddHy[3]);
        hermite_1d(xi[2], Hz[0], Hz[1], Hz[2], Hz[3], dHz[0], dHz[1], dHz[2], dHz[3],
                   ddHz[0], ddHz[1], ddHz[2], ddHz[3]);

        const int cx[] = {0, 2, 2, 0, 0, 2, 2, 0};
        const int cy[] = {0, 0, 2, 2, 0, 0, 2, 2};
        const int cz[] = {0, 0, 0, 0, 2, 2, 2, 2};

        for (int c = 0; c < 8; ++c) {
            const std::size_t base = static_cast<std::size_t>(8 * c);
            const int ix = cx[c], iy = cy[c], iz = cz[c];
            const Real Vx = Hx[ix], Sx = Hx[ix + 1];
            const Real Vy = Hy[iy], Sy = Hy[iy + 1];
            const Real Vz = Hz[iz], Sz = Hz[iz + 1];
            const Real dVx = dHx[ix], dSx = dHx[ix + 1];
            const Real dVy = dHy[iy], dSy = dHy[iy + 1];
            const Real dVz = dHz[iz], dSz = dHz[iz + 1];

            // 8 DOFs per corner, 3 gradient components each
            // DOF 0: Vx*Vy*Vz
            gradients[base + 0] = Gradient{};
            gradients[base + 0][0] = dVx * Vy * Vz;
            gradients[base + 0][1] = Vx * dVy * Vz;
            gradients[base + 0][2] = Vx * Vy * dVz;

            // DOF 1: Sx*Vy*Vz
            gradients[base + 1] = Gradient{};
            gradients[base + 1][0] = dSx * Vy * Vz;
            gradients[base + 1][1] = Sx * dVy * Vz;
            gradients[base + 1][2] = Sx * Vy * dVz;

            // DOF 2: Vx*Sy*Vz
            gradients[base + 2] = Gradient{};
            gradients[base + 2][0] = dVx * Sy * Vz;
            gradients[base + 2][1] = Vx * dSy * Vz;
            gradients[base + 2][2] = Vx * Sy * dVz;

            // DOF 3: Vx*Vy*Sz
            gradients[base + 3] = Gradient{};
            gradients[base + 3][0] = dVx * Vy * Sz;
            gradients[base + 3][1] = Vx * dVy * Sz;
            gradients[base + 3][2] = Vx * Vy * dSz;

            // DOF 4: Sx*Sy*Vz
            gradients[base + 4] = Gradient{};
            gradients[base + 4][0] = dSx * Sy * Vz;
            gradients[base + 4][1] = Sx * dSy * Vz;
            gradients[base + 4][2] = Sx * Sy * dVz;

            // DOF 5: Sx*Vy*Sz
            gradients[base + 5] = Gradient{};
            gradients[base + 5][0] = dSx * Vy * Sz;
            gradients[base + 5][1] = Sx * dVy * Sz;
            gradients[base + 5][2] = Sx * Vy * dSz;

            // DOF 6: Vx*Sy*Sz
            gradients[base + 6] = Gradient{};
            gradients[base + 6][0] = dVx * Sy * Sz;
            gradients[base + 6][1] = Vx * dSy * Sz;
            gradients[base + 6][2] = Vx * Sy * dSz;

            // DOF 7: Sx*Sy*Sz
            gradients[base + 7] = Gradient{};
            gradients[base + 7][0] = dSx * Sy * Sz;
            gradients[base + 7][1] = Sx * dSy * Sz;
            gradients[base + 7][2] = Sx * Sy * dSz;
        }
        return;
    }

    throw BasisEvaluationException("HermiteBasis::evaluate_gradients: unsupported dimension",
                                   __FILE__, __LINE__, __func__);
}

} // namespace basis
} // namespace FE
} // namespace svmp
