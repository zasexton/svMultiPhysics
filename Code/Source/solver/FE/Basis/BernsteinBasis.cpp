/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BernsteinBasis.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace basis {

namespace {

bool is_line(ElementType t) {
    return t == ElementType::Line2 || t == ElementType::Line3;
}

bool is_triangle(ElementType t) {
    return t == ElementType::Triangle3 || t == ElementType::Triangle6;
}

bool is_quadrilateral(ElementType t) {
    return t == ElementType::Quad4 || t == ElementType::Quad8 || t == ElementType::Quad9;
}

bool is_hexahedron(ElementType t) {
    return t == ElementType::Hex8 || t == ElementType::Hex20 || t == ElementType::Hex27;
}

bool is_wedge(ElementType t) {
    return t == ElementType::Wedge6 || t == ElementType::Wedge15 || t == ElementType::Wedge18;
}

bool is_pyramid(ElementType t) {
    return t == ElementType::Pyramid5 || t == ElementType::Pyramid13 || t == ElementType::Pyramid14;
}

inline Real pow_int(Real base, int exp) {
    if (exp == 0) {
        return Real(1);
    }
    return std::pow(base, static_cast<Real>(exp));
}

} // namespace

BernsteinBasis::BernsteinBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 0) {
        throw FEException("BernsteinBasis requires non-negative order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (is_line(element_type_)) {
        dimension_ = 1;
        size_ = static_cast<std::size_t>(order_ + 1);
    } else if (is_quadrilateral(element_type_)) {
        dimension_ = 2;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1));
    } else if (is_triangle(element_type_)) {
        dimension_ = 2;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 2) / 2);
        for (int i = 0; i <= order_; ++i) {
            for (int j = 0; j <= order_ - i; ++j) {
                int k = order_ - i - j;
                simplex_indices_.push_back({i, j, k, 0});
                coefficients_.push_back(binomial(order_, i) *
                                        binomial(order_ - i, j));
            }
        }
    } else if (is_hexahedron(element_type_)) {
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1) * (order_ + 1));
    } else if (is_wedge(element_type_)) {
        // Wedge treated as triangle x line in Bernstein form
        dimension_ = 3;
        const std::size_t tri_count =
            static_cast<std::size_t>((order_ + 1) * (order_ + 2) / 2);
        size_ = tri_count * static_cast<std::size_t>(order_ + 1);
        for (int i = 0; i <= order_; ++i) {
            for (int j = 0; j <= order_ - i; ++j) {
                int k = order_ - i - j;
                simplex_indices_.push_back({i, j, k, 0});
                coefficients_.push_back(binomial(order_, i) *
                                        binomial(order_ - i, j));
            }
        }
    } else if (is_pyramid(element_type_)) {
        // Pyramid treated via tensor-product Bernstein in cube coordinates
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1) * (order_ + 1));
    } else if (element_type_ == ElementType::Point1) {
        dimension_ = 0;
        size_ = 1;
    } else {
        throw FEException("Unsupported element type for BernsteinBasis",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

Real BernsteinBasis::binomial(int n, int k) const {
    if (k < 0 || k > n) return Real(0);
    if (k == 0 || k == n) return Real(1);
    Real res = Real(1);
    for (int i = 1; i <= k; ++i) {
        res *= static_cast<Real>(n - (k - i));
        res /= static_cast<Real>(i);
    }
    return res;
}

void BernsteinBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                     std::vector<Real>& values) const {
    values.assign(size_, Real(0));

    if (element_type_ == ElementType::Point1) {
        values[0] = Real(1);
        return;
    }

    if (is_line(element_type_)) {
        Real t = (xi[0] + Real(1)) * Real(0.5);
        for (int i = 0; i <= order_; ++i) {
            values[static_cast<std::size_t>(i)] =
                binomial(order_, i) *
                pow_int(t, i) *
                pow_int(Real(1) - t, order_ - i);
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        Real tx = (xi[0] + Real(1)) * Real(0.5);
        Real ty = (xi[1] + Real(1)) * Real(0.5);
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);
        std::vector<Real> bx(n1d), by(n1d);
        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bx[idx] = binomial(order_, i) *
                      pow_int(tx, i) *
                      pow_int(Real(1) - tx, order_ - i);
            by[idx] = binomial(order_, i) *
                      pow_int(ty, i) *
                      pow_int(Real(1) - ty, order_ - i);
        }

        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                values[idx++] = bx[static_cast<std::size_t>(i)] *
                                by[static_cast<std::size_t>(j)];
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        Real tx = (xi[0] + Real(1)) * Real(0.5);
        Real ty = (xi[1] + Real(1)) * Real(0.5);
        Real tz = (xi[2] + Real(1)) * Real(0.5);
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);
        std::vector<Real> bx(n1d), by(n1d), bz(n1d);
        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bx[idx] = binomial(order_, i) *
                      pow_int(tx, i) *
                      pow_int(Real(1) - tx, order_ - i);
            by[idx] = binomial(order_, i) *
                      pow_int(ty, i) *
                      pow_int(Real(1) - ty, order_ - i);
            bz[idx] = binomial(order_, i) *
                      pow_int(tz, i) *
                      pow_int(Real(1) - tz, order_ - i);
        }

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    values[idx++] = bx[static_cast<std::size_t>(i)] *
                                    by[static_cast<std::size_t>(j)] *
                                    bz[static_cast<std::size_t>(k)];
                }
            }
        }
        return;
    }

    if (is_wedge(element_type_)) {
        // Wedge = triangle (in barycentric coords) x 1D line Bernstein in z
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        const std::size_t tri_count = simplex_indices_.size();
        std::vector<Real> tri_vals(tri_count, Real(0));
        for (std::size_t idx = 0; idx < tri_count; ++idx) {
            const auto& e = simplex_indices_[idx];
            Real coeff = coefficients_[idx];
            tri_vals[idx] = coeff *
                            pow_int(l0, e[0]) * pow_int(l1, e[1]) * pow_int(l2, e[2]);
        }

        const Real tz = (xi[2] + Real(1)) * Real(0.5);
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);
        std::vector<Real> bz(n1d);
        for (int k = 0; k <= order_; ++k) {
            const std::size_t idx = static_cast<std::size_t>(k);
            bz[idx] = binomial(order_, k) *
                      pow_int(tz, k) *
                      pow_int(Real(1) - tz, order_ - k);
        }

        std::size_t out_idx = 0;
        for (std::size_t k = 0; k < n1d; ++k) {
            for (std::size_t n = 0; n < tri_count; ++n) {
                values[out_idx++] = tri_vals[n] * bz[k];
            }
        }
        return;
    }

    if (is_triangle(element_type_)) {
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;
        for (std::size_t idx = 0; idx < simplex_indices_.size(); ++idx) {
            const auto& e = simplex_indices_[idx];
            Real coeff = coefficients_[idx];
            values[idx] = coeff *
                          pow_int(l0, e[0]) * pow_int(l1, e[1]) * pow_int(l2, e[2]);
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        // Map pyramid to unit cube and use tensor-product Bernstein
        const Real z = xi[2];
        Real tx = Real(0.5);
        Real ty = Real(0.5);
        if (std::abs(Real(1) - z) > Real(1e-12)) {
            const Real scale = Real(1) / (Real(1) - z);
            const Real u = xi[0] * scale; // in [-1,1]
            const Real v = xi[1] * scale;
            tx = (u + Real(1)) * Real(0.5);
            ty = (v + Real(1)) * Real(0.5);
        }
        const Real tz = (z + Real(1)) * Real(0.5);
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);
        std::vector<Real> bx(n1d), by(n1d), bz(n1d);
        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bx[idx] = binomial(order_, i) *
                      pow_int(tx, i) *
                      pow_int(Real(1) - tx, order_ - i);
            by[idx] = binomial(order_, i) *
                      pow_int(ty, i) *
                      pow_int(Real(1) - ty, order_ - i);
            bz[idx] = binomial(order_, i) *
                      pow_int(tz, i) *
                      pow_int(Real(1) - tz, order_ - i);
        }

        std::size_t idx_out = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    idx_out = static_cast<std::size_t>(
                        (k * (order_ + 1) + j) * (order_ + 1) + i);
                    values[idx_out] = bx[static_cast<std::size_t>(i)] *
                                      by[static_cast<std::size_t>(j)] *
                                      bz[static_cast<std::size_t>(k)];
                }
            }
        }
        return;
    }

    throw FEException("Unsupported element in BernsteinBasis::evaluate_values",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void BernsteinBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                        std::vector<Gradient>& gradients) const {
    gradients.assign(size_, Gradient{});

    if (element_type_ == ElementType::Point1) {
        // Point has no spatial gradient
        return;
    }

    if (is_line(element_type_)) {
        // 1D Bernstein: B_i(t) = C(n,i) * t^i * (1-t)^(n-i), t = (x+1)/2
        // dB/dx = dB/dt * dt/dx = dB/dt * 0.5
        // dB/dt = n * [B_{i-1,n-1}(t) - B_{i,n-1}(t)]
        const Real t = (xi[0] + Real(1)) * Real(0.5);
        const Real dtdx = Real(0.5);

        // Compute lower order basis (n-1) for derivative formula
        std::vector<Real> B_nm1(static_cast<std::size_t>(order_), Real(0));
        if (order_ >= 1) {
            for (int i = 0; i < order_; ++i) {
                B_nm1[static_cast<std::size_t>(i)] =
                    binomial(order_ - 1, i) *
                    pow_int(t, i) *
                    pow_int(Real(1) - t, order_ - 1 - i);
            }
        }

        for (int i = 0; i <= order_; ++i) {
            Real dBdt = Real(0);
            if (order_ >= 1) {
                Real B_left = (i > 0) ? B_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                Real B_right = (i < order_) ? B_nm1[static_cast<std::size_t>(i)] : Real(0);
                dBdt = Real(order_) * (B_left - B_right);
            }
            gradients[static_cast<std::size_t>(i)][0] = dBdt * dtdx;
            gradients[static_cast<std::size_t>(i)][1] = Real(0);
            gradients[static_cast<std::size_t>(i)][2] = Real(0);
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        const Real tx = (xi[0] + Real(1)) * Real(0.5);
        const Real ty = (xi[1] + Real(1)) * Real(0.5);
        const Real dtdx = Real(0.5);
        const Real dtdy = Real(0.5);
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);

        // Compute basis values at order n
        std::vector<Real> bx(n1d), by(n1d);
        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bx[idx] = binomial(order_, i) * pow_int(tx, i) * pow_int(Real(1) - tx, order_ - i);
            by[idx] = binomial(order_, i) * pow_int(ty, i) * pow_int(Real(1) - ty, order_ - i);
        }

        // Compute lower order basis (n-1) for derivatives
        std::vector<Real> bx_nm1(static_cast<std::size_t>(order_), Real(0));
        std::vector<Real> by_nm1(static_cast<std::size_t>(order_), Real(0));
        if (order_ >= 1) {
            for (int i = 0; i < order_; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                bx_nm1[idx] = binomial(order_ - 1, i) * pow_int(tx, i) * pow_int(Real(1) - tx, order_ - 1 - i);
                by_nm1[idx] = binomial(order_ - 1, i) * pow_int(ty, i) * pow_int(Real(1) - ty, order_ - 1 - i);
            }
        }

        // Compute derivatives: dB_i/dt = n * (B_{i-1,n-1} - B_{i,n-1})
        std::vector<Real> dbx(n1d, Real(0)), dby(n1d, Real(0));
        for (int i = 0; i <= order_; ++i) {
            if (order_ >= 1) {
                Real B_left = (i > 0) ? bx_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                Real B_right = (i < order_) ? bx_nm1[static_cast<std::size_t>(i)] : Real(0);
                dbx[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);

                B_left = (i > 0) ? by_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? by_nm1[static_cast<std::size_t>(i)] : Real(0);
                dby[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);
            }
        }

        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                gradients[idx][0] = dbx[static_cast<std::size_t>(i)] * dtdx *
                                    by[static_cast<std::size_t>(j)];
                gradients[idx][1] = bx[static_cast<std::size_t>(i)] *
                                    dby[static_cast<std::size_t>(j)] * dtdy;
                gradients[idx][2] = Real(0);
                ++idx;
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        const Real tx = (xi[0] + Real(1)) * Real(0.5);
        const Real ty = (xi[1] + Real(1)) * Real(0.5);
        const Real tz = (xi[2] + Real(1)) * Real(0.5);
        const Real dtdx = Real(0.5);
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);

        // Compute basis values
        std::vector<Real> bx(n1d), by(n1d), bz(n1d);
        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bx[idx] = binomial(order_, i) * pow_int(tx, i) * pow_int(Real(1) - tx, order_ - i);
            by[idx] = binomial(order_, i) * pow_int(ty, i) * pow_int(Real(1) - ty, order_ - i);
            bz[idx] = binomial(order_, i) * pow_int(tz, i) * pow_int(Real(1) - tz, order_ - i);
        }

        // Compute lower order basis for derivatives
        std::vector<Real> bx_nm1(static_cast<std::size_t>(order_), Real(0));
        std::vector<Real> by_nm1(static_cast<std::size_t>(order_), Real(0));
        std::vector<Real> bz_nm1(static_cast<std::size_t>(order_), Real(0));
        if (order_ >= 1) {
            for (int i = 0; i < order_; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                bx_nm1[idx] = binomial(order_ - 1, i) * pow_int(tx, i) * pow_int(Real(1) - tx, order_ - 1 - i);
                by_nm1[idx] = binomial(order_ - 1, i) * pow_int(ty, i) * pow_int(Real(1) - ty, order_ - 1 - i);
                bz_nm1[idx] = binomial(order_ - 1, i) * pow_int(tz, i) * pow_int(Real(1) - tz, order_ - 1 - i);
            }
        }

        // Compute derivatives
        std::vector<Real> dbx(n1d, Real(0)), dby(n1d, Real(0)), dbz(n1d, Real(0));
        for (int i = 0; i <= order_; ++i) {
            if (order_ >= 1) {
                Real B_left, B_right;
                B_left = (i > 0) ? bx_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? bx_nm1[static_cast<std::size_t>(i)] : Real(0);
                dbx[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);

                B_left = (i > 0) ? by_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? by_nm1[static_cast<std::size_t>(i)] : Real(0);
                dby[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);

                B_left = (i > 0) ? bz_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? bz_nm1[static_cast<std::size_t>(i)] : Real(0);
                dbz[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);
            }
        }

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    gradients[idx][0] = dbx[static_cast<std::size_t>(i)] * dtdx *
                                        by[static_cast<std::size_t>(j)] *
                                        bz[static_cast<std::size_t>(k)];
                    gradients[idx][1] = bx[static_cast<std::size_t>(i)] *
                                        dby[static_cast<std::size_t>(j)] * dtdx *
                                        bz[static_cast<std::size_t>(k)];
                    gradients[idx][2] = bx[static_cast<std::size_t>(i)] *
                                        by[static_cast<std::size_t>(j)] *
                                        dbz[static_cast<std::size_t>(k)] * dtdx;
                    ++idx;
                }
            }
        }
        return;
    }

    if (is_triangle(element_type_)) {
        // Triangle Bernstein: B_{ijk} = coeff * l0^i * l1^j * l2^k
        // where l0 = 1 - l1 - l2, l1 = xi, l2 = eta
        // dB/dxi = coeff * (-i*l0^(i-1)) * l1^j * l2^k + coeff * l0^i * j*l1^(j-1) * l2^k
        //        = coeff * l0^(i-1) * l1^(j-1) * l2^k * [-i*l1 + j*l0]   (simplified)
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        for (std::size_t idx = 0; idx < simplex_indices_.size(); ++idx) {
            const auto& e = simplex_indices_[idx];
            int ei = e[0];  // exponent for l0
            int ej = e[1];  // exponent for l1
            int ek = e[2];  // exponent for l2
            Real coeff = coefficients_[idx];

            // Base value components
            Real l0_pow = pow_int(l0, ei);
            Real l1_pow = pow_int(l1, ej);
            Real l2_pow = pow_int(l2, ek);

            // Derivatives: dl0/dl1 = -1, dl0/dl2 = -1
            // dB/dl1 = coeff * [d(l0^i)/dl0 * dl0/dl1 * l1^j * l2^k + l0^i * d(l1^j)/dl1 * l2^k]
            //        = coeff * [-i*l0^(i-1) * l1^j * l2^k + l0^i * j*l1^(j-1) * l2^k]
            Real dB_dl1 = Real(0);
            Real dB_dl2 = Real(0);

            if (ei > 0) {
                dB_dl1 += coeff * Real(-ei) * pow_int(l0, ei - 1) * l1_pow * l2_pow;
                dB_dl2 += coeff * Real(-ei) * pow_int(l0, ei - 1) * l1_pow * l2_pow;
            }
            if (ej > 0) {
                dB_dl1 += coeff * l0_pow * Real(ej) * pow_int(l1, ej - 1) * l2_pow;
            }
            if (ek > 0) {
                dB_dl2 += coeff * l0_pow * l1_pow * Real(ek) * pow_int(l2, ek - 1);
            }

            gradients[idx][0] = dB_dl1;
            gradients[idx][1] = dB_dl2;
            gradients[idx][2] = Real(0);
        }
        return;
    }

    if (is_wedge(element_type_)) {
        // Wedge = triangle (barycentric) x 1D Bernstein in z
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;
        const Real tz = (xi[2] + Real(1)) * Real(0.5);
        const Real dtdz = Real(0.5);

        const std::size_t tri_count = simplex_indices_.size();
        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);

        // Compute triangle basis values and derivatives
        std::vector<Real> tri_vals(tri_count);
        std::vector<Real> tri_dl1(tri_count), tri_dl2(tri_count);

        for (std::size_t idx = 0; idx < tri_count; ++idx) {
            const auto& e = simplex_indices_[idx];
            int ei = e[0], ej = e[1], ek = e[2];
            Real coeff = coefficients_[idx];

            Real l0_pow = pow_int(l0, ei);
            Real l1_pow = pow_int(l1, ej);
            Real l2_pow = pow_int(l2, ek);

            tri_vals[idx] = coeff * l0_pow * l1_pow * l2_pow;

            Real dB_dl1 = Real(0), dB_dl2 = Real(0);
            if (ei > 0) {
                dB_dl1 += coeff * Real(-ei) * pow_int(l0, ei - 1) * l1_pow * l2_pow;
                dB_dl2 += coeff * Real(-ei) * pow_int(l0, ei - 1) * l1_pow * l2_pow;
            }
            if (ej > 0) {
                dB_dl1 += coeff * l0_pow * Real(ej) * pow_int(l1, ej - 1) * l2_pow;
            }
            if (ek > 0) {
                dB_dl2 += coeff * l0_pow * l1_pow * Real(ek) * pow_int(l2, ek - 1);
            }
            tri_dl1[idx] = dB_dl1;
            tri_dl2[idx] = dB_dl2;
        }

        // Compute 1D Bernstein in z
        std::vector<Real> bz(n1d), dbz(n1d, Real(0));
        std::vector<Real> bz_nm1(static_cast<std::size_t>(order_), Real(0));

        for (int k = 0; k <= order_; ++k) {
            bz[static_cast<std::size_t>(k)] = binomial(order_, k) *
                pow_int(tz, k) * pow_int(Real(1) - tz, order_ - k);
        }
        if (order_ >= 1) {
            for (int k = 0; k < order_; ++k) {
                bz_nm1[static_cast<std::size_t>(k)] = binomial(order_ - 1, k) *
                    pow_int(tz, k) * pow_int(Real(1) - tz, order_ - 1 - k);
            }
            for (int k = 0; k <= order_; ++k) {
                Real B_left = (k > 0) ? bz_nm1[static_cast<std::size_t>(k - 1)] : Real(0);
                Real B_right = (k < order_) ? bz_nm1[static_cast<std::size_t>(k)] : Real(0);
                dbz[static_cast<std::size_t>(k)] = Real(order_) * (B_left - B_right);
            }
        }

        std::size_t out_idx = 0;
        for (std::size_t k = 0; k < n1d; ++k) {
            for (std::size_t n = 0; n < tri_count; ++n) {
                gradients[out_idx][0] = tri_dl1[n] * bz[k];
                gradients[out_idx][1] = tri_dl2[n] * bz[k];
                gradients[out_idx][2] = tri_vals[n] * dbz[k] * dtdz;
                ++out_idx;
            }
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        // Pyramid: collapsed coordinates u = x/(1-z), v = y/(1-z), w = z
        // Then tensor-product Bernstein in (tu, tv, tw) where t = (coord+1)/2
        const Real z = xi[2];
        const Real eps = Real(1e-12);
        const Real one_minus_z = Real(1) - z;

        Real u, v;
        Real du_dx, du_dz, dv_dy, dv_dz;
        if (std::abs(one_minus_z) > eps) {
            const Real scale = Real(1) / one_minus_z;
            u = xi[0] * scale;
            v = xi[1] * scale;
            du_dx = scale;
            du_dz = xi[0] / (one_minus_z * one_minus_z);
            dv_dy = scale;
            dv_dz = xi[1] / (one_minus_z * one_minus_z);
        } else {
            u = Real(0);
            v = Real(0);
            du_dx = Real(0);
            du_dz = Real(0);
            dv_dy = Real(0);
            dv_dz = Real(0);
        }

        const Real tu = (u + Real(1)) * Real(0.5);
        const Real tv = (v + Real(1)) * Real(0.5);
        const Real tw = (z + Real(1)) * Real(0.5);  // Note: tw uses z directly, not (1-z)
        const Real dtu_du = Real(0.5);
        const Real dtv_dv = Real(0.5);
        const Real dtw_dz = Real(0.5);

        const std::size_t n1d = static_cast<std::size_t>(order_ + 1);

        // Compute basis values
        std::vector<Real> bu(n1d), bv(n1d), bw(n1d);
        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            bu[idx] = binomial(order_, i) * pow_int(tu, i) * pow_int(Real(1) - tu, order_ - i);
            bv[idx] = binomial(order_, i) * pow_int(tv, i) * pow_int(Real(1) - tv, order_ - i);
            bw[idx] = binomial(order_, i) * pow_int(tw, i) * pow_int(Real(1) - tw, order_ - i);
        }

        // Compute lower order basis for derivatives
        std::vector<Real> bu_nm1(static_cast<std::size_t>(order_), Real(0));
        std::vector<Real> bv_nm1(static_cast<std::size_t>(order_), Real(0));
        std::vector<Real> bw_nm1(static_cast<std::size_t>(order_), Real(0));
        if (order_ >= 1) {
            for (int i = 0; i < order_; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                bu_nm1[idx] = binomial(order_ - 1, i) * pow_int(tu, i) * pow_int(Real(1) - tu, order_ - 1 - i);
                bv_nm1[idx] = binomial(order_ - 1, i) * pow_int(tv, i) * pow_int(Real(1) - tv, order_ - 1 - i);
                bw_nm1[idx] = binomial(order_ - 1, i) * pow_int(tw, i) * pow_int(Real(1) - tw, order_ - 1 - i);
            }
        }

        // Compute derivatives wrt parameter
        std::vector<Real> dbu(n1d, Real(0)), dbv(n1d, Real(0)), dbw(n1d, Real(0));
        for (int i = 0; i <= order_; ++i) {
            if (order_ >= 1) {
                Real B_left, B_right;
                B_left = (i > 0) ? bu_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? bu_nm1[static_cast<std::size_t>(i)] : Real(0);
                dbu[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);

                B_left = (i > 0) ? bv_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? bv_nm1[static_cast<std::size_t>(i)] : Real(0);
                dbv[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);

                B_left = (i > 0) ? bw_nm1[static_cast<std::size_t>(i - 1)] : Real(0);
                B_right = (i < order_) ? bw_nm1[static_cast<std::size_t>(i)] : Real(0);
                dbw[static_cast<std::size_t>(i)] = Real(order_) * (B_left - B_right);
            }
        }

        // N = Bu(tu(u(x,z))) * Bv(tv(v(y,z))) * Bw(tw(z))
        // dN/dx = dBu/dtu * dtu/du * du/dx * Bv * Bw
        // dN/dy = Bu * dBv/dtv * dtv/dv * dv/dy * Bw
        // dN/dz = dBu/dtu * dtu/du * du/dz * Bv * Bw
        //       + Bu * dBv/dtv * dtv/dv * dv/dz * Bw
        //       + Bu * Bv * dBw/dtw * dtw/dz

        std::size_t out_idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    Real Bu = bu[static_cast<std::size_t>(i)];
                    Real Bv = bv[static_cast<std::size_t>(j)];
                    Real Bw = bw[static_cast<std::size_t>(k)];
                    Real dBu = dbu[static_cast<std::size_t>(i)];
                    Real dBv = dbv[static_cast<std::size_t>(j)];
                    Real dBw = dbw[static_cast<std::size_t>(k)];

                    gradients[out_idx][0] = dBu * dtu_du * du_dx * Bv * Bw;
                    gradients[out_idx][1] = Bu * dBv * dtv_dv * dv_dy * Bw;
                    gradients[out_idx][2] = dBu * dtu_du * du_dz * Bv * Bw
                                          + Bu * dBv * dtv_dv * dv_dz * Bw
                                          + Bu * Bv * dBw * dtw_dz;
                    ++out_idx;
                }
            }
        }
        return;
    }

    throw FEException("Unsupported element in BernsteinBasis::evaluate_gradients",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

} // namespace basis
} // namespace FE
} // namespace svmp
