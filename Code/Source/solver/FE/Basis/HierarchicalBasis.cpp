/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "HierarchicalBasis.h"
#include "BasisTraits.h"
#include "BasisTolerance.h"
#include <cmath>
#include <span>

namespace svmp {
namespace FE {
namespace basis {

namespace {

struct LegendreFirstSequence {
    std::span<Real> values;
    std::span<Real> derivatives;
};

struct LegendreSecondSequence {
    std::span<Real> values;
    std::span<Real> derivatives;
    std::span<Real> second_derivatives;
};

struct LegendreScratch {
    std::array<std::vector<Real>, 3> values;
    std::array<std::vector<Real>, 3> derivatives;
    std::array<std::vector<Real>, 3> second_derivatives;
};

LegendreScratch& legendre_scratch() {
    thread_local LegendreScratch scratch;
    return scratch;
}

std::span<Real> resize_sequence(std::vector<Real>& storage, int order) {
    const auto count = static_cast<std::size_t>(order + 1);
    storage.resize(count);
    return std::span<Real>(storage.data(), count);
}

std::span<Real> fill_legendre_values(LegendreScratch& scratch,
                                     std::size_t slot,
                                     int order,
                                     Real x) {
    auto values = resize_sequence(scratch.values[slot], order);
    orthopoly::legendre_sequence_to(order, x, values);
    return values;
}

LegendreFirstSequence fill_legendre_first(LegendreScratch& scratch,
                                          std::size_t slot,
                                          int order,
                                          Real x) {
    auto values = resize_sequence(scratch.values[slot], order);
    auto derivatives = resize_sequence(scratch.derivatives[slot], order);
    orthopoly::legendre_sequence_with_derivatives_to(order, x, values, derivatives);
    return {values, derivatives};
}

LegendreSecondSequence fill_legendre_second(LegendreScratch& scratch,
                                            std::size_t slot,
                                            int order,
                                            Real x) {
    auto values = resize_sequence(scratch.values[slot], order);
    auto derivatives = resize_sequence(scratch.derivatives[slot], order);
    auto second_derivatives = resize_sequence(scratch.second_derivatives[slot], order);
    orthopoly::legendre_sequence_with_second_derivatives_to(
        order, x, values, derivatives, second_derivatives);
    return {values, derivatives, second_derivatives};
}

} // namespace

HierarchicalBasis::HierarchicalBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 0) {
        throw BasisConfigurationException("HierarchicalBasis requires non-negative order",
                                          __FILE__, __LINE__, __func__);
    }

    if (is_line(element_type_)) {
        dimension_ = 1;
        size_ = static_cast<std::size_t>(order_ + 1);
    } else if (is_quadrilateral(element_type_)) {
        dimension_ = 2;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1));
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                tensor_indices_.push_back({i, j, 0});
            }
        }
    } else if (is_hexahedron(element_type_)) {
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1) * (order_ + 1));
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    tensor_indices_.push_back({i, j, k});
                }
            }
        }
    } else if (is_triangle(element_type_) || is_wedge(element_type_)) {
        // Triangle modal basis in (x,y); wedge is triangle x 1D Legendre in z
        const bool wedge = is_wedge(element_type_);
        dimension_ = wedge ? 3 : 2;
        const std::size_t tri_count =
            static_cast<std::size_t>((order_ + 1) * (order_ + 2) / 2);
        size_ = wedge
                  ? tri_count * static_cast<std::size_t>(order_ + 1)
                  : tri_count;
        for (int p = 0; p <= order_; ++p) {
            for (int q = 0; q <= order_ - p; ++q) {
                simplex_indices_.push_back({p, q, 0, 0});
            }
        }
    } else if (is_tetrahedron(element_type_)) {
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 2) * (order_ + 3) / 6);
        for (int p = 0; p <= order_; ++p) {
            for (int q = 0; q <= order_ - p; ++q) {
                for (int r = 0; r <= order_ - p - q; ++r) {
                    simplex_indices_.push_back({p, q, r, 0});
                }
            }
        }
    } else if (is_pyramid(element_type_)) {
        // Pyramid treated via tensor-product Legendre in transformed coordinates
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1) * (order_ + 1));
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    tensor_indices_.push_back({i, j, k});
                }
            }
        }
    } else if (element_type_ == ElementType::Point1) {
        dimension_ = 0;
        size_ = 1;
    } else {
        throw BasisElementCompatibilityException("Unsupported element type for HierarchicalBasis",
                                                 __FILE__, __LINE__, __func__);
    }
}

void HierarchicalBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                        std::vector<Real>& values) const {
    values.resize(size_);

    if (element_type_ == ElementType::Point1) {
        values[0] = Real(1);
        return;
    }

    auto& scratch = legendre_scratch();

    if (is_line(element_type_)) {
        const auto seq = fill_legendre_values(scratch, 0, order_, xi[0]);
        for (std::size_t i = 0; i < seq.size(); ++i) {
            values[i] = seq[i];
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        const auto seq_x = fill_legendre_values(scratch, 0, order_, xi[0]);
        const auto seq_y = fill_legendre_values(scratch, 1, order_, xi[1]);
        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                values[idx++] = seq_x[static_cast<std::size_t>(i)] *
                                seq_y[static_cast<std::size_t>(j)];
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        const auto seq_x = fill_legendre_values(scratch, 0, order_, xi[0]);
        const auto seq_y = fill_legendre_values(scratch, 1, order_, xi[1]);
        const auto seq_z = fill_legendre_values(scratch, 2, order_, xi[2]);
        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    values[idx++] = seq_x[static_cast<std::size_t>(i)] *
                                    seq_y[static_cast<std::size_t>(j)] *
                                    seq_z[static_cast<std::size_t>(k)];
                }
            }
        }
        return;
    }

    if (is_wedge(element_type_)) {
        // Tensor-product modal basis on wedge: Dubiner (triangle) x Legendre (line)
        const auto seq_z = fill_legendre_values(scratch, 0, order_, xi[2]);
        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            int p = pq[0];
            int q = pq[1];
            const Real tri_val = orthopoly::dubiner(p, q, xi[0], xi[1]);
            for (int k = 0; k <= order_; ++k) {
                values[idx++] = tri_val * seq_z[static_cast<std::size_t>(k)];
            }
        }
        return;
    }

    if (is_triangle(element_type_)) {
        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            int p = pq[0];
            int q = pq[1];
            values[idx++] = orthopoly::dubiner(p, q, xi[0], xi[1]);
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        // Modal basis on pyramid via tensor Legendre in cube coordinates
        const Real z = xi[2];
        Real u = Real(0);
        Real v = Real(0);
        if (!detail::basis_near_zero(Real(1) - z)) {
            const Real scale = Real(1) / (Real(1) - z);
            u = xi[0] * scale;
            v = xi[1] * scale;
        }
        const auto seq_u = fill_legendre_values(scratch, 0, order_, u);
        const auto seq_v = fill_legendre_values(scratch, 1, order_, v);
        const auto seq_w = fill_legendre_values(scratch, 2, order_, z);
        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    values[idx++] = seq_u[static_cast<std::size_t>(i)] *
                                    seq_v[static_cast<std::size_t>(j)] *
                                    seq_w[static_cast<std::size_t>(k)];
                }
            }
        }
        return;
    }

    if (is_tetrahedron(element_type_)) {
        std::size_t idx = 0;
        for (const auto& pqr : simplex_indices_) {
            int p = pqr[0];
            int q = pqr[1];
            int r = pqr[2];
            values[idx++] = orthopoly::proriol(p, q, r, xi[0], xi[1], xi[2]);
        }
        return;
    }

    throw BasisEvaluationException("Unsupported element in HierarchicalBasis::evaluate_values",
                                   __FILE__, __LINE__, __func__);
}

void HierarchicalBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                           std::vector<Gradient>& gradients) const {
    gradients.resize(size_);

    if (element_type_ == ElementType::Point1) {
        // Point has no spatial gradient
        gradients[0] = Gradient{};
        return;
    }

    auto& scratch = legendre_scratch();

    if (is_line(element_type_)) {
        // 1D Legendre polynomials: gradient is just the derivative in x
        const auto seq = fill_legendre_first(scratch, 0, order_, xi[0]);
        for (std::size_t i = 0; i < size_; ++i) {
            gradients[i][0] = seq.derivatives[i];
            gradients[i][1] = Real(0);
            gradients[i][2] = Real(0);
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        // Tensor product: P_i(x) * P_j(y)
        // d/dx = P'_i(x) * P_j(y)
        // d/dy = P_i(x) * P'_j(y)
        const auto seq_x = fill_legendre_first(scratch, 0, order_, xi[0]);
        const auto seq_y = fill_legendre_first(scratch, 1, order_, xi[1]);

        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                gradients[idx][0] = seq_x.derivatives[static_cast<std::size_t>(i)] *
                                    seq_y.values[static_cast<std::size_t>(j)];
                gradients[idx][1] = seq_x.values[static_cast<std::size_t>(i)] *
                                    seq_y.derivatives[static_cast<std::size_t>(j)];
                gradients[idx][2] = Real(0);
                ++idx;
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        // Tensor product: P_i(x) * P_j(y) * P_k(z)
        const auto seq_x = fill_legendre_first(scratch, 0, order_, xi[0]);
        const auto seq_y = fill_legendre_first(scratch, 1, order_, xi[1]);
        const auto seq_z = fill_legendre_first(scratch, 2, order_, xi[2]);

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    Real Px = seq_x.values[static_cast<std::size_t>(i)];
                    Real Py = seq_y.values[static_cast<std::size_t>(j)];
                    Real Pz = seq_z.values[static_cast<std::size_t>(k)];
                    Real dPx = seq_x.derivatives[static_cast<std::size_t>(i)];
                    Real dPy = seq_y.derivatives[static_cast<std::size_t>(j)];
                    Real dPz = seq_z.derivatives[static_cast<std::size_t>(k)];

                    gradients[idx][0] = dPx * Py * Pz;
                    gradients[idx][1] = Px * dPy * Pz;
                    gradients[idx][2] = Px * Py * dPz;
                    ++idx;
                }
            }
        }
        return;
    }

    if (is_triangle(element_type_)) {
        // Dubiner basis with derivatives
        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            int p = pq[0];
            int q = pq[1];
            const auto tri = orthopoly::dubiner_derivatives(p, q, xi[0], xi[1]);
            gradients[idx][0] = tri.dxi;
            gradients[idx][1] = tri.deta;
            gradients[idx][2] = Real(0);
            ++idx;
        }
        return;
    }

    if (is_wedge(element_type_)) {
        // Wedge = Dubiner (triangle) x Legendre (line in z)
        const auto seq_z = fill_legendre_first(scratch, 0, order_, xi[2]);

        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            int p = pq[0];
            int q = pq[1];
            const auto tri = orthopoly::dubiner_derivatives(p, q, xi[0], xi[1]);

            for (int k = 0; k <= order_; ++k) {
                Real Pz = seq_z.values[static_cast<std::size_t>(k)];
                Real dPz = seq_z.derivatives[static_cast<std::size_t>(k)];

                gradients[idx][0] = tri.dxi * Pz;
                gradients[idx][1] = tri.deta * Pz;
                gradients[idx][2] = tri.value * dPz;
                ++idx;
            }
        }
        return;
    }

    if (is_tetrahedron(element_type_)) {
        // Proriol basis with derivatives
        std::size_t idx = 0;
        for (const auto& pqr : simplex_indices_) {
            int p = pqr[0];
            int q = pqr[1];
            int r = pqr[2];
            const auto tet = orthopoly::proriol_derivatives(p, q, r, xi[0], xi[1], xi[2]);
            gradients[idx] = tet.gradient;
            ++idx;
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        // Pyramid via collapsed coordinates: u = x/(1-z), v = y/(1-z), w = z
        // Need chain rule for the coordinate transformation
        const Real z = xi[2];
        const Real one_minus_z = Real(1) - z;

        Real u, v;
        Real du_dx, du_dz, dv_dy, dv_dz;

        if (!detail::basis_near_zero(one_minus_z)) {
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

        const auto seq_u = fill_legendre_first(scratch, 0, order_, u);
        const auto seq_v = fill_legendre_first(scratch, 1, order_, v);
        const auto seq_w = fill_legendre_first(scratch, 2, order_, z);

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    Real Pu = seq_u.values[static_cast<std::size_t>(i)];
                    Real Pv = seq_v.values[static_cast<std::size_t>(j)];
                    Real Pw = seq_w.values[static_cast<std::size_t>(k)];
                    Real dPu = seq_u.derivatives[static_cast<std::size_t>(i)];
                    Real dPv = seq_v.derivatives[static_cast<std::size_t>(j)];
                    Real dPw = seq_w.derivatives[static_cast<std::size_t>(k)];

                    // N = Pu(u(x,z)) * Pv(v(y,z)) * Pw(z)
                    // dN/dx = dPu/du * du/dx * Pv * Pw
                    // dN/dy = Pu * dPv/dv * dv/dy * Pw
                    // dN/dz = dPu/du * du/dz * Pv * Pw + Pu * dPv/dv * dv/dz * Pw + Pu * Pv * dPw/dz

                    gradients[idx][0] = dPu * du_dx * Pv * Pw;
                    gradients[idx][1] = Pu * dPv * dv_dy * Pw;
                    gradients[idx][2] = dPu * du_dz * Pv * Pw
                                      + Pu * dPv * dv_dz * Pw
                                      + Pu * Pv * dPw;
                    ++idx;
                }
            }
        }
        return;
    }

    throw BasisEvaluationException("Unsupported element in HierarchicalBasis::evaluate_gradients",
                                   __FILE__, __LINE__, __func__);
}

void HierarchicalBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                          std::vector<Hessian>& hessians) const {
    hessians.resize(size_);

    if (element_type_ == ElementType::Point1) {
        hessians[0] = Hessian{};
        return;
    }

    auto& scratch = legendre_scratch();

    if (is_line(element_type_)) {
        const auto seq = fill_legendre_second(scratch, 0, order_, xi[0]);
        for (std::size_t i = 0; i < size_; ++i) {
            hessians[i] = Hessian{};
            hessians[i](0, 0) = seq.second_derivatives[i];
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        const auto seq_x = fill_legendre_second(scratch, 0, order_, xi[0]);
        const auto seq_y = fill_legendre_second(scratch, 1, order_, xi[1]);
        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                hessians[idx++] = make_symmetric_hessian(
                    seq_x.second_derivatives[static_cast<std::size_t>(i)] *
                        seq_y.values[static_cast<std::size_t>(j)],
                    seq_x.values[static_cast<std::size_t>(i)] *
                        seq_y.second_derivatives[static_cast<std::size_t>(j)],
                    Real(0),
                    seq_x.derivatives[static_cast<std::size_t>(i)] *
                        seq_y.derivatives[static_cast<std::size_t>(j)],
                    Real(0),
                    Real(0));
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        const auto seq_x = fill_legendre_second(scratch, 0, order_, xi[0]);
        const auto seq_y = fill_legendre_second(scratch, 1, order_, xi[1]);
        const auto seq_z = fill_legendre_second(scratch, 2, order_, xi[2]);
        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    hessians[idx++] = make_symmetric_hessian(
                        seq_x.second_derivatives[static_cast<std::size_t>(i)] *
                            seq_y.values[static_cast<std::size_t>(j)] *
                            seq_z.values[static_cast<std::size_t>(k)],
                        seq_x.values[static_cast<std::size_t>(i)] *
                            seq_y.second_derivatives[static_cast<std::size_t>(j)] *
                            seq_z.values[static_cast<std::size_t>(k)],
                        seq_x.values[static_cast<std::size_t>(i)] *
                            seq_y.values[static_cast<std::size_t>(j)] *
                            seq_z.second_derivatives[static_cast<std::size_t>(k)],
                        seq_x.derivatives[static_cast<std::size_t>(i)] *
                            seq_y.derivatives[static_cast<std::size_t>(j)] *
                            seq_z.values[static_cast<std::size_t>(k)],
                        seq_x.derivatives[static_cast<std::size_t>(i)] *
                            seq_y.values[static_cast<std::size_t>(j)] *
                            seq_z.derivatives[static_cast<std::size_t>(k)],
                        seq_x.values[static_cast<std::size_t>(i)] *
                            seq_y.derivatives[static_cast<std::size_t>(j)] *
                            seq_z.derivatives[static_cast<std::size_t>(k)]);
                }
            }
        }
        return;
    }

    if (is_triangle(element_type_)) {
        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            const auto jet = orthopoly::dubiner_with_second_derivatives(pq[0], pq[1], xi[0], xi[1]);
            hessians[idx++] = make_symmetric_hessian(
                jet.dxx, jet.dyy, Real(0), jet.dxy, Real(0), Real(0));
        }
        return;
    }

    if (is_wedge(element_type_)) {
        const auto seq_z = fill_legendre_second(scratch, 0, order_, xi[2]);
        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            const auto tri = orthopoly::dubiner_with_second_derivatives(pq[0], pq[1], xi[0], xi[1]);
            for (int k = 0; k <= order_; ++k) {
                const Real vz = seq_z.values[static_cast<std::size_t>(k)];
                const Real dz = seq_z.derivatives[static_cast<std::size_t>(k)];
                const Real d2z = seq_z.second_derivatives[static_cast<std::size_t>(k)];
                hessians[idx++] = make_symmetric_hessian(tri.dxx * vz,
                                                         tri.dyy * vz,
                                                         tri.value * d2z,
                                                         tri.dxy * vz,
                                                         tri.dxi * dz,
                                                         tri.deta * dz);
            }
        }
        return;
    }

    if (is_tetrahedron(element_type_)) {
        std::size_t idx = 0;
        for (const auto& pqr : simplex_indices_) {
            const auto jet = orthopoly::proriol_with_second_derivatives(
                pqr[0], pqr[1], pqr[2], xi[0], xi[1], xi[2]);
            hessians[idx++] = jet.hessian;
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        const Real z = xi[2];
        const Real one_minus_z = Real(1) - z;

        Real u = Real(0);
        Real v = Real(0);
        Real u_x = Real(0);
        Real u_z = Real(0);
        Real u_xz = Real(0);
        Real u_zz = Real(0);
        Real v_y = Real(0);
        Real v_z = Real(0);
        Real v_yz = Real(0);
        Real v_zz = Real(0);

        if (!detail::basis_near_zero(one_minus_z)) {
            const Real inv = Real(1) / one_minus_z;
            const Real inv2 = inv * inv;
            const Real inv3 = inv2 * inv;
            u = xi[0] * inv;
            v = xi[1] * inv;
            u_x = inv;
            u_z = xi[0] * inv2;
            u_xz = inv2;
            u_zz = Real(2) * xi[0] * inv3;
            v_y = inv;
            v_z = xi[1] * inv2;
            v_yz = inv2;
            v_zz = Real(2) * xi[1] * inv3;
        }

        const auto seq_u = fill_legendre_second(scratch, 0, order_, u);
        const auto seq_v = fill_legendre_second(scratch, 1, order_, v);
        const auto seq_w = fill_legendre_second(scratch, 2, order_, z);

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            const Real W = seq_w.values[kk];
            const Real dW = seq_w.derivatives[kk];
            const Real d2W = seq_w.second_derivatives[kk];
            for (int j = 0; j <= order_; ++j) {
                const std::size_t jj = static_cast<std::size_t>(j);
                const Real V = seq_v.values[jj];
                const Real dV = seq_v.derivatives[jj];
                const Real d2V = seq_v.second_derivatives[jj];
                for (int i = 0; i <= order_; ++i) {
                    const std::size_t ii = static_cast<std::size_t>(i);
                    const Real U = seq_u.values[ii];
                    const Real dU = seq_u.derivatives[ii];
                    const Real d2U = seq_u.second_derivatives[ii];

                    Hessian H{};
                    H(0, 0) = d2U * u_x * u_x * V * W;
                    H(1, 1) = U * d2V * v_y * v_y * W;
                    H(0, 1) = dU * u_x * dV * v_y * W;
                    H(1, 0) = H(0, 1);

                    H(0, 2) =
                        (d2U * u_x * u_z + dU * u_xz) * V * W +
                        dU * u_x * dV * v_z * W +
                        dU * u_x * V * dW;
                    H(2, 0) = H(0, 2);

                    H(1, 2) =
                        dU * u_z * dV * v_y * W +
                        U * (d2V * v_y * v_z + dV * v_yz) * W +
                        U * dV * v_y * dW;
                    H(2, 1) = H(1, 2);

                    H(2, 2) =
                        (d2U * u_z * u_z + dU * u_zz) * V * W +
                        U * (d2V * v_z * v_z + dV * v_zz) * W +
                        U * V * d2W +
                        Real(2) * dU * u_z * dV * v_z * W +
                        Real(2) * dU * u_z * V * dW +
                        Real(2) * U * dV * v_z * dW;
                    hessians[idx++] = H;
                }
            }
        }
        return;
    }

    throw BasisEvaluationException("Unsupported element in HierarchicalBasis::evaluate_hessians",
                                   __FILE__, __LINE__, __func__);
}

} // namespace basis
} // namespace FE
} // namespace svmp
