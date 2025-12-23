/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "HierarchicalBasis.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace basis {

namespace {

bool is_line(ElementType type) {
    return type == ElementType::Line2 || type == ElementType::Line3;
}

bool is_triangle(ElementType type) {
    return type == ElementType::Triangle3 || type == ElementType::Triangle6;
}

bool is_quadrilateral(ElementType type) {
    return type == ElementType::Quad4 || type == ElementType::Quad8 || type == ElementType::Quad9;
}

bool is_tetrahedron(ElementType type) {
    return type == ElementType::Tetra4 || type == ElementType::Tetra10;
}

bool is_hexahedron(ElementType type) {
    return type == ElementType::Hex8 || type == ElementType::Hex20 || type == ElementType::Hex27;
}

bool is_wedge(ElementType type) {
    return type == ElementType::Wedge6 || type == ElementType::Wedge15 || type == ElementType::Wedge18;
}

bool is_pyramid(ElementType type) {
    return type == ElementType::Pyramid5 || type == ElementType::Pyramid13 || type == ElementType::Pyramid14;
}

} // namespace

HierarchicalBasis::HierarchicalBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 0) {
        throw FEException("HierarchicalBasis requires non-negative order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
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
        throw FEException("Unsupported element type for HierarchicalBasis",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

void HierarchicalBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                        std::vector<Real>& values) const {
    values.assign(size_, Real(0));

    if (element_type_ == ElementType::Point1) {
        values[0] = Real(1);
        return;
    }

    if (is_line(element_type_)) {
        auto seq = orthopoly::legendre_sequence(order_, xi[0]);
        for (std::size_t i = 0; i < seq.size(); ++i) {
            values[i] = seq[i];
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        auto seq_x = orthopoly::legendre_sequence(order_, xi[0]);
        auto seq_y = orthopoly::legendre_sequence(order_, xi[1]);
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
        auto seq_x = orthopoly::legendre_sequence(order_, xi[0]);
        auto seq_y = orthopoly::legendre_sequence(order_, xi[1]);
        auto seq_z = orthopoly::legendre_sequence(order_, xi[2]);
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
        auto seq_z = orthopoly::legendre_sequence(order_, xi[2]);
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
        if (std::abs(Real(1) - z) > Real(1e-12)) {
            const Real scale = Real(1) / (Real(1) - z);
            u = xi[0] * scale;
            v = xi[1] * scale;
        }
        auto seq_u = orthopoly::legendre_sequence(order_, u);
        auto seq_v = orthopoly::legendre_sequence(order_, v);
        auto seq_w = orthopoly::legendre_sequence(order_, z);
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

    throw FEException("Unsupported element in HierarchicalBasis::evaluate_values",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void HierarchicalBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                           std::vector<Gradient>& gradients) const {
    gradients.assign(size_, Gradient{});

    if (element_type_ == ElementType::Point1) {
        // Point has no spatial gradient
        return;
    }

    if (is_line(element_type_)) {
        // 1D Legendre polynomials: gradient is just the derivative in x
        auto [vals, derivs] = orthopoly::legendre_sequence_with_derivatives(order_, xi[0]);
        for (std::size_t i = 0; i < size_; ++i) {
            gradients[i][0] = derivs[i];
            gradients[i][1] = Real(0);
            gradients[i][2] = Real(0);
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        // Tensor product: P_i(x) * P_j(y)
        // d/dx = P'_i(x) * P_j(y)
        // d/dy = P_i(x) * P'_j(y)
        auto [vals_x, derivs_x] = orthopoly::legendre_sequence_with_derivatives(order_, xi[0]);
        auto [vals_y, derivs_y] = orthopoly::legendre_sequence_with_derivatives(order_, xi[1]);

        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_; ++i) {
                gradients[idx][0] = derivs_x[static_cast<std::size_t>(i)] *
                                    vals_y[static_cast<std::size_t>(j)];
                gradients[idx][1] = vals_x[static_cast<std::size_t>(i)] *
                                    derivs_y[static_cast<std::size_t>(j)];
                gradients[idx][2] = Real(0);
                ++idx;
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        // Tensor product: P_i(x) * P_j(y) * P_k(z)
        auto [vals_x, derivs_x] = orthopoly::legendre_sequence_with_derivatives(order_, xi[0]);
        auto [vals_y, derivs_y] = orthopoly::legendre_sequence_with_derivatives(order_, xi[1]);
        auto [vals_z, derivs_z] = orthopoly::legendre_sequence_with_derivatives(order_, xi[2]);

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    Real Px = vals_x[static_cast<std::size_t>(i)];
                    Real Py = vals_y[static_cast<std::size_t>(j)];
                    Real Pz = vals_z[static_cast<std::size_t>(k)];
                    Real dPx = derivs_x[static_cast<std::size_t>(i)];
                    Real dPy = derivs_y[static_cast<std::size_t>(j)];
                    Real dPz = derivs_z[static_cast<std::size_t>(k)];

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
            auto [val, dxi, deta] = orthopoly::dubiner_with_derivatives(p, q, xi[0], xi[1]);
            gradients[idx][0] = dxi;
            gradients[idx][1] = deta;
            gradients[idx][2] = Real(0);
            ++idx;
        }
        return;
    }

    if (is_wedge(element_type_)) {
        // Wedge = Dubiner (triangle) x Legendre (line in z)
        auto [vals_z, derivs_z] = orthopoly::legendre_sequence_with_derivatives(order_, xi[2]);

        std::size_t idx = 0;
        for (const auto& pq : simplex_indices_) {
            int p = pq[0];
            int q = pq[1];
            auto [tri_val, dtri_dxi, dtri_deta] =
                orthopoly::dubiner_with_derivatives(p, q, xi[0], xi[1]);

            for (int k = 0; k <= order_; ++k) {
                Real Pz = vals_z[static_cast<std::size_t>(k)];
                Real dPz = derivs_z[static_cast<std::size_t>(k)];

                gradients[idx][0] = dtri_dxi * Pz;
                gradients[idx][1] = dtri_deta * Pz;
                gradients[idx][2] = tri_val * dPz;
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
            auto [val, dxi, deta, dzeta] =
                orthopoly::proriol_with_derivatives(p, q, r, xi[0], xi[1], xi[2]);
            gradients[idx][0] = dxi;
            gradients[idx][1] = deta;
            gradients[idx][2] = dzeta;
            ++idx;
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        // Pyramid via collapsed coordinates: u = x/(1-z), v = y/(1-z), w = z
        // Need chain rule for the coordinate transformation
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

        auto [vals_u, derivs_u] = orthopoly::legendre_sequence_with_derivatives(order_, u);
        auto [vals_v, derivs_v] = orthopoly::legendre_sequence_with_derivatives(order_, v);
        auto [vals_w, derivs_w] = orthopoly::legendre_sequence_with_derivatives(order_, z);

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_; ++j) {
                for (int i = 0; i <= order_; ++i) {
                    Real Pu = vals_u[static_cast<std::size_t>(i)];
                    Real Pv = vals_v[static_cast<std::size_t>(j)];
                    Real Pw = vals_w[static_cast<std::size_t>(k)];
                    Real dPu = derivs_u[static_cast<std::size_t>(i)];
                    Real dPv = derivs_v[static_cast<std::size_t>(j)];
                    Real dPw = derivs_w[static_cast<std::size_t>(k)];

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

    throw FEException("Unsupported element in HierarchicalBasis::evaluate_gradients",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

} // namespace basis
} // namespace FE
} // namespace svmp
