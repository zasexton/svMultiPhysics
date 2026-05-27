/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "HermiteBasis.h"

#include "Basis/BasisExceptions.h"

#include <array>

namespace svmp {
namespace FE {
namespace basis {

namespace {

using CubicCoefficients = std::array<Real, 4>;

constexpr std::array<CubicCoefficients, 4> kHermiteCoefficients{{
    CubicCoefficients{Real(1), Real(0), Real(-3), Real(2)},
    CubicCoefficients{Real(0), Real(1), Real(-2), Real(1)},
    CubicCoefficients{Real(0), Real(0), Real(3), Real(-2)},
    CubicCoefficients{Real(0), Real(0), Real(-1), Real(1)},
}};

constexpr std::array<Real, 4> kHermiteModeScale{
    Real(1), Real(2), Real(1), Real(2)};

[[nodiscard]] constexpr Real evaluate_cubic(const CubicCoefficients& coeffs,
                                            Real t) noexcept {
    return ((coeffs[3] * t + coeffs[2]) * t + coeffs[1]) * t + coeffs[0];
}

[[nodiscard]] constexpr Real evaluate_cubic_derivative(
    const CubicCoefficients& coeffs,
    Real t) noexcept {
    return (Real(3) * coeffs[3] * t + Real(2) * coeffs[2]) * t + coeffs[1];
}

[[nodiscard]] constexpr Real evaluate_cubic_second_derivative(
    const CubicCoefficients& coeffs,
    Real t) noexcept {
    return Real(6) * coeffs[3] * t + Real(2) * coeffs[2];
}

constexpr void hermite_1d(const Real s,
                          Real& H1, Real& H2, Real& H3, Real& H4,
                          Real& dH1_ds, Real& dH2_ds,
                          Real& dH3_ds, Real& dH4_ds,
                          Real& ddH1_ds2, Real& ddH2_ds2,
                          Real& ddH3_ds2, Real& ddH4_ds2) {
    // Map reference coordinate s ∈ [-1,1] to t ∈ [0,1]
    const Real t  = Real(0.5) * (s + Real(1));

    const Real dt_ds = Real(0.5);

    // Scale slope modes so that derivatives are taken with respect to s,
    // i.e., dH2/ds = 1 at the left node and dH4/ds = 1 at the right node.
    H1 = kHermiteModeScale[0] * evaluate_cubic(kHermiteCoefficients[0], t);
    H2 = kHermiteModeScale[1] * evaluate_cubic(kHermiteCoefficients[1], t);
    H3 = kHermiteModeScale[2] * evaluate_cubic(kHermiteCoefficients[2], t);
    H4 = kHermiteModeScale[3] * evaluate_cubic(kHermiteCoefficients[3], t);

    dH1_ds = kHermiteModeScale[0]
             * evaluate_cubic_derivative(kHermiteCoefficients[0], t) * dt_ds;
    dH2_ds = kHermiteModeScale[1]
             * evaluate_cubic_derivative(kHermiteCoefficients[1], t) * dt_ds;
    dH3_ds = kHermiteModeScale[2]
             * evaluate_cubic_derivative(kHermiteCoefficients[2], t) * dt_ds;
    dH4_ds = kHermiteModeScale[3]
             * evaluate_cubic_derivative(kHermiteCoefficients[3], t) * dt_ds;

    const Real dt_ds_sq = dt_ds * dt_ds;
    ddH1_ds2 = kHermiteModeScale[0]
               * evaluate_cubic_second_derivative(kHermiteCoefficients[0], t)
               * dt_ds_sq;
    ddH2_ds2 = kHermiteModeScale[1]
               * evaluate_cubic_second_derivative(kHermiteCoefficients[1], t)
               * dt_ds_sq;
    ddH3_ds2 = kHermiteModeScale[2]
               * evaluate_cubic_second_derivative(kHermiteCoefficients[2], t)
               * dt_ds_sq;
    ddH4_ds2 = kHermiteModeScale[3]
               * evaluate_cubic_second_derivative(kHermiteCoefficients[3], t)
               * dt_ds_sq;
}

constexpr std::size_t kMaxHermiteDofs = 64u;

struct HermiteAxis {
    std::array<Real, 4> value{};
    std::array<Real, 4> first{};
    std::array<Real, 4> second{};
};

constexpr HermiteAxis hermite_axis(const Real s) {
    HermiteAxis axis;
    hermite_1d(s,
               axis.value[0],
               axis.value[1],
               axis.value[2],
               axis.value[3],
               axis.first[0],
               axis.first[1],
               axis.first[2],
               axis.first[3],
               axis.second[0],
               axis.second[1],
               axis.second[2],
               axis.second[3]);
    return axis;
}

void write_gradient(Real* SVMP_RESTRICT gradients_out,
                    std::size_t dof,
                    Real dx,
                    Real dy,
                    Real dz) {
    if (gradients_out == nullptr) {
        return;
    }
    Real* g = gradients_out + dof * 3u;
    g[0] = dx;
    g[1] = dy;
    g[2] = dz;
}

void write_hessian(Real* SVMP_RESTRICT hessians_out,
                   std::size_t dof,
                   Real xx,
                   Real yy,
                   Real zz,
                   Real xy,
                   Real xz,
                   Real yz) {
    if (hessians_out == nullptr) {
        return;
    }
    Real* H = hessians_out + dof * 9u;
    H[0] = xx;
    H[1] = xy;
    H[2] = xz;
    H[3] = xy;
    H[4] = yy;
    H[5] = yz;
    H[6] = xz;
    H[7] = yz;
    H[8] = zz;
}

void write_hermite_tensor_to(int dimension,
                             const math::Vector<Real, 3>& xi,
                             Real* SVMP_RESTRICT values_out,
                             Real* SVMP_RESTRICT gradients_out,
                             Real* SVMP_RESTRICT hessians_out) {
    const HermiteAxis x = hermite_axis(xi[0]);
    static constexpr std::array<int, 4> line_mode{{0, 2, 1, 3}};

    if (dimension == 1) {
        for (std::size_t dof = 0; dof < line_mode.size(); ++dof) {
            const int ix = line_mode[dof];
            if (values_out != nullptr) {
                values_out[dof] = x.value[static_cast<std::size_t>(ix)];
            }
            write_gradient(gradients_out,
                           dof,
                           x.first[static_cast<std::size_t>(ix)],
                           Real(0),
                           Real(0));
            write_hessian(hessians_out,
                          dof,
                          x.second[static_cast<std::size_t>(ix)],
                          Real(0),
                          Real(0),
                          Real(0),
                          Real(0),
                          Real(0));
        }
        return;
    }

    const HermiteAxis y = hermite_axis(xi[1]);
    static constexpr std::array<int, 4> quad_x_mode{{0, 2, 2, 0}};
    static constexpr std::array<int, 4> quad_y_mode{{0, 0, 2, 2}};

    if (dimension == 2) {
        for (std::size_t corner = 0; corner < quad_x_mode.size(); ++corner) {
            const int ix0 = quad_x_mode[corner];
            const int iy0 = quad_y_mode[corner];
            const std::size_t base = 4u * corner;
            for (std::size_t local = 0; local < 4u; ++local) {
                const int ix = ix0 + ((local & 1u) != 0u ? 1 : 0);
                const int iy = iy0 + ((local & 2u) != 0u ? 1 : 0);
                const auto ux = static_cast<std::size_t>(ix);
                const auto uy = static_cast<std::size_t>(iy);
                const std::size_t dof = base + local;

                if (values_out != nullptr) {
                    values_out[dof] = x.value[ux] * y.value[uy];
                }
                write_gradient(gradients_out,
                               dof,
                               x.first[ux] * y.value[uy],
                               x.value[ux] * y.first[uy],
                               Real(0));
                write_hessian(hessians_out,
                              dof,
                              x.second[ux] * y.value[uy],
                              x.value[ux] * y.second[uy],
                              Real(0),
                              x.first[ux] * y.first[uy],
                              Real(0),
                              Real(0));
            }
        }
        return;
    }

    const HermiteAxis z = hermite_axis(xi[2]);
    static constexpr std::array<int, 8> hex_x_mode{{0, 2, 2, 0, 0, 2, 2, 0}};
    static constexpr std::array<int, 8> hex_y_mode{{0, 0, 2, 2, 0, 0, 2, 2}};
    static constexpr std::array<int, 8> hex_z_mode{{0, 0, 0, 0, 2, 2, 2, 2}};
    static constexpr std::array<std::array<int, 3>, 8> local_offset{{
        {{0, 0, 0}},
        {{1, 0, 0}},
        {{0, 1, 0}},
        {{0, 0, 1}},
        {{1, 1, 0}},
        {{1, 0, 1}},
        {{0, 1, 1}},
        {{1, 1, 1}},
    }};

    for (std::size_t corner = 0; corner < hex_x_mode.size(); ++corner) {
        const std::size_t base = 8u * corner;
        for (std::size_t local = 0; local < local_offset.size(); ++local) {
            const int ix = hex_x_mode[corner] + local_offset[local][0];
            const int iy = hex_y_mode[corner] + local_offset[local][1];
            const int iz = hex_z_mode[corner] + local_offset[local][2];
            const auto ux = static_cast<std::size_t>(ix);
            const auto uy = static_cast<std::size_t>(iy);
            const auto uz = static_cast<std::size_t>(iz);
            const std::size_t dof = base + local;

            if (values_out != nullptr) {
                values_out[dof] = x.value[ux] * y.value[uy] * z.value[uz];
            }
            write_gradient(gradients_out,
                           dof,
                           x.first[ux] * y.value[uy] * z.value[uz],
                           x.value[ux] * y.first[uy] * z.value[uz],
                           x.value[ux] * y.value[uy] * z.first[uz]);
            write_hessian(hessians_out,
                          dof,
                          x.second[ux] * y.value[uy] * z.value[uz],
                          x.value[ux] * y.second[uy] * z.value[uz],
                          x.value[ux] * y.value[uy] * z.second[uz],
                          x.first[ux] * y.first[uy] * z.value[uz],
                          x.first[ux] * y.value[uy] * z.first[uz],
                          x.value[ux] * y.first[uy] * z.first[uz]);
        }
    }
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
    evaluate_values_to(xi, values.data());
}

void HermiteBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                      std::vector<Gradient>& gradients) const {
    gradients.resize(size_);
    std::array<Real, kMaxHermiteDofs * 3u> raw{};
    evaluate_gradients_to(xi, raw.data());
    for (std::size_t dof = 0; dof < size_; ++dof) {
        gradients[dof] = Gradient{};
        gradients[dof][0] = raw[dof * 3u + 0u];
        gradients[dof][1] = raw[dof * 3u + 1u];
        gradients[dof][2] = raw[dof * 3u + 2u];
    }
}

void HermiteBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                     std::vector<Hessian>& hessians) const {
    hessians.resize(size_);
    std::array<Real, kMaxHermiteDofs * 9u> raw{};
    evaluate_hessians_to(xi, raw.data());
    for (std::size_t dof = 0; dof < size_; ++dof) {
        Hessian H{};
        for (std::size_t row = 0; row < 3u; ++row) {
            for (std::size_t col = 0; col < 3u; ++col) {
                H(row, col) = raw[dof * 9u + row * 3u + col];
            }
        }
        hessians[dof] = H;
    }
}

void HermiteBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                std::vector<Real>& values,
                                std::vector<Gradient>& gradients,
                                std::vector<Hessian>& hessians) const {
    values.resize(size_);
    gradients.resize(size_);
    hessians.resize(size_);

    std::array<Real, kMaxHermiteDofs * 3u> raw_gradients{};
    std::array<Real, kMaxHermiteDofs * 9u> raw_hessians{};
    write_hermite_tensor_to(dimension_,
                            xi,
                            values.data(),
                            raw_gradients.data(),
                            raw_hessians.data());

    for (std::size_t dof = 0; dof < size_; ++dof) {
        gradients[dof] = Gradient{};
        gradients[dof][0] = raw_gradients[dof * 3u + 0u];
        gradients[dof][1] = raw_gradients[dof * 3u + 1u];
        gradients[dof][2] = raw_gradients[dof * 3u + 2u];

        Hessian H{};
        for (std::size_t row = 0; row < 3u; ++row) {
            for (std::size_t col = 0; col < 3u; ++col) {
                H(row, col) = raw_hessians[dof * 9u + row * 3u + col];
            }
        }
        hessians[dof] = H;
    }
}

void HermiteBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                      Real* SVMP_RESTRICT values_out) const {
    write_hermite_tensor_to(dimension_, xi, values_out, nullptr, nullptr);
}

void HermiteBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                         Real* SVMP_RESTRICT gradients_out) const {
    write_hermite_tensor_to(dimension_, xi, nullptr, gradients_out, nullptr);
}

void HermiteBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                        Real* SVMP_RESTRICT hessians_out) const {
    write_hermite_tensor_to(dimension_, xi, nullptr, nullptr, hessians_out);
}

void HermiteBasis::evaluate_all_to(const math::Vector<Real, 3>& xi,
                                   Real* SVMP_RESTRICT values_out,
                                   Real* SVMP_RESTRICT gradients_out,
                                   Real* SVMP_RESTRICT hessians_out) const {
    write_hermite_tensor_to(dimension_, xi, values_out, gradients_out, hessians_out);
}

void HermiteBasis::evaluate_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (output_stride < num_qpts) {
        throw BasisConfigurationException(
            "HermiteBasis strided evaluation requires output_stride >= points.size()",
            __FILE__, __LINE__, __func__);
    }

    std::array<Real, kMaxHermiteDofs> values{};
    std::array<Real, kMaxHermiteDofs * 3u> gradients{};
    std::array<Real, kMaxHermiteDofs * 9u> hessians{};

    for (std::size_t q = 0; q < num_qpts; ++q) {
        write_hermite_tensor_to(dimension_,
                                points[q],
                                values_out != nullptr ? values.data() : nullptr,
                                gradients_out != nullptr ? gradients.data() : nullptr,
                                hessians_out != nullptr ? hessians.data() : nullptr);

        if (values_out != nullptr) {
            for (std::size_t dof = 0; dof < size_; ++dof) {
                values_out[dof * output_stride + q] = values[dof];
            }
        }
        if (gradients_out != nullptr) {
            for (std::size_t row = 0; row < size_ * 3u; ++row) {
                gradients_out[row * output_stride + q] = gradients[row];
            }
        }
        if (hessians_out != nullptr) {
            for (std::size_t row = 0; row < size_ * 9u; ++row) {
                hessians_out[row * output_stride + q] = hessians[row];
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
