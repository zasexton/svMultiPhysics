/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BernsteinBasis.h"
#include "BasisTraits.h"
#include "Math/IntegerMath.h"
#include "Math/PyramidTensorCoordinates.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

using math::binomial_real;
using PyramidCoordinate = math::PyramidTensorCoordinate;
using math::pyramid_tensor_coordinates;

namespace {

struct BernsteinEvaluationScratch {
    std::array<std::vector<Real>, 3> axis_values;
    std::array<std::vector<Real>, 3> axis_first;
    std::array<std::vector<Real>, 3> axis_second;
    std::array<std::vector<Real>, 4> powers;
    std::vector<Real> lower_degree;
    std::vector<Real> second_lower_degree;
    std::vector<Real> simplex_values;
    std::vector<Gradient> simplex_gradients;
    std::vector<Hessian> simplex_hessians;
};

BernsteinEvaluationScratch& bernstein_scratch() {
    static thread_local BernsteinEvaluationScratch scratch;
    return scratch;
}

void bernstein_values_1d(int order, Real t, std::vector<Real>& values) {
    const std::size_t n = static_cast<std::size_t>(order);
    values.resize(n + 1u);
    values[0] = Real(1);

    const Real u = Real(1) - t;
    for (std::size_t degree = 1; degree <= n; ++degree) {
        Real saved = Real(0);
        for (std::size_t i = 0; i < degree; ++i) {
            const Real previous = values[i];
            values[i] = saved + u * previous;
            saved = t * previous;
        }
        values[degree] = saved;
    }
}

void bernstein_1d_with_derivatives(int order,
                                   Real t,
                                   Real dt_dxi,
                                   std::vector<Real>& values,
                                   std::vector<Real>& first,
                                   std::vector<Real>& second,
                                   std::vector<Real>& lower_degree,
                                   std::vector<Real>& second_lower_degree) {
    const std::size_t n = static_cast<std::size_t>(order);
    values.resize(n + 1u);
    first.resize(n + 1u);
    second.resize(n + 1u);

    values[0] = Real(1);
    if (order >= 1 && order - 1 == 0) {
        lower_degree.assign(1u, Real(1));
    }
    if (order >= 2 && order - 2 == 0) {
        second_lower_degree.assign(1u, Real(1));
    }

    const Real u = Real(1) - t;
    for (std::size_t degree = 1; degree <= n; ++degree) {
        Real saved = Real(0);
        for (std::size_t i = 0; i < degree; ++i) {
            const Real previous = values[i];
            values[i] = saved + u * previous;
            saved = t * previous;
        }
        values[degree] = saved;

        if (order >= 2 && degree == static_cast<std::size_t>(order - 2)) {
            second_lower_degree.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(degree + 1u));
        }
        if (order >= 1 && degree == static_cast<std::size_t>(order - 1)) {
            lower_degree.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(degree + 1u));
        }
    }

    if (order >= 1) {
        const Real scale = Real(order) * dt_dxi;
        for (int i = 0; i <= order; ++i) {
            const Real left = (i > 0) ? lower_degree[static_cast<std::size_t>(i - 1)] : Real(0);
            const Real right = (i < order) ? lower_degree[static_cast<std::size_t>(i)] : Real(0);
            first[static_cast<std::size_t>(i)] = scale * (left - right);
        }
    } else {
        first[0] = Real(0);
    }

    if (order >= 2) {
        const Real scale = Real(order) * Real(order - 1) * dt_dxi * dt_dxi;
        for (int i = 0; i <= order; ++i) {
            const Real left = (i >= 2) ? second_lower_degree[static_cast<std::size_t>(i - 2)] : Real(0);
            const Real center = (i >= 1 && i - 1 <= order - 2)
                                    ? second_lower_degree[static_cast<std::size_t>(i - 1)]
                                    : Real(0);
            const Real right = (i <= order - 2)
                                   ? second_lower_degree[static_cast<std::size_t>(i)]
                                   : Real(0);
            second[static_cast<std::size_t>(i)] = scale * (left - Real(2) * center + right);
        }
    } else {
        std::fill(second.begin(), second.end(), Real(0));
    }
}

struct BernsteinBatchScratch {
    std::array<std::vector<Real>, 3> axis_values;
    std::array<std::vector<Real>, 3> axis_first;
    std::array<std::vector<Real>, 3> axis_second;
    std::array<std::vector<PyramidCoordinate>, 3> pyramid_coordinates;
    std::vector<Real> values;
    std::vector<Real> first;
    std::vector<Real> second;
    std::vector<Real> lower_degree;
    std::vector<Real> second_lower_degree;
};

BernsteinBatchScratch& bernstein_batch_scratch() {
    static thread_local BernsteinBatchScratch scratch;
    return scratch;
}

void powers_for(Real value, int order, std::vector<Real>& powers) {
    powers.resize(static_cast<std::size_t>(order) + 1u);
    powers[0] = Real(1);
    for (int p = 1; p <= order; ++p) {
        powers[static_cast<std::size_t>(p)] = powers[static_cast<std::size_t>(p - 1)] * value;
    }
}

void fill_simplex_powers(int order,
                         const std::array<Real, 4>& lambdas,
                         int count,
                         std::array<std::vector<Real>, 4>& powers) {
    for (int i = 0; i < count; ++i) {
        powers_for(lambdas[static_cast<std::size_t>(i)], order, powers[static_cast<std::size_t>(i)]);
    }
}

Real simplex_power_product(const std::array<int, 4>& exponents,
                           const std::array<std::vector<Real>, 4>& powers,
                           int count,
                           int skip_a,
                           int skip_b) {
    Real product = Real(1);
    for (int i = 0; i < count; ++i) {
        if (i == skip_a || i == skip_b) {
            continue;
        }
        product *= powers[static_cast<std::size_t>(i)][static_cast<std::size_t>(exponents[static_cast<std::size_t>(i)])];
    }
    return product;
}

void simplex_term(const std::array<int, 4>& exponents,
                  Real coefficient,
                  const std::array<std::vector<Real>, 4>& powers,
                  const std::array<std::array<Real, 3>, 4>& lambda_gradients,
                  int count,
                  int dimension,
                  Real* value,
                  Gradient* gradient,
                  Hessian* hessian) {
    if (value) {
        *value = coefficient * simplex_power_product(exponents, powers, count, -1, -1);
    }

    if (gradient) {
        Gradient out{};
        for (int a = 0; a < count; ++a) {
            const int ea = exponents[static_cast<std::size_t>(a)];
            if (ea == 0) {
                continue;
            }
            const Real common = coefficient * Real(ea) *
                                powers[static_cast<std::size_t>(a)][static_cast<std::size_t>(ea - 1)] *
                                simplex_power_product(exponents, powers, count, a, -1);
            for (int d = 0; d < dimension; ++d) {
                out[static_cast<std::size_t>(d)] +=
                    common * lambda_gradients[static_cast<std::size_t>(a)][static_cast<std::size_t>(d)];
            }
        }
        *gradient = out;
    }

    if (hessian) {
        Hessian out{};
        for (int a = 0; a < count; ++a) {
            const int ea = exponents[static_cast<std::size_t>(a)];
            if (ea >= 2) {
                const Real common = coefficient * Real(ea) * Real(ea - 1) *
                                    powers[static_cast<std::size_t>(a)][static_cast<std::size_t>(ea - 2)] *
                                    simplex_power_product(exponents, powers, count, a, -1);
                for (int r = 0; r < dimension; ++r) {
                    for (int c = 0; c < dimension; ++c) {
                        out(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) +=
                            common *
                            lambda_gradients[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                            lambda_gradients[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)];
                    }
                }
            }

            if (ea == 0) {
                continue;
            }
            for (int b = 0; b < count; ++b) {
                if (a == b) {
                    continue;
                }
                const int eb = exponents[static_cast<std::size_t>(b)];
                if (eb == 0) {
                    continue;
                }
                const Real common = coefficient * Real(ea) * Real(eb) *
                                    powers[static_cast<std::size_t>(a)][static_cast<std::size_t>(ea - 1)] *
                                    powers[static_cast<std::size_t>(b)][static_cast<std::size_t>(eb - 1)] *
                                    simplex_power_product(exponents, powers, count, a, b);
                for (int r = 0; r < dimension; ++r) {
                    for (int c = 0; c < dimension; ++c) {
                        out(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) +=
                            common *
                            lambda_gradients[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                            lambda_gradients[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                    }
                }
            }
        }
        *hessian = out;
    }
}

struct BernsteinSpanWriter {
    std::span<Real> values;
    std::span<Gradient> gradients;
    std::span<Hessian> hessians;

    bool wants_values() const noexcept { return !values.empty(); }
    bool wants_gradients() const noexcept { return !gradients.empty(); }
    bool wants_hessians() const noexcept { return !hessians.empty(); }

    void write_value(std::size_t index, Real value) const {
        values[index] = value;
    }

    void write_gradient(std::size_t index, const Gradient& gradient) const {
        gradients[index] = gradient;
    }

    void write_hessian(std::size_t index, const Hessian& hessian) const {
        hessians[index] = hessian;
    }
};

struct BernsteinRawWriter {
    std::size_t stride{1};
    std::size_t offset{0};
    Real* values{nullptr};
    Real* gradients{nullptr};
    Real* hessians{nullptr};

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void write_value(std::size_t index, Real value) const {
        values[index * stride + offset] = value;
    }

    void write_gradient(std::size_t index, const Gradient& gradient) const {
        const std::size_t base = index * 3u;
        gradients[(base + 0u) * stride + offset] = gradient[0];
        gradients[(base + 1u) * stride + offset] = gradient[1];
        gradients[(base + 2u) * stride + offset] = gradient[2];
    }

    void write_hessian(std::size_t index, const Hessian& hessian) const {
        store_hessian_strided(hessian, hessians + index * 9u * stride, stride, offset);
    }
};

} // namespace

BernsteinBasis::BernsteinBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 0) {
        throw BasisConfigurationException("BernsteinBasis requires non-negative order",
                                          __FILE__, __LINE__, __func__);
    }

    binomial_coefficients_.resize(static_cast<std::size_t>(order_) + 1u);
    for (int i = 0; i <= order_; ++i) {
        binomial_coefficients_[static_cast<std::size_t>(i)] = binomial_real(order_, i);
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
    } else if (is_tetrahedron(element_type_)) {
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 2) * (order_ + 3) / 6);
        for (int i = 0; i <= order_; ++i) {
            for (int j = 0; j <= order_ - i; ++j) {
                for (int k = 0; k <= order_ - i - j; ++k) {
                    const int l = order_ - i - j - k;
                    simplex_indices_.push_back({i, j, k, l});
                    coefficients_.push_back(binomial(order_, i) *
                                            binomial(order_ - i, j) *
                                            binomial(order_ - i - j, k));
                }
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
        throw BasisElementCompatibilityException("Unsupported element type for BernsteinBasis",
                                                 __FILE__, __LINE__, __func__);
    }
}

Real BernsteinBasis::binomial(int n, int k) const {
    if (k < 0 || k > n) return Real(0);
    if (n == order_ && static_cast<std::size_t>(k) < binomial_coefficients_.size()) {
        return binomial_coefficients_[static_cast<std::size_t>(k)];
    }
    if (k == 0 || k == n) return Real(1);
    return binomial_real(n, k);
}

template <typename Writer>
void BernsteinBasis::evaluate_into_writer(const math::Vector<Real, 3>& xi,
                                          Writer& writer) const {
    const bool need_values = writer.wants_values();
    const bool need_gradients = writer.wants_gradients();
    const bool need_hessians = writer.wants_hessians();
    const bool need_derivatives = need_gradients || need_hessians;

    if (element_type_ == ElementType::Point1) {
        if (need_values) {
            writer.write_value(0, Real(1));
        }
        return;
    }

    auto& scratch = bernstein_scratch();

    if (is_line(element_type_)) {
        const Real t = (xi[0] + Real(1)) * Real(0.5);
        auto& bx = scratch.axis_values[0];
        auto& dbx = scratch.axis_first[0];
        auto& ddbx = scratch.axis_second[0];
        if (need_derivatives) {
            bernstein_1d_with_derivatives(order_, t, Real(0.5), bx, dbx, ddbx,
                                          scratch.lower_degree, scratch.second_lower_degree);
        } else {
            bernstein_values_1d(order_, t, bx);
        }

        for (int i = 0; i <= order_; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            if (need_values) {
                writer.write_value(idx, bx[idx]);
            }
            if (need_gradients) {
                Gradient gradient{};
                gradient[0] = dbx[idx];
                writer.write_gradient(idx, gradient);
            }
            if (need_hessians) {
                Hessian hessian{};
                hessian(0u, 0u) = ddbx[idx];
                writer.write_hessian(idx, hessian);
            }
        }
        return;
    }

    if (is_quadrilateral(element_type_)) {
        const Real tx = (xi[0] + Real(1)) * Real(0.5);
        const Real ty = (xi[1] + Real(1)) * Real(0.5);
        auto& bx = scratch.axis_values[0];
        auto& by = scratch.axis_values[1];
        auto& dbx = scratch.axis_first[0];
        auto& dby = scratch.axis_first[1];
        auto& ddbx = scratch.axis_second[0];
        auto& ddby = scratch.axis_second[1];
        if (need_derivatives) {
            bernstein_1d_with_derivatives(order_, tx, Real(0.5), bx, dbx, ddbx,
                                          scratch.lower_degree, scratch.second_lower_degree);
            bernstein_1d_with_derivatives(order_, ty, Real(0.5), by, dby, ddby,
                                          scratch.lower_degree, scratch.second_lower_degree);
        } else {
            bernstein_values_1d(order_, tx, bx);
            bernstein_values_1d(order_, ty, by);
        }

        std::size_t idx = 0;
        for (int j = 0; j <= order_; ++j) {
            const std::size_t jj = static_cast<std::size_t>(j);
            for (int i = 0; i <= order_; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                if (need_values) {
                    writer.write_value(idx, bx[ii] * by[jj]);
                }
                if (need_gradients) {
                    Gradient gradient{};
                    gradient[0] = dbx[ii] * by[jj];
                    gradient[1] = bx[ii] * dby[jj];
                    writer.write_gradient(idx, gradient);
                }
                if (need_hessians) {
                    Hessian hessian{};
                    hessian(0u, 0u) = ddbx[ii] * by[jj];
                    hessian(1u, 1u) = bx[ii] * ddby[jj];
                    const Real xy = dbx[ii] * dby[jj];
                    hessian(0u, 1u) = xy;
                    hessian(1u, 0u) = xy;
                    writer.write_hessian(idx, hessian);
                }
                ++idx;
            }
        }
        return;
    }

    if (is_hexahedron(element_type_)) {
        const Real tx = (xi[0] + Real(1)) * Real(0.5);
        const Real ty = (xi[1] + Real(1)) * Real(0.5);
        const Real tz = (xi[2] + Real(1)) * Real(0.5);
        auto& bx = scratch.axis_values[0];
        auto& by = scratch.axis_values[1];
        auto& bz = scratch.axis_values[2];
        auto& dbx = scratch.axis_first[0];
        auto& dby = scratch.axis_first[1];
        auto& dbz = scratch.axis_first[2];
        auto& ddbx = scratch.axis_second[0];
        auto& ddby = scratch.axis_second[1];
        auto& ddbz = scratch.axis_second[2];
        if (need_derivatives) {
            bernstein_1d_with_derivatives(order_, tx, Real(0.5), bx, dbx, ddbx,
                                          scratch.lower_degree, scratch.second_lower_degree);
            bernstein_1d_with_derivatives(order_, ty, Real(0.5), by, dby, ddby,
                                          scratch.lower_degree, scratch.second_lower_degree);
            bernstein_1d_with_derivatives(order_, tz, Real(0.5), bz, dbz, ddbz,
                                          scratch.lower_degree, scratch.second_lower_degree);
        } else {
            bernstein_values_1d(order_, tx, bx);
            bernstein_values_1d(order_, ty, by);
            bernstein_values_1d(order_, tz, bz);
        }

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            for (int j = 0; j <= order_; ++j) {
                const std::size_t jj = static_cast<std::size_t>(j);
                for (int i = 0; i <= order_; ++i) {
                    const std::size_t ii = static_cast<std::size_t>(i);
                    if (need_values) {
                        writer.write_value(idx, bx[ii] * by[jj] * bz[kk]);
                    }
                    if (need_gradients) {
                        Gradient gradient{};
                        gradient[0] = dbx[ii] * by[jj] * bz[kk];
                        gradient[1] = bx[ii] * dby[jj] * bz[kk];
                        gradient[2] = bx[ii] * by[jj] * dbz[kk];
                        writer.write_gradient(idx, gradient);
                    }
                    if (need_hessians) {
                        Hessian hessian{};
                        hessian(0u, 0u) = ddbx[ii] * by[jj] * bz[kk];
                        hessian(1u, 1u) = bx[ii] * ddby[jj] * bz[kk];
                        hessian(2u, 2u) = bx[ii] * by[jj] * ddbz[kk];
                        const Real xy = dbx[ii] * dby[jj] * bz[kk];
                        const Real xz = dbx[ii] * by[jj] * dbz[kk];
                        const Real yz = bx[ii] * dby[jj] * dbz[kk];
                        hessian(0u, 1u) = xy;
                        hessian(1u, 0u) = xy;
                        hessian(0u, 2u) = xz;
                        hessian(2u, 0u) = xz;
                        hessian(1u, 2u) = yz;
                        hessian(2u, 1u) = yz;
                        writer.write_hessian(idx, hessian);
                    }
                    ++idx;
                }
            }
        }
        return;
    }

    if (is_triangle(element_type_)) {
        const std::array<Real, 4> lambdas{Real(1) - xi[0] - xi[1], xi[0], xi[1], Real(0)};
        const std::array<std::array<Real, 3>, 4> lambda_gradients{{
            {{Real(-1), Real(-1), Real(0)}},
            {{Real(1), Real(0), Real(0)}},
            {{Real(0), Real(1), Real(0)}},
            {{Real(0), Real(0), Real(0)}}
        }};
        fill_simplex_powers(order_, lambdas, 3, scratch.powers);
        for (std::size_t idx = 0; idx < simplex_indices_.size(); ++idx) {
            Real value{};
            Gradient gradient{};
            Hessian hessian{};
            simplex_term(simplex_indices_[idx], coefficients_[idx], scratch.powers,
                         lambda_gradients, 3, 2,
                         need_values ? &value : nullptr,
                         need_gradients ? &gradient : nullptr,
                         need_hessians ? &hessian : nullptr);
            if (need_values) {
                writer.write_value(idx, value);
            }
            if (need_gradients) {
                writer.write_gradient(idx, gradient);
            }
            if (need_hessians) {
                writer.write_hessian(idx, hessian);
            }
        }
        return;
    }

    if (is_tetrahedron(element_type_)) {
        const std::array<Real, 4> lambdas{Real(1) - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]};
        const std::array<std::array<Real, 3>, 4> lambda_gradients{{
            {{Real(-1), Real(-1), Real(-1)}},
            {{Real(1), Real(0), Real(0)}},
            {{Real(0), Real(1), Real(0)}},
            {{Real(0), Real(0), Real(1)}}
        }};
        fill_simplex_powers(order_, lambdas, 4, scratch.powers);
        for (std::size_t idx = 0; idx < simplex_indices_.size(); ++idx) {
            Real value{};
            Gradient gradient{};
            Hessian hessian{};
            simplex_term(simplex_indices_[idx], coefficients_[idx], scratch.powers,
                         lambda_gradients, 4, 3,
                         need_values ? &value : nullptr,
                         need_gradients ? &gradient : nullptr,
                         need_hessians ? &hessian : nullptr);
            if (need_values) {
                writer.write_value(idx, value);
            }
            if (need_gradients) {
                writer.write_gradient(idx, gradient);
            }
            if (need_hessians) {
                writer.write_hessian(idx, hessian);
            }
        }
        return;
    }

    if (is_wedge(element_type_)) {
        const std::array<Real, 4> lambdas{Real(1) - xi[0] - xi[1], xi[0], xi[1], Real(0)};
        const std::array<std::array<Real, 3>, 4> lambda_gradients{{
            {{Real(-1), Real(-1), Real(0)}},
            {{Real(1), Real(0), Real(0)}},
            {{Real(0), Real(1), Real(0)}},
            {{Real(0), Real(0), Real(0)}}
        }};
        fill_simplex_powers(order_, lambdas, 3, scratch.powers);

        const std::size_t tri_count = simplex_indices_.size();
        scratch.simplex_values.resize(tri_count);
        if (need_derivatives) {
            scratch.simplex_gradients.resize(tri_count);
        }
        if (need_hessians) {
            scratch.simplex_hessians.resize(tri_count);
        }
        for (std::size_t tri = 0; tri < tri_count; ++tri) {
            simplex_term(simplex_indices_[tri], coefficients_[tri], scratch.powers,
                         lambda_gradients, 3, 2,
                         &scratch.simplex_values[tri],
                         need_derivatives ? &scratch.simplex_gradients[tri] : nullptr,
                         need_hessians ? &scratch.simplex_hessians[tri] : nullptr);
        }

        const Real tz = (xi[2] + Real(1)) * Real(0.5);
        auto& bz = scratch.axis_values[2];
        auto& dbz = scratch.axis_first[2];
        auto& ddbz = scratch.axis_second[2];
        if (need_derivatives) {
            bernstein_1d_with_derivatives(order_, tz, Real(0.5), bz, dbz, ddbz,
                                          scratch.lower_degree, scratch.second_lower_degree);
        } else {
            bernstein_values_1d(order_, tz, bz);
        }

        std::size_t out_idx = 0;
        for (int k = 0; k <= order_; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            for (std::size_t tri = 0; tri < tri_count; ++tri) {
                if (need_values) {
                    writer.write_value(out_idx, scratch.simplex_values[tri] * bz[kk]);
                }
                if (need_gradients) {
                    Gradient gradient{};
                    gradient[0] = scratch.simplex_gradients[tri][0] * bz[kk];
                    gradient[1] = scratch.simplex_gradients[tri][1] * bz[kk];
                    gradient[2] = scratch.simplex_values[tri] * dbz[kk];
                    writer.write_gradient(out_idx, gradient);
                }
                if (need_hessians) {
                    Hessian hessian{};
                    const auto& tri_hessian = scratch.simplex_hessians[tri];
                    hessian(0u, 0u) = tri_hessian(0u, 0u) * bz[kk];
                    hessian(0u, 1u) = tri_hessian(0u, 1u) * bz[kk];
                    hessian(1u, 0u) = hessian(0u, 1u);
                    hessian(1u, 1u) = tri_hessian(1u, 1u) * bz[kk];
                    hessian(0u, 2u) = scratch.simplex_gradients[tri][0] * dbz[kk];
                    hessian(2u, 0u) = hessian(0u, 2u);
                    hessian(1u, 2u) = scratch.simplex_gradients[tri][1] * dbz[kk];
                    hessian(2u, 1u) = hessian(1u, 2u);
                    hessian(2u, 2u) = scratch.simplex_values[tri] * ddbz[kk];
                    writer.write_hessian(out_idx, hessian);
                }
                ++out_idx;
            }
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        const auto coordinates = pyramid_tensor_coordinates(xi);

        auto& bx = scratch.axis_values[0];
        auto& by = scratch.axis_values[1];
        auto& bz = scratch.axis_values[2];
        auto& dbx = scratch.axis_first[0];
        auto& dby = scratch.axis_first[1];
        auto& dbz = scratch.axis_first[2];
        auto& ddbx = scratch.axis_second[0];
        auto& ddby = scratch.axis_second[1];
        auto& ddbz = scratch.axis_second[2];
        if (need_derivatives) {
            bernstein_1d_with_derivatives(order_, coordinates[0].value, Real(1), bx, dbx, ddbx,
                                          scratch.lower_degree, scratch.second_lower_degree);
            bernstein_1d_with_derivatives(order_, coordinates[1].value, Real(1), by, dby, ddby,
                                          scratch.lower_degree, scratch.second_lower_degree);
            bernstein_1d_with_derivatives(order_, coordinates[2].value, Real(1), bz, dbz, ddbz,
                                          scratch.lower_degree, scratch.second_lower_degree);
        } else {
            bernstein_values_1d(order_, coordinates[0].value, bx);
            bernstein_values_1d(order_, coordinates[1].value, by);
            bernstein_values_1d(order_, coordinates[2].value, bz);
        }

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            for (int j = 0; j <= order_; ++j) {
                const std::size_t jj = static_cast<std::size_t>(j);
                for (int i = 0; i <= order_; ++i) {
                    const std::size_t ii = static_cast<std::size_t>(i);
                    const std::array<Real, 3> axis_values{bx[ii], by[jj], bz[kk]};
                    const std::array<Real, 3> product_except_axis{
                        axis_values[1] * axis_values[2],
                        axis_values[0] * axis_values[2],
                        axis_values[0] * axis_values[1]
                    };
                    auto product_except_pair = [&](int skip_a, int skip_b) {
                        return axis_values[static_cast<std::size_t>(3 - skip_a - skip_b)];
                    };

                    if (need_values) {
                        writer.write_value(idx, axis_values[0] * axis_values[1] * axis_values[2]);
                    }
                    if (need_gradients) {
                        const std::array<Real, 3> axis_first{dbx[ii], dby[jj], dbz[kk]};
                        Gradient gradient{};
                        for (int d = 0; d < 3; ++d) {
                            Real entry = Real(0);
                            for (int axis = 0; axis < 3; ++axis) {
                                const auto aa = static_cast<std::size_t>(axis);
                                entry += axis_first[static_cast<std::size_t>(axis)] *
                                         coordinates[static_cast<std::size_t>(axis)].first[static_cast<std::size_t>(d)] *
                                         product_except_axis[aa];
                            }
                            gradient[static_cast<std::size_t>(d)] = entry;
                        }
                        writer.write_gradient(idx, gradient);
                    }
                    if (need_hessians) {
                        const std::array<Real, 3> axis_first{dbx[ii], dby[jj], dbz[kk]};
                        const std::array<Real, 3> axis_second{ddbx[ii], ddby[jj], ddbz[kk]};
                        Hessian hessian{};
                        for (int r = 0; r < 3; ++r) {
                            const auto rr = static_cast<std::size_t>(r);
                            for (int c = 0; c < 3; ++c) {
                                const auto cc = static_cast<std::size_t>(c);
                                Real entry = Real(0);
                                for (int axis = 0; axis < 3; ++axis) {
                                    const auto aa = static_cast<std::size_t>(axis);
                                    entry += (axis_second[aa] * coordinates[aa].first[rr] * coordinates[aa].first[cc] +
                                              axis_first[aa] * coordinates[aa].second(rr, cc)) *
                                             product_except_axis[aa];
                                    for (int other = 0; other < 3; ++other) {
                                        if (other == axis) {
                                            continue;
                                        }
                                        const auto oo = static_cast<std::size_t>(other);
                                        entry += axis_first[aa] * coordinates[aa].first[rr] *
                                                 axis_first[oo] * coordinates[oo].first[cc] *
                                                 product_except_pair(axis, other);
                                    }
                                }
                                hessian(rr, cc) = entry;
                            }
                        }
                        writer.write_hessian(idx, hessian);
                    }
                    ++idx;
                }
            }
        }
        return;
    }

    throw BasisEvaluationException("Unsupported element in BernsteinBasis::evaluate_into",
                                   __FILE__, __LINE__, __func__);
}

void BernsteinBasis::evaluate_into_raw(const math::Vector<Real, 3>& xi,
                                       std::size_t output_stride,
                                       std::size_t output_offset,
                                       Real* SVMP_RESTRICT values_out,
                                       Real* SVMP_RESTRICT gradients_out,
                                       Real* SVMP_RESTRICT hessians_out) const {
    BernsteinRawWriter writer{output_stride, output_offset, values_out, gradients_out, hessians_out};
    evaluate_into_writer(xi, writer);
}

void BernsteinBasis::evaluate_into(const math::Vector<Real, 3>& xi,
                                   std::span<Real> values,
                                   std::span<Gradient> gradients,
                                   std::span<Hessian> hessians) const {
    BernsteinSpanWriter writer{values, gradients, hessians};
    evaluate_into_writer(xi, writer);
}

void BernsteinBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                     std::vector<Real>& values) const {
    values.resize(size_);
    evaluate_into(xi, std::span<Real>(values.data(), values.size()),
                  std::span<Gradient>{}, std::span<Hessian>{});
}

void BernsteinBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                        std::vector<Gradient>& gradients) const {
    gradients.resize(size_);
    if (element_type_ == ElementType::Point1) {
        gradients[0] = Gradient{};
        return;
    }
    evaluate_into(xi, std::span<Real>{},
                  std::span<Gradient>(gradients.data(), gradients.size()),
                  std::span<Hessian>{});
}

void BernsteinBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                       std::vector<Hessian>& hessians) const {
    hessians.resize(size_);
    if (element_type_ == ElementType::Point1) {
        hessians[0] = Hessian{};
        return;
    }
    evaluate_into(xi, std::span<Real>{}, std::span<Gradient>{},
                  std::span<Hessian>(hessians.data(), hessians.size()));
}

void BernsteinBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                  std::vector<Real>& values,
                                  std::vector<Gradient>& gradients,
                                  std::vector<Hessian>& hessians) const {
    values.resize(size_);
    gradients.resize(size_);
    hessians.resize(size_);
    if (element_type_ == ElementType::Point1) {
        values[0] = Real(1);
        gradients[0] = Gradient{};
        hessians[0] = Hessian{};
        return;
    }
    evaluate_into(xi,
                  std::span<Real>(values.data(), values.size()),
                  std::span<Gradient>(gradients.data(), gradients.size()),
                  std::span<Hessian>(hessians.data(), hessians.size()));
}

void BernsteinBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                        Real* SVMP_RESTRICT values_out) const {
    evaluate_into_raw(xi, 1u, 0u, values_out, nullptr, nullptr);
}

void BernsteinBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                           Real* SVMP_RESTRICT gradients_out) const {
    evaluate_into_raw(xi, 1u, 0u, nullptr, gradients_out, nullptr);
}

void BernsteinBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                          Real* SVMP_RESTRICT hessians_out) const {
    evaluate_into_raw(xi, 1u, 0u, nullptr, nullptr, hessians_out);
}

void BernsteinBasis::evaluate_at_quadrature_points(
    const std::vector<math::Vector<Real, 3>>& points,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    evaluate_at_quadrature_points_strided(points, points.size(), values_out, gradients_out, hessians_out);
}

void BernsteinBasis::evaluate_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    BASIS_CHECK_CONFIG(output_stride >= points.size(),
                       "BernsteinBasis::evaluate_at_quadrature_points_strided: output stride is smaller than point count");

    const std::size_t num_qpts = points.size();
    if (num_qpts == 0u ||
        (values_out == nullptr && gradients_out == nullptr && hessians_out == nullptr)) {
        return;
    }

    const bool need_derivatives = gradients_out != nullptr || hessians_out != nullptr;
    const auto axis_count = static_cast<std::size_t>(order_ + 1);
    auto& scratch = bernstein_batch_scratch();

    auto fill_axis = [&](std::size_t axis, Real scale) {
        auto& values = scratch.axis_values[axis];
        auto& first = scratch.axis_first[axis];
        auto& second = scratch.axis_second[axis];
        values.resize(axis_count * num_qpts);
        if (need_derivatives) {
            first.resize(axis_count * num_qpts);
            second.resize(axis_count * num_qpts);
        } else {
            first.clear();
            second.clear();
        }

        for (std::size_t q = 0; q < num_qpts; ++q) {
            const Real t = (points[q][axis] + Real(1)) * Real(0.5);
            if (need_derivatives) {
                bernstein_1d_with_derivatives(order_, t, scale,
                                              scratch.values,
                                              scratch.first,
                                              scratch.second,
                                              scratch.lower_degree,
                                              scratch.second_lower_degree);
                for (std::size_t i = 0; i < axis_count; ++i) {
                    values[i * num_qpts + q] = scratch.values[i];
                    first[i * num_qpts + q] = scratch.first[i];
                    second[i * num_qpts + q] = scratch.second[i];
                }
            } else {
                bernstein_values_1d(order_, t, scratch.values);
                for (std::size_t i = 0; i < axis_count; ++i) {
                    values[i * num_qpts + q] = scratch.values[i];
                }
            }
        }
    };

    auto fill_pyramid_axes = [&]() {
        for (std::size_t axis = 0; axis < 3u; ++axis) {
            scratch.pyramid_coordinates[axis].resize(num_qpts);
            scratch.axis_values[axis].resize(axis_count * num_qpts);
            if (need_derivatives) {
                scratch.axis_first[axis].resize(axis_count * num_qpts);
                scratch.axis_second[axis].resize(axis_count * num_qpts);
            } else {
                scratch.axis_first[axis].clear();
                scratch.axis_second[axis].clear();
            }
        }

        for (std::size_t q = 0; q < num_qpts; ++q) {
            const auto coordinates = pyramid_tensor_coordinates(points[q]);
            for (std::size_t axis = 0; axis < 3u; ++axis) {
                scratch.pyramid_coordinates[axis][q] = coordinates[axis];
                if (need_derivatives) {
                    bernstein_1d_with_derivatives(order_, coordinates[axis].value, Real(1),
                                                  scratch.values,
                                                  scratch.first,
                                                  scratch.second,
                                                  scratch.lower_degree,
                                                  scratch.second_lower_degree);
                    for (std::size_t i = 0; i < axis_count; ++i) {
                        scratch.axis_values[axis][i * num_qpts + q] = scratch.values[i];
                        scratch.axis_first[axis][i * num_qpts + q] = scratch.first[i];
                        scratch.axis_second[axis][i * num_qpts + q] = scratch.second[i];
                    }
                } else {
                    bernstein_values_1d(order_, coordinates[axis].value, scratch.values);
                    for (std::size_t i = 0; i < axis_count; ++i) {
                        scratch.axis_values[axis][i * num_qpts + q] = scratch.values[i];
                    }
                }
            }
        }
    };

    auto zero_gradient = [&](Real* gradient_row, std::size_t q) {
        gradient_row[0u * output_stride + q] = Real(0);
        gradient_row[1u * output_stride + q] = Real(0);
        gradient_row[2u * output_stride + q] = Real(0);
    };

    auto zero_hessian = [&](Real* hessian_row, std::size_t q) {
        for (std::size_t component = 0; component < 9u; ++component) {
            hessian_row[component * output_stride + q] = Real(0);
        }
    };

    if (element_type_ == ElementType::Point1) {
        if (values_out != nullptr) {
            std::fill_n(values_out, num_qpts, Real(1));
        }
        if (gradients_out != nullptr) {
            for (std::size_t component = 0; component < 3u; ++component) {
                std::fill_n(gradients_out + component * output_stride, num_qpts, Real(0));
            }
        }
        if (hessians_out != nullptr) {
            for (std::size_t component = 0; component < 9u; ++component) {
                std::fill_n(hessians_out + component * output_stride, num_qpts, Real(0));
            }
        }
        return;
    }

    if (is_line(element_type_) || is_quadrilateral(element_type_) || is_hexahedron(element_type_)) {
        const int dim = dimension_;
        for (int axis = 0; axis < dim; ++axis) {
            fill_axis(static_cast<std::size_t>(axis), Real(0.5));
        }

        std::size_t idx = 0;
        for (int k = 0; k < (dim == 3 ? order_ + 1 : 1); ++k) {
            const auto kk = static_cast<std::size_t>(k);
            for (int j = 0; j < (dim >= 2 ? order_ + 1 : 1); ++j) {
                const auto jj = static_cast<std::size_t>(j);
                for (int i = 0; i <= order_; ++i) {
                    const auto ii = static_cast<std::size_t>(i);
                    Real* value_row = values_out ? values_out + idx * output_stride : nullptr;
                    Real* gradient_row = gradients_out ? gradients_out + idx * 3u * output_stride : nullptr;
                    Real* hessian_row = hessians_out ? hessians_out + idx * 9u * output_stride : nullptr;

                    for (std::size_t q = 0; q < num_qpts; ++q) {
                        const Real bx = scratch.axis_values[0][ii * num_qpts + q];
                        const Real by = (dim >= 2)
                            ? scratch.axis_values[1][jj * num_qpts + q]
                            : Real(1);
                        const Real bz = (dim == 3)
                            ? scratch.axis_values[2][kk * num_qpts + q]
                            : Real(1);

                        if (value_row != nullptr) {
                            value_row[q] = bx * by * bz;
                        }

                        if (gradient_row != nullptr) {
                            zero_gradient(gradient_row, q);
                            const Real dbx = scratch.axis_first[0][ii * num_qpts + q];
                            gradient_row[0u * output_stride + q] = dbx * by * bz;
                            if (dim >= 2) {
                                const Real dby = scratch.axis_first[1][jj * num_qpts + q];
                                gradient_row[1u * output_stride + q] = bx * dby * bz;
                            }
                            if (dim == 3) {
                                const Real dbz = scratch.axis_first[2][kk * num_qpts + q];
                                gradient_row[2u * output_stride + q] = bx * by * dbz;
                            }
                        }

                        if (hessian_row != nullptr) {
                            zero_hessian(hessian_row, q);
                            const Real dbx = scratch.axis_first[0][ii * num_qpts + q];
                            const Real ddbx = scratch.axis_second[0][ii * num_qpts + q];
                            hessian_row[0u * output_stride + q] = ddbx * by * bz;
                            if (dim >= 2) {
                                const Real dby = scratch.axis_first[1][jj * num_qpts + q];
                                const Real ddby = scratch.axis_second[1][jj * num_qpts + q];
                                const Real xy = dbx * dby * bz;
                                hessian_row[1u * output_stride + q] = xy;
                                hessian_row[3u * output_stride + q] = xy;
                                hessian_row[4u * output_stride + q] = bx * ddby * bz;
                            }
                            if (dim == 3) {
                                const Real dby = scratch.axis_first[1][jj * num_qpts + q];
                                const Real dbz = scratch.axis_first[2][kk * num_qpts + q];
                                const Real ddbz = scratch.axis_second[2][kk * num_qpts + q];
                                const Real xz = dbx * by * dbz;
                                const Real yz = bx * dby * dbz;
                                hessian_row[2u * output_stride + q] = xz;
                                hessian_row[6u * output_stride + q] = xz;
                                hessian_row[5u * output_stride + q] = yz;
                                hessian_row[7u * output_stride + q] = yz;
                                hessian_row[8u * output_stride + q] = bx * by * ddbz;
                            }
                        }
                    }
                    ++idx;
                }
            }
        }
        return;
    }

    if (is_pyramid(element_type_)) {
        fill_pyramid_axes();

        std::size_t idx = 0;
        for (int k = 0; k <= order_; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            for (int j = 0; j <= order_; ++j) {
                const auto jj = static_cast<std::size_t>(j);
                for (int i = 0; i <= order_; ++i) {
                    const auto ii = static_cast<std::size_t>(i);
                    Real* value_row = values_out ? values_out + idx * output_stride : nullptr;
                    Real* gradient_row = gradients_out ? gradients_out + idx * 3u * output_stride : nullptr;
                    Real* hessian_row = hessians_out ? hessians_out + idx * 9u * output_stride : nullptr;

                    for (std::size_t q = 0; q < num_qpts; ++q) {
                        const std::array<Real, 3> axis_values{
                            scratch.axis_values[0][ii * num_qpts + q],
                            scratch.axis_values[1][jj * num_qpts + q],
                            scratch.axis_values[2][kk * num_qpts + q]
                        };
                        const std::array<Real, 3> product_except_axis{
                            axis_values[1] * axis_values[2],
                            axis_values[0] * axis_values[2],
                            axis_values[0] * axis_values[1]
                        };
                        if (value_row != nullptr) {
                            value_row[q] = axis_values[0] * axis_values[1] * axis_values[2];
                        }

                        if (gradient_row != nullptr) {
                            const std::array<Real, 3> axis_first{
                                scratch.axis_first[0][ii * num_qpts + q],
                                scratch.axis_first[1][jj * num_qpts + q],
                                scratch.axis_first[2][kk * num_qpts + q]
                            };
                            for (std::size_t d = 0; d < 3u; ++d) {
                                Real entry = Real(0);
                                for (std::size_t axis = 0; axis < 3u; ++axis) {
                                    entry += axis_first[axis] *
                                             scratch.pyramid_coordinates[axis][q].first[d] *
                                             product_except_axis[axis];
                                }
                                gradient_row[d * output_stride + q] = entry;
                            }
                        }

                        if (hessian_row != nullptr) {
                            const std::array<Real, 3> axis_first{
                                scratch.axis_first[0][ii * num_qpts + q],
                                scratch.axis_first[1][jj * num_qpts + q],
                                scratch.axis_first[2][kk * num_qpts + q]
                            };
                            const std::array<Real, 3> axis_second{
                                scratch.axis_second[0][ii * num_qpts + q],
                                scratch.axis_second[1][jj * num_qpts + q],
                                scratch.axis_second[2][kk * num_qpts + q]
                            };
                            const auto product_except_pair = [&](std::size_t a, std::size_t b) {
                                return axis_values[3u - a - b];
                            };

                            for (std::size_t r = 0; r < 3u; ++r) {
                                for (std::size_t c = 0; c < 3u; ++c) {
                                    Real entry = Real(0);
                                    for (std::size_t axis = 0; axis < 3u; ++axis) {
                                        const auto& coordinate = scratch.pyramid_coordinates[axis][q];
                                        entry += (axis_second[axis] * coordinate.first[r] * coordinate.first[c] +
                                                  axis_first[axis] * coordinate.second(r, c)) *
                                                 product_except_axis[axis];
                                        for (std::size_t other = 0; other < 3u; ++other) {
                                            if (other == axis) {
                                                continue;
                                            }
                                            entry += axis_first[axis] *
                                                     scratch.pyramid_coordinates[axis][q].first[r] *
                                                     axis_first[other] *
                                                     scratch.pyramid_coordinates[other][q].first[c] *
                                                     product_except_pair(axis, other);
                                        }
                                    }
                                    hessian_row[(r * 3u + c) * output_stride + q] = entry;
                                }
                            }
                        }
                    }
                    ++idx;
                }
            }
        }
        return;
    }

    for (std::size_t q = 0; q < points.size(); ++q) {
        evaluate_into_raw(points[q], output_stride, q, values_out, gradients_out, hessians_out);
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
