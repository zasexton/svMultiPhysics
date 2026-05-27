/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/BSplineBasis.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace svmp {
namespace FE {
namespace basis {

namespace {

struct BSplineDerivativeScratch {
    std::vector<Real> ndu;
    std::vector<Real> left;
    std::vector<Real> right;
    std::vector<Real> a;
    std::vector<Real> ders;
    std::vector<Real> values;
    std::vector<Real> first;
    std::vector<Real> second;
};

BSplineDerivativeScratch& bspline_derivative_scratch() {
    static thread_local BSplineDerivativeScratch scratch;
    return scratch;
}

Real reference_coordinate_tolerance() {
    return std::numeric_limits<Real>::epsilon() * Real(128);
}

Real snap_reference_coordinate(Real xi) {
    const Real tol = reference_coordinate_tolerance();
    if (xi < Real(-1) - tol || xi > Real(1) + tol) {
        throw BasisEvaluationException("BSplineBasis: reference coordinate lies outside [-1,1]",
                                       __FILE__, __LINE__, __func__);
    }
    if (std::abs(xi + Real(1)) <= tol) {
        return Real(-1);
    }
    if (std::abs(xi - Real(1)) <= tol) {
        return Real(1);
    }
    return xi;
}

Real denominator_tolerance(Real scale) {
    return std::numeric_limits<Real>::epsilon() * Real(128) *
           std::max(scale, std::numeric_limits<Real>::min());
}

Real knot_scale(const std::vector<Real>& knots) {
    Real scale = Real(1);
    if (!knots.empty()) {
        scale = std::max(scale, std::abs(knots.back() - knots.front()));
    }
    for (Real knot : knots) {
        scale = std::max(scale, std::abs(knot));
    }
    return scale;
}

std::uint64_t real_identity_word(Real value) noexcept {
    if (value == Real(0)) {
        value = Real(0);
    }
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(Real));
    return bits;
}

Real knot_interval_tolerance(const std::vector<Real>& knots, int degree) {
    const Real degree_scale = static_cast<Real>(std::max(degree, 1));
    return std::numeric_limits<Real>::epsilon() * Real(128) *
           degree_scale *
           std::max(knot_scale(knots), std::numeric_limits<Real>::min());
}

Real divide_if_resolved(Real numerator, Real denominator, Real tolerance) {
    return (std::abs(denominator) > tolerance) ? (numerator / denominator) : Real(0);
}

void validate_rational_denominator(Real denominator, Real scale, const char* operation) {
    if (!std::isfinite(denominator) ||
        std::abs(denominator) <= denominator_tolerance(scale)) {
        throw BasisEvaluationException(std::string("BSplineBasis: invalid rational denominator during ") +
                                           operation,
                                       __FILE__, __LINE__, __func__);
    }
}

int find_span_index(const std::vector<Real>& knots,
                    int degree,
                    std::size_t num_basis,
                    Real u) {
    const int n = static_cast<int>(num_basis) - 1;
    const std::size_t p = static_cast<std::size_t>(degree);

    const Real u_min = knots[p];
    const Real u_max = knots[static_cast<std::size_t>(n) + 1];

    const Real eps = knot_interval_tolerance(knots, degree);
    if (u >= u_max - eps) [[unlikely]] {
        return n;
    }
    if (u <= u_min + eps) [[unlikely]] {
        return degree;
    }

    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < knots[static_cast<std::size_t>(mid)] ||
           u >= knots[static_cast<std::size_t>(mid) + 1]) {
        if (u < knots[static_cast<std::size_t>(mid)]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    while (mid < n &&
           knots[static_cast<std::size_t>(mid) + 1] -
                   knots[static_cast<std::size_t>(mid)] <= eps) {
        ++mid;
    }
    return mid;
}

void basis_funs_and_derivatives_flat(const std::vector<Real>& knots,
                                     int degree,
                                     int span,
                                     Real u,
                                     int max_derivative,
                                     BSplineDerivativeScratch& scratch) {
    const int p = degree;
    const int ndu_size = p + 1;
    const std::size_t n = static_cast<std::size_t>(ndu_size);
    const std::size_t derivative_rows = static_cast<std::size_t>(max_derivative + 1);

    scratch.ndu.resize(n * n);
    scratch.left.resize(n);
    scratch.right.resize(n);
    scratch.a.resize(2u * n);
    scratch.ders.resize(derivative_rows * n);
    const Real interval_tol = knot_interval_tolerance(knots, degree);

    auto ndu = [&](int row, int col) -> Real& {
        return scratch.ndu[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(col)];
    };
    auto ders = [&](int row, int col) -> Real& {
        return scratch.ders[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(col)];
    };
    auto a = [&](int row, int col) -> Real& {
        return scratch.a[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(col)];
    };

    ndu(0, 0) = Real(1);
    for (int j = 1; j <= p; ++j) {
        scratch.left[static_cast<std::size_t>(j)] =
            u - knots[static_cast<std::size_t>(span + 1 - j)];
        scratch.right[static_cast<std::size_t>(j)] =
            knots[static_cast<std::size_t>(span + j)] - u;

        Real saved = Real(0);
        for (int r = 0; r < j; ++r) {
            const Real denom = scratch.right[static_cast<std::size_t>(r) + 1] +
                               scratch.left[static_cast<std::size_t>(j - r)];
            ndu(j, r) = denom;
            const Real temp = divide_if_resolved(ndu(r, j - 1), denom, interval_tol);
            ndu(r, j) = saved + scratch.right[static_cast<std::size_t>(r) + 1] * temp;
            saved = scratch.left[static_cast<std::size_t>(j - r)] * temp;
        }
        ndu(j, j) = saved;
    }

    for (int j = 0; j <= p; ++j) {
        ders(0, j) = ndu(j, p);
    }

    for (int r = 0; r <= p; ++r) {
        int s1 = 0;
        int s2 = 1;
        a(0, 0) = Real(1);

        for (int k = 1; k <= max_derivative; ++k) {
            Real d = Real(0);
            const int rk = r - k;
            const int pk = p - k;

            if (r >= k) {
                a(s2, 0) = divide_if_resolved(a(s1, 0), ndu(pk + 1, rk), interval_tol);
                d = a(s2, 0) * ndu(rk, pk);
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : p - r;

            for (int j = j1; j <= j2; ++j) {
                a(s2, j) = divide_if_resolved(a(s1, j) - a(s1, j - 1),
                                               ndu(pk + 1, rk + j),
                                               interval_tol);
                d += a(s2, j) * ndu(rk + j, pk);
            }

            if (r <= pk) {
                a(s2, k) = divide_if_resolved(-a(s1, k - 1),
                                               ndu(pk + 1, r),
                                               interval_tol);
                d += a(s2, k) * ndu(r, pk);
            }

            ders(k, r) = d;
            std::swap(s1, s2);
        }
    }

    int scale = p;
    for (int k = 1; k <= max_derivative; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders(k, j) *= Real(scale);
        }
        scale *= (p - k);
    }
}

} // namespace

BSplineBasis::BSplineBasis(int degree, std::vector<Real> knots)
    : degree_(degree),
      knots_(std::move(knots)) {
    if (degree_ < 0) {
        throw BasisConfigurationException("BSplineBasis requires non-negative degree",
                                          __FILE__, __LINE__, __func__);
    }
    if (knots_.size() < static_cast<std::size_t>(degree_) + 2) {
        throw BasisConfigurationException("BSplineBasis: knot vector too short for degree",
                                          __FILE__, __LINE__, __func__);
    }
    if (!std::is_sorted(knots_.begin(), knots_.end())) {
        throw BasisConfigurationException("BSplineBasis: knot vector must be non-decreasing",
                                          __FILE__, __LINE__, __func__);
    }

    num_basis_ = knots_.size() - static_cast<std::size_t>(degree_) - 1;
    if (num_basis_ == 0u) {
        throw BasisConfigurationException("BSplineBasis: invalid number of basis functions",
                                          __FILE__, __LINE__, __func__);
    }

    u_min_ = knots_[static_cast<std::size_t>(degree_)];
    u_max_ = knots_[num_basis_];
    if (!(u_max_ > u_min_)) {
        throw BasisConfigurationException("BSplineBasis: invalid parametric domain from knots",
                                          __FILE__, __LINE__, __func__);
    }
    rebuild_cache_identity();
}

BSplineBasis::BSplineBasis(int degree, std::vector<Real> knots, std::vector<Real> weights)
    : BSplineBasis(degree, std::move(knots)) {
    if (weights.size() != num_basis_) {
        throw BasisConfigurationException("BSplineBasis: weights size must equal number of basis functions",
                                          __FILE__, __LINE__, __func__);
    }
    for (Real weight : weights) {
        if (!std::isfinite(weight) || weight <= Real(0)) {
            throw BasisConfigurationException("BSplineBasis: rational weights must be finite and positive",
                                              __FILE__, __LINE__, __func__);
        }
    }
    semantic_type_ = BasisType::NURBS;
    weights_ = std::move(weights);
    rebuild_cache_identity();
}

void BSplineBasis::rebuild_cache_identity() {
    std::ostringstream oss;
    oss << BasisFunction::cache_identity()
        << "|degree=" << degree_
        << "|knots=" << knots_.size()
        << "|weights=" << weights_.size();

    oss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    for (Real knot : knots_) {
        oss << "|k=" << knot;
    }
    for (Real weight : weights_) {
        oss << "|w=" << weight;
    }
    cache_identity_ = oss.str();

    cache_identity_words_.clear();
    cache_identity_words_.reserve(6u + knots_.size() + weights_.size());
    cache_identity_words_.push_back(0x4253506c696e6531ULL); // "BSpline1"
    cache_identity_words_.push_back(static_cast<std::uint64_t>(semantic_type_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(degree_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(num_basis_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(knots_.size()));
    for (Real knot : knots_) {
        cache_identity_words_.push_back(real_identity_word(knot));
    }
    cache_identity_words_.push_back(static_cast<std::uint64_t>(weights_.size()));
    for (Real weight : weights_) {
        cache_identity_words_.push_back(real_identity_word(weight));
    }
    const auto fingerprint = compute_basis_identity_fingerprint(cache_identity_words_);
    cache_identity_hash_a_ = fingerprint.hash_a;
    cache_identity_hash_b_ = fingerprint.hash_b;
}

std::string BSplineBasis::cache_identity() const {
    return cache_identity_;
}

bool BSplineBasis::cache_identity_words(std::vector<std::uint64_t>& words) const {
    words.insert(words.end(), cache_identity_words_.begin(), cache_identity_words_.end());
    return true;
}

bool BSplineBasis::cache_identity_fingerprint(std::uint64_t& hash_a,
                                              std::uint64_t& hash_b) const {
    hash_a = cache_identity_hash_a_;
    hash_b = cache_identity_hash_b_;
    return !cache_identity_words_.empty();
}

void BSplineBasis::evaluate_point_strided(const math::Vector<Real, 3>& xi,
                                          std::size_t output_stride,
                                          std::size_t q,
                                          Real* SVMP_RESTRICT values_out,
                                          Real* SVMP_RESTRICT gradients_out,
                                          Real* SVMP_RESTRICT hessians_out,
                                          OutputInitialization initialization) const {
    const bool clear_outputs = initialization == OutputInitialization::ClearAllRequestedRows;
    for (std::size_t i = 0; clear_outputs && values_out != nullptr && i < num_basis_; ++i) {
        values_out[i * output_stride + q] = Real(0);
    }
    for (std::size_t i = 0; clear_outputs && gradients_out != nullptr && i < num_basis_; ++i) {
        for (std::size_t component = 0; component < 3u; ++component) {
            gradients_out[(i * 3u + component) * output_stride + q] = Real(0);
        }
    }
    for (std::size_t i = 0; clear_outputs && hessians_out != nullptr && i < num_basis_; ++i) {
        for (std::size_t component = 0; component < 9u; ++component) {
            hessians_out[(i * 9u + component) * output_stride + q] = Real(0);
        }
    }
    if (num_basis_ == 0) {
        return;
    }

    static thread_local std::vector<Real> active_values;
    static thread_local std::vector<Real> active_first;
    static thread_local std::vector<Real> active_second;

    const auto active = evaluate_active_support(
        xi,
        active_values,
        (gradients_out != nullptr || hessians_out != nullptr) ? &active_first : nullptr,
        hessians_out != nullptr ? &active_second : nullptr);

    for (std::size_t offset = 0; offset < active.count; ++offset) {
        const std::size_t i = active.first_index + offset;
        if (values_out != nullptr) {
            values_out[i * output_stride + q] = active_values[offset];
        }
        if (gradients_out != nullptr) {
            gradients_out[(i * 3u) * output_stride + q] = active_first[offset];
        }
        if (hessians_out != nullptr) {
            hessians_out[(i * 9u) * output_stride + q] = active_second[offset];
        }
    }
}

void BSplineBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                  std::vector<Real>& values) const {
    values.resize(num_basis_);
    if (num_basis_ == 0) {
        return;
    }
    evaluate_point_strided(xi, 1u, 0u, values.data(), nullptr, nullptr);
}

void BSplineBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                      std::vector<Gradient>& gradients) const {
    gradients.assign(num_basis_, Gradient{});
    if (num_basis_ == 0) {
        return;
    }
    auto& scratch = bspline_derivative_scratch();
    const auto active = evaluate_active_support(xi, scratch.values, &scratch.first, nullptr);
    for (std::size_t offset = 0; offset < active.count; ++offset) {
        gradients[active.first_index + offset][0] = scratch.first[offset];
    }
}

void BSplineBasis::evaluate_values_and_gradients(const math::Vector<Real, 3>& xi,
                                                 std::vector<Real>& values,
                                                 std::vector<Gradient>& gradients) const {
    values.assign(num_basis_, Real(0));
    gradients.assign(num_basis_, Gradient{});
    if (num_basis_ == 0) {
        return;
    }

    auto& scratch = bspline_derivative_scratch();
    const auto active = evaluate_active_support(xi, scratch.values, &scratch.first, nullptr);
    for (std::size_t offset = 0; offset < active.count; ++offset) {
        const std::size_t i = active.first_index + offset;
        values[i] = scratch.values[offset];
        gradients[i][0] = scratch.first[offset];
    }
}

BSplineBasis::ActiveSupportRange BSplineBasis::evaluate_active_support(
    const math::Vector<Real, 3>& xi,
    std::vector<Real>& values,
    std::vector<Real>* first_derivatives,
    std::vector<Real>* second_derivatives) const {
    values.clear();
    if (first_derivatives != nullptr) {
        first_derivatives->clear();
    }
    if (second_derivatives != nullptr) {
        second_derivatives->clear();
    }
    if (num_basis_ == 0) {
        return {};
    }

    const bool need_first = first_derivatives != nullptr || second_derivatives != nullptr;
    const bool need_second = second_derivatives != nullptr;
    const int max_derivative =
        need_second ? std::min(2, degree_) : (need_first ? std::min(1, degree_) : 0);

    const Real xi0 = snap_reference_coordinate(xi[0]);
    const Real du_dxi = (u_max_ - u_min_) * Real(0.5);
    const Real du_dxi_sq = du_dxi * du_dxi;
    const Real u = u_min_ + (xi0 + Real(1)) * Real(0.5) * (u_max_ - u_min_);
    const int span = find_span_index(knots_, degree_, num_basis_, u);
    const int first_span_index = span - degree_;
    const std::size_t first_index =
        first_span_index <= 0 ? 0u : static_cast<std::size_t>(first_span_index);
    const std::size_t last_index =
        std::min(num_basis_, static_cast<std::size_t>(span + 1));
    const ActiveSupportRange range{first_index, last_index - first_index};

    values.resize(range.count);
    auto& scratch = bspline_derivative_scratch();
    basis_funs_and_derivatives_flat(knots_, degree_, span, u, max_derivative, scratch);

    auto& first_local = scratch.first;
    auto& second_local = scratch.second;
    if (need_first) {
        first_local.resize(range.count);
        if (max_derivative == 0) {
            std::fill(first_local.begin(), first_local.end(), Real(0));
        }
    }
    if (need_second) {
        second_local.resize(range.count);
        if (max_derivative < 2) {
            std::fill(second_local.begin(), second_local.end(), Real(0));
        }
    }

    const std::size_t local_count = static_cast<std::size_t>(degree_ + 1);
    for (int r = 0; r <= degree_; ++r) {
        const int global = first_span_index + r;
        if (global < 0 || global >= static_cast<int>(num_basis_)) {
            continue;
        }
        const std::size_t global_index = static_cast<std::size_t>(global);
        const std::size_t output_index = global_index - range.first_index;
        const std::size_t local_index = static_cast<std::size_t>(r);
        values[output_index] = scratch.ders[local_index];
        if (max_derivative >= 1) {
            first_local[output_index] =
                scratch.ders[local_count + local_index] * du_dxi;
        }
        if (max_derivative >= 2) {
            second_local[output_index] =
                scratch.ders[2u * local_count + local_index] * du_dxi_sq;
        }
    }

    if (weights_.empty()) {
        if (first_derivatives != nullptr) {
            first_derivatives->resize(range.count);
            std::copy(first_local.begin(), first_local.end(), first_derivatives->begin());
        }
        if (second_derivatives != nullptr) {
            second_derivatives->resize(range.count);
            std::copy(second_local.begin(), second_local.end(), second_derivatives->begin());
        }
        return range;
    }

    Real W = Real(0);
    Real scale = Real(0);
    Real dW = Real(0);
    Real ddW = Real(0);
    for (std::size_t offset = 0; offset < range.count; ++offset) {
        const std::size_t global = range.first_index + offset;
        const Real weighted_value = values[offset] * weights_[global];
        W += weighted_value;
        scale += std::abs(weighted_value);
        if (need_first) {
            dW += first_local[offset] * weights_[global];
        }
        if (need_second) {
            ddW += second_local[offset] * weights_[global];
        }
    }

    validate_rational_denominator(W, scale, "active-support evaluation");

    const Real inv_W = Real(1) / W;
    const Real inv_W2 = inv_W * inv_W;
    const Real inv_W3 = inv_W2 * inv_W;
    if (first_derivatives != nullptr) {
        first_derivatives->resize(range.count);
    }
    if (second_derivatives != nullptr) {
        second_derivatives->resize(range.count);
    }
    for (std::size_t offset = 0; offset < range.count; ++offset) {
        const std::size_t global = range.first_index + offset;
        const Real wi = weights_[global];
        const Real A = values[offset] * wi;
        const Real dA = need_first ? first_local[offset] * wi : Real(0);
        const Real ddA = need_second ? second_local[offset] * wi : Real(0);
        values[offset] = A * inv_W;
        if (first_derivatives != nullptr) {
            (*first_derivatives)[offset] = (dA * W - A * dW) * inv_W2;
        }
        if (second_derivatives != nullptr) {
            (*second_derivatives)[offset] =
                (ddA * W * W - A * ddW * W - Real(2) * dA * W * dW +
                 Real(2) * A * dW * dW) *
                inv_W3;
        }
    }
    return range;
}

void BSplineBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                     std::vector<Hessian>& hessians) const {
    hessians.assign(num_basis_, Hessian{});
    if (num_basis_ == 0 || degree_ == 0) {
        return;
    }

    auto& scratch = bspline_derivative_scratch();
    const auto active = evaluate_active_support(xi, scratch.values, &scratch.first, &scratch.second);
    for (std::size_t offset = 0; offset < active.count; ++offset) {
        hessians[active.first_index + offset](0, 0) = scratch.second[offset];
    }
}

void BSplineBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                std::vector<Real>& values,
                                std::vector<Gradient>& gradients,
                                std::vector<Hessian>& hessians) const {
    values.assign(num_basis_, Real(0));
    gradients.assign(num_basis_, Gradient{});
    hessians.assign(num_basis_, Hessian{});
    if (num_basis_ == 0) {
        return;
    }

    auto& scratch = bspline_derivative_scratch();
    const auto active = evaluate_active_support(xi, scratch.values, &scratch.first, &scratch.second);
    for (std::size_t offset = 0; offset < active.count; ++offset) {
        const std::size_t i = active.first_index + offset;
        values[i] = scratch.values[offset];
        gradients[i][0] = scratch.first[offset];
        hessians[i](0, 0) = scratch.second[offset];
    }
}

void BSplineBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                      Real* SVMP_RESTRICT values_out) const {
    evaluate_point_strided(xi, 1u, 0u, values_out, nullptr, nullptr);
}

void BSplineBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                         Real* SVMP_RESTRICT gradients_out) const {
    evaluate_point_strided(xi, 1u, 0u, nullptr, gradients_out, nullptr);
}

void BSplineBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                        Real* SVMP_RESTRICT hessians_out) const {
    evaluate_point_strided(xi, 1u, 0u, nullptr, nullptr, hessians_out);
}

void BSplineBasis::evaluate_at_quadrature_points(
    const std::vector<math::Vector<Real, 3>>& points,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    evaluate_at_quadrature_points_strided(points,
                                          points.size(),
                                          values_out,
                                          gradients_out,
                                          hessians_out);
}

void BSplineBasis::evaluate_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    if (points.empty()) {
        return;
    }
    if (output_stride < points.size()) {
        throw BasisEvaluationException("BSplineBasis: output stride is smaller than quadrature point count",
                                       __FILE__, __LINE__, __func__);
    }
    for (std::size_t q = 0; q < points.size(); ++q) {
        evaluate_point_strided(points[q],
                               output_stride,
                               q,
                               values_out,
                               gradients_out,
                               hessians_out);
    }
}

void BSplineBasis::fill_scalar_cache_entry(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    if (output_stride < points.size()) {
        throw BasisEvaluationException("BSplineBasis: output stride is smaller than quadrature point count",
                                       __FILE__, __LINE__, __func__);
    }
    for (std::size_t q = 0; q < points.size(); ++q) {
        evaluate_point_strided(points[q],
                               output_stride,
                               q,
                               values_out,
                               gradients_out,
                               hessians_out,
                               OutputInitialization::CallerPrecleared);
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
