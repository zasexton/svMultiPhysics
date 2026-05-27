/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/NURBSTensorBasis.h"

#include <algorithm>
#include <array>
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

enum class OutputInitialization {
    ClearAllRequestedRows,
    CallerPrecleared
};

struct AxisEvaluationScratch {
    std::vector<Real> values;
    std::vector<Real> first_derivatives;
    std::vector<Real> second_derivatives;
    std::size_t first_active{0};
    std::size_t active_count{0};
};

struct NURBSTensorEvaluationScratch {
    std::array<AxisEvaluationScratch, 3> axes;
};

NURBSTensorEvaluationScratch& nurbs_tensor_scratch() {
    static thread_local NURBSTensorEvaluationScratch scratch;
    return scratch;
}

math::Vector<Real, 3> axis_coordinate(const math::Vector<Real, 3>& xi, int axis) {
    math::Vector<Real, 3> coord{};
    coord[0] = xi[static_cast<std::size_t>(axis)];
    return coord;
}

Real denominator_tolerance(Real scale) {
    return std::numeric_limits<Real>::epsilon() * Real(128) *
           std::max(scale, std::numeric_limits<Real>::min());
}

void validate_rational_denominator(Real denominator, Real scale, const char* operation) {
    if (!std::isfinite(denominator) ||
        std::abs(denominator) <= denominator_tolerance(scale)) {
        throw BasisEvaluationException(std::string("NURBSTensorBasis: invalid rational denominator during ") +
                                           operation,
                                       __FILE__, __LINE__, __func__);
    }
}

std::uint64_t real_identity_word(Real value) noexcept {
    if (value == Real(0)) {
        value = Real(0);
    }
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(Real));
    return bits;
}

struct NURBSTensorVectorWriter {
    std::vector<Real>* values{nullptr};
    std::vector<Gradient>* gradients{nullptr};
    std::vector<Hessian>* hessians{nullptr};

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void reset(std::size_t size) const {
        if (values != nullptr) {
            values->assign(size, Real(0));
        }
        if (gradients != nullptr) {
            gradients->assign(size, Gradient{});
        }
        if (hessians != nullptr) {
            hessians->assign(size, Hessian{});
        }
    }

    void value(std::size_t idx, Real v) const {
        (*values)[idx] = v;
    }

    void gradient(std::size_t idx, int component, Real v) const {
        (*gradients)[idx][static_cast<std::size_t>(component)] = v;
    }

    void hessian(std::size_t idx, int row, int col, Real v) const {
        (*hessians)[idx](static_cast<std::size_t>(row),
                         static_cast<std::size_t>(col)) = v;
    }
};

struct NURBSTensorRawWriter {
    std::size_t q{0};
    std::size_t stride{1};
    Real* values{nullptr};
    Real* gradients{nullptr};
    Real* hessians{nullptr};
    OutputInitialization initialization{OutputInitialization::ClearAllRequestedRows};

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void reset(std::size_t size) const {
        if (initialization == OutputInitialization::CallerPrecleared) {
            return;
        }
        if (values != nullptr) {
            for (std::size_t dof = 0; dof < size; ++dof) {
                values[dof * stride + q] = Real(0);
            }
        }
        if (gradients != nullptr) {
            for (std::size_t dof = 0; dof < size; ++dof) {
                for (std::size_t component = 0; component < 3u; ++component) {
                    gradients[(dof * 3u + component) * stride + q] = Real(0);
                }
            }
        }
        if (hessians != nullptr) {
            for (std::size_t dof = 0; dof < size; ++dof) {
                for (std::size_t component = 0; component < 9u; ++component) {
                    hessians[(dof * 9u + component) * stride + q] = Real(0);
                }
            }
        }
    }

    void value(std::size_t idx, Real v) const {
        values[idx * stride + q] = v;
    }

    void gradient(std::size_t idx, int component, Real v) const {
        gradients[(idx * 3u + static_cast<std::size_t>(component)) * stride + q] = v;
    }

    void hessian(std::size_t idx, int row, int col, Real v) const {
        hessians[(idx * 9u + static_cast<std::size_t>(row * 3 + col)) * stride + q] = v;
    }
};

struct NURBSTensorCompactWriter {
    std::vector<std::size_t>* indices{nullptr};
    std::vector<Real>* values{nullptr};
    std::vector<Gradient>* gradients{nullptr};
    std::vector<Hessian>* hessians{nullptr};
    mutable std::size_t last_index{std::numeric_limits<std::size_t>::max()};
    mutable std::size_t last_position{0};

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void reset(std::size_t) const {
        indices->clear();
        if (values != nullptr) {
            values->clear();
        }
        if (gradients != nullptr) {
            gradients->clear();
        }
        if (hessians != nullptr) {
            hessians->clear();
        }
        last_index = std::numeric_limits<std::size_t>::max();
        last_position = 0;
    }

    std::size_t ensure_position(std::size_t idx) const {
        if (last_index == idx) {
            return last_position;
        }
        indices->push_back(idx);
        if (values != nullptr) {
            values->push_back(Real(0));
        }
        if (gradients != nullptr) {
            gradients->push_back(Gradient{});
        }
        if (hessians != nullptr) {
            hessians->push_back(Hessian{});
        }
        last_index = idx;
        last_position = indices->size() - 1u;
        return last_position;
    }

    void value(std::size_t idx, Real v) const {
        (*values)[ensure_position(idx)] = v;
    }

    void gradient(std::size_t idx, int component, Real v) const {
        (*gradients)[ensure_position(idx)][static_cast<std::size_t>(component)] = v;
    }

    void hessian(std::size_t idx, int row, int col, Real v) const {
        (*hessians)[ensure_position(idx)](static_cast<std::size_t>(row),
                                          static_cast<std::size_t>(col)) = v;
    }
};

template <typename Writer>
void evaluate_nurbs_tensor_active_support(const NURBSTensorBasis& basis,
                                          const math::Vector<Real, 3>& xi,
                                          const Writer& writer) {
    const bool want_values = writer.wants_values();
    const bool want_gradients = writer.wants_gradients();
    const bool want_hessians = writer.wants_hessians();
    const bool need_gradients = want_gradients || want_hessians;
    const int dimension = basis.dimension();

    writer.reset(basis.size());

    auto& scratch = nurbs_tensor_scratch();
    for (int axis = 0; axis < dimension; ++axis) {
        auto& axis_data = scratch.axes[static_cast<std::size_t>(axis)];
        const math::Vector<Real, 3> coord = axis_coordinate(xi, axis);
        const auto& axis_basis = basis.axis_basis(axis);
        BSplineBasis::ActiveSupportRange axis_range;
        if (want_hessians) {
            axis_range = axis_basis.evaluate_active_support(
                coord,
                axis_data.values,
                &axis_data.first_derivatives,
                &axis_data.second_derivatives);
        } else if (need_gradients) {
            axis_range = axis_basis.evaluate_active_support(
                coord,
                axis_data.values,
                &axis_data.first_derivatives,
                nullptr);
            axis_data.second_derivatives.clear();
        } else {
            axis_range = axis_basis.evaluate_active_support(
                coord,
                axis_data.values,
                nullptr,
                nullptr);
            axis_data.first_derivatives.clear();
            axis_data.second_derivatives.clear();
        }
        axis_data.first_active = axis_range.first_index;
        axis_data.active_count = axis_range.count;
    }

    Real denom = Real(0);
    Real scale = Real(0);
    Gradient denom_gradient{};
    Hessian denom_hessian{};

    const auto& weights = basis.weights();
    const auto& axis_sizes = basis.axis_sizes();
    const auto& ax = scratch.axes[0];
    const auto& ay = scratch.axes[1];
    const std::size_t nx = axis_sizes[0];
    const std::size_t ny = axis_sizes[1];
    const std::size_t ix_end = ax.first_active + ax.active_count;
    const std::size_t iy_end = ay.first_active + ay.active_count;

    if (dimension == 2) {
        for (std::size_t j = ay.first_active; j < iy_end; ++j) {
            const std::size_t jy = j - ay.first_active;
            const Real vy = ay.values[jy];
            const Real gy = need_gradients ? ay.first_derivatives[jy] : Real(0);
            const Real hy = want_hessians ? ay.second_derivatives[jy] : Real(0);
            for (std::size_t i = ax.first_active; i < ix_end; ++i) {
                const std::size_t ix = i - ax.first_active;
                const std::size_t idx = j * nx + i;
                const Real weight = weights[idx];
                const Real vx = ax.values[ix];
                const Real A = vx * vy * weight;
                denom += A;
                scale += std::abs(A);

                if (need_gradients) {
                    const Real gx = ax.first_derivatives[ix];
                    denom_gradient[0] += gx * vy * weight;
                    denom_gradient[1] += vx * gy * weight;
                    if (want_hessians) {
                        const Real hx = ax.second_derivatives[ix];
                        const Real hxy = gx * gy;
                        denom_hessian(0, 0) += hx * vy * weight;
                        denom_hessian(1, 1) += vx * hy * weight;
                        denom_hessian(0, 1) += hxy * weight;
                        denom_hessian(1, 0) += hxy * weight;
                    }
                }
            }
        }

        validate_rational_denominator(denom, scale, "active-support evaluation");

        const Real inv_denom = Real(1) / denom;
        const Real inv_denom_sq = inv_denom * inv_denom;
        const Real inv_denom_cu = inv_denom_sq * inv_denom;
        const Real denom_sq = denom * denom;
        for (std::size_t j = ay.first_active; j < iy_end; ++j) {
            const std::size_t jy = j - ay.first_active;
            const Real vy = ay.values[jy];
            const Real gy = need_gradients ? ay.first_derivatives[jy] : Real(0);
            const Real hy = want_hessians ? ay.second_derivatives[jy] : Real(0);
            for (std::size_t i = ax.first_active; i < ix_end; ++i) {
                const std::size_t ix = i - ax.first_active;
                const std::size_t idx = j * nx + i;
                const Real weight = weights[idx];
                const Real vx = ax.values[ix];
                const Real A = vx * vy * weight;
                if (want_values) {
                    writer.value(idx, A * inv_denom);
                }
                if (need_gradients) {
                    const Real gx = ax.first_derivatives[ix];
                    const Real dA0 = gx * vy * weight;
                    const Real dA1 = vx * gy * weight;
                    if (want_gradients) {
                        writer.gradient(idx, 0, (dA0 * denom - A * denom_gradient[0]) * inv_denom_sq);
                        writer.gradient(idx, 1, (dA1 * denom - A * denom_gradient[1]) * inv_denom_sq);
                    }
                    if (want_hessians) {
                        const Real hx = ax.second_derivatives[ix];
                        const Real ddA00 = hx * vy * weight;
                        const Real ddA11 = vx * hy * weight;
                        const Real ddA01 = gx * gy * weight;
                        writer.hessian(idx, 0, 0,
                                       (ddA00 * denom_sq
                                        - A * denom_hessian(0, 0) * denom
                                        - Real(2) * dA0 * denom * denom_gradient[0]
                                        + Real(2) * A * denom_gradient[0] * denom_gradient[0]) *
                                       inv_denom_cu);
                        writer.hessian(idx, 1, 1,
                                       (ddA11 * denom_sq
                                        - A * denom_hessian(1, 1) * denom
                                        - Real(2) * dA1 * denom * denom_gradient[1]
                                        + Real(2) * A * denom_gradient[1] * denom_gradient[1]) *
                                       inv_denom_cu);
                        const Real mixed =
                            (ddA01 * denom_sq
                             - A * denom_hessian(0, 1) * denom
                             - dA0 * denom * denom_gradient[1]
                             - dA1 * denom * denom_gradient[0]
                             + Real(2) * A * denom_gradient[0] * denom_gradient[1]) *
                            inv_denom_cu;
                        writer.hessian(idx, 0, 1, mixed);
                        writer.hessian(idx, 1, 0, mixed);
                    }
                }
            }
        }
        return;
    }

    const auto& az = scratch.axes[2];
    const std::size_t iz_end = az.first_active + az.active_count;

    for (std::size_t k = az.first_active; k < iz_end; ++k) {
        const std::size_t kz = k - az.first_active;
        const Real vz = az.values[kz];
        const Real gz = need_gradients ? az.first_derivatives[kz] : Real(0);
        const Real hz = want_hessians ? az.second_derivatives[kz] : Real(0);
        for (std::size_t j = ay.first_active; j < iy_end; ++j) {
            const std::size_t jy = j - ay.first_active;
            const Real vy = ay.values[jy];
            const Real gy = need_gradients ? ay.first_derivatives[jy] : Real(0);
            const Real hy = want_hessians ? ay.second_derivatives[jy] : Real(0);
            for (std::size_t i = ax.first_active; i < ix_end; ++i) {
                const std::size_t ix = i - ax.first_active;
                const std::size_t idx = (k * ny + j) * nx + i;
                const Real weight = weights[idx];
                const Real vx = ax.values[ix];
                const Real A = vx * vy * vz * weight;
                denom += A;
                scale += std::abs(A);

                if (need_gradients) {
                    const Real gx = ax.first_derivatives[ix];
                    denom_gradient[0] += gx * vy * vz * weight;
                    denom_gradient[1] += vx * gy * vz * weight;
                    denom_gradient[2] += vx * vy * gz * weight;
                    if (want_hessians) {
                        const Real hx = ax.second_derivatives[ix];
                        denom_hessian(0, 0) += hx * vy * vz * weight;
                        denom_hessian(1, 1) += vx * hy * vz * weight;
                        denom_hessian(2, 2) += vx * vy * hz * weight;
                        const Real hxy = gx * gy * vz;
                        const Real hxz = gx * vy * gz;
                        const Real hyz = vx * gy * gz;
                        denom_hessian(0, 1) += hxy * weight;
                        denom_hessian(1, 0) += hxy * weight;
                        denom_hessian(0, 2) += hxz * weight;
                        denom_hessian(2, 0) += hxz * weight;
                        denom_hessian(1, 2) += hyz * weight;
                        denom_hessian(2, 1) += hyz * weight;
                    }
                }
            }
        }
    }

    validate_rational_denominator(denom, scale, "active-support evaluation");

    const Real inv_denom = Real(1) / denom;
    const Real inv_denom_sq = inv_denom * inv_denom;
    const Real inv_denom_cu = inv_denom_sq * inv_denom;
    const Real denom_sq = denom * denom;
    for (std::size_t k = az.first_active; k < iz_end; ++k) {
        const std::size_t kz = k - az.first_active;
        const Real vz = az.values[kz];
        const Real gz = need_gradients ? az.first_derivatives[kz] : Real(0);
        const Real hz = want_hessians ? az.second_derivatives[kz] : Real(0);
        for (std::size_t j = ay.first_active; j < iy_end; ++j) {
            const std::size_t jy = j - ay.first_active;
            const Real vy = ay.values[jy];
            const Real gy = need_gradients ? ay.first_derivatives[jy] : Real(0);
            const Real hy = want_hessians ? ay.second_derivatives[jy] : Real(0);
            for (std::size_t i = ax.first_active; i < ix_end; ++i) {
                const std::size_t ix = i - ax.first_active;
                const std::size_t idx = (k * ny + j) * nx + i;
                const Real weight = weights[idx];
                const Real vx = ax.values[ix];
                const Real A = vx * vy * vz * weight;
                if (want_values) {
                    writer.value(idx, A * inv_denom);
                }
                if (need_gradients) {
                    const Real gx = ax.first_derivatives[ix];
                    const Real dA0 = gx * vy * vz * weight;
                    const Real dA1 = vx * gy * vz * weight;
                    const Real dA2 = vx * vy * gz * weight;
                    if (want_gradients) {
                        writer.gradient(idx, 0, (dA0 * denom - A * denom_gradient[0]) * inv_denom_sq);
                        writer.gradient(idx, 1, (dA1 * denom - A * denom_gradient[1]) * inv_denom_sq);
                        writer.gradient(idx, 2, (dA2 * denom - A * denom_gradient[2]) * inv_denom_sq);
                    }
                    if (want_hessians) {
                        const Real hx = ax.second_derivatives[ix];
                        const Real ddA[3][3] = {
                            {hx * vy * vz * weight, gx * gy * vz * weight, gx * vy * gz * weight},
                            {gx * gy * vz * weight, vx * hy * vz * weight, vx * gy * gz * weight},
                            {gx * vy * gz * weight, vx * gy * gz * weight, vx * vy * hz * weight}
                        };
                        const Real dA[3] = {dA0, dA1, dA2};
                        for (int a = 0; a < 3; ++a) {
                            const std::size_t sa = static_cast<std::size_t>(a);
                            for (int b = 0; b < 3; ++b) {
                                const std::size_t sb = static_cast<std::size_t>(b);
                                writer.hessian(
                                    idx, a, b,
                                    (ddA[a][b] * denom_sq
                                     - A * denom_hessian(sa, sb) * denom
                                     - dA[a] * denom * denom_gradient[sb]
                                     - dA[b] * denom * denom_gradient[sa]
                                     + Real(2) * A * denom_gradient[sa] * denom_gradient[sb]) *
                                    inv_denom_cu);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace

NURBSTensorBasis::NURBSTensorBasis(BSplineBasis bx,
                                   BSplineBasis by,
                                   std::vector<Real> weights,
                                   std::vector<int> tensor_extents) {
    initialize({std::move(bx), std::move(by)}, std::move(weights), std::move(tensor_extents));
}

NURBSTensorBasis::NURBSTensorBasis(BSplineBasis bx,
                                   BSplineBasis by,
                                   BSplineBasis bz,
                                   std::vector<Real> weights,
                                   std::vector<int> tensor_extents) {
    initialize({std::move(bx), std::move(by), std::move(bz)}, std::move(weights), std::move(tensor_extents));
}

void NURBSTensorBasis::initialize(std::vector<BSplineBasis> axes,
                                  std::vector<Real> weights,
                                  std::vector<int> tensor_extents) {
    if (axes.size() != 2u && axes.size() != 3u) {
        throw BasisConfigurationException("NURBSTensorBasis requires 2D or 3D tensor-product axes",
                                          __FILE__, __LINE__, __func__);
    }

    dimension_ = static_cast<int>(axes.size());
    element_type_ = (dimension_ == 2) ? ElementType::Quad4 : ElementType::Hex8;
    order_ = 0;
    size_ = 1u;

    axes_ = std::move(axes);
    axis_sizes_.resize(static_cast<std::size_t>(dimension_));
    tensor_extents_.resize(static_cast<std::size_t>(dimension_));

    for (int axis = 0; axis < dimension_; ++axis) {
        if (axes_[static_cast<std::size_t>(axis)].is_rational()) {
            throw BasisConfigurationException("NURBSTensorBasis expects non-rational BSpline axes and a separate control-net weight array",
                                              __FILE__, __LINE__, __func__);
        }

        const std::size_t axis_size = axes_[static_cast<std::size_t>(axis)].size();
        axis_sizes_[static_cast<std::size_t>(axis)] = axis_size;
        order_ = std::max(order_, axes_[static_cast<std::size_t>(axis)].order());
        size_ *= axis_size;
    }

    if (weights.size() != size_) {
        throw BasisConfigurationException("NURBSTensorBasis: weights size must match tensor-product basis size",
                                          __FILE__, __LINE__, __func__);
    }
    for (Real weight : weights) {
        if (!std::isfinite(weight) || weight <= Real(0)) {
            throw BasisConfigurationException("NURBSTensorBasis: rational weights must be finite and positive",
                                              __FILE__, __LINE__, __func__);
        }
    }

    if (!tensor_extents.empty()) {
        if (tensor_extents.size() != static_cast<std::size_t>(dimension_)) {
            throw BasisConfigurationException("NURBSTensorBasis: tensor_extents must match tensor dimension",
                                              __FILE__, __LINE__, __func__);
        }
        for (int axis = 0; axis < dimension_; ++axis) {
            if (tensor_extents[static_cast<std::size_t>(axis)] != static_cast<int>(axis_sizes_[static_cast<std::size_t>(axis)])) {
                throw BasisConfigurationException("NURBSTensorBasis: tensor_extents must agree with spline axis basis sizes",
                                                  __FILE__, __LINE__, __func__);
            }
        }
        tensor_extents_ = std::move(tensor_extents);
    } else {
        for (int axis = 0; axis < dimension_; ++axis) {
            tensor_extents_[static_cast<std::size_t>(axis)] =
                static_cast<int>(axis_sizes_[static_cast<std::size_t>(axis)]);
        }
    }

    weights_ = std::move(weights);
    rebuild_cache_identity();
}

void NURBSTensorBasis::rebuild_cache_identity() {
    std::ostringstream oss;
    oss << BasisFunction::cache_identity()
        << "|axes=" << dimension_;
    for (int axis = 0; axis < dimension_; ++axis) {
        oss << "|axis" << axis << '=' << axes_[static_cast<std::size_t>(axis)].cache_identity()
            << "|extent" << axis << '=' << tensor_extents_[static_cast<std::size_t>(axis)];
    }

    oss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    for (Real weight : weights_) {
        oss << "|w=" << weight;
    }
    cache_identity_ = oss.str();

    cache_identity_words_.clear();
    cache_identity_words_.reserve(8u + axes_.size() * 16u + weights_.size());
    cache_identity_words_.push_back(0x4e5552425354656eULL); // "NURBSTen"
    cache_identity_words_.push_back(static_cast<std::uint64_t>(dimension_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(order_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(size_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(axes_.size()));
    for (std::size_t axis = 0; axis < axes_.size(); ++axis) {
        std::vector<std::uint64_t> axis_words;
        if (!axes_[axis].cache_identity_words(axis_words)) {
            cache_identity_words_.clear();
            cache_identity_hash_a_ = 0;
            cache_identity_hash_b_ = 0;
            return;
        }
        cache_identity_words_.push_back(static_cast<std::uint64_t>(axis_words.size()));
        cache_identity_words_.insert(cache_identity_words_.end(), axis_words.begin(), axis_words.end());
        cache_identity_words_.push_back(static_cast<std::uint64_t>(tensor_extents_[axis]));
    }
    cache_identity_words_.push_back(static_cast<std::uint64_t>(weights_.size()));
    for (Real weight : weights_) {
        cache_identity_words_.push_back(real_identity_word(weight));
    }
    const auto fingerprint = compute_basis_identity_fingerprint(cache_identity_words_);
    cache_identity_hash_a_ = fingerprint.hash_a;
    cache_identity_hash_b_ = fingerprint.hash_b;
}

std::string NURBSTensorBasis::cache_identity() const {
    return cache_identity_;
}

bool NURBSTensorBasis::cache_identity_words(std::vector<std::uint64_t>& words) const {
    if (cache_identity_words_.empty()) {
        return false;
    }
    words.insert(words.end(), cache_identity_words_.begin(), cache_identity_words_.end());
    return true;
}

bool NURBSTensorBasis::cache_identity_fingerprint(std::uint64_t& hash_a,
                                                  std::uint64_t& hash_b) const {
    if (cache_identity_words_.empty()) {
        return false;
    }
    hash_a = cache_identity_hash_a_;
    hash_b = cache_identity_hash_b_;
    return true;
}

NURBSTensorBasis::ActiveTensorSupportRange NURBSTensorBasis::active_tensor_support(
    const math::Vector<Real, 3>& xi) const {
    ActiveTensorSupportRange range;
    for (int axis = 0; axis < dimension_; ++axis) {
        std::vector<Real> axis_values;
        const auto axis_range =
            axes_[static_cast<std::size_t>(axis)].evaluate_active_support(
                axis_coordinate(xi, axis), axis_values);
        range.first_indices[static_cast<std::size_t>(axis)] = axis_range.first_index;
        range.counts[static_cast<std::size_t>(axis)] = axis_range.count;
    }
    return range;
}

NURBSTensorBasis::ActiveTensorSupportRange NURBSTensorBasis::evaluate_active_support(
    const math::Vector<Real, 3>& xi,
    std::vector<std::size_t>& global_indices,
    std::vector<Real>* values,
    std::vector<Gradient>* gradients,
    std::vector<Hessian>* hessians) const {
    const ActiveTensorSupportRange range = active_tensor_support(xi);
    global_indices.clear();
    if (values != nullptr) {
        values->clear();
    }
    if (gradients != nullptr) {
        gradients->clear();
    }
    if (hessians != nullptr) {
        hessians->clear();
    }

    if (values == nullptr && gradients == nullptr && hessians == nullptr) {
        const std::size_t nx = axis_sizes_[0];
        const std::size_t ny = axis_sizes_[1];
        const std::size_t ix_end = range.first_indices[0] + range.counts[0];
        const std::size_t iy_end = range.first_indices[1] + range.counts[1];
        if (dimension_ == 2) {
            for (std::size_t j = range.first_indices[1]; j < iy_end; ++j) {
                for (std::size_t i = range.first_indices[0]; i < ix_end; ++i) {
                    global_indices.push_back(j * nx + i);
                }
            }
            return range;
        }

        const std::size_t iz_end = range.first_indices[2] + range.counts[2];
        for (std::size_t k = range.first_indices[2]; k < iz_end; ++k) {
            for (std::size_t j = range.first_indices[1]; j < iy_end; ++j) {
                for (std::size_t i = range.first_indices[0]; i < ix_end; ++i) {
                    global_indices.push_back((k * ny + j) * nx + i);
                }
            }
        }
        return range;
    }

    const NURBSTensorCompactWriter writer{&global_indices, values, gradients, hessians};
    evaluate_nurbs_tensor_active_support(*this, xi, writer);
    return range;
}

void NURBSTensorBasis::evaluate_rational_active_support(
    const math::Vector<Real, 3>& xi,
    std::vector<Real>* values,
    std::vector<Gradient>* gradients,
    std::vector<Hessian>* hessians) const {
    const NURBSTensorVectorWriter writer{values, gradients, hessians};
    evaluate_nurbs_tensor_active_support(*this, xi, writer);
}

void NURBSTensorBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                       std::vector<Real>& values) const {
    evaluate_rational_active_support(xi, &values, nullptr, nullptr);
}

void NURBSTensorBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                          std::vector<Gradient>& gradients) const {
    evaluate_rational_active_support(xi, nullptr, &gradients, nullptr);
}

void NURBSTensorBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                         std::vector<Hessian>& hessians) const {
    evaluate_rational_active_support(xi, nullptr, nullptr, &hessians);
}

void NURBSTensorBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values,
                                    std::vector<Gradient>& gradients,
                                    std::vector<Hessian>& hessians) const {
    evaluate_rational_active_support(xi, &values, &gradients, &hessians);
}

void NURBSTensorBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                          Real* SVMP_RESTRICT values_out) const {
    const NURBSTensorRawWriter writer{0u, 1u, values_out, nullptr, nullptr};
    evaluate_nurbs_tensor_active_support(*this, xi, writer);
}

void NURBSTensorBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                             Real* SVMP_RESTRICT gradients_out) const {
    const NURBSTensorRawWriter writer{0u, 1u, nullptr, gradients_out, nullptr};
    evaluate_nurbs_tensor_active_support(*this, xi, writer);
}

void NURBSTensorBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                            Real* SVMP_RESTRICT hessians_out) const {
    const NURBSTensorRawWriter writer{0u, 1u, nullptr, nullptr, hessians_out};
    evaluate_nurbs_tensor_active_support(*this, xi, writer);
}

void NURBSTensorBasis::evaluate_at_quadrature_points(
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

void NURBSTensorBasis::evaluate_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (output_stride < num_qpts) {
        throw BasisConfigurationException(
            "NURBSTensorBasis strided evaluation requires output_stride >= points.size()",
            __FILE__, __LINE__, __func__);
    }

    for (std::size_t q = 0; q < num_qpts; ++q) {
        const NURBSTensorRawWriter writer{q, output_stride, values_out, gradients_out, hessians_out};
        evaluate_nurbs_tensor_active_support(*this, points[q], writer);
    }
}

void NURBSTensorBasis::fill_scalar_cache_entry(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (output_stride < num_qpts) {
        throw BasisConfigurationException(
            "NURBSTensorBasis cache fill requires output_stride >= points.size()",
            __FILE__, __LINE__, __func__);
    }

    for (std::size_t q = 0; q < num_qpts; ++q) {
        const NURBSTensorRawWriter writer{
            q,
            output_stride,
            values_out,
            gradients_out,
            hessians_out,
            OutputInitialization::CallerPrecleared};
        evaluate_nurbs_tensor_active_support(*this, points[q], writer);
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
