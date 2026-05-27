/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_TENSORBASIS_H
#define SVMP_FE_BASIS_TENSORBASIS_H

/**
 * @file TensorBasis.h
 * @brief Tensor-product basis wrapper for quadrilateral and hexahedral elements
 */

#include "BasisFunction.h"
#include "BasisTolerance.h"
#include "NodeOrderingConventions.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace basis {

namespace detail {

template<typename T, typename = void>
struct has_nodes_method : std::false_type {};

template<typename T>
struct has_nodes_method<T, std::void_t<decltype(std::declval<const T&>().nodes())>> : std::true_type {};

inline bool coords_close(const math::Vector<Real, 3>& a,
                         const math::Vector<Real, 3>& b,
                         Real tol = basis_scaled_tolerance()) {
    return (std::abs(a[0] - b[0]) <= tol) &&
           (std::abs(a[1] - b[1]) <= tol) &&
           (std::abs(a[2] - b[2]) <= tol);
}

} // namespace detail

/**
 * @brief Generic tensor-product basis composed from 1D bases
 *
 * Basis1D must satisfy the BasisFunction interface on a line element.
 */
template<typename Basis1D>
class TensorProductBasis : public BasisFunction {
public:
    /// Construct isotropic tensor-product basis from a single 1D prototype
    explicit TensorProductBasis(const Basis1D& basis_1d, int dimension = 2)
        : bases_{basis_1d, basis_1d, basis_1d}, dimension_(dimension) {
        if (dimension_ != 1 && dimension_ != 2 && dimension_ != 3) {
            throw BasisConfigurationException("TensorProductBasis dimension must be 1, 2, or 3",
                                              __FILE__, __LINE__, __func__);
        }
        normalize_orders();
        build_indices();
        finalize_node_ordering();
        rebuild_cache_identity();
    }

    /// Anisotropic 2D tensor product
    TensorProductBasis(const Basis1D& bx, const Basis1D& by)
        : bases_{bx, by, bx}, dimension_(2) {
        normalize_orders();
        build_indices();
        finalize_node_ordering();
        rebuild_cache_identity();
    }

    /// Anisotropic 3D tensor product
    TensorProductBasis(const Basis1D& bx, const Basis1D& by, const Basis1D& bz)
        : bases_{bx, by, bz}, dimension_(3) {
        normalize_orders();
        build_indices();
        finalize_node_ordering();
        rebuild_cache_identity();
    }

    [[nodiscard]] BasisType basis_type() const noexcept override { return bases_[0].basis_type(); }
    [[nodiscard]] ElementType element_type() const noexcept override {
        if (dimension_ == 1) return ElementType::Line2;
        if (dimension_ == 2) return ElementType::Quad4;
        return ElementType::Hex8;
    }
    [[nodiscard]] int dimension() const noexcept override { return dimension_; }
    [[nodiscard]] int order() const noexcept override { return order_; }
    [[nodiscard]] std::size_t size() const noexcept override { return indices_.size(); }
    [[nodiscard]] const Basis1D& axis_basis(int axis) const noexcept {
        return bases_[static_cast<std::size_t>(axis)];
    }
    [[nodiscard]] std::vector<int> tensor_extents() const {
        std::vector<int> extents(static_cast<std::size_t>(dimension_));
        for (int axis = 0; axis < dimension_; ++axis) {
            extents[static_cast<std::size_t>(axis)] =
                static_cast<int>(bases_[static_cast<std::size_t>(axis)].size());
        }
        return extents;
    }
    [[nodiscard]] std::string cache_identity() const override {
        return cache_identity_;
    }

    [[nodiscard]] bool cache_identity_words(std::vector<std::uint64_t>& words) const override {
        if (cache_identity_words_.empty()) {
            return false;
        }
        words.insert(words.end(), cache_identity_words_.begin(), cache_identity_words_.end());
        return true;
    }

    bool cache_identity_fingerprint(std::uint64_t& hash_a,
                                    std::uint64_t& hash_b) const override {
        if (cache_identity_words_.empty()) {
            return false;
        }
        hash_a = cache_identity_hash_a_;
        hash_b = cache_identity_hash_b_;
        return true;
    }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(size());
        write_evaluation(xi, 0u, 1u, values.data(), nullptr, nullptr);
    }

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override {
        gradients.resize(size());
        auto& scratch = axis_scratch();
        fill_axis_scratch(xi, true, false, scratch);
        const auto& vx = scratch.values[0];
        const auto& vy = scratch.values[1];
        const auto& vz = scratch.values[2];
        const auto& gx = scratch.gradients[0];
        const auto& gy = scratch.gradients[1];
        const auto& gz = scratch.gradients[2];

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            const auto ix = static_cast<std::size_t>(id[0]);
            Gradient g{};
            if (dimension_ == 1) {
                g[0] = gx[ix * 3u];
            } else if (dimension_ == 2) {
                const auto iy = static_cast<std::size_t>(id[1]);
                g[0] = gx[ix * 3u] * vy[iy];
                g[1] = vx[ix] * gy[iy * 3u];
            } else {
                const auto iy = static_cast<std::size_t>(id[1]);
                const auto iz = static_cast<std::size_t>(id[2]);
                g[0] = gx[ix * 3u] * vy[iy] * vz[iz];
                g[1] = vx[ix] * gy[iy * 3u] * vz[iz];
                g[2] = vx[ix] * vy[iy] * gz[iz * 3u];
            }
            gradients[idx] = g;
        }
    }

    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override {
        hessians.resize(size());
        auto& scratch = axis_scratch();
        fill_axis_scratch(xi, true, true, scratch);
        const auto& vx = scratch.values[0];
        const auto& vy = scratch.values[1];
        const auto& vz = scratch.values[2];
        const auto& gx = scratch.gradients[0];
        const auto& gy = scratch.gradients[1];
        const auto& gz = scratch.gradients[2];
        const auto& hx = scratch.hessians[0];
        const auto& hy = scratch.hessians[1];
        const auto& hz = scratch.hessians[2];

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            const auto ix = static_cast<std::size_t>(id[0]);
            Hessian H{};
            if (dimension_ == 1) {
                H(0, 0) = hx[ix * 9u];
            } else if (dimension_ == 2) {
                const auto iy = static_cast<std::size_t>(id[1]);
                const Real dx = gx[ix * 3u];
                const Real dy = gy[iy * 3u];
                H(0, 0) = hx[ix * 9u] * vy[iy];
                H(1, 1) = vx[ix] * hy[iy * 9u];
                H(0, 1) = dx * dy;
                H(1, 0) = H(0, 1);
            } else {
                const auto iy = static_cast<std::size_t>(id[1]);
                const auto iz = static_cast<std::size_t>(id[2]);
                const Real dx = gx[ix * 3u];
                const Real dy = gy[iy * 3u];
                const Real dz = gz[iz * 3u];
                H(0, 0) = hx[ix * 9u] * vy[iy] * vz[iz];
                H(1, 1) = vx[ix] * hy[iy * 9u] * vz[iz];
                H(2, 2) = vx[ix] * vy[iy] * hz[iz * 9u];
                H(0, 1) = dx * dy * vz[iz];
                H(1, 0) = H(0, 1);
                H(0, 2) = dx * vy[iy] * dz;
                H(2, 0) = H(0, 2);
                H(1, 2) = vx[ix] * dy * dz;
                H(2, 1) = H(1, 2);
            }
            hessians[idx] = H;
        }
    }

    void evaluate_values_to(const math::Vector<Real, 3>& xi,
                            Real* SVMP_RESTRICT values_out) const override {
        write_evaluation(xi, 0u, 1u, values_out, nullptr, nullptr);
    }

    void evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                               Real* SVMP_RESTRICT gradients_out) const override {
        write_evaluation(xi, 0u, 1u, nullptr, gradients_out, nullptr);
    }

    void evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                              Real* SVMP_RESTRICT hessians_out) const override {
        write_evaluation(xi, 0u, 1u, nullptr, nullptr, hessians_out);
    }

    void evaluate_at_quadrature_points(
        const std::vector<math::Vector<Real, 3>>& points,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override {
        evaluate_at_quadrature_points_strided(points,
                                              points.size(),
                                              values_out,
                                              gradients_out,
                                              hessians_out);
    }

    void evaluate_at_quadrature_points_strided(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override {
        const std::size_t num_qpts = points.size();
        if (output_stride < points.size()) {
            throw BasisConfigurationException(
                "TensorProductBasis strided evaluation requires output_stride >= points.size()",
                __FILE__, __LINE__, __func__);
        }
        if (num_qpts == 0u ||
            (values_out == nullptr && gradients_out == nullptr && hessians_out == nullptr)) {
            return;
        }

        const bool need_gradients = gradients_out != nullptr || hessians_out != nullptr;
        const bool need_hessians = hessians_out != nullptr;
        auto& scratch = axis_batch_scratch();
        fill_axis_batch_scratch(points, need_gradients, need_hessians, scratch);

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            const auto ix = static_cast<std::size_t>(id[0]);
            const auto iy = static_cast<std::size_t>(id[1]);
            const auto iz = static_cast<std::size_t>(id[2]);

            Real* value_row = values_out ? values_out + idx * output_stride : nullptr;
            Real* gradient_row = gradients_out ? gradients_out + idx * 3u * output_stride : nullptr;
            Real* hessian_row = hessians_out ? hessians_out + idx * 9u * output_stride : nullptr;

            for (std::size_t q = 0; q < num_qpts; ++q) {
                const Real vx = axis_value(scratch, 0u, ix, q, num_qpts);
                const Real vy = (dimension_ >= 2)
                    ? axis_value(scratch, 1u, iy, q, num_qpts)
                    : Real(1);
                const Real vz = (dimension_ == 3)
                    ? axis_value(scratch, 2u, iz, q, num_qpts)
                    : Real(1);

                if (value_row != nullptr) {
                    value_row[q] = vx * vy * vz;
                }

                if (gradient_row != nullptr) {
                    gradient_row[0u * output_stride + q] =
                        axis_gradient(scratch, 0u, ix, q, num_qpts) * vy * vz;
                    gradient_row[1u * output_stride + q] = (dimension_ >= 2)
                        ? vx * axis_gradient(scratch, 1u, iy, q, num_qpts) * vz
                        : Real(0);
                    gradient_row[2u * output_stride + q] = (dimension_ == 3)
                        ? vx * vy * axis_gradient(scratch, 2u, iz, q, num_qpts)
                        : Real(0);
                }

                if (hessian_row != nullptr) {
                    const Real dx = axis_gradient(scratch, 0u, ix, q, num_qpts);
                    const Real dy = (dimension_ >= 2)
                        ? axis_gradient(scratch, 1u, iy, q, num_qpts)
                        : Real(0);
                    const Real dz = (dimension_ == 3)
                        ? axis_gradient(scratch, 2u, iz, q, num_qpts)
                        : Real(0);
                    const Real d2x = axis_hessian(scratch, 0u, ix, q, num_qpts);
                    const Real d2y = (dimension_ >= 2)
                        ? axis_hessian(scratch, 1u, iy, q, num_qpts)
                        : Real(0);
                    const Real d2z = (dimension_ == 3)
                        ? axis_hessian(scratch, 2u, iz, q, num_qpts)
                        : Real(0);

                    const Real h01 = dx * dy * vz;
                    const Real h02 = dx * vy * dz;
                    const Real h12 = vx * dy * dz;
                    hessian_row[0u * output_stride + q] = d2x * vy * vz;
                    hessian_row[1u * output_stride + q] = h01;
                    hessian_row[2u * output_stride + q] = h02;
                    hessian_row[3u * output_stride + q] = h01;
                    hessian_row[4u * output_stride + q] = vx * d2y * vz;
                    hessian_row[5u * output_stride + q] = h12;
                    hessian_row[6u * output_stride + q] = h02;
                    hessian_row[7u * output_stride + q] = h12;
                    hessian_row[8u * output_stride + q] = vx * vy * d2z;
                }
            }
        }
    }

private:
    struct AxisScratch {
        std::array<std::vector<Real>, 3> values;
        std::array<std::vector<Real>, 3> gradients;
        std::array<std::vector<Real>, 3> hessians;
    };

    struct AxisBatchScratch {
        std::array<std::vector<math::Vector<Real, 3>>, 3> points;
        std::array<std::vector<Real>, 3> values;
        std::array<std::vector<Real>, 3> gradients;
        std::array<std::vector<Real>, 3> hessians;
    };

    std::array<Basis1D, 3> bases_;
    int dimension_;
    int order_{0};
    std::vector<std::array<int, 3>> indices_;
    std::string cache_identity_;
    std::vector<std::uint64_t> cache_identity_words_;
    std::uint64_t cache_identity_hash_a_{0};
    std::uint64_t cache_identity_hash_b_{0};

    static AxisScratch& axis_scratch() {
        static thread_local AxisScratch scratch;
        return scratch;
    }

    static AxisBatchScratch& axis_batch_scratch() {
        static thread_local AxisBatchScratch scratch;
        return scratch;
    }

    static Real axis_value(const AxisBatchScratch& scratch,
                           std::size_t axis,
                           std::size_t basis_index,
                           std::size_t q,
                           std::size_t stride) {
        return scratch.values[axis][basis_index * stride + q];
    }

    static Real axis_gradient(const AxisBatchScratch& scratch,
                              std::size_t axis,
                              std::size_t basis_index,
                              std::size_t q,
                              std::size_t stride) {
        return scratch.gradients[axis][basis_index * 3u * stride + q];
    }

    static Real axis_hessian(const AxisBatchScratch& scratch,
                             std::size_t axis,
                             std::size_t basis_index,
                             std::size_t q,
                             std::size_t stride) {
        return scratch.hessians[axis][basis_index * 9u * stride + q];
    }

    void fill_axis_scratch(const math::Vector<Real, 3>& xi,
                           bool need_gradients,
                           bool need_hessians,
                           AxisScratch& scratch) const {
        for (int axis = 0; axis < dimension_; ++axis) {
            const auto a = static_cast<std::size_t>(axis);
            const std::size_t axis_size = bases_[a].size();
            const math::Vector<Real, 3> coord{
                xi[a],
                Real(0),
                Real(0)
            };

            scratch.values[a].resize(axis_size);
            bases_[a].evaluate_values_to(coord, scratch.values[a].data());

            if (need_gradients) {
                scratch.gradients[a].resize(axis_size * 3u);
                bases_[a].evaluate_gradients_to(coord, scratch.gradients[a].data());
            } else {
                scratch.gradients[a].clear();
            }

            if (need_hessians) {
                scratch.hessians[a].resize(axis_size * 9u);
                bases_[a].evaluate_hessians_to(coord, scratch.hessians[a].data());
            } else {
                scratch.hessians[a].clear();
            }
        }
    }

    void fill_axis_batch_scratch(const std::vector<math::Vector<Real, 3>>& points,
                                 bool need_gradients,
                                 bool need_hessians,
                                 AxisBatchScratch& scratch) const {
        const std::size_t num_qpts = points.size();
        for (int axis = 0; axis < dimension_; ++axis) {
            const auto a = static_cast<std::size_t>(axis);
            const std::size_t axis_size = bases_[a].size();
            auto& axis_points = scratch.points[a];
            axis_points.resize(num_qpts);
            for (std::size_t q = 0; q < num_qpts; ++q) {
                axis_points[q] = math::Vector<Real, 3>{points[q][a], Real(0), Real(0)};
            }

            scratch.values[a].resize(axis_size * num_qpts);
            Real* gradients = nullptr;
            Real* hessians = nullptr;
            if (need_gradients) {
                scratch.gradients[a].resize(axis_size * 3u * num_qpts);
                gradients = scratch.gradients[a].data();
            } else {
                scratch.gradients[a].clear();
            }
            if (need_hessians) {
                scratch.hessians[a].resize(axis_size * 9u * num_qpts);
                hessians = scratch.hessians[a].data();
            } else {
                scratch.hessians[a].clear();
            }

            bases_[a].evaluate_at_quadrature_points_strided(
                axis_points,
                num_qpts,
                scratch.values[a].data(),
                gradients,
                hessians);
        }
    }

    void write_evaluation(const math::Vector<Real, 3>& xi,
                          std::size_t q,
                          std::size_t stride,
                          Real* SVMP_RESTRICT values_out,
                          Real* SVMP_RESTRICT gradients_out,
                          Real* SVMP_RESTRICT hessians_out) const {
        const bool need_gradients = gradients_out != nullptr || hessians_out != nullptr;
        const bool need_hessians = hessians_out != nullptr;
        auto& scratch = axis_scratch();
        fill_axis_scratch(xi, need_gradients, need_hessians, scratch);

        const auto& vx = scratch.values[0];
        const auto& vy = scratch.values[1];
        const auto& vz = scratch.values[2];
        const auto& gx = scratch.gradients[0];
        const auto& gy = scratch.gradients[1];
        const auto& gz = scratch.gradients[2];
        const auto& hx = scratch.hessians[0];
        const auto& hy = scratch.hessians[1];
        const auto& hz = scratch.hessians[2];

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            const auto ix = static_cast<std::size_t>(id[0]);
            const auto iy = static_cast<std::size_t>(id[1]);
            const auto iz = static_cast<std::size_t>(id[2]);

            if (values_out) {
                Real value = vx[ix];
                if (dimension_ >= 2) {
                    value *= vy[iy];
                }
                if (dimension_ == 3) {
                    value *= vz[iz];
                }
                values_out[idx * stride + q] = value;
            }

            if (gradients_out) {
                for (std::size_t component = 0; component < 3u; ++component) {
                    gradients_out[(idx * 3u + component) * stride + q] = Real(0);
                }
                if (dimension_ == 1) {
                    gradients_out[(idx * 3u + 0u) * stride + q] = gx[ix * 3u];
                } else if (dimension_ == 2) {
                    gradients_out[(idx * 3u + 0u) * stride + q] = gx[ix * 3u] * vy[iy];
                    gradients_out[(idx * 3u + 1u) * stride + q] = vx[ix] * gy[iy * 3u];
                } else {
                    gradients_out[(idx * 3u + 0u) * stride + q] = gx[ix * 3u] * vy[iy] * vz[iz];
                    gradients_out[(idx * 3u + 1u) * stride + q] = vx[ix] * gy[iy * 3u] * vz[iz];
                    gradients_out[(idx * 3u + 2u) * stride + q] = vx[ix] * vy[iy] * gz[iz * 3u];
                }
            }

            if (hessians_out) {
                for (std::size_t component = 0; component < 9u; ++component) {
                    hessians_out[(idx * 9u + component) * stride + q] = Real(0);
                }

                if (dimension_ == 1) {
                    hessians_out[(idx * 9u + 0u) * stride + q] = hx[ix * 9u];
                } else if (dimension_ == 2) {
                    const Real dx = gx[ix * 3u];
                    const Real dy = gy[iy * 3u];
                    hessians_out[(idx * 9u + 0u) * stride + q] = hx[ix * 9u] * vy[iy];
                    hessians_out[(idx * 9u + 4u) * stride + q] = vx[ix] * hy[iy * 9u];
                    hessians_out[(idx * 9u + 1u) * stride + q] = dx * dy;
                    hessians_out[(idx * 9u + 3u) * stride + q] = dx * dy;
                } else {
                    const Real dx = gx[ix * 3u];
                    const Real dy = gy[iy * 3u];
                    const Real dz = gz[iz * 3u];
                    hessians_out[(idx * 9u + 0u) * stride + q] = hx[ix * 9u] * vy[iy] * vz[iz];
                    hessians_out[(idx * 9u + 4u) * stride + q] = vx[ix] * hy[iy * 9u] * vz[iz];
                    hessians_out[(idx * 9u + 8u) * stride + q] = vx[ix] * vy[iy] * hz[iz * 9u];
                    hessians_out[(idx * 9u + 1u) * stride + q] = dx * dy * vz[iz];
                    hessians_out[(idx * 9u + 3u) * stride + q] = dx * dy * vz[iz];
                    hessians_out[(idx * 9u + 2u) * stride + q] = dx * vy[iy] * dz;
                    hessians_out[(idx * 9u + 6u) * stride + q] = dx * vy[iy] * dz;
                    hessians_out[(idx * 9u + 5u) * stride + q] = vx[ix] * dy * dz;
                    hessians_out[(idx * 9u + 7u) * stride + q] = vx[ix] * dy * dz;
                }
            }
        }
    }

    void normalize_orders() {
        order_ = bases_[0].order();
        if (dimension_ >= 2) {
            order_ = std::max(order_, bases_[1].order());
        }
        if (dimension_ == 3) {
            order_ = std::max(order_, bases_[2].order());
        }
    }

    void build_indices() {
        indices_.clear();
        const int nx = static_cast<int>(bases_[0].size());
        const int ny = (dimension_ >= 2) ? static_cast<int>(bases_[1].size()) : 1;
        const int nz = (dimension_ == 3) ? static_cast<int>(bases_[2].size()) : 1;
        if (dimension_ == 1) {
            for (int i = 0; i < nx; ++i) {
                indices_.push_back({i, 0, 0});
            }
            return;
        }
        if (dimension_ == 2) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    indices_.push_back({i, j, 0});
                }
            }
            return;
        }
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    indices_.push_back({i, j, k});
                }
            }
        }
    }

    void finalize_node_ordering() {
        if (dimension_ == 1) {
            return;
        }
        if (bases_[0].basis_type() != BasisType::Lagrange) {
            return;
        }
        if constexpr (!detail::has_nodes_method<Basis1D>::value) {
            return;
        } else {
            const int ox = bases_[0].order();
            const int oy = (dimension_ >= 2) ? bases_[1].order() : ox;
            const int oz = (dimension_ == 3) ? bases_[2].order() : ox;
            if ((dimension_ == 2 && ox != oy) || (dimension_ == 3 && (ox != oy || ox != oz))) {
                return;
            }

            ElementType ordering_type = ElementType::Unknown;
            if (dimension_ == 2) {
                if (ox == 1) ordering_type = ElementType::Quad4;
                if (ox == 2) ordering_type = ElementType::Quad9;
            } else if (dimension_ == 3) {
                if (ox == 1) ordering_type = ElementType::Hex8;
                if (ox == 2) ordering_type = ElementType::Hex27;
            }
            if (ordering_type == ElementType::Unknown) {
                return;
            }
            if (indices_.size() != ReferenceNodeLayout::num_nodes(ordering_type)) {
                return;
            }

            const auto& nx = bases_[0].nodes();
            const auto& ny = bases_[1].nodes();
            const auto& nz = bases_[2].nodes();
            if (nx.size() != bases_[0].size()) {
                return;
            }
            if (dimension_ >= 2 && ny.size() != bases_[1].size()) {
                return;
            }
            if (dimension_ == 3 && nz.size() != bases_[2].size()) {
                return;
            }

            std::vector<math::Vector<Real, 3>> internal_nodes;
            internal_nodes.reserve(indices_.size());
            for (const auto& id : indices_) {
                math::Vector<Real, 3> p{Real(0), Real(0), Real(0)};
                p[0] = nx[static_cast<std::size_t>(id[0])][0];
                if (dimension_ >= 2) {
                    p[1] = ny[static_cast<std::size_t>(id[1])][0];
                }
                if (dimension_ == 3) {
                    p[2] = nz[static_cast<std::size_t>(id[2])][0];
                }
                internal_nodes.push_back(p);
            }

            std::vector<std::array<int, 3>> reordered(indices_.size());
            std::vector<bool> used(indices_.size(), false);
            for (std::size_t ext = 0; ext < indices_.size(); ++ext) {
                const auto target = ReferenceNodeLayout::get_node_coords(ordering_type, ext);
                bool found = false;
                for (std::size_t in = 0; in < indices_.size(); ++in) {
                    if (used[in]) {
                        continue;
                    }
                    if (detail::coords_close(internal_nodes[in], target)) {
                        reordered[ext] = indices_[in];
                        used[in] = true;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw BasisNodeOrderingException("TensorProductBasis: failed to align tensor-product nodes with NodeOrderingConventions",
                                                     __FILE__, __LINE__, __func__);
                }
            }

            indices_ = std::move(reordered);
        }
    }

    void rebuild_cache_identity() {
        std::ostringstream oss;
        oss << BasisFunction::cache_identity() << "|axes=" << dimension_;

        std::vector<std::uint64_t> words;
        words.reserve(12u + static_cast<std::size_t>(dimension_) * 12u);
        words.push_back(0x74656e736f727031ULL); // "tensorp1"
        words.push_back(static_cast<std::uint64_t>(dimension_));
        words.push_back(static_cast<std::uint64_t>(order_));
        words.push_back(static_cast<std::uint64_t>(indices_.size()));

        bool has_structured_identity = true;
        for (int axis = 0; axis < dimension_; ++axis) {
            const auto& basis = bases_[static_cast<std::size_t>(axis)];
            oss << "|axis" << axis
                << ":type=" << static_cast<int>(basis.basis_type())
                << ":element=" << static_cast<int>(basis.element_type())
                << ":dimension=" << basis.dimension()
                << ":order=" << basis.order()
                << ":size=" << basis.size()
                << ":vector=" << (basis.is_vector_valued() ? 1 : 0);

            words.push_back(static_cast<std::uint64_t>(axis));
            words.push_back(static_cast<std::uint64_t>(static_cast<int>(basis.basis_type())));
            words.push_back(static_cast<std::uint64_t>(static_cast<int>(basis.element_type())));
            words.push_back(static_cast<std::uint64_t>(basis.dimension()));
            words.push_back(static_cast<std::uint64_t>(basis.order()));
            words.push_back(static_cast<std::uint64_t>(basis.size()));
            words.push_back(basis.is_vector_valued() ? 1ULL : 0ULL);

            if (basis.cache_identity_is_structural()) {
                oss << ":structural";
                words.push_back(0ULL);
                continue;
            }

            std::vector<std::uint64_t> axis_words;
            if (!basis.cache_identity_words(axis_words)) {
                has_structured_identity = false;
                oss << ":fallback=" << basis.cache_identity();
                continue;
            }
            const auto axis_fingerprint = compute_basis_identity_fingerprint(axis_words);
            oss << ":words=" << axis_words.size()
                << ':' << axis_fingerprint.hash_a
                << ':' << axis_fingerprint.hash_b;
            words.push_back(static_cast<std::uint64_t>(axis_words.size()));
            words.insert(words.end(), axis_words.begin(), axis_words.end());
        }

        cache_identity_ = oss.str();
        if (has_structured_identity) {
            cache_identity_words_ = std::move(words);
            const auto fingerprint = compute_basis_identity_fingerprint(cache_identity_words_);
            cache_identity_hash_a_ = fingerprint.hash_a;
            cache_identity_hash_b_ = fingerprint.hash_b;
        } else {
            cache_identity_words_.clear();
            cache_identity_hash_a_ = 0;
            cache_identity_hash_b_ = 0;
        }
    }
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_TENSORBASIS_H
