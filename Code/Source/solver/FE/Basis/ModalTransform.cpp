/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "ModalTransform.h"

#include "Math/DenseLinearAlgebra.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace basis {

struct ModalTransform::SolverStorage {
    math::DenseLUSolver solver;
};

struct ModalTransform::TransformData {
    std::size_t size{0};
    std::vector<Real> vandermonde_flat;
    std::shared_ptr<const SolverStorage> vandermonde_solver;
    std::vector<std::vector<Real>> vandermonde_rows;
    mutable std::once_flag vandermonde_inv_once;
    mutable std::vector<std::vector<Real>> vandermonde_inv;
};

namespace {

struct ModalTransformCacheKey {
    std::string modal_identity;
    std::string nodal_identity;

    bool operator==(const ModalTransformCacheKey& other) const noexcept {
        return modal_identity == other.modal_identity &&
               nodal_identity == other.nodal_identity;
    }
};

struct ModalTransformCacheKeyHash {
    std::size_t operator()(const ModalTransformCacheKey& key) const noexcept {
        const auto string_hash = std::hash<std::string>{};
        std::size_t seed = string_hash(key.modal_identity);
        seed ^= string_hash(key.nodal_identity) + 0x9e3779b97f4a7c15ULL +
                (seed << 6u) + (seed >> 2u);
        return seed;
    }
};

ModalTransformCacheKey make_modal_transform_cache_key(const BasisFunction& modal_basis,
                                                      const LagrangeBasis& nodal_basis) {
    return ModalTransformCacheKey{modal_basis.cache_identity(),
                                  nodal_basis.cache_identity()};
}

} // namespace

ModalTransform::ModalTransform(const BasisFunction& modal_basis,
                               const LagrangeBasis& nodal_basis)
    : modal_(modal_basis),
      nodal_(nodal_basis),
      transform_data_(get_or_build_transform_data(modal_basis, nodal_basis)) {
    if (modal_.size() != nodal_.size()) {
        throw BasisConfigurationException("ModalTransform requires modal/nodal bases of equal size",
                                          __FILE__, __LINE__, __func__);
    }
}

std::shared_ptr<const ModalTransform::TransformData>
ModalTransform::get_or_build_transform_data(const BasisFunction& modal_basis,
                                            const LagrangeBasis& nodal_basis) {
    if (modal_basis.size() != nodal_basis.size()) {
        throw BasisConfigurationException("ModalTransform requires modal/nodal bases of equal size",
                                          __FILE__, __LINE__, __func__);
    }

    static std::mutex cache_mutex;
    static std::unordered_map<ModalTransformCacheKey,
                              std::shared_ptr<const TransformData>,
                              ModalTransformCacheKeyHash> cache;

    const auto key = make_modal_transform_cache_key(modal_basis, nodal_basis);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }
    }

    const std::size_t n = nodal_basis.size();
    auto data = std::make_shared<TransformData>();
    data->size = n;
    data->vandermonde_flat.assign(n * n, Real(0));
    data->vandermonde_rows.assign(n, std::vector<Real>(n, Real(0)));
    const auto& nodes = nodal_basis.nodes();
    std::vector<Real> row;

    for (std::size_t i = 0; i < n; ++i) {
        modal_basis.evaluate_values(nodes[i], row);
        if (row.size() != n) {
            throw BasisConstructionException("Modal basis returned unexpected size during Vandermonde assembly",
                                             __FILE__, __LINE__, __func__);
        }
        std::copy(row.begin(), row.end(), data->vandermonde_rows[i].begin());
        std::copy(row.begin(), row.end(),
                  data->vandermonde_flat.begin() + static_cast<std::ptrdiff_t>(i * n));
    }

    auto storage = std::make_shared<SolverStorage>();
    storage->solver =
        math::factor_dense_matrix(data->vandermonde_flat, n, "ModalTransform Vandermonde");
    data->vandermonde_solver = std::move(storage);
    data->vandermonde_inv.clear();

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [iter, inserted] = cache.emplace(key, data);
        (void)inserted;
        return iter->second;
    }
}

void ModalTransform::materialize_vandermonde_inverse() const {
    const auto& data = *transform_data_;
    std::call_once(data.vandermonde_inv_once, [&data]() {
        const std::size_t n = data.size;
        std::vector<Real> inverse_flat(n * n, Real(0));
        for (std::size_t diag = 0; diag < n; ++diag) {
            inverse_flat[diag * n + diag] = Real(1);
        }
        data.vandermonde_solver->solver.solve_in_place(
            std::span<Real>(inverse_flat.data(), inverse_flat.size()), n);

        data.vandermonde_inv.assign(n, std::vector<Real>(n, Real(0)));
        for (std::size_t row = 0; row < n; ++row) {
            const Real* src = inverse_flat.data() + row * n;
            std::copy(src, src + n, data.vandermonde_inv[row].begin());
        }
    });
}

const std::vector<std::vector<Real>>& ModalTransform::vandermonde() const noexcept {
    return transform_data_->vandermonde_rows;
}

const std::vector<std::vector<Real>>& ModalTransform::vandermonde_inverse() const {
    materialize_vandermonde_inverse();
    return transform_data_->vandermonde_inv;
}

std::vector<Real> ModalTransform::modal_to_nodal(const std::vector<Real>& modal_coeffs) const {
    const std::size_t n = transform_data_->size;
    if (modal_coeffs.size() != n) {
        throw BasisEvaluationException("modal_to_nodal: size mismatch",
                                       __FILE__, __LINE__, __func__);
    }
    std::vector<Real> nodal(n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        const Real* row = transform_data_->vandermonde_flat.data() + i * n;
        for (std::size_t j = 0; j < n; ++j) {
            nodal[i] += row[j] * modal_coeffs[j];
        }
    }
    return nodal;
}

std::vector<Real> ModalTransform::nodal_to_modal(const std::vector<Real>& nodal_values) const {
    const std::size_t n = transform_data_->size;
    if (nodal_values.size() != n) {
        throw BasisEvaluationException("nodal_to_modal: size mismatch",
                                       __FILE__, __LINE__, __func__);
    }
    return transform_data_->vandermonde_solver->solver.solve(
        std::span<const Real>(nodal_values.data(), nodal_values.size()));
}

Real ModalTransform::condition_number() const {
    const std::size_t n = transform_data_->size;
    const auto diagnostics =
        math::dense_matrix_diagnostics(std::span<const Real>(transform_data_->vandermonde_flat.data(),
                                                               transform_data_->vandermonde_flat.size()),
                                         n, n, "ModalTransform Vandermonde");
    if (std::isfinite(diagnostics.condition_estimate)) {
        return diagnostics.condition_estimate;
    }
    return transform_data_->vandermonde_solver->solver.diagnostics.condition_estimate;
}

} // namespace basis
} // namespace FE
} // namespace svmp
