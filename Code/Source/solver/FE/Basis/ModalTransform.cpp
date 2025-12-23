/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "ModalTransform.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace basis {

ModalTransform::ModalTransform(const BasisFunction& modal_basis,
                               const LagrangeBasis& nodal_basis)
    : modal_(modal_basis), nodal_(nodal_basis) {
    if (modal_.size() != nodal_.size()) {
        throw FEException("ModalTransform requires modal/nodal bases of equal size",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    compute_vandermonde();
    invert_vandermonde();
}

void ModalTransform::compute_vandermonde() {
    const std::size_t n = nodal_.size();
    vandermonde_.assign(n, std::vector<Real>(n, Real(0)));
    const auto& nodes = nodal_.nodes();

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Real> row;
        modal_.evaluate_values(nodes[i], row);
        if (row.size() != n) {
            throw FEException("Modal basis returned unexpected size during Vandermonde assembly",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        vandermonde_[i] = row;
    }
}

void ModalTransform::invert_vandermonde() {
    const std::size_t n = vandermonde_.size();
    vandermonde_inv_.assign(n, std::vector<Real>(n, Real(0)));
    std::vector<std::vector<Real>> mat = vandermonde_;

    for (std::size_t i = 0; i < n; ++i) {
        vandermonde_inv_[i][i] = Real(1);
    }

    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivoting for robustness against node/basis ordering and
        // improved numerical stability.
        std::size_t pivot_row = col;
        Real max_abs = std::abs(mat[col][col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const Real cand = std::abs(mat[row][col]);
            if (cand > max_abs) {
                max_abs = cand;
                pivot_row = row;
            }
        }

        if (max_abs < Real(1e-14)) {
            throw FEException("Singular Vandermonde encountered in ModalTransform",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        if (pivot_row != col) {
            std::swap(mat[pivot_row], mat[col]);
            std::swap(vandermonde_inv_[pivot_row], vandermonde_inv_[col]);
        }

        const Real pivot = mat[col][col];
        const Real inv_pivot = Real(1) / pivot;
        for (std::size_t j = 0; j < n; ++j) {
            mat[col][j] *= inv_pivot;
            vandermonde_inv_[col][j] *= inv_pivot;
        }

        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) continue;
            Real factor = mat[row][col];
            for (std::size_t j = 0; j < n; ++j) {
                mat[row][j]       -= factor * mat[col][j];
                vandermonde_inv_[row][j] -= factor * vandermonde_inv_[col][j];
            }
        }
    }
}

std::vector<Real> ModalTransform::modal_to_nodal(const std::vector<Real>& modal_coeffs) const {
    const std::size_t n = vandermonde_.size();
    if (modal_coeffs.size() != n) {
        throw FEException("modal_to_nodal: size mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    std::vector<Real> nodal(n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            nodal[i] += vandermonde_[i][j] * modal_coeffs[j];
        }
    }
    return nodal;
}

std::vector<Real> ModalTransform::nodal_to_modal(const std::vector<Real>& nodal_values) const {
    const std::size_t n = vandermonde_inv_.size();
    if (nodal_values.size() != n) {
        throw FEException("nodal_to_modal: size mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    std::vector<Real> modal(n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            modal[i] += vandermonde_inv_[i][j] * nodal_values[j];
        }
    }
    return modal;
}

Real ModalTransform::condition_number() const {
    auto norm_inf = [](const std::vector<std::vector<Real>>& m) {
        Real max_row = Real(0);
        for (const auto& row : m) {
            Real sum = Real(0);
            for (Real v : row) sum += std::abs(v);
            max_row = std::max(max_row, sum);
        }
        return max_row;
    };
    return norm_inf(vandermonde_) * norm_inf(vandermonde_inv_);
}

} // namespace basis
} // namespace FE
} // namespace svmp
