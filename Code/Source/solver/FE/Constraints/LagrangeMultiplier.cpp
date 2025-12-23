/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LagrangeMultiplier.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction
// ============================================================================

LagrangeMultiplier::LagrangeMultiplier() = default;

LagrangeMultiplier::LagrangeMultiplier(const LagrangeMultiplierOptions& options)
    : options_(options) {}

LagrangeMultiplier::LagrangeMultiplier(const AffineConstraints& constraints,
                                        const LagrangeMultiplierOptions& options)
    : options_(options)
{
    initialize(constraints);
}

LagrangeMultiplier::~LagrangeMultiplier() = default;

LagrangeMultiplier::LagrangeMultiplier(LagrangeMultiplier&& other) noexcept = default;
LagrangeMultiplier& LagrangeMultiplier::operator=(LagrangeMultiplier&& other) noexcept = default;

// ============================================================================
// Setup
// ============================================================================

void LagrangeMultiplier::initialize(const AffineConstraints& constraints) {
    if (!constraints.isClosed()) {
        CONSTRAINT_THROW("LagrangeMultiplier requires closed AffineConstraints");
    }

    constraints_.clear();
    finalized_ = false;

    // Convert each constraint line to Lagrange form
    // u_slave = sum(a_i * u_master_i) + inhom
    // becomes: u_slave - sum(a_i * u_master_i) = inhom
    // which is: 1*u_slave + sum(-a_i * u_master_i) = inhom

    auto constrained_dofs = constraints.getConstrainedDofs();

    for (GlobalIndex slave : constrained_dofs) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        LagrangeConstraint lc;
        lc.constrained_dofs.push_back(slave);
        lc.coefficients.push_back(1.0);

        for (const auto& entry : constraint->entries) {
            lc.constrained_dofs.push_back(entry.master_dof);
            lc.coefficients.push_back(-entry.weight);
        }

        lc.rhs = constraint->inhomogeneity;
        constraints_.push_back(std::move(lc));
    }

    finalize();
}

void LagrangeMultiplier::addConstraint(std::span<const GlobalIndex> constrained_dofs,
                                        std::span<const double> coefficients,
                                        double rhs) {
    if (constrained_dofs.size() != coefficients.size()) {
        CONSTRAINT_THROW("DOFs and coefficients must have same size");
    }

    LagrangeConstraint lc;
    lc.constrained_dofs.assign(constrained_dofs.begin(), constrained_dofs.end());
    lc.coefficients.assign(coefficients.begin(), coefficients.end());
    lc.rhs = rhs;
    lc.multiplier_dof = -1;  // Will be assigned in finalize()

    constraints_.push_back(std::move(lc));
    finalized_ = false;
}

void LagrangeMultiplier::addConstraint(const LagrangeConstraint& constraint) {
    constraints_.push_back(constraint);
    finalized_ = false;
}

void LagrangeMultiplier::finalize() {
    if (finalized_) return;

    // Find maximum DOF index for later use
    max_dof_index_ = 0;
    for (const auto& lc : constraints_) {
        for (GlobalIndex dof : lc.constrained_dofs) {
            max_dof_index_ = std::max(max_dof_index_, dof);
        }
    }

    // Assign multiplier DOF indices
    if (options_.auto_assign_multiplier_dofs) {
        assignMultiplierDofs();
    }

    finalized_ = true;
}

void LagrangeMultiplier::assignMultiplierDofs() {
    GlobalIndex current_dof = options_.multiplier_dof_offset;
    for (auto& lc : constraints_) {
        lc.multiplier_dof = current_dof++;
    }
}

// ============================================================================
// Accessors
// ============================================================================

LagrangeStats LagrangeMultiplier::getStats() const {
    LagrangeStats stats;
    stats.n_constraints = numConstraints();
    stats.n_multiplier_dofs = numMultipliers();

    GlobalIndex nnz = 0;
    for (const auto& lc : constraints_) {
        nnz += static_cast<GlobalIndex>(lc.constrained_dofs.size());
    }
    stats.constraint_matrix_nnz = nnz;

    return stats;
}

// ============================================================================
// Assembly support
// ============================================================================

void LagrangeMultiplier::getConstraintMatrixCSR(
    std::vector<GlobalIndex>& row_offsets,
    std::vector<GlobalIndex>& col_indices,
    std::vector<double>& values) const
{
    row_offsets.clear();
    col_indices.clear();
    values.clear();

    row_offsets.reserve(static_cast<std::size_t>(numConstraints() + 1));
    row_offsets.push_back(0);

    for (const auto& lc : constraints_) {
        for (std::size_t j = 0; j < lc.constrained_dofs.size(); ++j) {
            col_indices.push_back(lc.constrained_dofs[j]);
            values.push_back(lc.coefficients[j]);
        }
        row_offsets.push_back(static_cast<GlobalIndex>(col_indices.size()));
    }
}

std::vector<double> LagrangeMultiplier::getConstraintRhs() const {
    std::vector<double> rhs;
    rhs.reserve(constraints_.size());
    for (const auto& lc : constraints_) {
        rhs.push_back(lc.rhs);
    }
    return rhs;
}

void LagrangeMultiplier::applyTranspose(std::span<const double> lambda,
                                         std::span<double> result) const {
    // result = B^T * lambda
    // B^T has columns = constraints, rows = DOFs
    // (B^T)_{j,i} = B_{i,j} = coefficient of DOF j in constraint i

    std::fill(result.begin(), result.end(), 0.0);

    for (std::size_t i = 0; i < constraints_.size(); ++i) {
        const auto& lc = constraints_[i];
        double lambda_i = lambda[i];

        for (std::size_t k = 0; k < lc.constrained_dofs.size(); ++k) {
            GlobalIndex j = lc.constrained_dofs[k];
            if (j >= 0 && static_cast<std::size_t>(j) < result.size()) {
                result[static_cast<std::size_t>(j)] += lc.coefficients[k] * lambda_i;
            }
        }
    }
}

void LagrangeMultiplier::applyConstraintMatrix(std::span<const double> u,
                                                std::span<double> result) const {
    // result = B * u
    // result_i = sum_j B_{i,j} * u_j

    if (result.size() < constraints_.size()) {
        CONSTRAINT_THROW("Result vector too small");
    }

    for (std::size_t i = 0; i < constraints_.size(); ++i) {
        const auto& lc = constraints_[i];
        double sum = 0.0;

        for (std::size_t k = 0; k < lc.constrained_dofs.size(); ++k) {
            GlobalIndex j = lc.constrained_dofs[k];
            if (j >= 0 && static_cast<std::size_t>(j) < u.size()) {
                sum += lc.coefficients[k] * u[static_cast<std::size_t>(j)];
            }
        }

        result[i] = sum;
    }
}

std::vector<double> LagrangeMultiplier::computeResidual(std::span<const double> u) const {
    std::vector<double> residual(constraints_.size());
    applyConstraintMatrix(u, residual);

    // r = B*u - g
    for (std::size_t i = 0; i < constraints_.size(); ++i) {
        residual[i] -= constraints_[i].rhs;
    }

    return residual;
}

bool LagrangeMultiplier::checkSatisfaction(std::span<const double> u,
                                            double tolerance) const {
    auto residual = computeResidual(u);
    for (double r : residual) {
        if (std::abs(r) > tolerance) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Saddle-point system support
// ============================================================================

void LagrangeMultiplier::applySaddlePointOperator(
    const std::function<void(std::span<const double>, std::span<double>)>& A_apply,
    std::span<const double> u,
    std::span<const double> lambda,
    std::span<double> result_u,
    std::span<double> result_lambda) const
{
    // result_u = A*u + B^T*lambda
    A_apply(u, result_u);

    // Add B^T * lambda contribution
    std::vector<double> bt_lambda(result_u.size(), 0.0);
    applyTranspose(lambda, bt_lambda);
    for (std::size_t i = 0; i < result_u.size(); ++i) {
        result_u[i] += bt_lambda[i];
    }

    // result_lambda = B*u - s*lambda
    applyConstraintMatrix(u, result_lambda);

    // Subtract stabilization term
    if (std::abs(options_.stabilization_param) > 1e-15) {
        for (std::size_t i = 0; i < result_lambda.size(); ++i) {
            result_lambda[i] -= options_.stabilization_param * lambda[i];
        }
    }
}

std::function<void(std::span<const double>, std::span<double>)>
LagrangeMultiplier::createSaddlePointOperator(
    std::function<void(std::span<const double>, std::span<double>)> A_apply,
    GlobalIndex n_primal_dofs) const
{
    return [this, A_apply, n_primal_dofs](std::span<const double> x,
                                           std::span<double> y) {
        std::size_t n_primal = static_cast<std::size_t>(n_primal_dofs);
        std::size_t n_mult = constraints_.size();

        // Extract u and lambda from combined vector
        std::span<const double> u = x.subspan(0, n_primal);
        std::span<const double> lambda = x.subspan(n_primal, n_mult);

        // Output views
        std::span<double> result_u = y.subspan(0, n_primal);
        std::span<double> result_lambda = y.subspan(n_primal, n_mult);

        applySaddlePointOperator(A_apply, u, lambda, result_u, result_lambda);
    };
}

// ============================================================================
// Multiplier interpretation
// ============================================================================

std::vector<double> LagrangeMultiplier::extractMultipliers(
    std::span<const double> combined,
    GlobalIndex n_primal_dofs) const
{
    std::size_t n_primal = static_cast<std::size_t>(n_primal_dofs);
    std::size_t n_mult = constraints_.size();

    if (combined.size() < n_primal + n_mult) {
        CONSTRAINT_THROW("Combined vector too small");
    }

    return std::vector<double>(combined.begin() + static_cast<std::ptrdiff_t>(n_primal),
                               combined.begin() + static_cast<std::ptrdiff_t>(n_primal + n_mult));
}

std::vector<double> LagrangeMultiplier::extractPrimal(
    std::span<const double> combined,
    GlobalIndex n_primal_dofs) const
{
    std::size_t n_primal = static_cast<std::size_t>(n_primal_dofs);

    if (combined.size() < n_primal) {
        CONSTRAINT_THROW("Combined vector too small");
    }

    return std::vector<double>(combined.begin(),
                               combined.begin() + static_cast<std::ptrdiff_t>(n_primal));
}

std::vector<double> LagrangeMultiplier::computeConstraintForces(
    std::span<const double> lambda) const
{
    if (lambda.size() < constraints_.size()) {
        CONSTRAINT_THROW("Lambda vector too small");
    }

    // Force at DOF j = sum_i B_{i,j} * lambda_i = (B^T * lambda)_j
    std::vector<double> forces(static_cast<std::size_t>(max_dof_index_ + 1), 0.0);
    applyTranspose(lambda, forces);
    return forces;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
