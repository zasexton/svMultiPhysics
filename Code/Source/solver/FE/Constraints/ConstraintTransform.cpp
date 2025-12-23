/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstraintTransform.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction
// ============================================================================

ConstraintTransform::ConstraintTransform() = default;

ConstraintTransform::ConstraintTransform(const AffineConstraints& constraints,
                                          GlobalIndex n_full_dofs,
                                          const TransformOptions& options)
    : options_(options), n_full_(n_full_dofs)
{
    initialize(constraints, n_full_dofs);
}

ConstraintTransform::~ConstraintTransform() = default;

ConstraintTransform::ConstraintTransform(ConstraintTransform&& other) noexcept = default;
ConstraintTransform& ConstraintTransform::operator=(ConstraintTransform&& other) noexcept = default;

// ============================================================================
// Initialization
// ============================================================================

void ConstraintTransform::initialize(const AffineConstraints& constraints,
                                      GlobalIndex n_full_dofs)
{
    if (!constraints.isClosed()) {
        CONSTRAINT_THROW("ConstraintTransform requires closed AffineConstraints");
    }

    n_full_ = n_full_dofs;

    // Build full-to-reduced and reduced-to-full mappings
    full_to_reduced_.resize(static_cast<std::size_t>(n_full_), -1);
    reduced_to_full_.clear();
    reduced_to_full_.reserve(static_cast<std::size_t>(n_full_ - static_cast<GlobalIndex>(constraints.numConstraints())));

    GlobalIndex reduced_idx = 0;
    for (GlobalIndex full_idx = 0; full_idx < n_full_; ++full_idx) {
        if (!constraints.isConstrained(full_idx)) {
            full_to_reduced_[static_cast<std::size_t>(full_idx)] = reduced_idx;
            reduced_to_full_.push_back(full_idx);
            ++reduced_idx;
        }
    }
    n_reduced_ = reduced_idx;

    // Build projection
    buildProjection(constraints);

    // Allocate scratch space
    work_full_.resize(static_cast<std::size_t>(n_full_));
    work_full2_.resize(static_cast<std::size_t>(n_full_));

    initialized_ = true;
}

void ConstraintTransform::buildProjection(const AffineConstraints& constraints)
{
    projection_rows_.resize(static_cast<std::size_t>(n_full_));
    inhomogeneity_.resize(static_cast<std::size_t>(n_full_), 0.0);

    for (GlobalIndex full_idx = 0; full_idx < n_full_; ++full_idx) {
        ProjectionRow& row = projection_rows_[static_cast<std::size_t>(full_idx)];
        row.full_index = full_idx;

        auto constraint = constraints.getConstraint(full_idx);
        if (!constraint) {
            // Unconstrained: identity mapping
            GlobalIndex reduced_idx = full_to_reduced_[static_cast<std::size_t>(full_idx)];
            row.reduced_indices.push_back(reduced_idx);
            row.weights.push_back(1.0);
            row.inhomogeneity = 0.0;
        } else {
            // Constrained: u_full = sum(weight * u_master) + inhom
            // Since constraint is closed, masters are unconstrained
            row.inhomogeneity = constraint->inhomogeneity;
            inhomogeneity_[static_cast<std::size_t>(full_idx)] = constraint->inhomogeneity;

            for (const auto& entry : constraint->entries) {
                GlobalIndex master_full = entry.master_dof;
                GlobalIndex master_reduced = full_to_reduced_[static_cast<std::size_t>(master_full)];

                if (master_reduced < 0) {
                    // Master is also constrained - this shouldn't happen after close()
                    CONSTRAINT_THROW_DOF("Master DOF is constrained (closure incomplete)",
                                         master_full);
                }

                // Check if this reduced index already exists (merge entries)
                auto it = std::find(row.reduced_indices.begin(),
                                   row.reduced_indices.end(),
                                   master_reduced);
                if (it != row.reduced_indices.end()) {
                    std::size_t idx = static_cast<std::size_t>(it - row.reduced_indices.begin());
                    row.weights[idx] += entry.weight;
                } else {
                    row.reduced_indices.push_back(master_reduced);
                    row.weights.push_back(entry.weight);
                }
            }
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

ReductionStats ConstraintTransform::getStats() const
{
    ReductionStats stats;
    stats.n_full_dofs = n_full_;
    stats.n_reduced_dofs = n_reduced_;
    stats.n_constrained = n_full_ - n_reduced_;

    // Count non-zeros in projection
    GlobalIndex nnz = 0;
    for (const auto& row : projection_rows_) {
        nnz += static_cast<GlobalIndex>(row.reduced_indices.size());
    }
    stats.projection_nnz = nnz;

    if (n_full_ > 0) {
        stats.reduction_ratio = static_cast<double>(n_reduced_) / static_cast<double>(n_full_);
    }

    return stats;
}

// ============================================================================
// Projection operations
// ============================================================================

void ConstraintTransform::applyProjection(std::span<const double> z_reduced,
                                           std::span<double> u_full) const
{
    if (!initialized_) {
        CONSTRAINT_THROW("ConstraintTransform not initialized");
    }

    if (static_cast<GlobalIndex>(z_reduced.size()) < n_reduced_) {
        CONSTRAINT_THROW("Reduced vector too small");
    }
    if (static_cast<GlobalIndex>(u_full.size()) < n_full_) {
        CONSTRAINT_THROW("Full vector too small");
    }

    // u = P * z + c
    for (GlobalIndex i = 0; i < n_full_; ++i) {
        const auto& row = projection_rows_[static_cast<std::size_t>(i)];
        double val = row.inhomogeneity;

        for (std::size_t j = 0; j < row.reduced_indices.size(); ++j) {
            val += row.weights[j] * z_reduced[static_cast<std::size_t>(row.reduced_indices[j])];
        }

        u_full[static_cast<std::size_t>(i)] = val;
    }
}

void ConstraintTransform::applyTranspose(std::span<const double> f_full,
                                          std::span<double> g_reduced) const
{
    if (!initialized_) {
        CONSTRAINT_THROW("ConstraintTransform not initialized");
    }

    if (static_cast<GlobalIndex>(f_full.size()) < n_full_) {
        CONSTRAINT_THROW("Full vector too small");
    }
    if (static_cast<GlobalIndex>(g_reduced.size()) < n_reduced_) {
        CONSTRAINT_THROW("Reduced vector too small");
    }

    // g = P^T * f
    // Zero output
    std::fill(g_reduced.begin(), g_reduced.begin() + n_reduced_, 0.0);

    // Accumulate contributions
    for (GlobalIndex i = 0; i < n_full_; ++i) {
        const auto& row = projection_rows_[static_cast<std::size_t>(i)];
        double f_i = f_full[static_cast<std::size_t>(i)];

        for (std::size_t j = 0; j < row.reduced_indices.size(); ++j) {
            g_reduced[static_cast<std::size_t>(row.reduced_indices[j])] += row.weights[j] * f_i;
        }
    }
}

// ============================================================================
// Solution expansion/restriction
// ============================================================================

std::vector<double> ConstraintTransform::expandSolution(
    std::span<const double> z_reduced) const
{
    std::vector<double> u_full(static_cast<std::size_t>(n_full_));
    applyProjection(z_reduced, u_full);
    return u_full;
}

std::vector<double> ConstraintTransform::restrictVector(
    std::span<const double> u_full) const
{
    if (!initialized_) {
        CONSTRAINT_THROW("ConstraintTransform not initialized");
    }

    std::vector<double> z_reduced(static_cast<std::size_t>(n_reduced_));

    for (GlobalIndex r = 0; r < n_reduced_; ++r) {
        GlobalIndex f = reduced_to_full_[static_cast<std::size_t>(r)];
        z_reduced[static_cast<std::size_t>(r)] = u_full[static_cast<std::size_t>(f)];
    }

    return z_reduced;
}

// ============================================================================
// Matrix-free operators
// ============================================================================

std::function<void(std::span<const double>, std::span<double>)>
ConstraintTransform::createReducedOperator(
    std::function<void(std::span<const double>, std::span<double>)> A_full) const
{
    return [this, A_full](std::span<const double> z_in, std::span<double> g_out) {
        applyReducedOperator(A_full, z_in, g_out);
    };
}

void ConstraintTransform::applyReducedOperator(
    const std::function<void(std::span<const double>, std::span<double>)>& A_full,
    std::span<const double> z_reduced,
    std::span<double> g_reduced) const
{
    if (!initialized_) {
        CONSTRAINT_THROW("ConstraintTransform not initialized");
    }

    // g = P^T A P z

    // Step 1: u = P z (expand to full)
    applyProjection(z_reduced, work_full_);

    // Step 2: v = A u (apply full operator)
    A_full(work_full_, work_full2_);

    // Step 3: g = P^T v (restrict to reduced)
    applyTranspose(work_full2_, g_reduced);
}

void ConstraintTransform::computeReducedRhs(
    const std::function<void(std::span<const double>, std::span<double>)>& A_full,
    std::span<const double> b_full,
    std::span<double> g_reduced) const
{
    if (!initialized_) {
        CONSTRAINT_THROW("ConstraintTransform not initialized");
    }

    // g = P^T (b - A c)

    // Step 1: v = A c (apply A to inhomogeneity)
    A_full(inhomogeneity_, work_full_);

    // Step 2: w = b - v
    for (GlobalIndex i = 0; i < n_full_; ++i) {
        work_full_[static_cast<std::size_t>(i)] =
            b_full[static_cast<std::size_t>(i)] - work_full_[static_cast<std::size_t>(i)];
    }

    // Step 3: g = P^T w
    applyTranspose(work_full_, g_reduced);
}

// ============================================================================
// CSR matrix access
// ============================================================================

void ConstraintTransform::getProjectionCSR(std::vector<GlobalIndex>& row_offsets,
                                            std::vector<GlobalIndex>& col_indices,
                                            std::vector<double>& values) const
{
    if (!initialized_) {
        CONSTRAINT_THROW("ConstraintTransform not initialized");
    }

    row_offsets.clear();
    col_indices.clear();
    values.clear();

    row_offsets.reserve(static_cast<std::size_t>(n_full_ + 1));
    row_offsets.push_back(0);

    for (const auto& row : projection_rows_) {
        for (std::size_t j = 0; j < row.reduced_indices.size(); ++j) {
            col_indices.push_back(row.reduced_indices[j]);
            values.push_back(row.weights[j]);
        }
        row_offsets.push_back(static_cast<GlobalIndex>(col_indices.size()));
    }
}

} // namespace constraints
} // namespace FE
} // namespace svmp
