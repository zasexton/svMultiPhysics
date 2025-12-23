/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "PenaltyMethod.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction
// ============================================================================

PenaltyMethod::PenaltyMethod() = default;

PenaltyMethod::PenaltyMethod(const PenaltyMethodOptions& options)
    : options_(options) {}

PenaltyMethod::PenaltyMethod(const AffineConstraints& constraints,
                              const PenaltyMethodOptions& options)
    : options_(options)
{
    initialize(constraints);
}

PenaltyMethod::~PenaltyMethod() = default;

PenaltyMethod::PenaltyMethod(PenaltyMethod&& other) noexcept = default;
PenaltyMethod& PenaltyMethod::operator=(PenaltyMethod&& other) noexcept = default;

// ============================================================================
// Setup
// ============================================================================

void PenaltyMethod::initialize(const AffineConstraints& constraints) {
    if (!constraints.isClosed()) {
        CONSTRAINT_THROW("PenaltyMethod requires closed AffineConstraints");
    }

    constraints_.clear();

    // Convert each constraint line to penalty form
    // u_slave = sum(a_i * u_master_i) + inhom
    // Penalty form: (u_slave - sum(a_i * u_master_i) - inhom)^2

    auto constrained_dofs = constraints.getConstrainedDofs();

    for (GlobalIndex slave : constrained_dofs) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        PenaltyConstraint pc;
        pc.dofs.push_back(slave);
        pc.coefficients.push_back(1.0);

        for (const auto& entry : constraint->entries) {
            pc.dofs.push_back(entry.master_dof);
            pc.coefficients.push_back(-entry.weight);
        }

        pc.rhs = constraint->inhomogeneity;
        pc.penalty = getEffectivePenalty(0.0);

        constraints_.push_back(std::move(pc));
    }
}

void PenaltyMethod::addConstraint(std::span<const GlobalIndex> dofs,
                                   std::span<const double> coefficients,
                                   double rhs,
                                   double penalty) {
    if (dofs.size() != coefficients.size()) {
        CONSTRAINT_THROW("DOFs and coefficients must have same size");
    }

    PenaltyConstraint pc;
    pc.dofs.assign(dofs.begin(), dofs.end());
    pc.coefficients.assign(coefficients.begin(), coefficients.end());
    pc.rhs = rhs;
    pc.penalty = getEffectivePenalty(penalty);

    constraints_.push_back(std::move(pc));
}

void PenaltyMethod::addConstraint(const PenaltyConstraint& constraint) {
    PenaltyConstraint pc = constraint;
    if (pc.penalty <= 0) {
        pc.penalty = getEffectivePenalty(0.0);
    }
    constraints_.push_back(std::move(pc));
}

void PenaltyMethod::addDirichletPenalty(GlobalIndex dof, double value, double penalty) {
    PenaltyConstraint pc;
    pc.dofs = {dof};
    pc.coefficients = {1.0};
    pc.rhs = value;
    pc.penalty = getEffectivePenalty(penalty);
    constraints_.push_back(std::move(pc));
}

void PenaltyMethod::addEqualityPenalty(GlobalIndex dof1, GlobalIndex dof2, double penalty) {
    PenaltyConstraint pc;
    pc.dofs = {dof1, dof2};
    pc.coefficients = {1.0, -1.0};
    pc.rhs = 0.0;
    pc.penalty = getEffectivePenalty(penalty);
    constraints_.push_back(std::move(pc));
}

void PenaltyMethod::clear() {
    constraints_.clear();
}

double PenaltyMethod::getEffectivePenalty(double specified_penalty) const {
    if (specified_penalty > 0) {
        return specified_penalty * options_.penalty_scaling;
    }
    return options_.default_penalty * options_.penalty_scaling;
}

// ============================================================================
// System modification
// ============================================================================

void PenaltyMethod::getPenaltyMatrixCSR(std::vector<GlobalIndex>& row_offsets,
                                         std::vector<GlobalIndex>& col_indices,
                                         std::vector<double>& values,
                                         GlobalIndex n_dofs) const {
    // Build penalty matrix contribution: sum_k alpha_k * b_k * b_k^T
    // where b_k is the k-th constraint vector

    // Use map to accumulate entries
    std::map<std::pair<GlobalIndex, GlobalIndex>, double> entries;

    for (const auto& pc : constraints_) {
        double alpha = pc.penalty;

        // Add alpha * b * b^T contribution
        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            for (std::size_t j = 0; j < pc.dofs.size(); ++j) {
                GlobalIndex row = pc.dofs[i];
                GlobalIndex col = pc.dofs[j];
                double val = alpha * pc.coefficients[i] * pc.coefficients[j];

                entries[{row, col}] += val;
            }
        }
    }

    // Convert to CSR
    row_offsets.clear();
    col_indices.clear();
    values.clear();

    row_offsets.resize(static_cast<std::size_t>(n_dofs + 1), 0);

    // Count entries per row
    for (const auto& [idx, val] : entries) {
        if (idx.first < n_dofs) {
            row_offsets[static_cast<std::size_t>(idx.first + 1)]++;
        }
    }

    // Cumulative sum
    for (GlobalIndex i = 1; i <= n_dofs; ++i) {
        row_offsets[static_cast<std::size_t>(i)] +=
            row_offsets[static_cast<std::size_t>(i - 1)];
    }

    // Fill arrays
    col_indices.resize(entries.size());
    values.resize(entries.size());

    std::vector<GlobalIndex> current_offset(row_offsets.begin(), row_offsets.end() - 1);

    for (const auto& [idx, val] : entries) {
        if (idx.first < n_dofs) {
            std::size_t pos = static_cast<std::size_t>(
                current_offset[static_cast<std::size_t>(idx.first)]++);
            col_indices[pos] = idx.second;
            values[pos] = val;
        }
    }
}

std::vector<double> PenaltyMethod::getPenaltyRhs(GlobalIndex n_dofs) const {
    std::vector<double> rhs(static_cast<std::size_t>(n_dofs), 0.0);

    // RHS contribution: sum_k alpha_k * g_k * b_k
    for (const auto& pc : constraints_) {
        double alpha_g = pc.penalty * pc.rhs;

        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            GlobalIndex dof = pc.dofs[i];
            if (dof >= 0 && dof < n_dofs) {
                rhs[static_cast<std::size_t>(dof)] += alpha_g * pc.coefficients[i];
            }
        }
    }

    return rhs;
}

void PenaltyMethod::applyPenaltyToMatrix(
    const std::function<void(GlobalIndex, GlobalIndex, double)>& add_entry) const
{
    // Add alpha * b * b^T for each constraint
    for (const auto& pc : constraints_) {
        double alpha = pc.penalty;

        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            for (std::size_t j = 0; j < pc.dofs.size(); ++j) {
                GlobalIndex row = pc.dofs[i];
                GlobalIndex col = pc.dofs[j];
                double val = alpha * pc.coefficients[i] * pc.coefficients[j];

                add_entry(row, col, val);
            }
        }
    }
}

void PenaltyMethod::applyPenaltyToRhs(std::span<double> rhs) const {
    for (const auto& pc : constraints_) {
        double alpha_g = pc.penalty * pc.rhs;

        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            GlobalIndex dof = pc.dofs[i];
            if (dof >= 0 && static_cast<std::size_t>(dof) < rhs.size()) {
                rhs[static_cast<std::size_t>(dof)] += alpha_g * pc.coefficients[i];
            }
        }
    }
}

std::function<void(std::span<const double>, std::span<double>)>
PenaltyMethod::createPenalizedOperator(
    std::function<void(std::span<const double>, std::span<double>)> A_apply,
    GlobalIndex n_dofs) const
{
    return [this, A_apply, n_dofs](std::span<const double> x, std::span<double> y) {
        // y = A*x
        A_apply(x, y);

        // y += alpha * B^T * B * x
        applyPenaltyOperator(x, y);
    };
}

void PenaltyMethod::applyPenaltyOperator(std::span<const double> x,
                                          std::span<double> y) const {
    // For each constraint: y += alpha * b * (b^T * x)
    for (const auto& pc : constraints_) {
        // Compute b^T * x
        double btx = 0.0;
        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            GlobalIndex dof = pc.dofs[i];
            if (dof >= 0 && static_cast<std::size_t>(dof) < x.size()) {
                btx += pc.coefficients[i] * x[static_cast<std::size_t>(dof)];
            }
        }

        // Add alpha * b * (b^T * x) to y
        double alpha_btx = pc.penalty * btx;
        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            GlobalIndex dof = pc.dofs[i];
            if (dof >= 0 && static_cast<std::size_t>(dof) < y.size()) {
                y[static_cast<std::size_t>(dof)] += alpha_btx * pc.coefficients[i];
            }
        }
    }
}

// ============================================================================
// Monitoring and statistics
// ============================================================================

std::vector<double> PenaltyMethod::computeResiduals(
    std::span<const double> solution) const
{
    std::vector<double> residuals;
    residuals.reserve(constraints_.size());

    for (const auto& pc : constraints_) {
        double sum = 0.0;
        for (std::size_t i = 0; i < pc.dofs.size(); ++i) {
            GlobalIndex dof = pc.dofs[i];
            if (dof >= 0 && static_cast<std::size_t>(dof) < solution.size()) {
                sum += pc.coefficients[i] * solution[static_cast<std::size_t>(dof)];
            }
        }
        residuals.push_back(sum - pc.rhs);
    }

    return residuals;
}

bool PenaltyMethod::checkSatisfaction(std::span<const double> solution,
                                       double tolerance) const {
    auto residuals = computeResiduals(solution);
    for (double r : residuals) {
        if (std::abs(r) > tolerance) {
            return false;
        }
    }
    return true;
}

PenaltyStats PenaltyMethod::computeStats(
    std::optional<std::span<const double>> solution) const
{
    PenaltyStats stats;
    stats.n_constraints = numConstraints();

    if (constraints_.empty()) {
        return stats;
    }

    // Penalty statistics
    stats.max_penalty = constraints_[0].penalty;
    stats.min_penalty = constraints_[0].penalty;

    for (const auto& pc : constraints_) {
        stats.max_penalty = std::max(stats.max_penalty, pc.penalty);
        stats.min_penalty = std::min(stats.min_penalty, pc.penalty);
    }

    // Residual statistics (if solution provided)
    if (solution.has_value()) {
        auto residuals = computeResiduals(*solution);

        stats.max_residual = 0.0;
        double sum_sq = 0.0;

        for (double r : residuals) {
            stats.max_residual = std::max(stats.max_residual, std::abs(r));
            sum_sq += r * r;
        }

        stats.rms_residual = std::sqrt(sum_sq / static_cast<double>(residuals.size()));
    }

    // Condition estimate
    stats.condition_estimate = stats.max_penalty;

    return stats;
}

double PenaltyMethod::estimateConditionContribution(
    double matrix_diagonal_estimate) const
{
    if (constraints_.empty() || matrix_diagonal_estimate <= 0) {
        return 1.0;
    }

    double max_penalty = 0.0;
    for (const auto& pc : constraints_) {
        max_penalty = std::max(max_penalty, pc.penalty);
    }

    // Condition contribution ~ max_penalty / matrix_scale
    return max_penalty / matrix_diagonal_estimate;
}

// ============================================================================
// Penalty scaling
// ============================================================================

void PenaltyMethod::autoScalePenalty(std::span<const double> diagonal_values,
                                      double scale_factor) {
    if (diagonal_values.empty()) return;

    double max_diag = 0.0;
    for (double d : diagonal_values) {
        max_diag = std::max(max_diag, std::abs(d));
    }

    double scaled_penalty = max_diag * scale_factor;

    for (auto& pc : constraints_) {
        pc.penalty = scaled_penalty;
    }

    options_.default_penalty = scaled_penalty;
}

void PenaltyMethod::scalePenalties(double factor) {
    for (auto& pc : constraints_) {
        pc.penalty *= factor;
    }
}

void PenaltyMethod::setUniformPenalty(double penalty) {
    for (auto& pc : constraints_) {
        pc.penalty = penalty;
    }
}

// ============================================================================
// Utility functions
// ============================================================================

double computeOptimalPenalty(double stiffness_estimate, double target_accuracy) {
    // Rule of thumb: penalty ~ stiffness / accuracy
    // This balances constraint satisfaction with conditioning
    if (stiffness_estimate <= 0 || target_accuracy <= 0) {
        return 1e6;  // Default
    }
    return stiffness_estimate / target_accuracy;
}

std::vector<double> adaptPenalties(std::span<const double> current_penalties,
                                    std::span<const double> residuals,
                                    double target_residual) {
    if (current_penalties.size() != residuals.size()) {
        CONSTRAINT_THROW("Penalties and residuals must have same size");
    }

    std::vector<double> new_penalties;
    new_penalties.reserve(current_penalties.size());

    for (std::size_t i = 0; i < current_penalties.size(); ++i) {
        double r = std::abs(residuals[i]);
        double p = current_penalties[i];

        if (r > target_residual) {
            // Increase penalty proportionally
            double factor = r / target_residual;
            new_penalties.push_back(p * factor);
        } else {
            // Keep current penalty (or slightly reduce for better conditioning)
            new_penalties.push_back(p);
        }
    }

    return new_penalties;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
