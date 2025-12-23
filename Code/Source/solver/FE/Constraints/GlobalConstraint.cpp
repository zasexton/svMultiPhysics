/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "GlobalConstraint.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction
// ============================================================================

GlobalConstraint::GlobalConstraint() = default;

GlobalConstraint::GlobalConstraint(std::vector<GlobalIndex> dofs,
                                    GlobalConstraintType type,
                                    const GlobalConstraintOptions& options)
    : dofs_(std::move(dofs)), global_type_(type), options_(options)
{
    // Initialize weights to uniform for zero/fixed mean
    if (global_type_ == GlobalConstraintType::ZeroMean ||
        global_type_ == GlobalConstraintType::FixedMean) {
        weights_.resize(dofs_.size(), 1.0);
        normalizeWeights();
    }

    if (global_type_ == GlobalConstraintType::FixedMean) {
        target_value_ = options_.pinned_value;
    }
}

GlobalConstraint::GlobalConstraint(std::vector<GlobalIndex> dofs,
                                    std::vector<double> weights,
                                    double target_value,
                                    const GlobalConstraintOptions& options)
    : dofs_(std::move(dofs)),
      weights_(std::move(weights)),
      global_type_(GlobalConstraintType::FixedMean),
      options_(options),
      target_value_(target_value)
{
    if (dofs_.size() != weights_.size()) {
        CONSTRAINT_THROW("DOFs and weights must have same size");
    }
    normalizeWeights();
}

// ============================================================================
// Copy/Move
// ============================================================================

GlobalConstraint::GlobalConstraint(const GlobalConstraint& other) = default;
GlobalConstraint::GlobalConstraint(GlobalConstraint&& other) noexcept = default;
GlobalConstraint& GlobalConstraint::operator=(const GlobalConstraint& other) = default;
GlobalConstraint& GlobalConstraint::operator=(GlobalConstraint&& other) noexcept = default;

// ============================================================================
// Constraint interface
// ============================================================================

void GlobalConstraint::apply(AffineConstraints& constraints) const {
    if (dofs_.empty()) {
        return;
    }

    switch (options_.strategy) {
        case GlobalConstraintStrategy::PinSingleDof: {
            // Select DOF to pin
            pinned_dof_ = selectDofToPin();

            // Add constraint: u_pinned = target_value
            constraints.addLine(pinned_dof_);
            constraints.setInhomogeneity(pinned_dof_, target_value_);
            break;
        }

        case GlobalConstraintStrategy::WeightedMean: {
            // Express first DOF in terms of others to enforce weighted mean
            // sum(w_i * u_i) = target
            // w_0 * u_0 = target - sum_{i>0}(w_i * u_i)
            // u_0 = target/w_0 - sum_{i>0}(w_i/w_0 * u_i)

            if (weights_.empty() || std::abs(weights_[0]) < 1e-15) {
                CONSTRAINT_THROW("First weight must be non-zero for WeightedMean strategy");
            }

            double w0_inv = 1.0 / weights_[0];

            constraints.addLine(dofs_[0]);
            constraints.setInhomogeneity(dofs_[0], target_value_ * w0_inv);

            for (std::size_t i = 1; i < dofs_.size(); ++i) {
                constraints.addEntry(dofs_[0], dofs_[i], -weights_[i] * w0_inv);
            }

            pinned_dof_ = dofs_[0];
            break;
        }

        case GlobalConstraintStrategy::LagrangeMultiplier:
        case GlobalConstraintStrategy::NullspaceProjection:
            // These strategies require system-level modifications
            // and cannot be expressed as simple DOF constraints.
            // They need to be handled at the assembly/solver level.
            // For now, fall back to PinSingleDof
            pinned_dof_ = selectDofToPin();
            constraints.addLine(pinned_dof_);
            constraints.setInhomogeneity(pinned_dof_, target_value_);
            break;
    }
}

ConstraintInfo GlobalConstraint::getInfo() const {
    ConstraintInfo info;
    info.name = "GlobalConstraint";
    info.type = ConstraintType::Global;
    info.num_constrained_dofs = (options_.strategy == GlobalConstraintStrategy::WeightedMean)
                                 ? 1 : (dofs_.empty() ? 0 : 1);
    info.is_time_dependent = false;
    info.is_homogeneous = (std::abs(target_value_) < 1e-15);
    return info;
}

GlobalConstraintInfo GlobalConstraint::getGlobalInfo() const {
    GlobalConstraintInfo info;
    info.type = global_type_;
    info.strategy = options_.strategy;
    info.pinned_dof = pinned_dof_;
    info.target_value = target_value_;
    info.nullspace_vector = getNullspaceVector();
    return info;
}

// ============================================================================
// Nullspace operations
// ============================================================================

std::vector<double> GlobalConstraint::getNullspaceVector() const {
    // Return weights normalized so that their sum equals 1
    std::vector<double> nullvec = weights_;
    double sum = std::accumulate(nullvec.begin(), nullvec.end(), 0.0);
    if (std::abs(sum) > 1e-15) {
        for (double& w : nullvec) {
            w /= sum;
        }
    }
    return nullvec;
}

void GlobalConstraint::projectToConstrainedSpace(std::span<double> vec) const {
    if (dofs_.empty()) return;

    // Compute current value of constraint
    double current = computeWeightedMean(vec, dofs_, weights_);

    // Subtract to make constraint satisfied
    double correction = current - target_value_;

    // For zero/fixed mean constraints, subtract the same correction from each DOF
    // This makes the mean shift by exactly -correction
    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        GlobalIndex dof = dofs_[i];
        if (dof >= 0 && static_cast<std::size_t>(dof) < vec.size()) {
            vec[static_cast<std::size_t>(dof)] -= correction;
        }
    }
}

bool GlobalConstraint::checkSatisfaction(std::span<const double> vec) const {
    return std::abs(computeResidual(vec)) < options_.tolerance;
}

double GlobalConstraint::computeResidual(std::span<const double> vec) const {
    if (dofs_.empty()) return 0.0;

    double sum = 0.0;
    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        GlobalIndex dof = dofs_[i];
        if (dof >= 0 && static_cast<std::size_t>(dof) < vec.size()) {
            sum += weights_[i] * vec[static_cast<std::size_t>(dof)];
        }
    }
    return sum - target_value_;
}

// ============================================================================
// Static factory methods
// ============================================================================

GlobalConstraint GlobalConstraint::zeroMean(std::vector<GlobalIndex> dofs,
                                             const GlobalConstraintOptions& options) {
    return GlobalConstraint(std::move(dofs), GlobalConstraintType::ZeroMean, options);
}

GlobalConstraint GlobalConstraint::fixedMean(std::vector<GlobalIndex> dofs,
                                              double target,
                                              const GlobalConstraintOptions& options) {
    GlobalConstraintOptions opts = options;
    opts.pinned_value = target;
    return GlobalConstraint(std::move(dofs), GlobalConstraintType::FixedMean, opts);
}

GlobalConstraint GlobalConstraint::volumeConservation(std::vector<GlobalIndex> dofs,
                                                       double initial_volume,
                                                       const GlobalConstraintOptions& options) {
    GlobalConstraintOptions opts = options;
    opts.pinned_value = initial_volume;
    return GlobalConstraint(std::move(dofs), GlobalConstraintType::VolumeConservation, opts);
}

GlobalConstraint GlobalConstraint::nullspacePinning(std::vector<GlobalIndex> dofs,
                                                     std::vector<double> nullspace_vector,
                                                     const GlobalConstraintOptions& options) {
    GlobalConstraintOptions opts = options;
    opts.strategy = GlobalConstraintStrategy::NullspaceProjection;
    return GlobalConstraint(std::move(dofs), std::move(nullspace_vector), 0.0, opts);
}

GlobalConstraint GlobalConstraint::pinDof(GlobalIndex dof, double value) {
    GlobalConstraintOptions options;
    options.strategy = GlobalConstraintStrategy::PinSingleDof;
    options.explicit_pin_dof = dof;
    options.pinned_value = value;
    return GlobalConstraint({dof}, GlobalConstraintType::NullspacePinning, options);
}

// ============================================================================
// Internal helpers
// ============================================================================

GlobalIndex GlobalConstraint::selectDofToPin() const {
    if (dofs_.empty()) {
        CONSTRAINT_THROW("No DOFs available for pinning");
    }

    // If explicit DOF specified, use it
    if (options_.explicit_pin_dof >= 0) {
        // Verify it's in our list
        auto it = std::find(dofs_.begin(), dofs_.end(), options_.explicit_pin_dof);
        if (it != dofs_.end()) {
            return options_.explicit_pin_dof;
        }
        // Fall through to default selection
    }

    // Default: choose first DOF (deterministic)
    // In practice, prefer_boundary_dof would query mesh connectivity
    // For now, just use the first DOF for determinism
    return dofs_[0];
}

void GlobalConstraint::normalizeWeights() {
    double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (std::abs(sum) > 1e-15 && std::abs(sum - 1.0) > 1e-15) {
        for (double& w : weights_) {
            w /= sum;
        }
    }
}

// ============================================================================
// Utility functions
// ============================================================================

double computeMean(std::span<const double> vec, std::span<const GlobalIndex> dofs) {
    if (dofs.empty()) return 0.0;

    double sum = 0.0;
    std::size_t count = 0;

    for (GlobalIndex dof : dofs) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < vec.size()) {
            sum += vec[static_cast<std::size_t>(dof)];
            ++count;
        }
    }

    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double computeWeightedMean(std::span<const double> vec,
                            std::span<const GlobalIndex> dofs,
                            std::span<const double> weights) {
    if (dofs.empty() || dofs.size() != weights.size()) return 0.0;

    double sum = 0.0;
    double weight_sum = 0.0;

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        GlobalIndex dof = dofs[i];
        if (dof >= 0 && static_cast<std::size_t>(dof) < vec.size()) {
            sum += weights[i] * vec[static_cast<std::size_t>(dof)];
            weight_sum += weights[i];
        }
    }

    return std::abs(weight_sum) > 1e-15 ? sum / weight_sum : 0.0;
}

void subtractMean(std::span<double> vec, std::span<const GlobalIndex> dofs) {
    double mean = computeMean(vec, dofs);

    for (GlobalIndex dof : dofs) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < vec.size()) {
            vec[static_cast<std::size_t>(dof)] -= mean;
        }
    }
}

} // namespace constraints
} // namespace FE
} // namespace svmp
