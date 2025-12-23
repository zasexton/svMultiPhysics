/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MultiPointConstraint.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction
// ============================================================================

MultiPointConstraint::MultiPointConstraint() = default;

MultiPointConstraint::MultiPointConstraint(std::vector<MPCEquation> equations,
                                            const MPCOptions& options)
    : equations_(std::move(equations)), options_(options) {}

// ============================================================================
// Copy/Move
// ============================================================================

MultiPointConstraint::MultiPointConstraint(const MultiPointConstraint& other) = default;
MultiPointConstraint::MultiPointConstraint(MultiPointConstraint&& other) noexcept = default;
MultiPointConstraint& MultiPointConstraint::operator=(const MultiPointConstraint& other) = default;
MultiPointConstraint& MultiPointConstraint::operator=(MultiPointConstraint&& other) noexcept = default;

// ============================================================================
// Constraint interface
// ============================================================================

void MultiPointConstraint::apply(AffineConstraints& constraints) const {
    // Apply explicit constraints directly
    for (const auto& ec : explicit_constraints_) {
        constraints.addLine(ec.slave_dof);
        for (const auto& [master, weight] : ec.masters) {
            if (std::abs(weight) > options_.coefficient_tolerance) {
                constraints.addEntry(ec.slave_dof, master, weight);
            }
        }
        if (std::abs(ec.inhomogeneity) > options_.coefficient_tolerance) {
            constraints.setInhomogeneity(ec.slave_dof, ec.inhomogeneity);
        }
    }

    // Convert and apply equations
    for (const auto& eq : equations_) {
        ExplicitConstraint ec = convertEquation(eq);
        constraints.addLine(ec.slave_dof);
        for (const auto& [master, weight] : ec.masters) {
            if (std::abs(weight) > options_.coefficient_tolerance) {
                constraints.addEntry(ec.slave_dof, master, weight);
            }
        }
        if (std::abs(ec.inhomogeneity) > options_.coefficient_tolerance) {
            constraints.setInhomogeneity(ec.slave_dof, ec.inhomogeneity);
        }
    }
}

ConstraintInfo MultiPointConstraint::getInfo() const {
    ConstraintInfo info;
    info.name = "MultiPointConstraint";
    info.type = ConstraintType::MultiPoint;
    info.num_constrained_dofs = numConstraints();
    info.is_time_dependent = false;
    info.is_homogeneous = true;  // Check if all inhomogeneities are zero

    for (const auto& ec : explicit_constraints_) {
        if (std::abs(ec.inhomogeneity) > options_.coefficient_tolerance) {
            info.is_homogeneous = false;
            break;
        }
    }
    if (info.is_homogeneous) {
        for (const auto& eq : equations_) {
            if (std::abs(eq.rhs) > options_.coefficient_tolerance) {
                info.is_homogeneous = false;
                break;
            }
        }
    }

    return info;
}

// ============================================================================
// Equation-based interface
// ============================================================================

void MultiPointConstraint::addEquation(const MPCEquation& equation) {
    equations_.push_back(equation);
}

void MultiPointConstraint::addEquation(std::span<const GlobalIndex> dofs,
                                        std::span<const double> coefficients,
                                        double rhs) {
    if (dofs.size() != coefficients.size()) {
        CONSTRAINT_THROW("DOFs and coefficients must have same size");
    }

    MPCEquation eq;
    eq.rhs = rhs;
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        eq.addTerm(dofs[i], coefficients[i]);
    }
    equations_.push_back(std::move(eq));
}

// ============================================================================
// Explicit slave/master interface
// ============================================================================

void MultiPointConstraint::addConstraint(GlobalIndex slave_dof,
                                          std::vector<std::pair<GlobalIndex, double>> masters,
                                          double inhomogeneity) {
    ExplicitConstraint ec;
    ec.slave_dof = slave_dof;
    ec.masters = std::move(masters);
    ec.inhomogeneity = inhomogeneity;
    explicit_constraints_.push_back(std::move(ec));
}

void MultiPointConstraint::addConstraint(GlobalIndex slave_dof,
                                          GlobalIndex master_dof,
                                          double weight,
                                          double inhomogeneity) {
    ExplicitConstraint ec;
    ec.slave_dof = slave_dof;
    ec.masters = {{master_dof, weight}};
    ec.inhomogeneity = inhomogeneity;
    explicit_constraints_.push_back(std::move(ec));
}

// ============================================================================
// Common constraint patterns
// ============================================================================

void MultiPointConstraint::addRigidLink(std::span<const GlobalIndex> slave_dofs,
                                         GlobalIndex master_dof) {
    for (GlobalIndex slave : slave_dofs) {
        if (slave != master_dof) {
            addConstraint(slave, master_dof, 1.0, 0.0);
        }
    }
}

void MultiPointConstraint::addAverage(GlobalIndex slave_dof,
                                       std::span<const GlobalIndex> master_dofs) {
    if (master_dofs.empty()) return;

    double weight = 1.0 / static_cast<double>(master_dofs.size());
    std::vector<std::pair<GlobalIndex, double>> masters;
    masters.reserve(master_dofs.size());

    for (GlobalIndex m : master_dofs) {
        masters.emplace_back(m, weight);
    }

    addConstraint(slave_dof, std::move(masters), 0.0);
}

void MultiPointConstraint::addWeightedAverage(GlobalIndex slave_dof,
                                               std::span<const GlobalIndex> master_dofs,
                                               std::span<const double> weights) {
    if (master_dofs.size() != weights.size()) {
        CONSTRAINT_THROW("Master DOFs and weights must have same size");
    }
    if (master_dofs.empty()) return;

    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (std::abs(sum) < 1e-15) {
        CONSTRAINT_THROW("Sum of weights is zero");
    }

    std::vector<std::pair<GlobalIndex, double>> masters;
    masters.reserve(master_dofs.size());

    for (std::size_t i = 0; i < master_dofs.size(); ++i) {
        masters.emplace_back(master_dofs[i], weights[i] / sum);
    }

    addConstraint(slave_dof, std::move(masters), 0.0);
}

void MultiPointConstraint::addLinearCombination(
    GlobalIndex slave_dof,
    std::span<const std::pair<GlobalIndex, double>> terms,
    double constant) {
    std::vector<std::pair<GlobalIndex, double>> masters(terms.begin(), terms.end());
    addConstraint(slave_dof, std::move(masters), constant);
}

// ============================================================================
// Accessors
// ============================================================================

void MultiPointConstraint::clear() {
    equations_.clear();
    explicit_constraints_.clear();
}

// ============================================================================
// Validation
// ============================================================================

std::string MultiPointConstraint::validate() const {
    std::unordered_set<GlobalIndex> slave_set;

    // Check explicit constraints
    for (const auto& ec : explicit_constraints_) {
        if (ec.slave_dof < 0) {
            return "Invalid slave DOF index";
        }
        if (slave_set.find(ec.slave_dof) != slave_set.end()) {
            return "Duplicate slave DOF: " + std::to_string(ec.slave_dof);
        }
        slave_set.insert(ec.slave_dof);

        for (const auto& [master, weight] : ec.masters) {
            if (master < 0) {
                return "Invalid master DOF index";
            }
            if (master == ec.slave_dof) {
                return "Master DOF equals slave DOF: " + std::to_string(master);
            }
        }
    }

    // Check equations
    for (const auto& eq : equations_) {
        if (eq.terms.empty()) {
            return "Empty MPC equation";
        }

        // Check for zero equation
        bool all_zero = true;
        for (const auto& term : eq.terms) {
            if (std::abs(term.coefficient) > options_.coefficient_tolerance) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            return "MPC equation has all zero coefficients";
        }

        // Check DOF validity
        for (const auto& term : eq.terms) {
            if (term.dof < 0) {
                return "Invalid DOF index in equation";
            }
        }
    }

    return "";  // Valid
}

// ============================================================================
// Static factory methods
// ============================================================================

MultiPointConstraint MultiPointConstraint::rigidLink(
    std::span<const GlobalIndex> slave_dofs,
    GlobalIndex master_dof) {
    MultiPointConstraint mpc;
    mpc.addRigidLink(slave_dofs, master_dof);
    return mpc;
}

MultiPointConstraint MultiPointConstraint::average(
    GlobalIndex constrained_dof,
    std::span<const GlobalIndex> averaged_dofs) {
    MultiPointConstraint mpc;
    mpc.addAverage(constrained_dof, averaged_dofs);
    return mpc;
}

// ============================================================================
// Internal helpers
// ============================================================================

MultiPointConstraint::ExplicitConstraint
MultiPointConstraint::convertEquation(const MPCEquation& eq) const {
    ExplicitConstraint result;

    if (eq.terms.empty()) {
        CONSTRAINT_THROW("Empty MPC equation cannot be converted");
    }

    // Find slave DOF
    std::size_t slave_idx = 0;
    double max_abs_coeff = 0.0;

    if (options_.auto_select_slave && options_.prefer_largest_coefficient) {
        // Select DOF with largest absolute coefficient as slave
        for (std::size_t i = 0; i < eq.terms.size(); ++i) {
            double abs_coeff = std::abs(eq.terms[i].coefficient);
            if (abs_coeff > max_abs_coeff) {
                max_abs_coeff = abs_coeff;
                slave_idx = i;
            }
        }
    }

    double slave_coeff = eq.terms[slave_idx].coefficient;
    if (std::abs(slave_coeff) < options_.coefficient_tolerance) {
        CONSTRAINT_THROW("Slave coefficient is zero");
    }

    result.slave_dof = eq.terms[slave_idx].dof;
    result.inhomogeneity = eq.rhs / slave_coeff;

    // Build master list: u_slave = rhs/c_slave - sum_{j != slave}(c_j/c_slave * u_j)
    for (std::size_t i = 0; i < eq.terms.size(); ++i) {
        if (i != slave_idx) {
            double weight = -eq.terms[i].coefficient / slave_coeff;
            if (std::abs(weight) > options_.coefficient_tolerance) {
                result.masters.emplace_back(eq.terms[i].dof, weight);
            }
        }
    }

    return result;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
