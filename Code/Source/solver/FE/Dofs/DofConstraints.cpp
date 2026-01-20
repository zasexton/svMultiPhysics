/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofConstraints.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

DofConstraints::DofConstraints() = default;
DofConstraints::~DofConstraints() = default;

DofConstraints::DofConstraints(DofConstraints&&) noexcept = default;
DofConstraints& DofConstraints::operator=(DofConstraints&&) noexcept = default;

// =============================================================================
// Adding Constraints
// =============================================================================

void DofConstraints::addDirichletBC(GlobalIndex dof, double value) {
    is_closed_ = false;

    ConstraintLine line;
    line.constrained_dof = dof;
    line.inhomogeneity = value;
    line.type = ConstraintType::Dirichlet;
    // No entries (master DOFs) for Dirichlet

    constraints_[dof] = std::move(line);

    // Rebuild constrained DOF set
    std::vector<GlobalIndex> dofs;
    for (const auto& [d, c] : constraints_) {
        dofs.push_back(d);
    }
    constrained_dofs_ = IndexSet(std::move(dofs));
}

void DofConstraints::addDirichletBC(std::span<const GlobalIndex> dofs, double value) {
    for (auto dof : dofs) {
        addDirichletBC(dof, value);
    }
}

void DofConstraints::addDirichletBC(std::span<const GlobalIndex> dofs,
                                     std::span<const double> values) {
    if (dofs.size() != values.size()) {
        throw FEException("DofConstraints::addDirichletBC: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        addDirichletBC(dofs[i], values[i]);
    }
}

void DofConstraints::addPeriodicBC(GlobalIndex master, GlobalIndex slave) {
    is_closed_ = false;

    ConstraintLine line;
    line.constrained_dof = slave;
    line.entries.push_back({master, 1.0});
    line.inhomogeneity = 0.0;
    line.type = ConstraintType::Periodic;

    constraints_[slave] = std::move(line);

    // Rebuild constrained DOF set
    std::vector<GlobalIndex> dofs;
    for (const auto& [d, c] : constraints_) {
        dofs.push_back(d);
    }
    constrained_dofs_ = IndexSet(std::move(dofs));
}

void DofConstraints::addPeriodicBC(std::span<const GlobalIndex> masters,
                                    std::span<const GlobalIndex> slaves) {
    if (masters.size() != slaves.size()) {
        throw FEException("DofConstraints::addPeriodicBC: size mismatch");
    }

    for (std::size_t i = 0; i < masters.size(); ++i) {
        addPeriodicBC(masters[i], slaves[i]);
    }
}

void DofConstraints::addLinearConstraint(GlobalIndex constrained_dof,
                                          std::span<const GlobalIndex> dofs,
                                          std::span<const double> coefficients,
                                          double inhomogeneity) {
    if (dofs.size() != coefficients.size()) {
        throw FEException("DofConstraints::addLinearConstraint: size mismatch");
    }

    is_closed_ = false;

    ConstraintLine line;
    line.constrained_dof = constrained_dof;
    line.inhomogeneity = inhomogeneity;
    line.type = ConstraintType::Linear;

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        if (std::abs(coefficients[i]) > 1e-15) {
            line.entries.push_back({dofs[i], coefficients[i]});
        }
    }

    constraints_[constrained_dof] = std::move(line);

    // Rebuild constrained DOF set
    std::vector<GlobalIndex> all_dofs;
    for (const auto& [d, c] : constraints_) {
        all_dofs.push_back(d);
    }
    constrained_dofs_ = IndexSet(std::move(all_dofs));
}

void DofConstraints::addHangingNodeConstraint(GlobalIndex hanging_dof,
                                               std::span<const GlobalIndex> parent_dofs,
                                               std::span<const double> weights) {
    if (parent_dofs.size() != weights.size()) {
        throw FEException("DofConstraints::addHangingNodeConstraint: size mismatch");
    }

    is_closed_ = false;

    ConstraintLine line;
    line.constrained_dof = hanging_dof;
    line.inhomogeneity = 0.0;
    line.type = ConstraintType::HangingNode;

    for (std::size_t i = 0; i < parent_dofs.size(); ++i) {
        if (std::abs(weights[i]) > 1e-15) {
            line.entries.push_back({parent_dofs[i], weights[i]});
        }
    }

    constraints_[hanging_dof] = std::move(line);

    // Rebuild constrained DOF set
    std::vector<GlobalIndex> all_dofs;
    for (const auto& [d, c] : constraints_) {
        all_dofs.push_back(d);
    }
    constrained_dofs_ = IndexSet(std::move(all_dofs));
}

void DofConstraints::clear() {
    constraints_.clear();
    constrained_dofs_ = IndexSet();
    is_closed_ = false;
}

// =============================================================================
// Query Methods
// =============================================================================

bool DofConstraints::isConstrained(GlobalIndex dof) const noexcept {
    return constraints_.count(dof) > 0;
}

std::optional<ConstraintLine> DofConstraints::getConstraintLine(GlobalIndex dof) const {
    auto it = constraints_.find(dof);
    if (it != constraints_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<double> DofConstraints::getDirichletValue(GlobalIndex dof) const {
    auto it = constraints_.find(dof);
    if (it != constraints_.end() && it->second.isDirichlet()) {
        return it->second.inhomogeneity;
    }
    return std::nullopt;
}

// =============================================================================
// Constraint Application
// =============================================================================

void DofConstraints::applyConstraints(AbstractMatrix& matrix, AbstractVector& rhs) const {
    // Apply constraints using symmetric elimination
    // For each constrained DOF i with constraint u_i = c_j * u_j + b:
    // 1. Modify RHS: rhs_k -= A_ki * b for all k
    // 2. Modify matrix: A_kj += A_ki * c_j for all k, j
    // 3. Zero row i and set A_ii = 1
    // 4. Set rhs_i = b

    for (const auto& [dof, constraint] : constraints_) {
        if (constraint.isDirichlet()) {
            // Simple Dirichlet: set row to identity, set RHS to value
            matrix.setRowToIdentity(dof);
            GlobalIndex indices[1] = {dof};
            double values[1] = {constraint.inhomogeneity};
            rhs.setValues(std::span<const GlobalIndex>(indices, 1),
                          std::span<const double>(values, 1));
        } else {
            // Linear constraint: more complex elimination
            // For now, use simple row zeroing (not fully symmetric)
            matrix.zeroRow(dof);

            // Set diagonal to 1
            GlobalIndex rows[1] = {dof};
            GlobalIndex cols[1] = {dof};
            double vals[1] = {1.0};
            matrix.addValues(std::span<const GlobalIndex>(rows, 1),
                            std::span<const GlobalIndex>(cols, 1),
                            std::span<const double>(vals, 1));

            // Add contributions from master DOFs
            for (const auto& entry : constraint.entries) {
                GlobalIndex mcols[1] = {entry.dof};
                double mvals[1] = {-entry.coefficient};
                matrix.addValues(std::span<const GlobalIndex>(rows, 1),
                                std::span<const GlobalIndex>(mcols, 1),
                                std::span<const double>(mvals, 1));
            }

            // Set RHS to inhomogeneity
            GlobalIndex rhs_idx[1] = {dof};
            double rhs_val[1] = {constraint.inhomogeneity};
            rhs.setValues(std::span<const GlobalIndex>(rhs_idx, 1),
                          std::span<const double>(rhs_val, 1));
        }
    }
}

void DofConstraints::applyToMatrix(AbstractMatrix& matrix) const {
    for (const auto& [dof, constraint] : constraints_) {
        matrix.setRowToIdentity(dof);
    }
}

void DofConstraints::applyToRhs(AbstractVector& rhs) const {
    for (const auto& [dof, constraint] : constraints_) {
        if (constraint.isDirichlet()) {
            GlobalIndex indices[1] = {dof};
            double values[1] = {constraint.inhomogeneity};
            rhs.setValues(std::span<const GlobalIndex>(indices, 1),
                          std::span<const double>(values, 1));
        } else {
            // For linear constraints, compute value from masters
            double value = constraint.inhomogeneity;
            for (const auto& entry : constraint.entries) {
                value += entry.coefficient * rhs.getValue(entry.dof);
            }
            GlobalIndex indices[1] = {dof};
            double values[1] = {value};
            rhs.setValues(std::span<const GlobalIndex>(indices, 1),
                          std::span<const double>(values, 1));
        }
    }
}

void DofConstraints::applySolutionConstraints(AbstractVector& solution) const {
    // After solving, recover constrained DOF values
    for (const auto& [dof, constraint] : constraints_) {
        double value = constraint.inhomogeneity;
        for (const auto& entry : constraint.entries) {
            value += entry.coefficient * solution.getValue(entry.dof);
        }
        GlobalIndex indices[1] = {dof};
        double values[1] = {value};
        solution.setValues(std::span<const GlobalIndex>(indices, 1),
                           std::span<const double>(values, 1));
    }
}

// =============================================================================
// Condensation
// =============================================================================

void DofConstraints::buildConstraintMatrix(
    GlobalIndex n_total_dofs,
    std::vector<GlobalIndex>& row_offsets,
    std::vector<GlobalIndex>& col_indices,
    std::vector<double>& values) const {

    // Build C such that u_full = C * u_reduced + u_inhom
    // C has n_total_dofs rows and n_unconstrained columns

    row_offsets.clear();
    col_indices.clear();
    values.clear();

    // Get reduced mapping
    auto reduced_map = getReducedMapping(n_total_dofs);

    row_offsets.reserve(static_cast<std::size_t>(n_total_dofs + 1));
    row_offsets.push_back(0);

    for (GlobalIndex i = 0; i < n_total_dofs; ++i) {
        auto it = constraints_.find(i);
        if (it == constraints_.end()) {
            // Unconstrained DOF: identity
            GlobalIndex reduced_idx = reduced_map[static_cast<std::size_t>(i)];
            if (reduced_idx >= 0) {
                col_indices.push_back(reduced_idx);
                values.push_back(1.0);
            }
        } else {
            // Constrained DOF: add master DOF contributions
            const auto& constraint = it->second;
            for (const auto& entry : constraint.entries) {
                GlobalIndex master_reduced = reduced_map[static_cast<std::size_t>(entry.dof)];
                if (master_reduced >= 0) {
                    col_indices.push_back(master_reduced);
                    values.push_back(entry.coefficient);
                }
            }
        }
        row_offsets.push_back(static_cast<GlobalIndex>(col_indices.size()));
    }
}

std::vector<GlobalIndex> DofConstraints::getReducedMapping(GlobalIndex n_total_dofs) const {
    std::vector<GlobalIndex> mapping(static_cast<std::size_t>(n_total_dofs), -1);

    GlobalIndex reduced_idx = 0;
    for (GlobalIndex i = 0; i < n_total_dofs; ++i) {
        if (constraints_.count(i) == 0) {
            mapping[static_cast<std::size_t>(i)] = reduced_idx++;
        }
    }

    return mapping;
}

GlobalIndex DofConstraints::numUnconstrainedDofs(GlobalIndex n_total_dofs) const {
    return n_total_dofs - static_cast<GlobalIndex>(constraints_.size());
}

// =============================================================================
// Closure and Validation
// =============================================================================

void DofConstraints::close() {
    if (is_closed_) return;

    // Resolve chains transitively: if A depends on B and B depends on C,
    // then A should directly depend on C (and only on unconstrained DOFs).
    //
    // This must be independent of map iteration order, so we close
    // dependencies recursively and memoize closed lines.
    std::unordered_set<GlobalIndex> visiting;
    std::unordered_set<GlobalIndex> closed;
    closed.reserve(constraints_.size());

    for (const auto& kv : constraints_) {
        visiting.clear();
        closeConstraint(kv.first, visiting, closed);
    }

    // After closure, validate to catch any circular dependencies.
    const auto errors = validate();
    if (!errors.empty()) {
        throw FEException("DofConstraints::close: validation failed:\n" + errors);
    }

    is_closed_ = true;
}

void DofConstraints::closeConstraint(GlobalIndex dof,
                                      std::unordered_set<GlobalIndex>& visiting,
                                      std::unordered_set<GlobalIndex>& closed) {
    if (closed.count(dof) > 0) {
        return;
    }

    auto it_line = constraints_.find(dof);
    if (it_line == constraints_.end()) {
        return;
    }

    // Prevent infinite recursion / circular dependencies.
    if (visiting.count(dof) > 0) {
        return;
    }
    visiting.insert(dof);

    auto& line = it_line->second;

    // First, ensure all constrained masters are themselves closed.
    for (const auto& entry : line.entries) {
        if (constraints_.count(entry.dof) > 0) {
            closeConstraint(entry.dof, visiting, closed);
        }
    }

    // Now substitute any constrained masters using their closed form.
    std::unordered_map<GlobalIndex, double> coeffs;
    coeffs.reserve(line.entries.size());

    double new_inhom = line.inhomogeneity;

    for (const auto& entry : line.entries) {
        auto it_master = constraints_.find(entry.dof);
        if (it_master != constraints_.end()) {
            const auto& master = it_master->second;
            new_inhom += entry.coefficient * master.inhomogeneity;

            for (const auto& master_entry : master.entries) {
                coeffs[master_entry.dof] += entry.coefficient * master_entry.coefficient;
            }
        } else {
            coeffs[entry.dof] += entry.coefficient;
        }
    }

    std::vector<ConstraintEntry> new_entries;
    new_entries.reserve(coeffs.size());

    for (const auto& [master_dof, coeff] : coeffs) {
        if (std::abs(coeff) > 1e-15) {
            new_entries.push_back({master_dof, coeff});
        }
    }

    // Deterministic order for reproducibility.
    std::sort(new_entries.begin(), new_entries.end(),
              [](const ConstraintEntry& a, const ConstraintEntry& b) {
                  return a.dof < b.dof;
              });

    line.entries = std::move(new_entries);
    line.inhomogeneity = new_inhom;

    visiting.erase(dof);
    closed.insert(dof);
}

std::string DofConstraints::validate() const {
    std::ostringstream errors;

    // Check for circular dependencies
    for (const auto& [dof, constraint] : constraints_) {
        std::unordered_set<GlobalIndex> visited;
        visited.insert(dof);

        std::vector<GlobalIndex> to_check;
        for (const auto& entry : constraint.entries) {
            to_check.push_back(entry.dof);
        }

        while (!to_check.empty()) {
            GlobalIndex check_dof = to_check.back();
            to_check.pop_back();

            if (visited.count(check_dof) > 0) {
                errors << "Circular dependency detected involving DOF " << dof << "\n";
                break;
            }
            visited.insert(check_dof);

            auto it = constraints_.find(check_dof);
            if (it != constraints_.end()) {
                for (const auto& entry : it->second.entries) {
                    to_check.push_back(entry.dof);
                }
            }
        }
    }

    // Check for negative DOF indices
    for (const auto& [dof, constraint] : constraints_) {
        if (dof < 0) {
            errors << "Invalid constrained DOF index: " << dof << "\n";
        }
        for (const auto& entry : constraint.entries) {
            if (entry.dof < 0) {
                errors << "Invalid master DOF index: " << entry.dof
                       << " in constraint for DOF " << dof << "\n";
            }
        }
    }

    return errors.str();
}

} // namespace dofs
} // namespace FE
} // namespace svmp
