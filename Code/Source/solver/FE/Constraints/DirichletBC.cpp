/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DirichletBC.h"

#include <algorithm>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction - Direct DOF specification
// ============================================================================

DirichletBC::DirichletBC(GlobalIndex dof, double value)
    : dofs_{dof}, values_{value} {}

DirichletBC::DirichletBC(std::vector<GlobalIndex> dofs, double value)
    : dofs_(std::move(dofs)), values_(dofs_.size(), value) {}

DirichletBC::DirichletBC(std::vector<GlobalIndex> dofs, std::vector<double> values)
    : dofs_(std::move(dofs)), values_(std::move(values)) {
    if (dofs_.size() != values_.size()) {
        CONSTRAINT_THROW("DOFs and values must have same size");
    }
}

DirichletBC::DirichletBC(std::span<const GlobalIndex> dofs, std::span<const double> values)
    : dofs_(dofs.begin(), dofs.end()), values_(values.begin(), values.end()) {
    if (dofs_.size() != values_.size()) {
        CONSTRAINT_THROW("DOFs and values must have same size");
    }
}

// ============================================================================
// Construction - With options
// ============================================================================

DirichletBC::DirichletBC(std::vector<GlobalIndex> boundary_dofs,
                          double value,
                          const DirichletBCOptions& options)
    : dofs_(std::move(boundary_dofs)),
      values_(dofs_.size(), value),
      options_(options) {}

DirichletBC::DirichletBC(std::vector<GlobalIndex> boundary_dofs,
                          std::vector<std::array<double, 3>> dof_coordinates,
                          DirichletFunction func,
                          const DirichletBCOptions& options)
    : dofs_(std::move(boundary_dofs)),
      values_(dofs_.size()),
      coordinates_(std::move(dof_coordinates)),
      spatial_func_(std::move(func)),
      options_(options) {
    if (dofs_.size() != coordinates_.size()) {
        CONSTRAINT_THROW("DOFs and coordinates must have same size");
    }
    evaluateFunction();
}

DirichletBC::DirichletBC(std::vector<GlobalIndex> boundary_dofs,
                          std::vector<std::array<double, 3>> dof_coordinates,
                          TimeDependentDirichletFunction func,
                          double initial_time,
                          const DirichletBCOptions& options)
    : dofs_(std::move(boundary_dofs)),
      values_(dofs_.size()),
      coordinates_(std::move(dof_coordinates)),
      time_dependent_func_(std::move(func)),
      options_(options),
      current_time_(initial_time) {
    if (dofs_.size() != coordinates_.size()) {
        CONSTRAINT_THROW("DOFs and coordinates must have same size");
    }
    evaluateFunctionAtTime(initial_time);
}

// ============================================================================
// Copy/Move
// ============================================================================

DirichletBC::DirichletBC(const DirichletBC& other) = default;
DirichletBC::DirichletBC(DirichletBC&& other) noexcept = default;

DirichletBC& DirichletBC::operator=(const DirichletBC& other) {
    if (this != &other) {
        Constraint::operator=(other);
        dofs_ = other.dofs_;
        values_ = other.values_;
        coordinates_ = other.coordinates_;
        spatial_func_ = other.spatial_func_;
        time_dependent_func_ = other.time_dependent_func_;
        options_ = other.options_;
        current_time_ = other.current_time_;
    }
    return *this;
}

DirichletBC& DirichletBC::operator=(DirichletBC&& other) noexcept {
    if (this != &other) {
        Constraint::operator=(std::move(other));
        dofs_ = std::move(other.dofs_);
        values_ = std::move(other.values_);
        coordinates_ = std::move(other.coordinates_);
        spatial_func_ = std::move(other.spatial_func_);
        time_dependent_func_ = std::move(other.time_dependent_func_);
        options_ = std::move(other.options_);
        current_time_ = other.current_time_;
    }
    return *this;
}

// ============================================================================
// Constraint interface
// ============================================================================

void DirichletBC::apply(AffineConstraints& constraints) const {
    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        constraints.addLine(dofs_[i]);
        constraints.setInhomogeneity(dofs_[i], values_[i]);
    }
}

ConstraintInfo DirichletBC::getInfo() const {
    ConstraintInfo info;
    info.name = "DirichletBC";
    info.type = ConstraintType::Dirichlet;
    info.num_constrained_dofs = dofs_.size();
    info.is_time_dependent = time_dependent_func_.has_value();

    // Check if all values are zero
    info.is_homogeneous = true;
    for (double v : values_) {
        if (std::abs(v) > 1e-15) {
            info.is_homogeneous = false;
            break;
        }
    }

    return info;
}

bool DirichletBC::updateValues(AffineConstraints& constraints, double time) const {
    if (!time_dependent_func_) {
        return false;
    }

    // Update current time and re-evaluate
    current_time_ = time;
    evaluateFunctionAtTime(time);

    // Update constraints
    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        constraints.updateInhomogeneity(dofs_[i], values_[i]);
    }

    return true;
}

// ============================================================================
// Modification
// ============================================================================

void DirichletBC::setValue(double value) {
    std::fill(values_.begin(), values_.end(), value);
}

void DirichletBC::setValues(std::vector<double> values) {
    if (values.size() != dofs_.size()) {
        CONSTRAINT_THROW("Values size must match DOFs size");
    }
    values_ = std::move(values);
}

void DirichletBC::setTime(double time) {
    if (time_dependent_func_) {
        current_time_ = time;
        evaluateFunctionAtTime(time);
    }
}

void DirichletBC::addDof(GlobalIndex dof, double value) {
    dofs_.push_back(dof);
    values_.push_back(value);
}

void DirichletBC::addDofs(std::span<const GlobalIndex> dofs, std::span<const double> values) {
    if (dofs.size() != values.size()) {
        CONSTRAINT_THROW("DOFs and values must have same size");
    }
    dofs_.insert(dofs_.end(), dofs.begin(), dofs.end());
    values_.insert(values_.end(), values.begin(), values.end());
}

// ============================================================================
// Factory methods
// ============================================================================

DirichletBC DirichletBC::homogeneous(std::vector<GlobalIndex> dofs) {
    return DirichletBC(std::move(dofs), 0.0);
}

DirichletBC DirichletBC::singleComponent(std::vector<GlobalIndex> dofs,
                                          double value,
                                          int component,
                                          int total_components) {
    DirichletBCOptions options;
    options.component_mask = ComponentMask(component, total_components);
    return DirichletBC(std::move(dofs), value, options);
}

// ============================================================================
// Internal helpers
// ============================================================================

void DirichletBC::evaluateFunction() {
    if (!spatial_func_) return;

    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        const auto& coord = coordinates_[i];
        values_[i] = (*spatial_func_)(coord[0], coord[1], coord[2]);
    }
}

void DirichletBC::evaluateFunctionAtTime(double time) const {
    if (!time_dependent_func_) return;

    for (std::size_t i = 0; i < dofs_.size(); ++i) {
        const auto& coord = coordinates_[i];
        const_cast<std::vector<double>&>(values_)[i] =
            (*time_dependent_func_)(coord[0], coord[1], coord[2], time);
    }
}

} // namespace constraints
} // namespace FE
} // namespace svmp
