/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINT_H

/**
 * @file Constraint.h
 * @brief Abstract base class for constraint definitions
 *
 * This header defines the polymorphic Constraint interface that all constraint
 * types derive from. The design supports a clean separation between:
 *
 * 1. Constraint definition (what DOFs are constrained and how)
 * 2. Constraint application (enforcement via AffineConstraints)
 *
 * Constraint subclasses:
 * - DirichletBC: Essential boundary conditions (algebraic)
 * - PeriodicBC: Periodic boundary constraints (algebraic)
 * - HangingNodeConstraint: Mesh adaptivity constraints (algebraic)
 * - MultiPointConstraint: General linear DOF relations (algebraic)
 * - GlobalConstraint: Global/integral constraints (algebraic)
 *
 * Boundary term classes (NOT Constraint subclasses):
 * - NeumannBC: Natural boundary conditions (variational)
 * - RobinBC: Mixed boundary conditions (variational)
 *
 * The key distinction:
 * - Algebraic constraints (Constraint) define DOF relationships
 * - Boundary terms define contributions to the weak form
 */

#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <string>
#include <memory>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofHandler;
    class DofMap;
}

namespace constraints {

/**
 * @brief Type of constraint for classification
 */
enum class ConstraintType : std::uint8_t {
    Dirichlet,        ///< Essential BC: u = value
    Periodic,         ///< Periodic BC: u_slave = u_master (possibly with transformation)
    HangingNode,      ///< From mesh adaptivity: u_hanging = interpolation(u_parents)
    MultiPoint,       ///< General MPC: u_slave = sum(a_i * u_i) + b
    Global,           ///< Global constraint: integral condition on all DOFs
    Custom            ///< User-defined constraint type
};

/**
 * @brief Convert constraint type to string
 */
inline const char* constraintTypeToString(ConstraintType type) noexcept {
    switch (type) {
        case ConstraintType::Dirichlet:    return "Dirichlet";
        case ConstraintType::Periodic:     return "Periodic";
        case ConstraintType::HangingNode:  return "HangingNode";
        case ConstraintType::MultiPoint:   return "MultiPoint";
        case ConstraintType::Global:       return "Global";
        case ConstraintType::Custom:       return "Custom";
        default:                           return "Unknown";
    }
}

/**
 * @brief Mask for selecting field components
 *
 * Used to specify which components of a vector field are constrained.
 * For example, fixing only the x-component of velocity.
 */
class ComponentMask {
public:
    /**
     * @brief Default constructor - all components selected
     */
    ComponentMask() : all_selected_(true), num_components_(0) {}

    /**
     * @brief Construct with explicit component selection
     */
    explicit ComponentMask(std::vector<bool> mask)
        : mask_(std::move(mask)), all_selected_(false), num_components_(mask_.size()) {}

    /**
     * @brief Construct selecting a single component
     */
    ComponentMask(int component, int total_components)
        : mask_(static_cast<std::size_t>(total_components), false),
          all_selected_(false),
          num_components_(static_cast<std::size_t>(total_components)) {
        if (component >= 0 && component < total_components) {
            mask_[static_cast<std::size_t>(component)] = true;
        }
    }

    /**
     * @brief Check if a component is selected
     */
    [[nodiscard]] bool operator[](std::size_t component) const {
        if (all_selected_) return true;
        if (component >= mask_.size()) return false;
        return mask_[component];
    }

    /**
     * @brief Check if all components are selected
     */
    [[nodiscard]] bool allSelected() const noexcept { return all_selected_; }

    /**
     * @brief Get number of components in the mask
     */
    [[nodiscard]] std::size_t size() const noexcept { return num_components_; }

    /**
     * @brief Count selected components
     */
    [[nodiscard]] std::size_t countSelected() const {
        if (all_selected_) return num_components_;
        std::size_t count = 0;
        for (bool b : mask_) {
            if (b) ++count;
        }
        return count;
    }

    /**
     * @brief Get indices of selected components
     */
    [[nodiscard]] std::vector<int> getSelectedComponents() const {
        std::vector<int> result;
        if (all_selected_) {
            for (std::size_t i = 0; i < num_components_; ++i) {
                result.push_back(static_cast<int>(i));
            }
        } else {
            for (std::size_t i = 0; i < mask_.size(); ++i) {
                if (mask_[i]) {
                    result.push_back(static_cast<int>(i));
                }
            }
        }
        return result;
    }

    /**
     * @brief Static factory for mask selecting all components
     */
    static ComponentMask all(int num_components) {
        return ComponentMask(std::vector<bool>(static_cast<std::size_t>(num_components), true));
    }

    /**
     * @brief Static factory for mask selecting no components
     */
    static ComponentMask none(int num_components) {
        return ComponentMask(std::vector<bool>(static_cast<std::size_t>(num_components), false));
    }

private:
    std::vector<bool> mask_;
    bool all_selected_;
    std::size_t num_components_;
};

/**
 * @brief Information about constraint application
 */
struct ConstraintInfo {
    std::string name;                    ///< Human-readable name
    ConstraintType type;                 ///< Type classification
    std::size_t num_constrained_dofs{0}; ///< Number of DOFs affected
    bool is_time_dependent{false};       ///< Whether values change with time
    bool is_homogeneous{true};           ///< Whether all inhomogeneities are zero
};

/**
 * @brief Abstract base class for constraint definitions
 *
 * Constraint objects encapsulate the definition of DOF constraints.
 * They provide a `apply()` method that registers the constraint lines
 * with an AffineConstraints object.
 *
 * Typical workflow:
 * @code
 *   // Create constraint
 *   auto bc = std::make_unique<DirichletBC>(dof_handler, boundary_id, value);
 *
 *   // Apply to constraint manager
 *   AffineConstraints constraints;
 *   bc->apply(constraints);
 *
 *   // Finalize
 *   constraints.close();
 * @endcode
 *
 * Subclasses must implement:
 * - apply(): Register constraints with AffineConstraints
 * - getType(): Return the constraint type
 * - getInfo(): Return metadata about the constraint
 */
class Constraint {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~Constraint() = default;

    /**
     * @brief Apply this constraint to an AffineConstraints object
     *
     * This is the main entry point. Implementations should call
     * AffineConstraints::addLine(), addEntry(), setInhomogeneity()
     * to register all constraint lines.
     *
     * @param constraints The constraint manager to add to
     * @throws FEException if application fails
     */
    virtual void apply(AffineConstraints& constraints) const = 0;

    /**
     * @brief Get the constraint type
     */
    [[nodiscard]] virtual ConstraintType getType() const noexcept = 0;

    /**
     * @brief Get information about this constraint
     */
    [[nodiscard]] virtual ConstraintInfo getInfo() const = 0;

    /**
     * @brief Get the name of this constraint (for debugging/logging)
     */
    [[nodiscard]] virtual std::string getName() const {
        return constraintTypeToString(getType());
    }

    /**
     * @brief Update constraint values for a new time step
     *
     * For time-dependent constraints, this updates the values without
     * needing to rebuild the entire constraint structure.
     *
     * @param constraints The constraint manager to update
     * @param time The current simulation time
     * @return true if any values were updated
     */
    [[nodiscard]] virtual bool updateValues(AffineConstraints& constraints, double time) const {
        (void)constraints;
        (void)time;
        return false;  // Default: not time-dependent
    }

    /**
     * @brief Check if this constraint is time-dependent
     */
    [[nodiscard]] virtual bool isTimeDependent() const noexcept { return false; }

    /**
     * @brief Clone this constraint (polymorphic copy)
     */
    [[nodiscard]] virtual std::unique_ptr<Constraint> clone() const = 0;

protected:
    /**
     * @brief Protected default constructor (use derived classes)
     */
    Constraint() = default;

    /**
     * @brief Protected copy constructor
     */
    Constraint(const Constraint&) = default;

    /**
     * @brief Protected move constructor
     */
    Constraint(Constraint&&) = default;

    /**
     * @brief Protected copy assignment operator
     */
    Constraint& operator=(const Constraint&) = default;

    /**
     * @brief Protected move assignment operator
     */
    Constraint& operator=(Constraint&&) = default;
};

/**
 * @brief Container for multiple constraints
 *
 * Manages a collection of constraints and provides unified application.
 */
class ConstraintCollection {
public:
    /**
     * @brief Default constructor
     */
    ConstraintCollection() = default;

    /**
     * @brief Add a constraint to the collection
     */
    void add(std::unique_ptr<Constraint> constraint) {
        constraints_.push_back(std::move(constraint));
    }

    /**
     * @brief Add a constraint (by reference, will be cloned)
     */
    void add(const Constraint& constraint) {
        constraints_.push_back(constraint.clone());
    }

    /**
     * @brief Apply all constraints to an AffineConstraints object
     */
    void applyAll(AffineConstraints& affine_constraints) const {
        for (const auto& constraint : constraints_) {
            constraint->apply(affine_constraints);
        }
    }

    /**
     * @brief Update all time-dependent constraints
     *
     * @param affine_constraints The constraint manager
     * @param time Current simulation time
     * @return Number of constraints that were updated
     */
    std::size_t updateAll(AffineConstraints& affine_constraints, double time) const {
        std::size_t count = 0;
        for (const auto& constraint : constraints_) {
            if (constraint->updateValues(affine_constraints, time)) {
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Get number of constraints in collection
     */
    [[nodiscard]] std::size_t size() const noexcept { return constraints_.size(); }

    /**
     * @brief Check if collection is empty
     */
    [[nodiscard]] bool empty() const noexcept { return constraints_.empty(); }

    /**
     * @brief Clear all constraints
     */
    void clear() { constraints_.clear(); }

    /**
     * @brief Get constraint by index
     */
    [[nodiscard]] const Constraint& operator[](std::size_t idx) const {
        return *constraints_[idx];
    }

    /**
     * @brief Get begin iterator
     */
    [[nodiscard]] auto begin() const { return constraints_.begin(); }

    /**
     * @brief Get end iterator
     */
    [[nodiscard]] auto end() const { return constraints_.end(); }

private:
    std::vector<std::unique_ptr<Constraint>> constraints_;
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINT_H
