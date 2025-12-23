/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_DIRICHLETBC_H
#define SVMP_FE_CONSTRAINTS_DIRICHLETBC_H

/**
 * @file DirichletBC.h
 * @brief Essential (Dirichlet) boundary conditions
 *
 * DirichletBC represents essential boundary conditions that fix DOF values:
 *
 *   u_i = g(x)   on Gamma_D
 *
 * This is an ALGEBRAIC constraint - it defines a relationship between DOF values,
 * not a contribution to the weak form. The constraint is enforced by setting
 * u_i = value with no master DOFs in the constraint line.
 *
 * Features:
 * - Constant value BCs
 * - Function-based BCs (spatially varying)
 * - Time-dependent BCs
 * - Component-wise BCs (e.g., fix only x-velocity)
 * - Multiple boundary IDs
 *
 * Module boundary:
 * - This module OWNS the constraint definition
 * - This module does NOT OWN DOF coordinates (queries from DofHandler if needed)
 */

#include "Constraint.h"
#include "AffineConstraints.h"
#include "Core/Types.h"

#include <vector>
#include <span>
#include <functional>
#include <optional>
#include <array>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofHandler;
}

namespace constraints {

/**
 * @brief Function type for spatially-varying Dirichlet values
 *
 * Takes physical coordinates (x, y, z) and returns the BC value.
 */
using DirichletFunction = std::function<double(double x, double y, double z)>;

/**
 * @brief Function type for time-dependent Dirichlet values
 *
 * Takes coordinates and time, returns the BC value.
 */
using TimeDependentDirichletFunction =
    std::function<double(double x, double y, double z, double t)>;

/**
 * @brief Function type for vector-valued Dirichlet values
 *
 * Takes coordinates and returns a vector of values (one per component).
 */
using VectorDirichletFunction =
    std::function<std::vector<double>(double x, double y, double z)>;

/**
 * @brief Options for DirichletBC
 */
struct DirichletBCOptions {
    ComponentMask component_mask{};     ///< Which components to constrain (all by default)
    bool use_interpolation{true};       ///< Interpolate function at DOF coordinates
    double time{0.0};                   ///< Current time for time-dependent BCs
};

/**
 * @brief Essential (Dirichlet) boundary condition constraint
 *
 * DirichletBC constrains DOFs to prescribed values. It supports several
 * modes of specification:
 *
 * 1. **Direct DOFs with constant value:**
 *    @code
 *    DirichletBC bc(dofs, value);
 *    @endcode
 *
 * 2. **Direct DOFs with individual values:**
 *    @code
 *    DirichletBC bc(dofs, values);
 *    @endcode
 *
 * 3. **Function-based (requires DOF coordinates):**
 *    @code
 *    DirichletBC bc(dof_handler, boundary_id, [](double x, double y, double z) {
 *        return std::sin(x);
 *    });
 *    @endcode
 *
 * 4. **Time-dependent:**
 *    @code
 *    DirichletBC bc(dof_handler, boundary_id,
 *        [](double x, double y, double z, double t) {
 *            return t * std::sin(x);
 *        }, current_time);
 *    @endcode
 *
 * Usage:
 * @code
 *   AffineConstraints constraints;
 *   DirichletBC bc({0, 1, 2}, 1.0);  // Fix DOFs 0,1,2 to value 1.0
 *   bc.apply(constraints);
 *   constraints.close();
 * @endcode
 */
class DirichletBC : public Constraint {
public:
    // =========================================================================
    // Construction - Direct DOF specification
    // =========================================================================

    /**
     * @brief Construct with single DOF and value
     */
    DirichletBC(GlobalIndex dof, double value);

    /**
     * @brief Construct with multiple DOFs and same value
     */
    DirichletBC(std::vector<GlobalIndex> dofs, double value);

    /**
     * @brief Construct with multiple DOFs and corresponding values
     */
    DirichletBC(std::vector<GlobalIndex> dofs, std::vector<double> values);

    /**
     * @brief Construct from spans (non-owning)
     */
    DirichletBC(std::span<const GlobalIndex> dofs, std::span<const double> values);

    // =========================================================================
    // Construction - DofHandler-based (requires coordinate info)
    // =========================================================================

    /**
     * @brief Construct with constant value on boundary
     *
     * @param boundary_dofs DOFs on the boundary
     * @param value Constant value to prescribe
     * @param options BC options
     */
    DirichletBC(std::vector<GlobalIndex> boundary_dofs,
                double value,
                const DirichletBCOptions& options);

    /**
     * @brief Construct with spatially-varying function
     *
     * @param boundary_dofs DOFs on the boundary
     * @param dof_coordinates Coordinates of each DOF (size = 3 * boundary_dofs.size())
     * @param func Function returning value at each coordinate
     * @param options BC options
     */
    DirichletBC(std::vector<GlobalIndex> boundary_dofs,
                std::vector<std::array<double, 3>> dof_coordinates,
                DirichletFunction func,
                const DirichletBCOptions& options = {});

    /**
     * @brief Construct with time-dependent function
     *
     * @param boundary_dofs DOFs on the boundary
     * @param dof_coordinates Coordinates of each DOF
     * @param func Function returning value at each coordinate and time
     * @param initial_time Initial time for evaluation
     * @param options BC options
     */
    DirichletBC(std::vector<GlobalIndex> boundary_dofs,
                std::vector<std::array<double, 3>> dof_coordinates,
                TimeDependentDirichletFunction func,
                double initial_time,
                const DirichletBCOptions& options = {});

    /**
     * @brief Destructor
     */
    ~DirichletBC() override = default;

    /**
     * @brief Copy constructor
     */
    DirichletBC(const DirichletBC& other);

    /**
     * @brief Move constructor
     */
    DirichletBC(DirichletBC&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    DirichletBC& operator=(const DirichletBC& other);

    /**
     * @brief Move assignment
     */
    DirichletBC& operator=(DirichletBC&& other) noexcept;

    // =========================================================================
    // Constraint interface
    // =========================================================================

    /**
     * @brief Apply Dirichlet constraints to AffineConstraints
     */
    void apply(AffineConstraints& constraints) const override;

    /**
     * @brief Get constraint type (Dirichlet)
     */
    [[nodiscard]] ConstraintType getType() const noexcept override {
        return ConstraintType::Dirichlet;
    }

    /**
     * @brief Get constraint information
     */
    [[nodiscard]] ConstraintInfo getInfo() const override;

    /**
     * @brief Update values for new time (time-dependent BCs)
     */
    [[nodiscard]] bool updateValues(AffineConstraints& constraints, double time) const override;

    /**
     * @brief Check if this BC is time-dependent
     */
    [[nodiscard]] bool isTimeDependent() const noexcept override {
        return time_dependent_func_.has_value();
    }

    /**
     * @brief Clone this constraint
     */
    [[nodiscard]] std::unique_ptr<Constraint> clone() const override {
        return std::make_unique<DirichletBC>(*this);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get constrained DOF indices
     */
    [[nodiscard]] std::span<const GlobalIndex> getDofs() const noexcept {
        return dofs_;
    }

    /**
     * @brief Get prescribed values
     */
    [[nodiscard]] std::span<const double> getValues() const noexcept {
        return values_;
    }

    /**
     * @brief Get current time
     */
    [[nodiscard]] double getTime() const noexcept { return current_time_; }

    /**
     * @brief Get component mask
     */
    [[nodiscard]] const ComponentMask& getComponentMask() const noexcept {
        return options_.component_mask;
    }

    // =========================================================================
    // Modification
    // =========================================================================

    /**
     * @brief Set new constant value for all DOFs
     */
    void setValue(double value);

    /**
     * @brief Set new values for all DOFs
     */
    void setValues(std::vector<double> values);

    /**
     * @brief Set current time and re-evaluate function
     */
    void setTime(double time);

    /**
     * @brief Add more DOFs with a value
     */
    void addDof(GlobalIndex dof, double value);

    /**
     * @brief Add more DOFs with values
     */
    void addDofs(std::span<const GlobalIndex> dofs, std::span<const double> values);

    // =========================================================================
    // Factory methods
    // =========================================================================

    /**
     * @brief Create Dirichlet BC fixing all DOFs to zero
     */
    static DirichletBC homogeneous(std::vector<GlobalIndex> dofs);

    /**
     * @brief Create Dirichlet BC for a single component of a vector field
     */
    static DirichletBC singleComponent(std::vector<GlobalIndex> dofs,
                                        double value,
                                        int component,
                                        int total_components);

private:
    // Core data
    std::vector<GlobalIndex> dofs_;
    std::vector<double> values_;

    // Optional coordinate data (for function-based BCs)
    std::vector<std::array<double, 3>> coordinates_;

    // Function pointers (optional)
    std::optional<DirichletFunction> spatial_func_;
    std::optional<TimeDependentDirichletFunction> time_dependent_func_;

    // Options
    DirichletBCOptions options_;

    // Current time for time-dependent BCs
    mutable double current_time_{0.0};

    // Helper to evaluate function at coordinates
    void evaluateFunction();
    void evaluateFunctionAtTime(double time) const;
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_DIRICHLETBC_H
