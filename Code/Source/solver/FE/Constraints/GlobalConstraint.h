/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_GLOBALCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_GLOBALCONSTRAINT_H

/**
 * @file GlobalConstraint.h
 * @brief Global and integral constraints for finite element systems
 *
 * GlobalConstraint handles constraints that involve global properties of the
 * solution, such as:
 * - Zero-mean conditions (e.g., pressure uniqueness in incompressible flow)
 * - Volume conservation constraints
 * - Nullspace pinning for singular systems
 *
 * These constraints typically involve integral conditions over the domain:
 *   integral(u * phi) = c
 *
 * Implementation strategies:
 * 1. **Pin a DOF**: Set u_i = value (simplest, may break symmetry)
 * 2. **Mean-value constraint**: Add a Lagrange multiplier for integral(u) = 0
 * 3. **Nullspace projection**: Remove nullspace component from residual
 *
 * This class provides the ALGEBRAIC machinery for these constraints.
 *
 * @see AffineConstraints for constraint storage
 * @see LagrangeMultiplier for saddle-point formulations
 */

#include "Constraint.h"
#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief Strategy for enforcing global constraints
 */
enum class GlobalConstraintStrategy {
    PinSingleDof,        ///< Pin one DOF to fix the constant (simplest)
    LagrangeMultiplier,  ///< Add Lagrange multiplier for integral constraint
    NullspaceProjection, ///< Project out nullspace component
    WeightedMean         ///< Constrain weighted mean to specified value
};

/**
 * @brief Type of global constraint
 */
enum class GlobalConstraintType {
    ZeroMean,            ///< integral(u) = 0
    FixedMean,           ///< integral(u) = c
    VolumeConservation,  ///< integral(u) = V_0 (initial volume)
    NullspacePinning     ///< Remove specific nullspace component
};

/**
 * @brief Options for global constraint setup
 */
struct GlobalConstraintOptions {
    GlobalConstraintStrategy strategy{GlobalConstraintStrategy::PinSingleDof};
    double pinned_value{0.0};                  ///< Value for pinned DOF
    double tolerance{1e-14};                   ///< Tolerance for constraint satisfaction
    bool prefer_boundary_dof{true};            ///< Prefer boundary DOFs for pinning
    GlobalIndex explicit_pin_dof{-1};          ///< Explicitly specify DOF to pin (-1 = auto)
};

/**
 * @brief Information about a global constraint
 */
struct GlobalConstraintInfo {
    GlobalConstraintType type;
    GlobalConstraintStrategy strategy;
    GlobalIndex pinned_dof{-1};                ///< DOF that was pinned (if applicable)
    double target_value{0.0};                  ///< Target value for the constraint
    std::vector<double> nullspace_vector;      ///< Nullspace vector (if applicable)
};

/**
 * @brief Global/integral constraint for finite element systems
 *
 * GlobalConstraint handles constraints that involve global properties of the
 * solution. The most common use case is fixing pressure uniqueness in
 * incompressible flow by enforcing zero mean pressure.
 *
 * Usage:
 * @code
 *   // Zero-mean pressure constraint
 *   GlobalConstraint pressure_fix = GlobalConstraint::zeroMean(pressure_dofs);
 *
 *   // Apply to constraints
 *   AffineConstraints constraints;
 *   pressure_fix.apply(constraints);
 *   constraints.close();
 * @endcode
 *
 * For more sophisticated enforcement (e.g., Lagrange multipliers), see
 * the LagrangeMultiplier class.
 */
class GlobalConstraint : public Constraint {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (empty constraint)
     */
    GlobalConstraint();

    /**
     * @brief Construct with DOFs and type
     *
     * @param dofs DOFs involved in the global constraint
     * @param type Type of global constraint
     * @param options Constraint options
     */
    GlobalConstraint(std::vector<GlobalIndex> dofs,
                     GlobalConstraintType type,
                     const GlobalConstraintOptions& options = {});

    /**
     * @brief Construct with DOFs, weights, and target value
     *
     * For weighted mean constraint: sum(w_i * u_i) = target
     *
     * @param dofs DOFs involved
     * @param weights Weights for each DOF
     * @param target_value Target value for the weighted sum
     * @param options Constraint options
     */
    GlobalConstraint(std::vector<GlobalIndex> dofs,
                     std::vector<double> weights,
                     double target_value,
                     const GlobalConstraintOptions& options = {});

    /**
     * @brief Destructor
     */
    ~GlobalConstraint() override = default;

    /**
     * @brief Copy constructor
     */
    GlobalConstraint(const GlobalConstraint& other);

    /**
     * @brief Move constructor
     */
    GlobalConstraint(GlobalConstraint&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    GlobalConstraint& operator=(const GlobalConstraint& other);

    /**
     * @brief Move assignment
     */
    GlobalConstraint& operator=(GlobalConstraint&& other) noexcept;

    // =========================================================================
    // Constraint interface
    // =========================================================================

    /**
     * @brief Apply global constraint to AffineConstraints
     *
     * For PinSingleDof strategy: adds a single Dirichlet-like constraint.
     * For other strategies: may require additional system modifications.
     */
    void apply(AffineConstraints& constraints) const override;

    /**
     * @brief Get constraint type
     */
    [[nodiscard]] ConstraintType getType() const noexcept override {
        return ConstraintType::Global;
    }

    /**
     * @brief Get constraint information
     */
    [[nodiscard]] ConstraintInfo getInfo() const override;

    /**
     * @brief Clone this constraint
     */
    [[nodiscard]] std::unique_ptr<Constraint> clone() const override {
        return std::make_unique<GlobalConstraint>(*this);
    }

    // =========================================================================
    // Global constraint specific accessors
    // =========================================================================

    /**
     * @brief Get DOFs involved in this constraint
     */
    [[nodiscard]] const std::vector<GlobalIndex>& getDofs() const noexcept {
        return dofs_;
    }

    /**
     * @brief Get weights for DOFs
     */
    [[nodiscard]] const std::vector<double>& getWeights() const noexcept {
        return weights_;
    }

    /**
     * @brief Get constraint type
     */
    [[nodiscard]] GlobalConstraintType getGlobalType() const noexcept {
        return global_type_;
    }

    /**
     * @brief Get enforcement strategy
     */
    [[nodiscard]] GlobalConstraintStrategy getStrategy() const noexcept {
        return options_.strategy;
    }

    /**
     * @brief Get the DOF that was selected for pinning
     *
     * Only valid after apply() for PinSingleDof strategy.
     */
    [[nodiscard]] GlobalIndex getPinnedDof() const noexcept {
        return pinned_dof_;
    }

    /**
     * @brief Get target value for the constraint
     */
    [[nodiscard]] double getTargetValue() const noexcept {
        return target_value_;
    }

    /**
     * @brief Get detailed constraint information
     */
    [[nodiscard]] GlobalConstraintInfo getGlobalInfo() const;

    // =========================================================================
    // Nullspace operations
    // =========================================================================

    /**
     * @brief Get the nullspace vector for this constraint
     *
     * Returns the vector v such that the constraint is integral(v * u) = c.
     * For zero-mean: v = 1/N for all DOFs.
     *
     * @return Nullspace vector (size = n_dofs, indexed by DOF)
     */
    [[nodiscard]] std::vector<double> getNullspaceVector() const;

    /**
     * @brief Project a vector to remove the nullspace component
     *
     * For zero-mean: subtracts the mean value from all entries.
     *
     * @param vec Vector to project (modified in place)
     */
    void projectToConstrainedSpace(std::span<double> vec) const;

    /**
     * @brief Check if a vector satisfies the constraint
     *
     * @param vec Vector to check
     * @return True if constraint is satisfied within tolerance
     */
    [[nodiscard]] bool checkSatisfaction(std::span<const double> vec) const;

    /**
     * @brief Compute the constraint residual
     *
     * @param vec Vector to check
     * @return Constraint violation (should be near zero if satisfied)
     */
    [[nodiscard]] double computeResidual(std::span<const double> vec) const;

    // =========================================================================
    // Static factory methods
    // =========================================================================

    /**
     * @brief Create zero-mean constraint
     *
     * Enforces integral(u) = 0 over the specified DOFs.
     *
     * @param dofs DOFs involved in the constraint
     * @param options Constraint options
     * @return GlobalConstraint
     */
    static GlobalConstraint zeroMean(std::vector<GlobalIndex> dofs,
                                      const GlobalConstraintOptions& options = {});

    /**
     * @brief Create fixed-mean constraint
     *
     * Enforces integral(u) = target over the specified DOFs.
     *
     * @param dofs DOFs involved
     * @param target Target mean value
     * @param options Constraint options
     * @return GlobalConstraint
     */
    static GlobalConstraint fixedMean(std::vector<GlobalIndex> dofs,
                                       double target,
                                       const GlobalConstraintOptions& options = {});

    /**
     * @brief Create volume conservation constraint
     *
     * @param dofs Displacement DOFs
     * @param initial_volume Initial volume to conserve
     * @param options Constraint options
     * @return GlobalConstraint
     */
    static GlobalConstraint volumeConservation(std::vector<GlobalIndex> dofs,
                                                double initial_volume,
                                                const GlobalConstraintOptions& options = {});

    /**
     * @brief Create nullspace pinning constraint
     *
     * Pins the component of the solution in the direction of nullspace_vector.
     *
     * @param dofs DOFs involved
     * @param nullspace_vector The nullspace vector to pin
     * @param options Constraint options
     * @return GlobalConstraint
     */
    static GlobalConstraint nullspacePinning(std::vector<GlobalIndex> dofs,
                                              std::vector<double> nullspace_vector,
                                              const GlobalConstraintOptions& options = {});

    /**
     * @brief Create constraint by pinning a specific DOF
     *
     * Simplest strategy: just pin one DOF to a value.
     *
     * @param dof DOF to pin
     * @param value Value to pin to
     * @return GlobalConstraint
     */
    static GlobalConstraint pinDof(GlobalIndex dof, double value = 0.0);

private:
    // Data
    std::vector<GlobalIndex> dofs_;
    std::vector<double> weights_;
    GlobalConstraintType global_type_{GlobalConstraintType::ZeroMean};
    GlobalConstraintOptions options_;
    double target_value_{0.0};

    // Computed during apply()
    mutable GlobalIndex pinned_dof_{-1};

    // Internal helpers
    GlobalIndex selectDofToPin() const;
    void normalizeWeights();
};

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Compute mean value of a vector at specified DOFs
 *
 * @param vec Full vector
 * @param dofs DOF indices to include
 * @return Mean value
 */
[[nodiscard]] double computeMean(std::span<const double> vec,
                                  std::span<const GlobalIndex> dofs);

/**
 * @brief Compute weighted mean of a vector at specified DOFs
 *
 * @param vec Full vector
 * @param dofs DOF indices
 * @param weights Weights for each DOF
 * @return Weighted mean
 */
[[nodiscard]] double computeWeightedMean(std::span<const double> vec,
                                          std::span<const GlobalIndex> dofs,
                                          std::span<const double> weights);

/**
 * @brief Subtract mean from vector at specified DOFs
 *
 * @param vec Vector to modify
 * @param dofs DOF indices
 */
void subtractMean(std::span<double> vec, std::span<const GlobalIndex> dofs);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_GLOBALCONSTRAINT_H
