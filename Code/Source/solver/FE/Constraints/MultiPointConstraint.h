/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_MULTIPOINTCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_MULTIPOINTCONSTRAINT_H

/**
 * @file MultiPointConstraint.h
 * @brief General linear multi-point constraints (MPCs)
 *
 * MultiPointConstraint handles general linear constraints involving multiple
 * DOFs. These are the most general form of algebraic constraint:
 *
 *   sum_i (c_i * u_i) = b
 *
 * which is stored in canonical slave/master form:
 *
 *   u_slave = sum_j (a_j * u_master_j) + inhom
 *
 * Common use cases:
 * - Rigid body links (DOFs move together)
 * - Average value constraints (mean of several DOFs = constant)
 * - Interface coupling between subdomains
 * - User-defined linear relationships
 *
 * This class provides a convenient interface for constructing such constraints
 * and converting them to the canonical AffineConstraints form.
 *
 * @see AffineConstraints for constraint storage
 * @see HangingNodeConstraint for mesh adaptivity constraints
 */

#include "Constraint.h"
#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief A single term in an MPC equation: c * u_dof
 */
struct MPCTerm {
    GlobalIndex dof;     ///< DOF index
    double coefficient;  ///< Coefficient for this DOF

    MPCTerm() : dof(-1), coefficient(0.0) {}
    MPCTerm(GlobalIndex d, double c) : dof(d), coefficient(c) {}
};

/**
 * @brief A complete MPC equation: sum(c_i * u_i) = rhs
 */
struct MPCEquation {
    std::vector<MPCTerm> terms;  ///< Terms in the equation
    double rhs{0.0};             ///< Right-hand side value
    std::string name;            ///< Optional name for debugging

    /**
     * @brief Check if equation is valid
     */
    [[nodiscard]] bool isValid() const {
        return !terms.empty();
    }

    /**
     * @brief Add a term to the equation
     */
    void addTerm(GlobalIndex dof, double coeff) {
        terms.emplace_back(dof, coeff);
    }
};

/**
 * @brief Options for MPC handling
 */
struct MPCOptions {
    double coefficient_tolerance{1e-15};  ///< Tolerance for zero coefficients
    bool auto_select_slave{true};         ///< Automatically select slave DOF
    bool prefer_largest_coefficient{true}; ///< Prefer DOF with largest |coefficient| as slave
};

/**
 * @brief General multi-point constraint
 *
 * MultiPointConstraint provides a flexible interface for creating general
 * linear constraints between DOFs. Constraints can be specified either in
 * equation form or in explicit slave/master form.
 *
 * **Equation form**: sum(c_i * u_i) = b
 *   - The class automatically selects a slave DOF and converts to canonical form
 *
 * **Explicit form**: u_slave = sum(a_i * u_master_i) + inhom
 *   - User specifies slave and masters directly
 *
 * Usage (equation form):
 * @code
 *   MultiPointConstraint mpc;
 *
 *   // Constraint: u_0 + u_1 - 2*u_2 = 0 (u_2 is average of u_0 and u_1)
 *   MPCEquation eq;
 *   eq.addTerm(0, 1.0);
 *   eq.addTerm(1, 1.0);
 *   eq.addTerm(2, -2.0);
 *   eq.rhs = 0.0;
 *   mpc.addEquation(eq);
 *
 *   AffineConstraints constraints;
 *   mpc.apply(constraints);
 * @endcode
 *
 * Usage (explicit form):
 * @code
 *   MultiPointConstraint mpc;
 *
 *   // u_2 = 0.5*u_0 + 0.5*u_1  (same as above, explicit)
 *   mpc.addConstraint(2, {{0, 0.5}, {1, 0.5}});
 *
 *   AffineConstraints constraints;
 *   mpc.apply(constraints);
 * @endcode
 */
class MultiPointConstraint : public Constraint {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (empty constraint set)
     */
    MultiPointConstraint();

    /**
     * @brief Construct with MPC equations
     *
     * @param equations Vector of MPC equations
     * @param options MPC options
     */
    explicit MultiPointConstraint(std::vector<MPCEquation> equations,
                                   const MPCOptions& options = {});

    /**
     * @brief Destructor
     */
    ~MultiPointConstraint() override = default;

    /**
     * @brief Copy constructor
     */
    MultiPointConstraint(const MultiPointConstraint& other);

    /**
     * @brief Move constructor
     */
    MultiPointConstraint(MultiPointConstraint&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    MultiPointConstraint& operator=(const MultiPointConstraint& other);

    /**
     * @brief Move assignment
     */
    MultiPointConstraint& operator=(MultiPointConstraint&& other) noexcept;

    // =========================================================================
    // Constraint interface
    // =========================================================================

    /**
     * @brief Apply MPCs to AffineConstraints
     */
    void apply(AffineConstraints& constraints) const override;

    /**
     * @brief Get constraint type
     */
    [[nodiscard]] ConstraintType getType() const noexcept override {
        return ConstraintType::MultiPoint;
    }

    /**
     * @brief Get constraint information
     */
    [[nodiscard]] ConstraintInfo getInfo() const override;

    /**
     * @brief Clone this constraint
     */
    [[nodiscard]] std::unique_ptr<Constraint> clone() const override {
        return std::make_unique<MultiPointConstraint>(*this);
    }

    // =========================================================================
    // Equation-based interface
    // =========================================================================

    /**
     * @brief Add an MPC equation: sum(c_i * u_i) = b
     *
     * The equation will be converted to canonical form when apply() is called.
     *
     * @param equation The MPC equation
     */
    void addEquation(const MPCEquation& equation);

    /**
     * @brief Add an MPC equation from raw data
     *
     * @param dofs DOF indices
     * @param coefficients Coefficients for each DOF
     * @param rhs Right-hand side value
     */
    void addEquation(std::span<const GlobalIndex> dofs,
                     std::span<const double> coefficients,
                     double rhs = 0.0);

    /**
     * @brief Get all equations
     */
    [[nodiscard]] const std::vector<MPCEquation>& getEquations() const noexcept {
        return equations_;
    }

    // =========================================================================
    // Explicit slave/master interface
    // =========================================================================

    /**
     * @brief Add constraint in explicit slave/master form
     *
     * u_slave = sum(a_i * u_master_i) + inhom
     *
     * @param slave_dof Slave DOF index
     * @param masters Vector of (master_dof, weight) pairs
     * @param inhomogeneity Constant term
     */
    void addConstraint(GlobalIndex slave_dof,
                       std::vector<std::pair<GlobalIndex, double>> masters,
                       double inhomogeneity = 0.0);

    /**
     * @brief Add constraint with single master (simple equality)
     *
     * u_slave = weight * u_master + inhom
     */
    void addConstraint(GlobalIndex slave_dof,
                       GlobalIndex master_dof,
                       double weight = 1.0,
                       double inhomogeneity = 0.0);

    // =========================================================================
    // Common constraint patterns
    // =========================================================================

    /**
     * @brief Rigid link constraint: DOFs move together
     *
     * u_slave = u_master (no relative motion)
     *
     * @param slave_dofs DOFs to constrain
     * @param master_dof Reference DOF
     */
    void addRigidLink(std::span<const GlobalIndex> slave_dofs,
                      GlobalIndex master_dof);

    /**
     * @brief Average constraint: slave = average of masters
     *
     * u_slave = (1/n) * sum(u_master_i)
     *
     * @param slave_dof Slave DOF
     * @param master_dofs Master DOFs
     */
    void addAverage(GlobalIndex slave_dof,
                    std::span<const GlobalIndex> master_dofs);

    /**
     * @brief Weighted average constraint
     *
     * u_slave = sum(w_i * u_master_i) / sum(w_i)
     *
     * @param slave_dof Slave DOF
     * @param master_dofs Master DOFs
     * @param weights Weights for each master
     */
    void addWeightedAverage(GlobalIndex slave_dof,
                             std::span<const GlobalIndex> master_dofs,
                             std::span<const double> weights);

    /**
     * @brief Linear combination constraint
     *
     * u_slave = c_1 * u_1 + c_2 * u_2 + ... + c_n * u_n + b
     *
     * @param slave_dof Slave DOF
     * @param terms Vector of (dof, coefficient) pairs
     * @param constant Constant term
     */
    void addLinearCombination(GlobalIndex slave_dof,
                               std::span<const std::pair<GlobalIndex, double>> terms,
                               double constant = 0.0);

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get number of constraints
     */
    [[nodiscard]] std::size_t numConstraints() const noexcept {
        return equations_.size() + explicit_constraints_.size();
    }

    /**
     * @brief Check if empty
     */
    [[nodiscard]] bool empty() const noexcept {
        return equations_.empty() && explicit_constraints_.empty();
    }

    /**
     * @brief Clear all constraints
     */
    void clear();

    /**
     * @brief Get options
     */
    [[nodiscard]] const MPCOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Set options
     */
    void setOptions(const MPCOptions& options) {
        options_ = options;
    }

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate all constraints
     *
     * @return Error message, or empty string if valid
     */
    [[nodiscard]] std::string validate() const;

    // =========================================================================
    // Static factory methods
    // =========================================================================

    /**
     * @brief Create rigid link constraint
     *
     * @param slave_dofs DOFs to constrain
     * @param master_dof Reference DOF
     * @return MultiPointConstraint
     */
    static MultiPointConstraint rigidLink(std::span<const GlobalIndex> slave_dofs,
                                           GlobalIndex master_dof);

    /**
     * @brief Create average constraint
     *
     * @param constrained_dof DOF to be constrained to average
     * @param averaged_dofs DOFs whose average is computed
     * @return MultiPointConstraint
     */
    static MultiPointConstraint average(GlobalIndex constrained_dof,
                                         std::span<const GlobalIndex> averaged_dofs);

private:
    // Stored in equation form
    std::vector<MPCEquation> equations_;

    // Stored in explicit slave/master form
    struct ExplicitConstraint {
        GlobalIndex slave_dof;
        std::vector<std::pair<GlobalIndex, double>> masters;
        double inhomogeneity{0.0};
    };
    std::vector<ExplicitConstraint> explicit_constraints_;

    MPCOptions options_;

    // Convert equation to slave/master form
    ExplicitConstraint convertEquation(const MPCEquation& eq) const;
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_MULTIPOINTCONSTRAINT_H
