/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_ROBINBC_H
#define SVMP_FE_CONSTRAINTS_ROBINBC_H

/**
 * @file RobinBC.h
 * @brief Robin (mixed) boundary conditions
 *
 * RobinBC handles mixed boundary conditions of the form:
 *
 *   alpha * u + beta * (du/dn) = g   on boundary
 *
 * where:
 * - alpha is the Dirichlet-like coefficient
 * - beta is the Neumann-like coefficient
 * - du/dn is the outward normal derivative
 * - g is the prescribed boundary value
 *
 * Special cases:
 * - alpha = 1, beta = 0: Dirichlet BC (u = g)
 * - alpha = 0, beta = 1: Neumann BC (du/dn = g)
 * - alpha, beta != 0: Robin/mixed BC
 *
 * Robin conditions appear in:
 * - Heat transfer: convection boundary (alpha = h, g = h*T_inf)
 * - Absorbing boundaries for wave equations
 * - Elastic foundation (Winkler) conditions
 * - Impedance boundary conditions
 *
 * IMPORTANT: RobinBC is NOT an algebraic constraint on DOFs!
 * It is a boundary condition that contributes to both the system matrix
 * and the RHS during assembly. Similar to NeumannBC, it should be
 * integrated during the assembly loop over boundary elements.
 *
 * @see NeumannBC for pure Neumann conditions
 * @see DirichletBC for essential boundary conditions
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "AffineConstraints.h"  // For CONSTRAINT_THROW macro

#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <array>
#include <optional>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief Robin BC data at a single point
 */
struct RobinData {
    double alpha{1.0};   ///< Coefficient of u
    double beta{1.0};    ///< Coefficient of du/dn
    double g{0.0};       ///< RHS value

    /**
     * @brief Check if this is a valid Robin BC
     */
    [[nodiscard]] bool isValid() const {
        return std::abs(alpha) + std::abs(beta) > 1e-15;  // At least one nonzero
    }

    /**
     * @brief Check if this is a pure Dirichlet BC
     */
    [[nodiscard]] bool isDirichlet() const {
        return std::abs(beta) < 1e-15 && std::abs(alpha) > 1e-15;
    }

    /**
     * @brief Check if this is a pure Neumann BC
     */
    [[nodiscard]] bool isNeumann() const {
        return std::abs(alpha) < 1e-15 && std::abs(beta) > 1e-15;
    }
};

/**
 * @brief Function type for spatially-varying Robin coefficients
 *
 * Takes (x, y, z, time) and returns RobinData
 */
using RobinFunction = std::function<RobinData(double, double, double, double)>;

/**
 * @brief Options for Robin BC
 */
struct RobinBCOptions {
    int boundary_id{-1};                ///< Boundary marker (-1 = all boundaries)
    std::optional<int> component;       ///< Component (for vector problems)
    double tolerance{1e-15};            ///< Tolerance for zero detection
};

/**
 * @brief Robin (mixed) boundary condition
 *
 * RobinBC represents a mixed boundary condition that contributes to both
 * the system matrix and RHS. During assembly on boundary elements:
 *
 * Matrix contribution (from alpha*u term):
 *   A_local += integral(alpha * N_i * N_j) over boundary
 *
 * RHS contribution (from g term):
 *   f_local += integral(g * N_i) over boundary
 *
 * Note: The beta * du/dn term is typically absorbed into the weak form
 * and does not require explicit boundary integration for standard FE.
 *
 * Usage:
 * @code
 *   // Convective heat transfer: h*(T - T_inf) + q_n = 0
 *   // Rewritten: h*T + q_n = h*T_inf
 *   double h = 10.0;      // Heat transfer coefficient
 *   double T_inf = 300.0; // Ambient temperature
 *
 *   RobinBC robin;
 *   robin.setConstant(h, 1.0, h * T_inf);  // alpha=h, beta=1, g=h*T_inf
 *   robin.setBoundaryId(CONVECTION_BOUNDARY);
 *
 *   // During boundary assembly:
 *   RobinData data = robin.evaluate(x, y, z, time);
 *   // Add data.alpha * integral(N_i * N_j) to local matrix
 *   // Add data.g * integral(N_i) to local RHS
 * @endcode
 */
class RobinBC {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (zero Robin BC)
     */
    RobinBC();

    /**
     * @brief Construct with constant coefficients
     *
     * @param alpha Coefficient of u
     * @param beta Coefficient of du/dn
     * @param g RHS value
     * @param options BC options
     */
    RobinBC(double alpha, double beta, double g,
            const RobinBCOptions& options = {});

    /**
     * @brief Construct with spatially-varying coefficients
     *
     * @param func Function returning Robin coefficients
     * @param options BC options
     */
    RobinBC(RobinFunction func, const RobinBCOptions& options = {});

    /**
     * @brief Destructor
     */
    ~RobinBC();

    /**
     * @brief Copy constructor
     */
    RobinBC(const RobinBC& other);

    /**
     * @brief Move constructor
     */
    RobinBC(RobinBC&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    RobinBC& operator=(const RobinBC& other);

    /**
     * @brief Move assignment
     */
    RobinBC& operator=(RobinBC&& other) noexcept;

    // =========================================================================
    // Setup
    // =========================================================================

    /**
     * @brief Set constant Robin coefficients
     */
    void setConstant(double alpha, double beta, double g);

    /**
     * @brief Set spatially-varying Robin coefficients
     */
    void setFunction(RobinFunction func);

    /**
     * @brief Set boundary marker
     */
    void setBoundaryId(int boundary_id) {
        options_.boundary_id = boundary_id;
    }

    /**
     * @brief Set component (for vector problems)
     */
    void setComponent(int component) {
        options_.component = component;
    }

    /**
     * @brief Set options
     */
    void setOptions(const RobinBCOptions& options) {
        options_ = options;
    }

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * @brief Evaluate Robin coefficients at a point
     *
     * @param x, y, z Spatial coordinates
     * @param time Time (for time-dependent BCs)
     * @return Robin data at the point
     */
    [[nodiscard]] RobinData evaluate(double x, double y, double z,
                                      double time = 0.0) const;

    /**
     * @brief Evaluate alpha coefficient at a point
     */
    [[nodiscard]] double evaluateAlpha(double x, double y, double z,
                                        double time = 0.0) const;

    /**
     * @brief Evaluate g (RHS) at a point
     */
    [[nodiscard]] double evaluateG(double x, double y, double z,
                                    double time = 0.0) const;

    /**
     * @brief Evaluate at multiple points
     *
     * @param coords Point coordinates (n x 3 array)
     * @param time Time
     * @return Robin data at each point
     */
    [[nodiscard]] std::vector<RobinData> evaluateMultiple(
        std::span<const std::array<double, 3>> coords,
        double time = 0.0) const;

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get boundary marker
     */
    [[nodiscard]] int getBoundaryId() const noexcept {
        return options_.boundary_id;
    }

    /**
     * @brief Get component (if set)
     */
    [[nodiscard]] std::optional<int> getComponent() const noexcept {
        return options_.component;
    }

    /**
     * @brief Get options
     */
    [[nodiscard]] const RobinBCOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Check if this is a constant Robin BC
     */
    [[nodiscard]] bool isConstant() const noexcept {
        return !robin_func_;
    }

    /**
     * @brief Check if time-dependent
     */
    [[nodiscard]] bool isTimeDependent() const noexcept {
        return !isConstant();  // Function-based may be time-dependent
    }

    /**
     * @brief Get constant data (if constant)
     */
    [[nodiscard]] const RobinData& getConstantData() const {
        if (!isConstant()) {
            CONSTRAINT_THROW("Robin BC is not constant");
        }
        return constant_data_;
    }

    // =========================================================================
    // Assembly helpers
    // =========================================================================

    /**
     * @brief Compute matrix contribution at a quadrature point
     *
     * For a single quadrature point with shape functions N, returns
     * the contribution to the local matrix from the alpha * u term.
     *
     * @param shape_values Shape function values at the point
     * @param alpha Alpha coefficient at the point
     * @param weight Quadrature weight * Jacobian
     * @param local_matrix Output local matrix (n_dofs x n_dofs)
     */
    static void computeMatrixContribution(
        std::span<const double> shape_values,
        double alpha,
        double weight,
        std::span<double> local_matrix);

    /**
     * @brief Compute RHS contribution at a quadrature point
     *
     * @param shape_values Shape function values at the point
     * @param g RHS value at the point
     * @param weight Quadrature weight * Jacobian
     * @param local_rhs Output local RHS (n_dofs)
     */
    static void computeRhsContribution(
        std::span<const double> shape_values,
        double g,
        double weight,
        std::span<double> local_rhs);

    // =========================================================================
    // Static factory methods
    // =========================================================================

    /**
     * @brief Create convective heat transfer BC
     *
     * For heat transfer: h*(T - T_inf) = -q_n
     *
     * @param heat_transfer_coeff Heat transfer coefficient h
     * @param ambient_temperature Ambient temperature T_inf
     * @param options BC options
     * @return RobinBC
     */
    static RobinBC convective(double heat_transfer_coeff,
                               double ambient_temperature,
                               const RobinBCOptions& options = {});

    /**
     * @brief Create absorbing boundary condition
     *
     * For wave equations: du/dt + c * du/dn = 0
     *
     * @param wave_speed Wave speed c
     * @param options BC options
     * @return RobinBC
     */
    static RobinBC absorbing(double wave_speed,
                              const RobinBCOptions& options = {});

    /**
     * @brief Create elastic foundation (Winkler) BC
     *
     * For structural: k * u + sigma_n = 0
     *
     * @param spring_constant Foundation stiffness k
     * @param options BC options
     * @return RobinBC
     */
    static RobinBC elasticFoundation(double spring_constant,
                                      const RobinBCOptions& options = {});

    /**
     * @brief Create impedance boundary condition
     *
     * General form: Z * u + du/dn = g
     *
     * @param impedance Impedance Z
     * @param rhs RHS value g
     * @param options BC options
     * @return RobinBC
     */
    static RobinBC impedance(double impedance, double rhs = 0.0,
                              const RobinBCOptions& options = {});

private:
    RobinData constant_data_;
    RobinFunction robin_func_;
    RobinBCOptions options_;
};

// ============================================================================
// Vector Robin BC (for vector-valued problems)
// ============================================================================

/**
 * @brief Robin data for vector-valued problems
 */
struct VectorRobinData {
    std::array<double, 3> alpha{{1.0, 1.0, 1.0}};  ///< Component-wise alpha
    std::array<double, 3> beta{{1.0, 1.0, 1.0}};   ///< Component-wise beta
    std::array<double, 3> g{{0.0, 0.0, 0.0}};      ///< Component-wise RHS
};

/**
 * @brief Function type for vector Robin coefficients
 */
using VectorRobinFunction = std::function<VectorRobinData(double, double, double, double)>;

/**
 * @brief Vector Robin boundary condition
 *
 * For vector-valued problems where each component may have different
 * Robin coefficients.
 */
class VectorRobinBC {
public:
    /**
     * @brief Default constructor
     */
    VectorRobinBC();

    /**
     * @brief Construct with constant coefficients
     */
    VectorRobinBC(const VectorRobinData& data,
                   const RobinBCOptions& options = {});

    /**
     * @brief Construct with function
     */
    VectorRobinBC(VectorRobinFunction func,
                   const RobinBCOptions& options = {});

    /**
     * @brief Destructor
     */
    ~VectorRobinBC();

    /**
     * @brief Copy constructor
     */
    VectorRobinBC(const VectorRobinBC& other);

    /**
     * @brief Move constructor
     */
    VectorRobinBC(VectorRobinBC&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    VectorRobinBC& operator=(const VectorRobinBC& other);

    /**
     * @brief Move assignment
     */
    VectorRobinBC& operator=(VectorRobinBC&& other) noexcept;

    /**
     * @brief Evaluate at a point
     */
    [[nodiscard]] VectorRobinData evaluate(double x, double y, double z,
                                            double time = 0.0) const;

    /**
     * @brief Get boundary marker
     */
    [[nodiscard]] int getBoundaryId() const noexcept {
        return options_.boundary_id;
    }

    /**
     * @brief Set boundary marker
     */
    void setBoundaryId(int boundary_id) {
        options_.boundary_id = boundary_id;
    }

private:
    VectorRobinData constant_data_;
    VectorRobinFunction robin_func_;
    RobinBCOptions options_;
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_ROBINBC_H
