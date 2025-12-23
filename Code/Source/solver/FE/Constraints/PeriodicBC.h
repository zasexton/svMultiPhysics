/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_PERIODICBC_H
#define SVMP_FE_CONSTRAINTS_PERIODICBC_H

/**
 * @file PeriodicBC.h
 * @brief Periodic boundary constraints
 *
 * PeriodicBC implements periodic boundary conditions that relate DOFs on
 * opposite boundaries. For a periodic domain:
 *
 *   u(x + L) = u(x)   (standard periodicity)
 *   u(x + L) = -u(x)  (anti-periodicity)
 *
 * The constraint form is: u_slave = T * u_master
 * where T is an optional transformation (identity for scalar, rotation for vectors).
 *
 * Features:
 * - Coordinate-based DOF matching
 * - Support for translation, rotation, and general transformations
 * - Anti-periodic boundary conditions
 * - Component-wise periodicity for vector fields
 *
 * This is an ALGEBRAIC constraint - it defines DOF relationships,
 * not contributions to the weak form.
 */

#include "Constraint.h"
#include "AffineConstraints.h"
#include "Core/Types.h"

#include <vector>
#include <span>
#include <functional>
#include <optional>
#include <array>
#include <memory>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief Transformation type for periodic boundaries
 */
enum class PeriodicTransformType {
    Translation,     ///< Simple translation (u_slave = u_master)
    Rotation,        ///< Rotation (for vector fields on rotational periodicity)
    Reflection,      ///< Reflection (for anti-periodic BCs)
    General          ///< User-defined transformation matrix
};

/**
 * @brief Options for periodic boundary conditions
 */
struct PeriodicBCOptions {
    double matching_tolerance{1e-10};    ///< Tolerance for coordinate matching
    bool anti_periodic{false};           ///< Anti-periodic (sign flip)
    ComponentMask component_mask{};       ///< Which components to make periodic
    PeriodicTransformType transform_type{PeriodicTransformType::Translation};
};

/**
 * @brief Periodic pair: slave DOF -> master DOF mapping
 */
struct PeriodicPair {
    GlobalIndex slave_dof;
    GlobalIndex master_dof;
    double weight{1.0};  ///< Usually 1.0 (or -1.0 for anti-periodic)
};

/**
 * @brief Periodic boundary condition constraint
 *
 * PeriodicBC establishes constraints between DOFs on matching periodic
 * boundaries. The slave boundary DOFs are expressed in terms of master
 * boundary DOFs.
 *
 * Usage modes:
 *
 * 1. **Direct pair specification:**
 *    @code
 *    PeriodicBC bc(slave_dofs, master_dofs);
 *    @endcode
 *
 * 2. **Coordinate-based matching:**
 *    @code
 *    PeriodicBC bc(slave_dofs, slave_coords, master_dofs, master_coords,
 *                  translation_vector, options);
 *    @endcode
 *
 * 3. **With rotation (for cylindrical/angular periodicity):**
 *    @code
 *    PeriodicBC bc(slave_dofs, slave_coords, master_dofs, master_coords,
 *                  rotation_matrix, options);
 *    @endcode
 */
class PeriodicBC : public Constraint {
public:
    // =========================================================================
    // Construction - Direct pair specification
    // =========================================================================

    /**
     * @brief Default constructor - creates empty periodic BC for later population
     */
    PeriodicBC() = default;

    /**
     * @brief Construct with explicit DOF pairs
     *
     * @param slave_dofs Slave (constrained) DOF indices
     * @param master_dofs Corresponding master DOF indices
     */
    PeriodicBC(std::vector<GlobalIndex> slave_dofs,
               std::vector<GlobalIndex> master_dofs);

    /**
     * @brief Construct with pairs and options
     */
    PeriodicBC(std::vector<GlobalIndex> slave_dofs,
               std::vector<GlobalIndex> master_dofs,
               const PeriodicBCOptions& options);

    /**
     * @brief Construct from periodic pairs
     */
    explicit PeriodicBC(std::vector<PeriodicPair> pairs);

    // =========================================================================
    // Construction - Coordinate-based matching
    // =========================================================================

    /**
     * @brief Construct with coordinate matching and translation
     *
     * DOFs are matched based on coordinates after applying translation.
     *
     * @param slave_dofs DOFs on slave boundary
     * @param slave_coords Coordinates of slave DOFs
     * @param master_dofs DOFs on master boundary
     * @param master_coords Coordinates of master DOFs
     * @param translation Vector from slave to master coordinates
     * @param options BC options
     */
    PeriodicBC(std::vector<GlobalIndex> slave_dofs,
               std::vector<std::array<double, 3>> slave_coords,
               std::vector<GlobalIndex> master_dofs,
               std::vector<std::array<double, 3>> master_coords,
               std::array<double, 3> translation,
               const PeriodicBCOptions& options = {});

    /**
     * @brief Construct with general coordinate transformation
     *
     * @param slave_dofs DOFs on slave boundary
     * @param slave_coords Coordinates of slave DOFs
     * @param master_dofs DOFs on master boundary
     * @param master_coords Coordinates of master DOFs
     * @param transform Function mapping slave coords to expected master coords
     * @param options BC options
     */
    PeriodicBC(std::vector<GlobalIndex> slave_dofs,
               std::vector<std::array<double, 3>> slave_coords,
               std::vector<GlobalIndex> master_dofs,
               std::vector<std::array<double, 3>> master_coords,
               std::function<std::array<double, 3>(std::array<double, 3>)> transform,
               const PeriodicBCOptions& options = {});

    /**
     * @brief Destructor
     */
    ~PeriodicBC() override = default;

    /**
     * @brief Copy constructor
     */
    PeriodicBC(const PeriodicBC& other);

    /**
     * @brief Move constructor
     */
    PeriodicBC(PeriodicBC&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    PeriodicBC& operator=(const PeriodicBC& other);

    /**
     * @brief Move assignment
     */
    PeriodicBC& operator=(PeriodicBC&& other) noexcept;

    // =========================================================================
    // Constraint interface
    // =========================================================================

    /**
     * @brief Apply periodic constraints to AffineConstraints
     */
    void apply(AffineConstraints& constraints) const override;

    /**
     * @brief Get constraint type
     */
    [[nodiscard]] ConstraintType getType() const noexcept override {
        return ConstraintType::Periodic;
    }

    /**
     * @brief Get constraint information
     */
    [[nodiscard]] ConstraintInfo getInfo() const override;

    /**
     * @brief Clone this constraint
     */
    [[nodiscard]] std::unique_ptr<Constraint> clone() const override {
        return std::make_unique<PeriodicBC>(*this);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get periodic pairs
     */
    [[nodiscard]] const std::vector<PeriodicPair>& getPairs() const noexcept {
        return pairs_;
    }

    /**
     * @brief Get number of periodic pairs
     */
    [[nodiscard]] std::size_t numPairs() const noexcept {
        return pairs_.size();
    }

    /**
     * @brief Get options
     */
    [[nodiscard]] const PeriodicBCOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Check if anti-periodic
     */
    [[nodiscard]] bool isAntiPeriodic() const noexcept {
        return options_.anti_periodic;
    }

    // =========================================================================
    // Modification
    // =========================================================================

    /**
     * @brief Add a periodic pair
     */
    void addPair(GlobalIndex slave_dof, GlobalIndex master_dof, double weight = 1.0);

    /**
     * @brief Add multiple pairs
     */
    void addPairs(std::span<const GlobalIndex> slave_dofs,
                  std::span<const GlobalIndex> master_dofs);

    // =========================================================================
    // Factory methods
    // =========================================================================

    /**
     * @brief Create periodic BC for x-direction periodicity
     *
     * @param left_dofs DOFs on left boundary (x = x_min)
     * @param left_coords Coordinates of left DOFs
     * @param right_dofs DOFs on right boundary (x = x_max)
     * @param right_coords Coordinates of right DOFs
     * @param domain_length Length in x-direction
     * @return PeriodicBC constraint
     */
    static PeriodicBC xPeriodic(
        std::vector<GlobalIndex> left_dofs,
        std::vector<std::array<double, 3>> left_coords,
        std::vector<GlobalIndex> right_dofs,
        std::vector<std::array<double, 3>> right_coords,
        double domain_length);

    /**
     * @brief Create periodic BC for y-direction periodicity
     */
    static PeriodicBC yPeriodic(
        std::vector<GlobalIndex> bottom_dofs,
        std::vector<std::array<double, 3>> bottom_coords,
        std::vector<GlobalIndex> top_dofs,
        std::vector<std::array<double, 3>> top_coords,
        double domain_length);

    /**
     * @brief Create periodic BC for z-direction periodicity
     */
    static PeriodicBC zPeriodic(
        std::vector<GlobalIndex> back_dofs,
        std::vector<std::array<double, 3>> back_coords,
        std::vector<GlobalIndex> front_dofs,
        std::vector<std::array<double, 3>> front_coords,
        double domain_length);

private:
    // Periodic pairs
    std::vector<PeriodicPair> pairs_;

    // Options
    PeriodicBCOptions options_;

    // Helper to match coordinates
    void matchCoordinates(
        std::span<const GlobalIndex> slave_dofs,
        std::span<const std::array<double, 3>> slave_coords,
        std::span<const GlobalIndex> master_dofs,
        std::span<const std::array<double, 3>> master_coords,
        const std::function<std::array<double, 3>(std::array<double, 3>)>& transform);
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_PERIODICBC_H
