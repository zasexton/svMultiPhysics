/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_NEUMANNBC_H
#define SVMP_FE_CONSTRAINTS_NEUMANNBC_H

/**
 * @file NeumannBC.h
 * @brief Natural (Neumann) boundary conditions
 *
 * NeumannBC represents natural boundary conditions that prescribe flux/traction:
 *
 *   n . (grad u) = g   (scalar case)
 *   sigma . n = t      (vector/elasticity case)
 *
 * IMPORTANT: This is NOT an algebraic constraint. It does NOT derive from Constraint.
 * Instead, it represents a contribution to the weak form that is integrated over
 * boundary faces during assembly.
 *
 * The weak form contribution is:
 *   integral_{Gamma_N} g * v dS   (added to RHS)
 *
 * Features:
 * - Constant flux/traction values
 * - Spatially varying flux/traction via functions
 * - Time-dependent Neumann conditions
 * - Vector-valued (traction) boundary conditions
 * - Integration with surface quadrature rules
 *
 * Module boundary:
 * - This module OWNS boundary term definition and storage
 * - This module does NOT OWN assembly (use with Assembly module)
 * - This module does NOT OWN quadrature (use with Quadrature module)
 */

#include "Core/Types.h"
#include "Core/FEException.h"

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
 * @brief Function type for scalar Neumann flux
 *
 * Takes physical coordinates (x, y, z) and returns the flux value.
 */
using NeumannFluxFunction = std::function<double(double x, double y, double z)>;

/**
 * @brief Function type for vector-valued traction
 *
 * Takes coordinates and returns traction vector.
 */
using TractionFunction =
    std::function<std::array<double, 3>(double x, double y, double z)>;

/**
 * @brief Function type for time-dependent Neumann flux
 */
using TimeDependentNeumannFunction =
    std::function<double(double x, double y, double z, double t)>;

/**
 * @brief Function type for time-dependent traction
 */
using TimeDependentTractionFunction =
    std::function<std::array<double, 3>(double x, double y, double z, double t)>;

/**
 * @brief Specification for a Neumann boundary condition
 *
 * This struct holds all information needed to apply a Neumann BC
 * during assembly. It can be:
 * - Constant scalar flux
 * - Constant vector traction
 * - Spatially varying (via function)
 * - Time-dependent
 */
struct NeumannSpec {
    // Boundary identification
    int boundary_id{-1};                    ///< Mesh boundary ID (face tag)
    std::string boundary_name;              ///< Optional boundary name

    // Type indicators
    bool is_vector_valued{false};           ///< True for traction, false for scalar flux
    bool is_time_dependent{false};          ///< True if value changes with time

    // Constant values (if not using functions)
    double constant_flux{0.0};              ///< Constant scalar flux value
    std::array<double, 3> constant_traction{{0.0, 0.0, 0.0}}; ///< Constant traction vector

    // Function-based values
    std::optional<NeumannFluxFunction> flux_function;
    std::optional<TractionFunction> traction_function;
    std::optional<TimeDependentNeumannFunction> time_flux_function;
    std::optional<TimeDependentTractionFunction> time_traction_function;

    /**
     * @brief Evaluate scalar flux at a point
     */
    [[nodiscard]] double evaluateFlux(double x, double y, double z, double t = 0.0) const {
        if (time_flux_function) {
            return (*time_flux_function)(x, y, z, t);
        }
        if (flux_function) {
            return (*flux_function)(x, y, z);
        }
        return constant_flux;
    }

    /**
     * @brief Evaluate traction at a point
     */
    [[nodiscard]] std::array<double, 3> evaluateTraction(double x, double y, double z,
                                                          double t = 0.0) const {
        if (time_traction_function) {
            return (*time_traction_function)(x, y, z, t);
        }
        if (traction_function) {
            return (*traction_function)(x, y, z);
        }
        return constant_traction;
    }
};

/**
 * @brief Natural (Neumann) boundary condition
 *
 * NeumannBC represents flux or traction boundary conditions that contribute
 * to the RHS of the weak form. Unlike DirichletBC, this is NOT an algebraic
 * constraint and does NOT modify the DOF relationships.
 *
 * Usage (with assembly):
 * @code
 *   // Create Neumann BC
 *   NeumannBC bc(boundary_id, flux_value);
 *
 *   // During assembly over boundary faces:
 *   for (auto& face : boundary_faces) {
 *       if (face.boundary_id == bc.getBoundaryId()) {
 *           // Integrate: integral_{face} flux * v dS
 *           for (int q = 0; q < n_quad_points; ++q) {
 *               auto [x, y, z] = quad_points[q];
 *               double flux = bc.evaluateFlux(x, y, z, time);
 *               for (int i = 0; i < n_shape_functions; ++i) {
 *                   rhs[i] += flux * N[i] * weight[q] * jacobian_det[q];
 *               }
 *           }
 *       }
 *   }
 * @endcode
 */
class NeumannBC {
public:
    // =========================================================================
    // Construction - Scalar flux
    // =========================================================================

    /**
     * @brief Construct with constant scalar flux on a boundary
     *
     * @param boundary_id Mesh boundary identifier
     * @param flux Constant flux value
     */
    NeumannBC(int boundary_id, double flux);

    /**
     * @brief Construct with spatially varying flux
     *
     * @param boundary_id Mesh boundary identifier
     * @param flux_func Function returning flux at coordinates
     */
    NeumannBC(int boundary_id, NeumannFluxFunction flux_func);

    /**
     * @brief Construct with time-dependent flux
     *
     * @param boundary_id Mesh boundary identifier
     * @param flux_func Function returning flux at coordinates and time
     */
    NeumannBC(int boundary_id, TimeDependentNeumannFunction flux_func);

    // =========================================================================
    // Construction - Vector traction
    // =========================================================================

    /**
     * @brief Construct with constant traction vector
     *
     * @param boundary_id Mesh boundary identifier
     * @param traction Traction vector (tx, ty, tz)
     */
    NeumannBC(int boundary_id, std::array<double, 3> traction);

    /**
     * @brief Construct with spatially varying traction
     *
     * @param boundary_id Mesh boundary identifier
     * @param traction_func Function returning traction at coordinates
     */
    NeumannBC(int boundary_id, TractionFunction traction_func);

    /**
     * @brief Construct with time-dependent traction
     *
     * @param boundary_id Mesh boundary identifier
     * @param traction_func Function returning traction at coordinates and time
     */
    NeumannBC(int boundary_id, TimeDependentTractionFunction traction_func);

    /**
     * @brief Construct from NeumannSpec
     */
    explicit NeumannBC(NeumannSpec spec);

    /**
     * @brief Default destructor
     */
    ~NeumannBC() = default;

    /**
     * @brief Copy constructor
     */
    NeumannBC(const NeumannBC& other);

    /**
     * @brief Move constructor
     */
    NeumannBC(NeumannBC&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    NeumannBC& operator=(const NeumannBC& other);

    /**
     * @brief Move assignment
     */
    NeumannBC& operator=(NeumannBC&& other) noexcept;

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * @brief Evaluate scalar flux at a point
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param time Current time (for time-dependent BCs)
     * @return Flux value
     */
    [[nodiscard]] double evaluateFlux(double x, double y, double z,
                                       double time = 0.0) const {
        return spec_.evaluateFlux(x, y, z, time);
    }

    /**
     * @brief Evaluate traction at a point
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param time Current time (for time-dependent BCs)
     * @return Traction vector
     */
    [[nodiscard]] std::array<double, 3> evaluateTraction(double x, double y, double z,
                                                          double time = 0.0) const {
        return spec_.evaluateTraction(x, y, z, time);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get boundary ID
     */
    [[nodiscard]] int getBoundaryId() const noexcept { return spec_.boundary_id; }

    /**
     * @brief Get boundary name (if set)
     */
    [[nodiscard]] const std::string& getBoundaryName() const noexcept {
        return spec_.boundary_name;
    }

    /**
     * @brief Check if this is a vector-valued (traction) BC
     */
    [[nodiscard]] bool isVectorValued() const noexcept { return spec_.is_vector_valued; }

    /**
     * @brief Check if this BC is time-dependent
     */
    [[nodiscard]] bool isTimeDependent() const noexcept { return spec_.is_time_dependent; }

    /**
     * @brief Get constant flux value (for constant BCs)
     */
    [[nodiscard]] double getConstantFlux() const noexcept { return spec_.constant_flux; }

    /**
     * @brief Get constant traction value (for constant BCs)
     */
    [[nodiscard]] const std::array<double, 3>& getConstantTraction() const noexcept {
        return spec_.constant_traction;
    }

    /**
     * @brief Get the underlying specification
     */
    [[nodiscard]] const NeumannSpec& getSpec() const noexcept { return spec_; }

    // =========================================================================
    // Modification
    // =========================================================================

    /**
     * @brief Set boundary name
     */
    void setBoundaryName(std::string name) { spec_.boundary_name = std::move(name); }

    /**
     * @brief Update constant flux value
     */
    void setFlux(double flux) {
        spec_.is_vector_valued = false;
        spec_.is_time_dependent = false;
        spec_.constant_flux = flux;
        spec_.flux_function.reset();
        spec_.time_flux_function.reset();
        spec_.traction_function.reset();
        spec_.time_traction_function.reset();
    }

    /**
     * @brief Update constant traction value
     */
    void setTraction(std::array<double, 3> traction) {
        spec_.is_vector_valued = true;
        spec_.is_time_dependent = false;
        spec_.constant_traction = traction;
        spec_.traction_function.reset();
        spec_.time_traction_function.reset();
        spec_.flux_function.reset();
        spec_.time_flux_function.reset();
    }

    // =========================================================================
    // Clone
    // =========================================================================

    /**
     * @brief Clone this Neumann BC
     */
    [[nodiscard]] std::unique_ptr<NeumannBC> clone() const {
        return std::make_unique<NeumannBC>(*this);
    }

private:
    NeumannSpec spec_;
};

/**
 * @brief Collection of Neumann boundary conditions
 *
 * Manages multiple Neumann BCs and provides lookup by boundary ID.
 */
class NeumannBCCollection {
public:
    /**
     * @brief Default constructor
     */
    NeumannBCCollection() = default;

    /**
     * @brief Add a Neumann BC to the collection
     */
    void add(NeumannBC bc) {
        bcs_.push_back(std::move(bc));
    }

    /**
     * @brief Add a Neumann BC (by unique_ptr)
     */
    void add(std::unique_ptr<NeumannBC> bc) {
        bcs_.push_back(std::move(*bc));
    }

    /**
     * @brief Find Neumann BC for a boundary ID
     *
     * @param boundary_id The boundary to look up
     * @return Pointer to NeumannBC if found, nullptr otherwise
     */
    [[nodiscard]] const NeumannBC* find(int boundary_id) const {
        for (const auto& bc : bcs_) {
            if (bc.getBoundaryId() == boundary_id) {
                return &bc;
            }
        }
        return nullptr;
    }

    /**
     * @brief Check if there's a Neumann BC for a boundary ID
     */
    [[nodiscard]] bool hasBC(int boundary_id) const {
        return find(boundary_id) != nullptr;
    }

    /**
     * @brief Get number of BCs
     */
    [[nodiscard]] std::size_t size() const noexcept { return bcs_.size(); }

    /**
     * @brief Check if empty
     */
    [[nodiscard]] bool empty() const noexcept { return bcs_.empty(); }

    /**
     * @brief Clear all BCs
     */
    void clear() { bcs_.clear(); }

    /**
     * @brief Get iterator to first BC
     */
    [[nodiscard]] auto begin() const { return bcs_.begin(); }

    /**
     * @brief Get iterator past last BC
     */
    [[nodiscard]] auto end() const { return bcs_.end(); }

private:
    std::vector<NeumannBC> bcs_;
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_NEUMANNBC_H
