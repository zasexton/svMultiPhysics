/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_BOUNDARY_REDUCTION_SERVICE_H
#define SVMP_FE_SYSTEMS_BOUNDARY_REDUCTION_SERVICE_H

/**
 * @file BoundaryReductionService.h
 * @brief Physics-agnostic boundary functional evaluation service.
 *
 * This service encapsulates the registration, compilation, and evaluation
 * of boundary functionals (true quadrature-weighted boundary integrals)
 * and their associated boundary measures.
 *
 * It is used by both the legacy CoupledBoundaryManager and the generalized
 * AuxiliaryInputRegistry to evaluate boundary-integral inputs.
 *
 * ## Design principles
 *
 * - All names, types, and comments are physics-agnostic.
 * - This is a neutral FE infrastructure service, not tied to any specific
 *   physics formulation.
 * - Supports Sum and Average reductions; extensible to Min/Max.
 * - MPI-safe: global reductions performed when MPI is initialized.
 */

#include "Core/Types.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEException.h"

#include "Assembly/FunctionalAssembler.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"  // for SymbolicOptions
#include "Spaces/FunctionSpace.h"
#include "Systems/SystemState.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class FunctionalKernel;
}

namespace systems {

class FESystem;

/**
 * @brief Physics-agnostic service for registering, compiling, and evaluating
 *        boundary functionals (true quadrature-weighted boundary integrals).
 *
 * This service is the single authoritative path for boundary-integral
 * evaluation in the FE library.  It supports:
 *
 * - Registration of named boundary functionals with integrand, marker,
 *   and reduction mode.
 * - Lazy compilation of functional kernels (interpreter or JIT).
 * - Evaluation of boundary integrals with MPI global reduction.
 * - Boundary measure (area/length) computation and caching.
 * - Multi-field secondary field bindings for integrands that reference
 *   fields beyond the primary field.
 */
class BoundaryReductionService {
public:
    BoundaryReductionService(FESystem& system, FieldId primary_field);
    ~BoundaryReductionService();

    BoundaryReductionService(const BoundaryReductionService&) = delete;
    BoundaryReductionService& operator=(const BoundaryReductionService&) = delete;
    BoundaryReductionService(BoundaryReductionService&&) noexcept = delete;
    BoundaryReductionService& operator=(BoundaryReductionService&&) noexcept = delete;

    // -----------------------------------------------------------------
    //  Registration
    // -----------------------------------------------------------------

    /**
     * @brief Register a boundary functional.
     *
     * The functional must have a non-empty name, valid integrand, and
     * boundary_marker >= 0.  Duplicate registrations with identical
     * properties are silently accepted; conflicting duplicates throw.
     *
     * @param functional The boundary functional definition.
     */
    void addBoundaryFunctional(forms::BoundaryFunctional functional);

    /**
     * @brief Check whether a functional with the given name is registered.
     */
    [[nodiscard]] bool hasFunctional(std::string_view name) const noexcept;

    /**
     * @brief Return the number of registered boundary functionals.
     */
    [[nodiscard]] std::size_t functionalCount() const noexcept { return functionals_.size(); }

    // -----------------------------------------------------------------
    //  Compilation options
    // -----------------------------------------------------------------

    /**
     * @brief Set compiler/JIT options for boundary functional kernels.
     *
     * Invalidates any already-compiled kernels so they can be recompiled.
     */
    void setCompilerOptions(const forms::SymbolicOptions& options);

    // -----------------------------------------------------------------
    //  Multi-field support
    // -----------------------------------------------------------------

    /**
     * @brief Register a secondary field for multi-field boundary evaluation.
     *
     * When boundary integrands depend on fields other than the primary field,
     * register those bindings here so the evaluator can bind their data.
     */
    void registerSecondaryField(const assembly::FieldSolutionBinding& binding);

    /**
     * @brief Set total DOFs per node for interleaved multi-field DOF layout.
     */
    void setDofPerNode(int dof_per_node) noexcept;

    // -----------------------------------------------------------------
    //  Evaluation
    // -----------------------------------------------------------------

    /**
     * @brief Compile and evaluate a named boundary functional.
     *
     * Compiles the functional kernel if not already compiled, then
     * evaluates the boundary integral with MPI reduction and applies
     * the reduction mode (Sum, Average).
     *
     * @param name  Name of the registered functional.
     * @param state System state providing solution, time, parameters.
     * @return The reduced scalar value.
     */
    [[nodiscard]] Real evaluateFunctional(std::string_view name, const SystemStateView& state);

    /**
     * @brief Evaluate a domain functional over an explicit cell subset.
     *
     * The named functional must be a domain functional.  The supplied cells are
     * mesh-global ids; the FunctionalAssembler will ignore cells not owned by
     * the current rank, and this service performs the same MPI sum reduction as
     * evaluateFunctional().
     */
    [[nodiscard]] Real evaluateFunctionalOverCells(
        std::string_view name,
        std::span<const GlobalIndex> cell_ids,
        const SystemStateView& state);

    /**
     * @brief Evaluate all registered boundary functionals.
     *
     * @param state System state.
     * @return Vector of results in registration order.
     */
    [[nodiscard]] std::vector<Real> evaluateAll(const SystemStateView& state);

    /**
     * @brief Compute and cache the geometric measure (area/length) of a boundary.
     *
     * @param boundary_marker The boundary label.
     * @param state System state (needed for assembler configuration).
     * @return The global boundary measure.
     */
    [[nodiscard]] Real boundaryMeasure(int boundary_marker, const SystemStateView& state);

    // -----------------------------------------------------------------
    //  Sensitivity (for monolithic coupling)
    // -----------------------------------------------------------------

    /**
     * @brief Compute the gradient dQ/du for a named boundary functional.
     *
     * Returns a sparse vector of (DOF_index, value) pairs representing
     * the derivative of the boundary integral with respect to each DOF.
     * Used for monolithic auxiliary-field coupling: the mixed Jacobian
     * contribution is dF_aux/dQ * dQ/du.
     *
     * @param name  Name of the registered functional.
     * @param state System state providing solution, time, parameters.
     * @return Vector of (global_dof_index, derivative_value) pairs.
     */
    struct SensitivityEntry {
        GlobalIndex dof{0};
        Real value{0.0};
    };

    [[nodiscard]] std::vector<SensitivityEntry> evaluateFunctionalGradient(
        std::string_view name,
        const SystemStateView& state,
        bool apply_constraints = true);

    /**
     * @brief Gradient w.r.t. a specific field (for multi-field integrands).
     *
     * Transforms DiscreteField/StateField nodes matching `target_field` to
     * TrialFunction, leaving other fields as constants.
     */
    [[nodiscard]] std::vector<SensitivityEntry> evaluateFunctionalGradient(
        std::string_view name,
        FieldId target_field,
        const SystemStateView& state,
        bool apply_constraints = true);

    /**
     * @brief Gradient of a domain functional over an explicit cell subset.
     *
     * Used by topology-Region entity-local monolithic auxiliary inputs, where
     * each auxiliary entity owns a different cell subset under one registered
     * functional name.
     */
    [[nodiscard]] std::vector<SensitivityEntry> evaluateFunctionalGradientOverCells(
        std::string_view name,
        FieldId target_field,
        std::span<const GlobalIndex> cell_ids,
        const SystemStateView& state,
        bool apply_constraints = true);

    // -----------------------------------------------------------------
    //  Accessors
    // -----------------------------------------------------------------

    /**
     * @brief Return the definition of a registered functional by name.
     */
    [[nodiscard]] const forms::BoundaryFunctional& functionalDef(std::string_view name) const;

    /**
     * @brief Return all registered functional definitions in registration order.
     */
    [[nodiscard]] std::vector<forms::BoundaryFunctional> allFunctionalDefs() const;

    /**
     * @brief Return the primary field ID.
     */
    [[nodiscard]] FieldId primaryField() const noexcept { return primary_field_; }

private:
    struct CompiledFunctional {
        forms::BoundaryFunctional def{};
        std::shared_ptr<assembly::FunctionalKernel> kernel{};
    };

    void compileFunctionalIfNeeded(CompiledFunctional& entry);
    void configureAssembler(assembly::FunctionalAssembler& assembler,
                            const SystemStateView& state,
                            bool bind_solution) const;
    Real evaluateFunctionalEntry(CompiledFunctional& entry, const SystemStateView& state);

    FESystem& system_;
    FieldId primary_field_{INVALID_FIELD_ID};

    std::vector<CompiledFunctional> functionals_{};
    std::unordered_map<std::string, std::size_t> name_to_functional_{};
    std::unordered_map<int, Real> boundary_measure_cache_{};

    forms::SymbolicOptions compiler_options_{};

    std::vector<assembly::FieldSolutionBinding> secondary_fields_{};
    int dof_per_node_{0};

    /// Default geometry space for field-free integrands (GEOMETRY_FIELD_ID).
    mutable std::shared_ptr<spaces::FunctionSpace> geometry_space_{};
    /// Scratch buffer for interleaved solution reordering (multi-field).
    mutable std::vector<Real> interleaved_sol_{};
    [[nodiscard]] const spaces::FunctionSpace& geometrySpace() const;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_BOUNDARY_REDUCTION_SERVICE_H
