#ifndef SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_MANAGER_H
#define SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_MANAGER_H

/**
 * @file CoupledBoundaryManager.h
 * @brief Orchestration for coupled boundary conditions (boundary functionals + aux state)
 *
 * @deprecated This class is the legacy coupled-boundary-specific path.
 * New formulations should use the generalized AuxiliaryState subsystem:
 *
 * - Define models via `AuxiliaryModelBuilder` (Systems/AuxiliaryModelBuilder.h)
 * - Deploy via `use(model)` (Systems/AuxiliaryBindings.h)
 * - Register inputs via `AuxiliaryInputRegistry` (Systems/AuxiliaryInputRegistry.h)
 * - Access via `FESystem::auxiliaryStateManager()` and `auxiliaryInputRegistry()`
 *
 * ## Migration paths
 *
 * ### Lumped boundary models (e.g., RCR, Windkessel)
 *
 * Old: `coupled.addAuxiliaryState(reg)` with scalar ODE RHS
 * New: Define via `AuxiliaryModelBuilder`, deploy with `use(model).scope(Global)`
 *      and bind boundary-reduction inputs.
 *
 * ### Per-boundary-entity models
 *
 * Old: Not supported in the legacy API (scalar-only)
 * New: Deploy with `use(model).scope(BoundaryEntity).region({BoundarySet, "outlet"})`
 *
 * ### Boundary functionals
 *
 * Old: `coupled.addBoundaryFunctional(...)` + `boundaryIntegralValue("Q")`
 * New: Register via `AuxiliaryInputRegistry` with `BoundaryReduction` producer,
 *      reference via `AuxiliaryInput("Q")` in formulation expressions.
 *
 * This class remains functional for backward compatibility but internally
 * owns its own `AuxiliaryState` and `BoundaryFunctionalResults`.  Future
 * releases will thin this to a forwarding shim over the generalized subsystem.
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Core/ParameterValue.h"

#include "Assembly/FunctionalAssembler.h"  // for FieldSolutionBinding

#include "Constraints/CoupledBCContext.h"
#include "Constraints/CoupledNeumannBC.h"
#include "Constraints/CoupledRobinBC.h"
#include "Forms/BoundaryFunctional.h"
#include "Systems/AuxiliaryState.h"
#include "Systems/SystemState.h"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class FunctionalKernel;
}

namespace systems {

class FESystem;

class CoupledBoundaryManager {
public:
    CoupledBoundaryManager(FESystem& system, FieldId primary_field);
    ~CoupledBoundaryManager();

    CoupledBoundaryManager(CoupledBoundaryManager&&) noexcept = delete;
    CoupledBoundaryManager& operator=(CoupledBoundaryManager&&) noexcept = delete;
    CoupledBoundaryManager(const CoupledBoundaryManager&) = delete;
    CoupledBoundaryManager& operator=(const CoupledBoundaryManager&) = delete;

    [[nodiscard]] FieldId primaryField() const noexcept { return primary_field_; }

    // ---------------------------------------------------------------------
    // Registration
    // ---------------------------------------------------------------------

    void addBoundaryFunctional(forms::BoundaryFunctional functional);
    void addCoupledNeumannBC(constraints::CoupledNeumannBC bc);
    void addCoupledRobinBC(constraints::CoupledRobinBC bc);
    /// @deprecated Use AuxiliaryModelBuilder + use(model) instead.
    /// This method registers a scalar ODE auxiliary state variable in the
    /// legacy flat AuxiliaryState buffer.  New formulations should define
    /// auxiliary models via AuxiliaryModelBuilder and deploy them via
    /// use(model).scope(Global) with boundary-reduction input bindings.
    void addAuxiliaryState(AuxiliaryStateRegistration registration);

    /**
     * @brief Compiler/JIT options for boundary functional kernels
     *
     * By default, coupled boundary integrals use the interpreter functional kernels.
     * When `options.jit.enable=true`, boundary functionals may be JIT-compiled with
     * interpreter fallback.
     */
    void setCompilerOptions(forms::SymbolicOptions options);

    // ---------------------------------------------------------------------
    // Parameter contract integration
    // ---------------------------------------------------------------------

    /**
     * @brief Parameter requirements implied by stored expressions (boundary functionals + ODEs)
     *
     * This allows Systems to register parameter contracts for coupled BC
     * infrastructure alongside regular AssemblyKernel requirements.
     */
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const;

    /**
     * @brief Resolve ParameterSymbol terminals to ParameterRef(slot) for JIT-friendly evaluation
     *
     * Must be called after Systems has finalized the ParameterRegistry slot layout.
     */
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param);

    // ---------------------------------------------------------------------
    // Assembly-time update
    // ---------------------------------------------------------------------

    /**
     * @brief Update integrals + auxiliary state for an upcoming assembly call
     *
     * Safe to call multiple times (e.g., during nonlinear iterations):
     * the auxiliary state work values are reset to the committed state each call.
     */
    void prepareForAssembly(const SystemStateView& state);

    // ---------------------------------------------------------------------
    // Time-step lifecycle
    // ---------------------------------------------------------------------

    void beginTimeStep();
    void commitTimeStep();

    // ---------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------

    /// @deprecated For new code, use FESystem::auxiliaryInputRegistry() instead.
    [[nodiscard]] const forms::BoundaryFunctionalResults& integrals() const noexcept { return integrals_; }
    /// @deprecated For new code, use FESystem::auxiliaryStateManager() instead.
    [[nodiscard]] const AuxiliaryState& auxiliaryState() const noexcept { return aux_state_; }
    /// @deprecated For new code, use FESystem::auxiliaryStateManager() instead.
    [[nodiscard]] AuxiliaryState& auxiliaryState() noexcept { return aux_state_; }

    [[nodiscard]] const constraints::CoupledBCContext& context() const noexcept { return ctx_; }
    [[nodiscard]] const constraints::CoupledBCContext* contextPtr() const noexcept { return &ctx_; }

    struct RegisteredBoundaryFunctional {
        forms::BoundaryFunctional def{};
        std::uint32_t slot{0u};
    };

    /**
     * @brief Return registered boundary functionals with their stable slot indices
     *
     * The returned `slot` indices correspond to the ordering of
     * `integrals().all()` and to `BoundaryIntegralRef(slot)` terminals used in
     * compiled FE/Forms kernels.
     */
    [[nodiscard]] std::vector<RegisteredBoundaryFunctional> registeredBoundaryFunctionals() const;

    /**
     * @brief Return the set of boundary markers that have coupled BC residual terms registered
     *
     * This includes markers referenced by CoupledNeumannBC and CoupledRobinBC registrations.
     * It is useful for building sparsity patterns for coupled-BC Jacobians without unnecessarily
     * densifying unrelated boundary terms.
     */
    [[nodiscard]] std::vector<int> coupledBoundaryMarkers() const;

    /**
     * @brief Geometric measure (area/length) of a boundary marker
     *
     * Cached per boundary marker. Used for `BoundaryFunctional::Reduction::Average`
     * and for coupled Jacobian computations.
     */
    Real boundaryMeasure(int boundary_marker, const SystemStateView& state);


    /**
     * @brief Compute auxiliary state values for a given set of boundary integrals
     *
     * This mirrors the ODE-advance logic in prepareForAssembly(), but does not
     * modify the stored auxiliary state. It is intended for Jacobian chain-rule
     * computations where the residual is differentiated with respect to
     * boundary-functionals while accounting for their induced auxiliary-state
     * changes.
     */
    [[nodiscard]] std::vector<Real> computeAuxiliaryStateForIntegrals(std::span<const Real> integrals_override,
                                                                      const SystemStateView& state) const;

    /**
     * @brief Compute sensitivities d(aux)/d(integrals) for a given set of boundary integrals
     *
     * Returns a row-major matrix with shape (aux_size x integrals_override.size()).
     * This mirrors the auxiliary advance logic used in prepareForAssembly(), including
     * the sequential update order of registered auxiliary variables.
     */
    [[nodiscard]] std::vector<Real> computeAuxiliarySensitivityForIntegrals(std::span<const Real> integrals_override,
                                                                            const SystemStateView& state) const;

    /**
     * @brief Register a secondary field for multi-field boundary functional evaluation
     *
     * When boundary functionals depend on fields other than the primary field,
     * register those fields here so the evaluator can bind their solution data.
     */
    void registerSecondaryField(const assembly::FieldSolutionBinding& binding);

    /**
     * @brief Set total DOFs per node for interleaved multi-field DOF layout
     */
    void setDofPerNode(int dof_per_node) noexcept;

private:
    struct CompiledFunctional {
        forms::BoundaryFunctional def{};
        std::shared_ptr<assembly::FunctionalKernel> kernel{};
    };

    void requireSetup() const;
    void compileFunctionalIfNeeded(const forms::BoundaryFunctional& functional);
    Real evaluateFunctional(const CompiledFunctional& entry, const SystemStateView& state);

    FESystem& system_;
    FieldId primary_field_{INVALID_FIELD_ID};

    std::vector<CompiledFunctional> functionals_{};
    std::unordered_map<std::string, std::size_t> name_to_functional_{};
    std::unordered_map<int, Real> boundary_measure_cache_{};

    std::vector<constraints::CoupledNeumannBC> coupled_neumann_{};
    std::vector<constraints::CoupledRobinBC> coupled_robin_{};
    std::vector<AuxiliaryStateRegistration> aux_registrations_{};

    forms::BoundaryFunctionalResults integrals_{};
    AuxiliaryState aux_state_{};
    constraints::CoupledBCContext ctx_{integrals_, aux_state_, 0.0, 0.0};

    forms::SymbolicOptions compiler_options_{};

    std::vector<assembly::FieldSolutionBinding> secondary_fields_{};
    int dof_per_node_{0};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_MANAGER_H
